// server.js - OpenAI-compatible proxy to NVIDIA NIM for Janitor AI

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false;

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = true;

// Default fallback model if nothing matches
const DEFAULT_FALLBACK_MODEL = 'meta/llama-3.1-70b-instruct';

// Model mapping
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1-terminus',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',
  'deepseek-r1-0528': 'deepseek-ai/deepseek-r1-0528'
};

// Axios instance
const nimClient = axios.create({
  baseURL: NIM_API_BASE,
  timeout: 45000,
  headers: { 'Content-Type': 'application/json' }
});

if (!NIM_API_KEY) {
  console.warn('⚠️  NIM_API_KEY is not set.');
}

// ---------------------
// 🔧 Middleware
// ---------------------
app.use(cors());
app.use(express.json({ limit: '2mb' }));   // ← FIXED (was express.json())

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI → NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// Model list
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map((id) => ({
    id,
    object: 'model',
    created: Math.floor(Date.now() / 1000),
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

// Resolve model
function resolveNimModel(requestedModel) {
  if (!requestedModel || typeof requestedModel !== 'string')
    return DEFAULT_FALLBACK_MODEL;

  if (MODEL_MAPPING[requestedModel])
    return MODEL_MAPPING[requestedModel];

  if (requestedModel.includes('/') || requestedModel.includes('-'))
    return requestedModel;

  const lower = requestedModel.toLowerCase();
  if (lower.includes('gpt-4') || lower.includes('opus') || lower.includes('405b'))
    return 'meta/llama-3.1-405b-instruct';
  if (lower.includes('claude') || lower.includes('gemini') || lower.includes('70b'))
    return 'meta/llama-3.1-70b-instruct';

  return DEFAULT_FALLBACK_MODEL;
}

// Chat completions
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body || {};

    if (!model || typeof model !== 'string')
      return res.status(400).json({ error: { message: 'Invalid model', type: 'invalid_request_error', code: 400 } });

    if (!Array.isArray(messages))
      return res.status(400).json({ error: { message: '"messages" must be an array.', type: 'invalid_request_error', code: 400 } });

    if (!NIM_API_KEY)
      return res.status(500).json({ error: { message: 'NIM_API_KEY missing', type: 'config_error', code: 500 } });

    const nimModel = resolveNimModel(model);

    const nimRequestBody = {
      model: nimModel,
      messages,
      temperature: typeof temperature === 'number' ? temperature : 0.6,
      max_tokens: typeof max_tokens === 'number' ? max_tokens : 9024,
      stream: !!stream,
      ...(ENABLE_THINKING_MODE && { chat_template_kwargs: { thinking: true } })
    };

    // STREAMING MODE
    if (stream) {
      const nimResponse = await nimClient.post('/chat/completions', nimRequestBody, {
        headers: { Authorization: `Bearer ${NIM_API_KEY}` },
        responseType: 'stream',
        validateStatus: (status) => status < 500
      });

      if (nimResponse.status >= 400)
        return res.status(nimResponse.status).json({ error: { message: `NIM error: ${nimResponse.statusText}`, type: 'nim_error', code: nimResponse.status } });

      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningOpen = false;

      nimResponse.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data:')) continue;

          const payload = line.slice(5).trim();

          if (payload === '[DONE]') {
            res.write('data: [DONE]\n\n');
            continue;
          }

          try {
            const data = JSON.parse(payload);
            const choice = data?.choices?.[0];

            if (choice && choice.delta) {
              const delta = choice.delta;

              if (!SHOW_REASONING) {
                delete delta.reasoning_content;
              } else {
                let newContent = '';
                const hasReasoning = delta.reasoning_content;
                const hasContent = delta.content;

                if (hasReasoning) {
                  if (!reasoningOpen) {
                    reasoningOpen = true;
                    newContent += '<think>\n' + hasReasoning;
                  } else {
                    newContent += hasReasoning;
                  }
                }

                if (hasContent) {
                  if (reasoningOpen) {
                    newContent += '\n</think>\n\n' + hasContent;
                    reasoningOpen = false;
                  } else {
                    newContent += hasContent;
                  }
                }

                delta.content = newContent;
                delete delta.reasoning_content;
              }
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch {
            res.write(`${line}\n\n`);
          }
        }
      });

      nimResponse.data.on('end', () => {
        if (SHOW_REASONING && reasoningOpen) {
          const closePacket = {
            object: 'chat.completion.chunk',
            created: Math.floor(Date.now() / 1000),
            model,
            choices: [{ delta: { content: '\n</think>\n\n' } }]
          };
          res.write(`data: ${JSON.stringify(closePacket)}\n\n`);
        }
        res.end();
      });

      nimResponse.data.on('error', (err) => {
        console.error('Stream error:', err.message);
        res.end();
      });

      return;
    }

    // NON-STREAMING MODE
    const nimResponse = await nimClient.post('/chat/completions', nimRequestBody, {
      headers: { Authorization: `Bearer ${NIM_API_KEY}` },
      validateStatus: (status) => status < 500
    });

    if (nimResponse.status >= 400)
      return res.status(nimResponse.status).json({ error: { message: `NIM error: ${nimResponse.statusText}`, type: 'nim_error', code: nimResponse.status } });

    const nimData = nimResponse.data;

    const choices = (nimData.choices || []).map((choice, idx) => {
      const msg = choice.message || {};
      let content = msg.content || '';

      if (SHOW_REASONING && msg.reasoning_content) {
        content = `<think>\n${msg.reasoning_content}\n</think>\n\n${msg.content || ''}`;
      }

      return {
        index: choice.index ?? idx,
        message: { role: msg.role || 'assistant', content },
        finish_reason: choice.finish_reason || 'stop'
      };
    });

    res.json({
      id: nimData.id || `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model,
      choices,
      usage: nimData.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    });

  } catch (error) {
    console.error('Proxy error:', error.message);

    const status = error.response?.status || 500;
    const msg =
      error.response?.data?.error?.message ||
      error.message ||
      'Internal server error';

    res.status(status).json({
      error: { message: msg, type: 'proxy_error', code: status }
    });
  }
});

// Catch-all
app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI → NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
