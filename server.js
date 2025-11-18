// server.js - OpenAI-compatible proxy to NVIDIA NIM for Janitor AI (RP-focused)

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const morgan = require('morgan');

const app = express();
const PORT = process.env.PORT || 3000;

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Toggles (env-driven)
const SHOW_REASONING = String(process.env.SHOW_REASONING || 'false').toLowerCase() === 'true';
const ENABLE_THINKING_MODE = String(process.env.ENABLE_THINKING_MODE || 'false').toLowerCase() === 'true';

// Default fallback model if nothing matches
const DEFAULT_FALLBACK_MODEL = process.env.DEFAULT_FALLBACK_MODEL || 'meta/llama-3.1-70b-instruct';

// Model mapping (OpenAI-ish model IDs → NIM model IDs)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1-terminus',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking'
};

// ---- Sanity checks ----
if (!NIM_API_KEY) {
  console.error('❌ NIM_API_KEY is not set. Cannot start server.');
  process.exit(1);
}

// Axios instance for NIM
const nimClient = axios.create({
  baseURL: NIM_API_BASE,
  timeout: Number(process.env.NIM_TIMEOUT_MS || 45000),
  headers: {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${NIM_API_KEY}`
  }
});

// ---- Helpers ----

// Resolve NIM model from incoming OpenAI-ish model
function resolveNimModel(requestedModel) {
  if (!requestedModel || typeof requestedModel !== 'string') return DEFAULT_FALLBACK_MODEL;

  // 1) explicit mapping
  if (MODEL_MAPPING[requestedModel]) return MODEL_MAPPING[requestedModel];

  // 2) assume they passed an actual NIM model id directly
  if (requestedModel.includes('/') || requestedModel.includes('-')) return requestedModel;

  // 3) crude heuristic fallback based on name
  const lower = requestedModel.toLowerCase();
  if (lower.includes('gpt-4') || lower.includes('opus') || lower.includes('405b')) {
    return 'meta/llama-3.1-405b-instruct';
  }
  if (lower.includes('claude') || lower.includes('gemini') || lower.includes('70b')) {
    return 'meta/llama-3.1-70b-instruct';
  }

  return DEFAULT_FALLBACK_MODEL;
}

// Wrap reasoning content into <think> tags for non-stream responses
function mergeReasoningNonStream(message) {
  const msg = message || {};
  let content = msg.content || '';

  if (SHOW_REASONING && msg.reasoning_content) {
    content =
      '<think>\n' +
      msg.reasoning_content +
      '\n</think>\n\n' +
      (msg.content || '');
  }

  return {
    role: msg.role || 'assistant',
    content
  };
}

// Transform a streaming chunk: drop/wrap reasoning, force original model id
function transformStreamChunk(chunk, requestedModel) {
  const choice = chunk?.choices?.[0];
  if (!choice || !choice.delta) return chunk;

  const delta = choice.delta;

  if (!SHOW_REASONING) {
    // Drop reasoning entirely
    if ('reasoning_content' in delta) {
      delete delta.reasoning_content;
    }
    chunk.model = requestedModel;
    return chunk;
  }

  // When SHOW_REASONING = true, we let the streaming wrapper decide how to inject <think>.
  // Here we just enforce the model name for Janitor.
  chunk.model = requestedModel;
  return chunk;
}

// ---- Middleware ----
app.use(cors());
app.use(express.json({ limit: '1mb' }));
app.use(morgan('tiny'));

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI → NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    nim_base: NIM_API_BASE
  });
});

// OpenAI-compatible: list models
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map((id) => ({
    id,
    object: 'model',
    created: Math.floor(Date.now() / 1000),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({
    object: 'list',
    data: models
  });
});

// Main chat completions endpoint
app.post('/v1/chat/completions', async (req, res) => {
  const { model, messages, temperature, max_tokens, stream } = req.body || {};

  // Basic validation so Janitor doesn't get random 500s
  if (!model || typeof model !== 'string') {
    return res.status(400).json({
      error: {
        message: 'Missing or invalid "model" (must be a string).',
        type: 'invalid_request_error',
        code: 400
      }
    });
  }

  if (!Array.isArray(messages)) {
    return res.status(400).json({
      error: {
        message: '"messages" must be an array.',
        type: 'invalid_request_error',
        code: 400
      }
    });
  }

  const nimModel = resolveNimModel(model);

  const nimRequestBody = {
    model: nimModel,
    messages,
    // RP-friendly defaults, but still overrideable
    temperature: typeof temperature === 'number' ? temperature : 0.8,
    max_tokens: typeof max_tokens === 'number' ? max_tokens : 9024,
    stream: !!stream,
    ...(ENABLE_THINKING_MODE && { chat_template_kwargs: { thinking: true } })
  };

  try {
    if (stream) {
      // Streaming: pipe NIM SSE → OpenAI-style SSE
      const nimResponse = await nimClient.post('/chat/completions', nimRequestBody, {
        responseType: 'stream',
        validateStatus: (status) => status < 500
      });

      if (nimResponse.status >= 400) {
        console.error('NIM streaming error:', nimResponse.status, nimResponse.statusText);
        return res.status(nimResponse.status).json({
          error: {
            message: `NIM error: ${nimResponse.statusText}`,
            type: 'nim_error',
            code: nimResponse.status
          }
        });
      }

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

            // enforce model + basic reasoning transform
            const transformed = transformStreamChunk(data, model);
            const choice = transformed?.choices?.[0];

            if (SHOW_REASONING && choice && choice.delta) {
              const delta = choice.delta;
              let newContent = '';
              const hasReasoning =
                typeof delta.reasoning_content === 'string' && delta.reasoning_content.length > 0;
              const hasContent =
                typeof delta.content === 'string' && delta.content.length > 0;

              if (hasReasoning) {
                if (!reasoningOpen) {
                  reasoningOpen = true;
                  newContent += '<think>\n' + delta.reasoning_content;
                } else {
                  newContent += delta.reasoning_content;
                }
              }

              if (hasContent) {
                if (reasoningOpen) {
                  newContent += '\n</think>\n\n' + delta.content;
                  reasoningOpen = false;
                } else {
                  newContent += delta.content;
                }
              }

              delta.content = newContent;
              delete delta.reasoning_content;
            } else if (choice && choice.delta && !SHOW_REASONING) {
              // drop reasoning_content in no-reasoning mode
              delete choice.delta.reasoning_content;
            }

            res.write(`data: ${JSON.stringify(transformed)}\n\n`);
          } catch (err) {
            console.error('Failed to parse NIM stream chunk:', err.message);
            // Forward raw line anyway so stream doesn’t die
            res.write(`${line}\n\n`);
          }
        }
      });

      nimResponse.data.on('end', () => {
        if (SHOW_REASONING && reasoningOpen) {
          const closePacket = {
            id: null,
            object: 'chat.completion.chunk',
            created: Math.floor(Date.now() / 1000),
            model,
            choices: [
              {
                index: 0,
                delta: { content: '\n</think>\n\n' },
                finish_reason: null
              }
            ]
          };
          res.write(`data: ${JSON.stringify(closePacket)}\n\n`);
        }
        res.end();
      });

      nimResponse.data.on('error', (err) => {
        console.error('Stream error from NIM:', err.message);
        res.end();
      });

      req.on('close', () => {
        if (nimResponse.data.destroy) {
          nimResponse.data.destroy();
        }
      });

    } else {
      // Non-streaming
      const nimResponse = await nimClient.post('/chat/completions', nimRequestBody, {
        validateStatus: (status) => status < 500
      });

      if (nimResponse.status >= 400) {
        console.error('NIM error:', nimResponse.status, nimResponse.statusText, nimResponse.data);
        return res.status(nimResponse.status).json({
          error: {
            message:
              nimResponse.data?.error?.message ||
              nimResponse.statusText ||
              'Unknown NIM error',
            type: 'nim_error',
            code: nimResponse.status
          }
        });
      }

      const nimData = nimResponse.data;

      const choices = (nimData.choices || []).map((choice, idx) => {
        const mergedMessage = mergeReasoningNonStream(choice.message);
        return {
          index: typeof choice.index === 'number' ? choice.index : idx,
          message: mergedMessage,
          finish_reason: choice.finish_reason || 'stop'
        };
      });

      const openaiStyleResponse = {
        id: nimData.id || `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model, // original requested model ID for Janitor
        choices,
        usage: nimData.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      res.json(openaiStyleResponse);
    }
  } catch (error) {
    const status = error.response?.status || 500;
    const body = error.response?.data;

    console.error('Proxy error:', {
      message: error.message,
      status,
      body
    });

    res.status(status).json({
      error: {
        message:
          body?.error?.message ||
          body?.message ||
          error.message ||
          'Internal server error',
        type: 'proxy_error',
        code: status
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI → NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
