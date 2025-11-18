// Full improved NIM <-> JanitorAI proxy server
// Implements: strict validation, better model resolution, fast-fail config,
// safer streaming, better error logging, token clamping, message validation,
// safer body limits, client disconnect handling, thinking mode gating, and more.

import express from 'express';
import cors from 'cors';
import axios from 'axios';

// ===== CONFIG =====
const NIM_API_KEY = process.env.NIM_API_KEY;
const NIM_API_BASE = process.env.NIM_API_BASE || "https://integrate.api.nvidia.com/v1/";
const PORT = parseInt(process.env.PORT || "3000");

// Hard fail if missing API key
if (!NIM_API_KEY) {
  console.error("FATAL: NIM_API_KEY is missing. Set it in Railway env vars.");
  process.exit(1);
}

// ===== FLAGS =====
const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

// ===== MODEL TABLE =====
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1-terminus',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',
};

// Models that support thinking
const THINKING_MODE_MODELS = new Set([
  'openai/gpt-oss-120b',
  'openai/gpt-oss-20b'
]);

const DEFAULT_FALLBACK_MODEL = 'meta/llama-3.1-70b-instruct';

function clamp(num, min, max) {
  return Math.min(Math.max(num, min), max);
}

function resolveNimModel(requested) {
  if (!requested || typeof requested !== 'string') {
    console.warn(`No model provided. Using fallback ${DEFAULT_FALLBACK_MODEL}`);
    return DEFAULT_FALLBACK_MODEL;
  }

  if (MODEL_MAPPING[requested]) {
    return MODEL_MAPPING[requested];
  }

  if (requested.includes('/') || requested.includes('-')) {
    console.log(`Using raw NIM model: ${requested}`);
    return requested;
  }

  const l = requested.toLowerCase();
  if (l.includes('gpt-4') || l.includes('opus') || l.includes('405b')) {
    console.warn(`Heuristic map: ${requested} → llama-3.1-405b`);
    return 'meta/llama-3.1-405b-instruct';
  }
  if (l.includes('claude') || l.includes('gemini') || l.includes('70b')) {
    console.warn(`Heuristic map: ${requested} → llama-3.1-70b`);
    return 'meta/llama-3.1-70b-instruct';
  }

  console.warn(`Unknown model '${requested}'. Using fallback.`);
  return DEFAULT_FALLBACK_MODEL;
}

// ===== EXPRESS SETUP =====
const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));

// Axios client
const nimClient = axios.create({
  baseURL: NIM_API_BASE,
  headers: {
    'Authorization': `Bearer ${NIM_API_KEY}`,
    'Content-Type': 'application/json'
  },
  timeout: 45000
});

// ===== HEALTH =====
app.get('/health', (req, res) => {
  res.json({ status: 'ok', SHOW_REASONING, ENABLE_THINKING_MODE });
});

// ===== VALIDATION =====
function validateMessages(messages) {
  if (!Array.isArray(messages) || messages.length === 0) return 'messages must be a non-empty array';
  for (const m of messages) {
    if (!m || typeof m !== 'object') return 'each message must be an object';
    if (!['system','user','assistant','tool'].includes(m.role)) return `invalid role '${m.role}'`;
    if (typeof m.content !== 'string') return 'message.content must be a string';
  }
  return null;
}

// ===== MAIN COMPLETIONS ENDPOINT =====
app.post('/v1/chat/completions', async (req, res) => {
  const { model, messages, temperature, max_tokens, stream } = req.body;

  const msgErr = validateMessages(messages);
  if (msgErr) {
    return res.status(400).json({ error: { message: msgErr, type: 'invalid_request_error', code: 400 }});
  }

  const nimModel = resolveNimModel(model);

  const temp = clamp(typeof temperature === 'number' ? temperature : 0.6, 0, 2);
  const maxT = clamp(typeof max_tokens === 'number' ? max_tokens : 1024, 1, 9024);

  const body = {
    model: nimModel,
    messages,
    temperature: temp,
    max_tokens: maxT,
    stream: !!stream,
  };

  if (ENABLE_THINKING_MODE && THINKING_MODE_MODELS.has(nimModel)) {
    body.chat_template_kwargs = { thinking: true };
  }

  try {
    if (stream) {
      const cancel = new AbortController();
      req.on('close', () => cancel.abort());

      const upstream = await nimClient.post('/chat/completions', body, {
        responseType: 'stream',
        signal: cancel.signal,
        validateStatus: s => s < 500
      });

      if (upstream.status >= 400) {
        console.error('NIM error (stream)', upstream.status);
        return res.status(502).json({ error: { message: 'Upstream NIM error', code: upstream.status }});
      }

      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      upstream.data.on('data', chunk => {
        buffer += chunk.toString();
        let parts = buffer.split('

');
        buffer = parts.pop();

        for (const part of parts) {
          if (!part.startsWith('data:')) continue;
          const json = part.slice(5).trim();
          if (json === '[DONE]') {
            res.write('data: [DONE]

');
            return;
          }

          try {
            const event = JSON.parse(json);
            if (event.choices?.length) {
              let delta = event.choices[0].delta;
              if (delta.reasoning_content) {
                if (SHOW_REASONING) {
                  delta.content = `<think>${delta.reasoning_content}</think>` + (delta.content || '');
                }
                delete delta.reasoning_content;
              }
            }
            res.write(`data: ${JSON.stringify(event)}

`);
          } catch (_) {}
        }
      });

      upstream.data.on('end', () => res.end());
      upstream.data.on('error', () => res.end());
      return;
    }

    // non-streaming
    const upstream = await nimClient.post('/chat/completions', body, { validateStatus: s => s < 500 });

    if (upstream.status >= 400) {
      console.error('NIM error (non-stream)', upstream.status, upstream.data);
      return res.status(502).json({ error: { message: upstream.data?.error?.message || 'Upstream NIM error', code: upstream.status }});
    }

    const d = upstream.data;

    if (!d.choices?.length || !d.choices[0].message?.content) {
      return res.status(500).json({ error: { message: 'Invalid NIM response: missing content' }});
    }

    let content = d.choices[0].message.content;
    if (SHOW_REASONING && d.choices[0].message.reasoning_content) {
      content = `<think>${d.choices[0].message.reasoning_content}</think>` + content;
    }

    res.json({
      id: d.id || 'proxy-chatcmpl',
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model,
      choices: [{ index: 0, message: { role: 'assistant', content }, finish_reason: d.choices[0].finish_reason || 'stop' }],
      usage: d.usage || {}
    });
  } catch (err) {
    console.error('Proxy error', err.response?.data || err.message);
    const status = err.response?.status || 500;
    res.status(status).json({ error: { message: err.response?.data?.error?.message || err.message, type: 'proxy_error', code: status }});
  }
});

// ===== MODEL LIST =====
app.get('/v1/models', (req, res) => {
  const out = Object.keys(MODEL_MAPPING).map(id => ({ id, object: 'model', owned_by: 'proxy', nim_id: MODEL_MAPPING[id] }));
  res.json({ object: 'list', data: out });
});

// ===== START SERVER =====
app.listen(PORT, () => console.log(`Proxy running on port ${PORT}`));
