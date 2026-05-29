# cocapn-sdk — One API Key, Any AI Model

**One API key to access OpenAI, Claude, DeepSeek, Gemini, and more. See exactly what it costs.**

## What This Gives You

- **One API key** — route to any model (DeepSeek, GPT-4o, Claude, Gemini) through a single endpoint
- **Cost tracking** — every response includes the exact cost, token counts, and provider used
- **Streaming support** — `chatStream()` delivers tokens as they arrive
- **Model catalog** — list available models with pricing per token
- **Usage dashboard** — query your spend by day, week, or month

## Quick Start

```bash
npm install cocapn
```

```javascript
const cocapn = require('cocapn');
cocapn.apiKey = process.env.COCAPN_API_KEY; // or pass { apiKey: '...' }

// Chat with any model
const response = await cocapn.chat('Explain transformers in one paragraph', {
  model: 'deepseek-chat',
  system: 'You are a concise teacher.'
});

console.log(response.text);      // "Transformers are..."
console.log(response.cost);      // 0.000042
console.log(response.tokens);    // { in: 15, out: 47 }
console.log(response.provider);  // "deepseek"
```

### Streaming

```javascript
await cocapn.chatStream('Tell me a story', (chunk) => {
  process.stdout.write(chunk);
}, { model: 'gpt-4o' });
```

### List Models

```javascript
const models = await cocapn.models();
// [{ id: 'deepseek-chat', provider: 'deepseek', costIn: 0.00000014, costOut: 0.00000028 }, ...]
```

### Usage Stats

```javascript
const usage = await cocapn.usage('week');
```

## API Reference

### `new Cocapn(options?)`
| Option | Default | Description |
|--------|---------|-------------|
| `apiKey` | `COCAPN_API_KEY` env | Your Cocapn API key |
| `baseURL` | `https://cocapn.ai` | API base URL |

### `chat(message, options?) → Promise<Response>`
Returns `{ text, cost, tokens: { in, out }, model, provider }`.

### `chatStream(message, onChunk, options?) → Promise<void>`
Streams text chunks to `onChunk(delta: string)`.

### `models() → Promise<Model[]>`
### `usage(period?) → Promise<Usage>`

## How It Fits
- [OpenConstruct Documentation](https://github.com/SuperInstance/openconstruct-docs) — ecosystem-wide docs and guides

The unified model gateway for the [SuperInstance fleet](https://github.com/SuperInstance):

- **[cocapn-py](https://github.com/SuperInstance/cocapn-py)** — Python SDK (same API, different language)
- **[api-gateway-1](https://github.com/SuperInstance/api-gateway-1)** — The gateway server backing this SDK
- **[cocapn](https://github.com/SuperInstance/cocapn)** — Core agent infrastructure
- **[cocapn-explain](https://github.com/SuperInstance/cocapn-explain)** — Agent explainability

## Installation

```bash
npm install cocapn
```

Requires Node.js 18+. MIT license.
