# cocapn — SDK

One API key, any AI model, see what it costs.

## Install

```bash
npm install cocapn
```

## Quick Start

```javascript
const cocapn = require('cocapn');

// Set your API key (or use COCAPN_API_KEY env var)
cocapn.apiKey = 'cocapn_your_key';

// Send a message
const response = await cocapn.chat('Explain quantum computing', {
  model: 'deepseek-chat',
});

console.log(response.text);      // "Quantum computing uses..."
console.log(response.cost);      // 0.0042
console.log(response.tokens);    // { in: 15, out: 847 }
```

## Streaming

```javascript
await cocapn.chatStream('Tell me a story', (chunk) => {
  process.stdout.write(chunk);
}, { model: 'claude-3-5-sonnet' });
```

## System Prompts

```javascript
const response = await cocapn.chat('Summarize this article', {
  model: 'gpt-4o',
  system: 'You are a helpful assistant that writes concise summaries.',
});
```

## Conversation History

```javascript
const history = [
  { role: 'user', content: 'My name is Casey' },
  { role: 'assistant', content: 'Hello Casey!' },
];

const response = await cocapn.chat('What is my name?', { history });
console.log(response.text); // "Your name is Casey."
```

## Models

```javascript
const models = await cocapn.models();
// [{ id: 'deepseek-chat', provider: 'deepseek', costIn: 0.14, costOut: 0.28 }, ...]
```

## Usage

```javascript
const usage = await cocapn.usage('week');
console.log(usage.totalCost);    // 0.42
console.log(usage.requests);     // 127
console.log(usage.byModel);      // { 'deepseek-chat': 0.12, 'gpt-4o': 0.30 }
```

## TypeScript

```typescript
import cocapn from 'cocapn';

const response: ChatResponse = await cocapn.chat('Hello', {
  model: 'deepseek-chat',
  temperature: 0.7,
});
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `COCAPN_API_KEY` | Your Cocapn API key |
| `COCAPN_BASE_URL` | Custom API URL (default: https://cocapn.ai) |

## API Key

Get your key at cocapn.ai after signing up. Free tier includes 50 requests/day.
