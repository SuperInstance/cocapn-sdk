/**
 * Cocapn SDK — One API key, any AI model, see what it costs.
 *
 * @example
 * const cocapn = require('cocapn');
 * cocapn.apiKey = 'cocapn_your_key';
 * const response = await cocapn.chat('Hello!', { model: 'deepseek-chat' });
 * console.log(response.text);      // "Hello! How can I help you?"
 * console.log(response.cost);      // 0.000042
 * console.log(response.tokens);    // { in: 15, out: 47 }
 */

const DEFAULT_BASE = 'https://cocapn.ai';

class Cocapn {
  constructor(options = {}) {
    this.apiKey = options.apiKey || process.env.COCAPN_API_KEY;
    this.baseURL = options.baseURL || DEFAULT_BASE;
  }

  /**
   * Send a chat message and get a response.
   * @param {string} message - The user message
   * @param {object} options - Options
   * @param {string} [options.model='deepseek-chat'] - Model to use
   * @param {string} [options.system] - System prompt
   * @param {number} [options.maxTokens=4096] - Max tokens
   * @param {number} [options.temperature] - Temperature
   * @param {string[]} [options.history] - Previous messages [{role, content}]
   * @returns {Promise<{text: string, cost: number, tokens: {in: number, out: number}, model: string, provider: string}>}
   */
  async chat(message, options = {}) {
    const model = options.model || 'deepseek-chat';
    const messages = [];
    if (options.system) messages.push({ role: 'system', content: options.system });
    if (options.history) messages.push(...options.history);
    messages.push({ role: 'user', content: message });

    const body = { model, messages, max_tokens: options.maxTokens || 4096 };
    if (options.temperature !== undefined) body.temperature = options.temperature;

    const resp = await this._fetch('/v1/chat/completions', body);
    return {
      text: resp.choices[0].message.content,
      cost: parseFloat(resp.cocapn_cost || '0'),
      tokens: {
        in: resp.usage?.prompt_tokens || 0,
        out: resp.usage?.completion_tokens || 0,
      },
      model: resp.model,
      provider: resp.cocapn_provider || model.split('-')[0],
    };
  }

  /**
   * Send a chat message with streaming.
   * @param {string} message - The user message
   * @param {function} onChunk - Called with each text chunk
   * @param {object} options - Same as chat()
   */
  async chatStream(message, onChunk, options = {}) {
    const model = options.model || 'deepseek-chat';
    const messages = [];
    if (options.system) messages.push({ role: 'system', content: options.system });
    if (options.history) messages.push(...options.history);
    messages.push({ role: 'user', content: message });

    const body = { model, messages, max_tokens: options.maxTokens || 4096, stream: true };
    if (options.temperature !== undefined) body.temperature = options.temperature;

    const resp = await this._fetchRaw('/v1/chat/completions', body);
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (line.startsWith('data: ') && line !== 'data: [DONE]') {
          try {
            const json = JSON.parse(line.slice(6));
            const delta = json.choices?.[0]?.delta?.content;
            if (delta) onChunk(delta);
          } catch {}
        }
      }
    }
  }

  /**
   * List available models.
   * @returns {Promise<Array<{id: string, provider: string, costIn: number, costOut: number}>>}
   */
  async models() {
    const resp = await this._fetch('/v1/models');
    return resp.data.map(m => ({
      id: m.id,
      provider: m.owned_by,
      costIn: m.cocapn_cost_in,
      costOut: m.cocapn_cost_out,
    }));
  }

  /**
   * Get usage stats.
   * @param {string} [period='day'] - 'day', 'week', or 'month'
   */
  async usage(period = 'day') {
    return this._fetch(`/v1/usage?period=${period}`);
  }

  // ─── Internal ───

  async _fetch(path, body) {
    const resp = await this._fetchRaw(path, body);
    const data = await resp.json();
    if (!resp.ok) {
      const err = new Error(data.error?.message || `HTTP ${resp.status}`);
      err.status = resp.status;
      err.data = data;
      throw err;
    }
    return data;
  }

  async _fetchRaw(path, body) {
    const headers = {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${this.apiKey}`,
    };
    const resp = await fetch(`${this.baseURL}${path}`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });
    return resp;
  }
}

// Singleton for convenience
const instance = new Cocapn();

// Export both class and singleton
module.exports = Cocapn;
module.exports.Cocapn = Cocapn;
module.exports.default = instance;

// Named exports
module.exports.chat = (message, options) => instance.chat(message, options);
module.exports.chatStream = (message, onChunk, options) => instance.chatStream(message, onChunk, options);
module.exports.models = () => instance.models();
module.exports.usage = (period) => instance.usage(period);
