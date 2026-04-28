/**
 * Cocapn SDK — TypeScript definitions
 */

export interface TokenCount {
  in: number;
  out: number;
}

export interface ChatResponse {
  text: string;
  cost: number;
  tokens: TokenCount;
  model: string;
  provider: string;
}

export interface Model {
  id: string;
  provider: string;
  costIn: number;
  costOut: number;
}

export interface UsageStats {
  period: string;
  totalCost: number;
  totalTokensIn: number;
  totalTokensOut: number;
  requests: number;
  byModel: Record<string, number>;
  tier: string;
}

export interface Message {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface ChatOptions {
  model?: string;
  system?: string;
  maxTokens?: number;
  temperature?: number;
  history?: Message[];
}

export class Cocapn {
  apiKey?: string;
  baseURL: string;

  constructor(options?: { apiKey?: string; baseURL?: string });

  chat(message: string, options?: ChatOptions): Promise<ChatResponse>;
  chatStream(message: string, onChunk: (text: string) => void, options?: ChatOptions): Promise<void>;
  models(): Promise<Model[]>;
  usage(period?: 'day' | 'week' | 'month'): Promise<UsageStats>;
}

declare const cocapn: Cocapn;
export default cocapn;

export function chat(message: string, options?: ChatOptions): Promise<ChatResponse>;
export function chatStream(message: string, onChunk: (text: string) => void, options?: ChatOptions): Promise<void>;
export function models(): Promise<Model[]>;
export function usage(period?: 'day' | 'week' | 'month'): Promise<UsageStats>;
