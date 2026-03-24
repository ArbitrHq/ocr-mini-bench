import Anthropic from '@anthropic-ai/sdk';
import { parseFirstJsonObject } from './parsing';
import {
  isRecord,
  isString,
  isArray,
  getStringProperty,
  getArrayProperty,
  getRecordProperty,
} from '../lib/type-guards';

export function readMistralOcrAnnotationAsText(data: Record<string, unknown>): string {
  const raw = data.document_annotation;
  if (isString(raw)) {
    const parsed = parseFirstJsonObject(raw);
    return parsed ? JSON.stringify(parsed) : raw;
  }
  if (isRecord(raw)) {
    return JSON.stringify(raw);
  }
  return '';
}

export function readMistralOcrMarkdown(data: Record<string, unknown>): string {
  const pages = getArrayProperty(data, 'pages') ?? [];
  return pages
    .map((page) => {
      if (!isRecord(page)) return '';
      return getStringProperty(page, 'markdown') ?? '';
    })
    .filter(Boolean)
    .join('\n\n');
}

export function readTextFromAnthropicContent(content: Anthropic.Messages.ContentBlock[]): string {
  return content
    .filter((block) => block.type === 'text')
    .map((block) => (block.type === 'text' ? block.text : ''))
    .join('\n')
    .trim();
}

export function readTextFromOpenAIResponse(data: Record<string, unknown>): string {
  const outputText = getStringProperty(data, 'output_text');
  if (outputText?.trim()) {
    return outputText.trim();
  }

  const output = getArrayProperty(data, 'output') ?? [];
  const chunks: string[] = [];

  for (const item of output) {
    if (!isRecord(item)) continue;
    const content = getArrayProperty(item, 'content') ?? [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (getStringProperty(part, 'type') === 'output_text') {
        const text = getStringProperty(part, 'text');
        if (text) chunks.push(text);
      }
    }
  }

  return chunks.join('\n').trim();
}

export function readTextFromGeminiResponse(data: Record<string, unknown>): string {
  const candidates = getArrayProperty(data, 'candidates') ?? [];
  const firstCandidate = candidates.length > 0 && isRecord(candidates[0]) ? candidates[0] : undefined;
  const content = firstCandidate ? getRecordProperty(firstCandidate, 'content') : undefined;
  const parts = content ? getArrayProperty(content, 'parts') ?? [] : [];
  return parts
    .map((part) => {
      if (!isRecord(part)) return '';
      return getStringProperty(part, 'text') ?? '';
    })
    .filter(Boolean)
    .join('\n')
    .trim();
}

export function readTextFromMistralChatResponse(data: Record<string, unknown>): string {
  const choices = getArrayProperty(data, 'choices') ?? [];
  const firstChoice = choices.length > 0 && isRecord(choices[0]) ? choices[0] : undefined;
  const message = firstChoice ? getRecordProperty(firstChoice, 'message') : undefined;

  if (!message) return '';

  const content = message.content;

  if (isString(content)) {
    return content.trim();
  }

  if (isArray(content)) {
    return content
      .map((part) => {
        if (!isRecord(part)) return '';
        return getStringProperty(part, 'text') ?? '';
      })
      .filter(Boolean)
      .join('\n')
      .trim();
  }

  return '';
}
