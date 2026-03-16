import Anthropic from '@anthropic-ai/sdk';
import { parseFirstJsonObject } from './parsing';

export function readMistralOcrAnnotationAsText(data: Record<string, unknown>): string {
  const raw = data.document_annotation;
  if (typeof raw === 'string') {
    const parsed = parseFirstJsonObject(raw);
    return parsed ? JSON.stringify(parsed) : raw;
  }
  if (raw && typeof raw === 'object') {
    return JSON.stringify(raw);
  }
  return '';
}

export function readMistralOcrMarkdown(data: Record<string, unknown>): string {
  const pages = Array.isArray(data.pages) ? data.pages : [];
  return pages
    .map((page) => {
      if (!page || typeof page !== 'object') return '';
      const pageObj = page as Record<string, unknown>;
      return typeof pageObj.markdown === 'string' ? pageObj.markdown : '';
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
  const outputText = data.output_text;
  if (typeof outputText === 'string' && outputText.trim()) {
    return outputText.trim();
  }

  const output = Array.isArray(data.output) ? data.output : [];
  const chunks: string[] = [];

  for (const item of output) {
    const itemObj = typeof item === 'object' && item ? (item as Record<string, unknown>) : null;
    const content = itemObj && Array.isArray(itemObj.content) ? itemObj.content : [];
    for (const part of content) {
      const partObj = typeof part === 'object' && part ? (part as Record<string, unknown>) : null;
      if (partObj?.type === 'output_text' && typeof partObj.text === 'string') {
        chunks.push(partObj.text);
      }
    }
  }

  return chunks.join('\n').trim();
}

export function readTextFromGeminiResponse(data: Record<string, unknown>): string {
  const candidates = Array.isArray(data.candidates) ? data.candidates : [];
  const firstCandidate =
    candidates.length > 0 && typeof candidates[0] === 'object' && candidates[0]
      ? (candidates[0] as Record<string, unknown>)
      : null;
  const content =
    firstCandidate?.content && typeof firstCandidate.content === 'object'
      ? (firstCandidate.content as Record<string, unknown>)
      : null;
  const parts = content && Array.isArray(content.parts) ? content.parts : [];
  return parts
    .map((part) => {
      if (!part || typeof part !== 'object') return '';
      const partObj = part as Record<string, unknown>;
      return typeof partObj.text === 'string' ? partObj.text : '';
    })
    .filter(Boolean)
    .join('\n')
    .trim();
}

export function readTextFromMistralChatResponse(data: Record<string, unknown>): string {
  const choices = Array.isArray(data.choices) ? data.choices : [];
  const firstChoice =
    choices.length > 0 && typeof choices[0] === 'object' && choices[0]
      ? (choices[0] as Record<string, unknown>)
      : null;
  const message =
    firstChoice?.message && typeof firstChoice.message === 'object'
      ? (firstChoice.message as Record<string, unknown>)
      : null;
  const content = message?.content;

  if (typeof content === 'string') {
    return content.trim();
  }

  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (!part || typeof part !== 'object') return '';
        const partObj = part as Record<string, unknown>;
        return typeof partObj.text === 'string' ? partObj.text : '';
      })
      .filter(Boolean)
      .join('\n')
      .trim();
  }

  return '';
}
