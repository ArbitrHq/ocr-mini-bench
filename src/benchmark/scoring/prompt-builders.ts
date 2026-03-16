import type { GroundTruthDocument } from '../types';

function buildKeyInstruction(document: GroundTruthDocument): string {
  return document.keys
    .map(
      (key) =>
        `- ${key.name} (type: ${key.data_type ?? 'unknown'}, critical: ${key.critical ? 'yes' : 'no'})`
    )
    .join('\n');
}

function injectRequestedKeys(template: string, keyInstruction: string): string {
  if (template.includes('{{REQUESTED_KEYS}}')) {
    return template.replace('{{REQUESTED_KEYS}}', keyInstruction);
  }
  return `${template.trim()}\n\nRequested keys:\n${keyInstruction}`;
}

export function buildBenchmarkSystemPrompt(template: string): string {
  return template.replace('{{REQUESTED_KEYS}}', '').trim();
}

export function buildBenchmarkUserPrompt(template: string, document: GroundTruthDocument): string {
  return injectRequestedKeys(template, buildKeyInstruction(document));
}
