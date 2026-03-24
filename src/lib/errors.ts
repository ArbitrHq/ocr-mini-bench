/**
 * Extracts a human-readable error message from an unknown error value.
 */
export function toErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return 'Model run failed.';
}
