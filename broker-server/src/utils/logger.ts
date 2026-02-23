/**
 * stderr-only logger.
 * CRITICAL: Never use console.log() — it corrupts the stdio JSON-RPC transport.
 * All logging must go to stderr.
 */

export function log(message: string): void {
  console.error(`[alpaca-broker] ${message}`);
}

export function logError(message: string, error?: unknown): void {
  const errMsg = error instanceof Error ? error.message : String(error);
  console.error(`[alpaca-broker] ERROR: ${message} — ${errMsg}`);
}

export function logWarn(message: string): void {
  console.error(`[alpaca-broker] WARN: ${message}`);
}
