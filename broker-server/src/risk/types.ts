export interface RiskCheckResult {
  allowed: boolean;
  reason?: string;
  warnings: string[];
  requiresConfirmation: boolean;
  confirmationMessage?: string;
}

export interface CircuitBreakerState {
  tripped: boolean;
  reason?: string;
  trippedAt?: Date;
  consecutiveLosses: number;
  peakEquity: number;
}
