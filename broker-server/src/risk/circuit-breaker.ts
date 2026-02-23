/**
 * Circuit Breaker — kill switches for catastrophic scenarios.
 *
 * Triggers:
 *   - Drawdown from peak equity exceeds threshold
 *   - 3+ consecutive losing trades (tracked per session)
 *   - API latency spike (tracked externally)
 */

import { getRiskConfig } from "../utils/config.js";
import { log, logWarn } from "../utils/logger.js";
import type { CircuitBreakerState } from "./types.js";

let state: CircuitBreakerState = {
  tripped: false,
  consecutiveLosses: 0,
  peakEquity: 0,
};

export function getCircuitBreakerState(): CircuitBreakerState {
  return { ...state };
}

export function resetCircuitBreaker(): void {
  state = {
    tripped: false,
    consecutiveLosses: 0,
    peakEquity: state.peakEquity,
  };
  log("Circuit breaker reset");
}

export function updateEquity(currentEquity: number): void {
  if (currentEquity > state.peakEquity) {
    state.peakEquity = currentEquity;
  }
}

export function checkCircuitBreaker(currentEquity: number): {
  tripped: boolean;
  reason?: string;
} {
  const config = getRiskConfig();

  // Update peak
  if (state.peakEquity === 0) {
    state.peakEquity = currentEquity;
  }
  updateEquity(currentEquity);

  // Already tripped
  if (state.tripped) {
    return {
      tripped: true,
      reason: `CIRCUIT BREAKER ACTIVE: ${state.reason}. Reset required to resume trading.`,
    };
  }

  // Check drawdown from peak
  if (state.peakEquity > 0) {
    const drawdownPct = (state.peakEquity - currentEquity) / state.peakEquity;
    if (drawdownPct >= config.drawdownKillSwitch) {
      state.tripped = true;
      state.reason = `Drawdown ${(drawdownPct * 100).toFixed(1)}% exceeds kill switch threshold ${(config.drawdownKillSwitch * 100).toFixed(1)}%`;
      state.trippedAt = new Date();
      logWarn(`CIRCUIT BREAKER TRIPPED: ${state.reason}`);
      return { tripped: true, reason: state.reason };
    }
  }

  // Check consecutive losses
  if (state.consecutiveLosses >= 3) {
    state.tripped = true;
    state.reason = `${state.consecutiveLosses} consecutive losing trades`;
    state.trippedAt = new Date();
    logWarn(`CIRCUIT BREAKER TRIPPED: ${state.reason}`);
    return { tripped: true, reason: state.reason };
  }

  return { tripped: false };
}

export function recordTradeResult(profitable: boolean): void {
  if (profitable) {
    state.consecutiveLosses = 0;
  } else {
    state.consecutiveLosses++;
    if (state.consecutiveLosses >= 3) {
      logWarn(
        `${state.consecutiveLosses} consecutive losses — circuit breaker may trip`
      );
    }
  }
}
