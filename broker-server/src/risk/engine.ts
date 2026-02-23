/**
 * Pre-trade Risk Engine.
 *
 * Runs before every order submission:
 * 1. Circuit breaker check
 * 2. Daily loss limit
 * 3. Order value vs max position size
 * 4. Max order quantity
 * 5. Buying power check
 * 6. Position concentration check
 * 7. Max open positions check
 * 8. Large order confirmation
 */

import { getAlpacaClient } from "../utils/alpaca-client.js";
import { getRiskConfig } from "../utils/config.js";
import { getDailyTracker } from "./daily-tracker.js";
import { checkCircuitBreaker, updateEquity } from "./circuit-breaker.js";
import { log, logWarn } from "../utils/logger.js";
import type { RiskCheckResult } from "./types.js";

export async function performPreTradeRiskCheck(params: {
  symbol: string;
  qty: number;
  side: "buy" | "sell";
  type: string;
  limitPrice?: number;
}): Promise<RiskCheckResult> {
  const config = getRiskConfig();
  const warnings: string[] = [];

  try {
    const client = getAlpacaClient();

    // 1. Get account info
    const account = await client.getAccount();
    const buyingPower = parseFloat(account.buying_power);
    const portfolioValue = parseFloat(account.portfolio_value);
    const equity = parseFloat(account.equity);

    // Update circuit breaker tracking
    updateEquity(equity);

    // 2. Circuit breaker
    const cb = checkCircuitBreaker(equity);
    if (cb.tripped) {
      return {
        allowed: false,
        reason: cb.reason,
        warnings,
        requiresConfirmation: false,
      };
    }

    // 3. Daily loss limit
    const tracker = getDailyTracker();
    tracker.recordStartOfDay(equity);
    const dailyPnL = tracker.getDailyPnL(equity);

    if (dailyPnL < -config.maxDailyLoss) {
      return {
        allowed: false,
        reason: `BLOCKED: Daily loss limit reached. Current daily P&L: $${dailyPnL.toFixed(2)}. Limit: -$${config.maxDailyLoss}`,
        warnings,
        requiresConfirmation: false,
      };
    }

    if (dailyPnL < -config.maxDailyLoss * 0.5) {
      warnings.push(
        `WARNING: Daily P&L is $${dailyPnL.toFixed(2)}, approaching loss limit of -$${config.maxDailyLoss}`
      );
    }

    // 4. Max order quantity
    if (params.qty > config.maxOrderQty) {
      return {
        allowed: false,
        reason: `BLOCKED: Order quantity ${params.qty} exceeds max ${config.maxOrderQty} shares`,
        warnings,
        requiresConfirmation: false,
      };
    }

    // 5. Estimate order value
    let estimatedValue: number;
    if (params.limitPrice) {
      estimatedValue = params.qty * params.limitPrice;
    } else {
      try {
        const lastTrade = await client.getLatestTrade(params.symbol);
        estimatedValue = params.qty * parseFloat(lastTrade.Price);
      } catch {
        // Fallback: use a conservative estimate
        logWarn(`Could not get latest trade for ${params.symbol}, using limit price or blocking`);
        return {
          allowed: false,
          reason: `BLOCKED: Could not determine current price for ${params.symbol}. Use a limit order.`,
          warnings,
          requiresConfirmation: false,
        };
      }
    }

    // 6. Max position size
    if (estimatedValue > config.maxPositionSize) {
      return {
        allowed: false,
        reason: `BLOCKED: Order value $${estimatedValue.toFixed(2)} exceeds max position size $${config.maxPositionSize}`,
        warnings,
        requiresConfirmation: false,
      };
    }

    // 7. Buying power check (for buy orders)
    if (params.side === "buy" && estimatedValue > buyingPower) {
      return {
        allowed: false,
        reason: `BLOCKED: Insufficient buying power. Need: $${estimatedValue.toFixed(2)}, Available: $${buyingPower.toFixed(2)}`,
        warnings,
        requiresConfirmation: false,
      };
    }

    // 8. Position concentration
    if (params.side === "buy") {
      let existingValue = 0;
      try {
        const position = await client.getPosition(params.symbol);
        existingValue = Math.abs(parseFloat(position.market_value));
      } catch {
        // No existing position
      }
      const totalExposure = existingValue + estimatedValue;
      const concentration = portfolioValue > 0 ? totalExposure / portfolioValue : 1;

      if (concentration > config.positionConcentrationLimit) {
        return {
          allowed: false,
          reason: `BLOCKED: Position concentration ${(concentration * 100).toFixed(1)}% exceeds limit ${(config.positionConcentrationLimit * 100).toFixed(1)}%`,
          warnings,
          requiresConfirmation: false,
        };
      }
      if (concentration > config.positionConcentrationLimit * 0.8) {
        warnings.push(
          `WARNING: Position will be ${(concentration * 100).toFixed(1)}% of portfolio, approaching limit`
        );
      }
    }

    // 9. Max open positions
    try {
      const positions = await client.getPositions();
      if (params.side === "buy" && positions.length >= config.maxOpenPositions) {
        // Check if we already hold this symbol
        const alreadyHeld = positions.some(
          (p: any) => p.symbol === params.symbol.toUpperCase()
        );
        if (!alreadyHeld) {
          return {
            allowed: false,
            reason: `BLOCKED: Already at max ${config.maxOpenPositions} open positions`,
            warnings,
            requiresConfirmation: false,
          };
        }
      }
    } catch {
      warnings.push("WARNING: Could not verify position count");
    }

    // 10. Large order confirmation
    let requiresConfirmation = false;
    let confirmationMessage: string | undefined;
    if (params.qty > config.largeOrderThreshold) {
      requiresConfirmation = true;
      confirmationMessage =
        `⚠️ LARGE ORDER: ${params.qty} shares of ${params.symbol} ` +
        `(~$${estimatedValue.toFixed(2)}) exceeds threshold of ${config.largeOrderThreshold} shares. ` +
        `Please confirm you want to proceed.`;
    }

    log(
      `Risk check PASSED for ${params.side} ${params.qty} ${params.symbol} ` +
        `(~$${estimatedValue.toFixed(2)}, ${warnings.length} warnings)`
    );

    return {
      allowed: true,
      warnings,
      requiresConfirmation,
      confirmationMessage,
    };
  } catch (error) {
    const errMsg = error instanceof Error ? error.message : String(error);
    return {
      allowed: false,
      reason: `Risk check error: ${errMsg}`,
      warnings,
      requiresConfirmation: false,
    };
  }
}
