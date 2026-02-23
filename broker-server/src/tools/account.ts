import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { getAlpacaClient } from "../utils/alpaca-client.js";
import { isPaperTrading } from "../utils/config.js";
import { getDailyTracker } from "../risk/daily-tracker.js";
import { getCircuitBreakerState, updateEquity } from "../risk/circuit-breaker.js";
import { logError } from "../utils/logger.js";

export function registerAccountTools(server: McpServer): void {
  server.tool(
    "get_account",
    "Get Alpaca account info: buying power, equity, portfolio value, daily P&L, circuit breaker status",
    {},
    async () => {
      try {
        const client = getAlpacaClient();
        const account = await client.getAccount();
        const equity = parseFloat(account.equity);

        // Track equity
        const tracker = getDailyTracker();
        tracker.recordStartOfDay(equity);
        const dailyPnL = tracker.getDailyPnL(equity);
        updateEquity(equity);

        const cbState = getCircuitBreakerState();

        const result = {
          id: account.id,
          status: account.status,
          buying_power: account.buying_power,
          cash: account.cash,
          portfolio_value: account.portfolio_value,
          equity: account.equity,
          long_market_value: account.long_market_value,
          short_market_value: account.short_market_value,
          pattern_day_trader: account.pattern_day_trader,
          day_trade_count: account.daytrade_count,
          daily_pnl: dailyPnL.toFixed(2),
          paper_trading: isPaperTrading(),
          circuit_breaker: {
            tripped: cbState.tripped,
            reason: cbState.reason || null,
            peak_equity: cbState.peakEquity.toFixed(2),
            consecutive_losses: cbState.consecutiveLosses,
          },
        };

        return {
          content: [{ type: "text" as const, text: JSON.stringify(result, null, 2) }],
        };
      } catch (err) {
        logError("get_account", err);
        return {
          content: [
            {
              type: "text" as const,
              text: `Error fetching account: ${err instanceof Error ? err.message : String(err)}`,
            },
          ],
        };
      }
    }
  );
}
