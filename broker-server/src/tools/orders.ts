import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { getAlpacaClient } from "../utils/alpaca-client.js";
import { performPreTradeRiskCheck } from "../risk/engine.js";
import { getRiskConfig } from "../utils/config.js";
import { logError, log } from "../utils/logger.js";

export function registerOrderTools(server: McpServer): void {
  const config = getRiskConfig();

  server.tool(
    "place_order",
    `Place a stock order through Alpaca with pre-trade risk checks.\n` +
      `Risk limits: max position $${config.maxPositionSize}, ` +
      `max daily loss $${config.maxDailyLoss}, ` +
      `concentration limit ${config.positionConcentrationLimit * 100}%, ` +
      `max qty ${config.maxOrderQty}. ` +
      `Orders > ${config.largeOrderThreshold} shares require confirmation.`,
    {
      symbol: z.string().describe("Stock ticker symbol (e.g. AAPL)"),
      qty: z.number().positive().describe("Number of shares"),
      side: z.enum(["buy", "sell"]).describe("Order side"),
      type: z
        .enum(["market", "limit", "stop", "stop_limit", "trailing_stop"])
        .describe("Order type"),
      time_in_force: z
        .enum(["day", "gtc", "opg", "ioc"])
        .describe("Time in force"),
      limit_price: z
        .number()
        .optional()
        .describe("Limit price (required for limit and stop_limit orders)"),
      stop_price: z
        .number()
        .optional()
        .describe("Stop price (required for stop and stop_limit orders)"),
      trail_percent: z
        .number()
        .optional()
        .describe("Trail percent for trailing_stop orders"),
      extended_hours: z
        .boolean()
        .optional()
        .describe("Allow extended hours trading"),
    },
    async (params) => {
      try {
        // Step 1: Risk checks
        const riskResult = await performPreTradeRiskCheck({
          symbol: params.symbol,
          qty: params.qty,
          side: params.side,
          type: params.type,
          limitPrice: params.limit_price,
        });

        if (!riskResult.allowed) {
          return {
            content: [
              { type: "text" as const, text: riskResult.reason! },
            ],
          };
        }

        // Step 2: Confirmation required?
        if (riskResult.requiresConfirmation) {
          const warningText =
            riskResult.warnings.length > 0
              ? "\n" + riskResult.warnings.join("\n")
              : "";
          return {
            content: [
              {
                type: "text" as const,
                text:
                  `${riskResult.confirmationMessage}${warningText}\n\n` +
                  `To proceed, ask the user for confirmation and then call place_order again with the same parameters.`,
              },
            ],
          };
        }

        // Step 3: Build and submit order
        const client = getAlpacaClient();
        const orderParams: Record<string, any> = {
          symbol: params.symbol.toUpperCase(),
          qty: params.qty,
          side: params.side,
          type: params.type,
          time_in_force: params.time_in_force,
        };

        if (params.limit_price !== undefined)
          orderParams.limit_price = params.limit_price;
        if (params.stop_price !== undefined)
          orderParams.stop_price = params.stop_price;
        if (params.trail_percent !== undefined)
          orderParams.trail_percent = params.trail_percent;
        if (params.extended_hours !== undefined)
          orderParams.extended_hours = params.extended_hours;

        log(
          `Submitting order: ${params.side} ${params.qty} ${params.symbol} ${params.type}`
        );
        const order = await client.createOrder(orderParams);

        const warningText =
          riskResult.warnings.length > 0
            ? "\n\nRisk Warnings:\n" + riskResult.warnings.join("\n")
            : "";

        return {
          content: [
            {
              type: "text" as const,
              text:
                JSON.stringify(
                  {
                    order_id: order.id,
                    client_order_id: order.client_order_id,
                    symbol: order.symbol,
                    qty: order.qty,
                    side: order.side,
                    type: order.type,
                    time_in_force: order.time_in_force,
                    status: order.status,
                    submitted_at: order.submitted_at,
                    limit_price: order.limit_price,
                    stop_price: order.stop_price,
                  },
                  null,
                  2
                ) + warningText,
            },
          ],
        };
      } catch (err) {
        logError("place_order", err);
        return {
          content: [
            {
              type: "text" as const,
              text: `Error placing order: ${err instanceof Error ? err.message : String(err)}`,
            },
          ],
        };
      }
    }
  );

  server.tool(
    "get_orders",
    "List orders. Filter by status (open, closed, all). Defaults to open orders.",
    {
      status: z
        .enum(["open", "closed", "all"])
        .optional()
        .describe("Order status filter. Defaults to 'open'."),
      limit: z
        .number()
        .optional()
        .describe("Max number of orders to return. Defaults to 50."),
      symbols: z
        .string()
        .optional()
        .describe("Comma-separated symbols to filter by"),
    },
    async ({ status, limit, symbols }) => {
      try {
        const client = getAlpacaClient();
        const params: Record<string, any> = {
          status: status || "open",
          limit: limit || 50,
        };
        if (symbols) {
          params.symbols = symbols
            .split(",")
            .map((s) => s.trim().toUpperCase());
        }

        const orders = await client.getOrders(params);

        const result = orders.map((o: any) => ({
          id: o.id,
          symbol: o.symbol,
          qty: o.qty,
          filled_qty: o.filled_qty,
          side: o.side,
          type: o.type,
          time_in_force: o.time_in_force,
          status: o.status,
          submitted_at: o.submitted_at,
          filled_at: o.filled_at,
          limit_price: o.limit_price,
          stop_price: o.stop_price,
          filled_avg_price: o.filled_avg_price,
        }));

        return {
          content: [
            { type: "text" as const, text: JSON.stringify(result, null, 2) },
          ],
        };
      } catch (err) {
        logError("get_orders", err);
        return {
          content: [
            {
              type: "text" as const,
              text: `Error fetching orders: ${err instanceof Error ? err.message : String(err)}`,
            },
          ],
        };
      }
    }
  );

  server.tool(
    "cancel_order",
    "Cancel a specific open order by order ID.",
    {
      order_id: z.string().describe("The order ID to cancel"),
    },
    async ({ order_id }) => {
      try {
        const client = getAlpacaClient();
        await client.cancelOrder(order_id);
        return {
          content: [
            {
              type: "text" as const,
              text: `Order ${order_id} cancelled successfully.`,
            },
          ],
        };
      } catch (err) {
        logError("cancel_order", err);
        return {
          content: [
            {
              type: "text" as const,
              text: `Error cancelling order ${order_id}: ${err instanceof Error ? err.message : String(err)}`,
            },
          ],
        };
      }
    }
  );

  server.tool(
    "cancel_all_orders",
    "Cancel ALL open orders. Use with caution.",
    {},
    async () => {
      try {
        const client = getAlpacaClient();
        const result = await client.cancelAllOrders();
        return {
          content: [
            {
              type: "text" as const,
              text: `All open orders cancelled. ${Array.isArray(result) ? result.length : 0} orders affected.`,
            },
          ],
        };
      } catch (err) {
        logError("cancel_all_orders", err);
        return {
          content: [
            {
              type: "text" as const,
              text: `Error cancelling orders: ${err instanceof Error ? err.message : String(err)}`,
            },
          ],
        };
      }
    }
  );
}
