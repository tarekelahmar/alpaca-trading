import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { getAlpacaClient } from "../utils/alpaca-client.js";
import { logError } from "../utils/logger.js";

export function registerPositionTools(server: McpServer): void {
  server.tool(
    "get_positions",
    "Get all open positions with current market value, P&L, and unrealized gains",
    {},
    async () => {
      try {
        const client = getAlpacaClient();
        const positions = await client.getPositions();

        const result = positions.map((p: any) => ({
          symbol: p.symbol,
          qty: p.qty,
          side: p.side,
          market_value: p.market_value,
          cost_basis: p.cost_basis,
          avg_entry_price: p.avg_entry_price,
          current_price: p.current_price,
          unrealized_pl: p.unrealized_pl,
          unrealized_plpc: p.unrealized_plpc,
          change_today: p.change_today,
        }));

        return {
          content: [
            { type: "text" as const, text: JSON.stringify(result, null, 2) },
          ],
        };
      } catch (err) {
        logError("get_positions", err);
        return {
          content: [
            {
              type: "text" as const,
              text: `Error fetching positions: ${err instanceof Error ? err.message : String(err)}`,
            },
          ],
        };
      }
    }
  );

  server.tool(
    "get_position",
    "Get detail for a single open position",
    {
      symbol: z.string().describe("Stock ticker symbol"),
    },
    async ({ symbol }) => {
      try {
        const client = getAlpacaClient();
        const position = await client.getPosition(symbol.toUpperCase());

        const result = {
          symbol: position.symbol,
          qty: position.qty,
          side: position.side,
          market_value: position.market_value,
          cost_basis: position.cost_basis,
          avg_entry_price: position.avg_entry_price,
          current_price: position.current_price,
          unrealized_pl: position.unrealized_pl,
          unrealized_plpc: position.unrealized_plpc,
          change_today: position.change_today,
        };

        return {
          content: [
            { type: "text" as const, text: JSON.stringify(result, null, 2) },
          ],
        };
      } catch (err) {
        logError("get_position", err);
        return {
          content: [
            {
              type: "text" as const,
              text: `Error fetching position for ${symbol}: ${err instanceof Error ? err.message : String(err)}`,
            },
          ],
        };
      }
    }
  );

  server.tool(
    "close_position",
    "Close or reduce an open position. Returns the closing order details.",
    {
      symbol: z.string().describe("Stock ticker symbol to close"),
      qty: z
        .number()
        .optional()
        .describe(
          "Number of shares to close. Omit to close entire position."
        ),
    },
    async ({ symbol, qty }) => {
      try {
        const client = getAlpacaClient();
        const options = qty ? { qty: qty.toString() } : {};
        const order = await client.closePosition(symbol.toUpperCase(), options);

        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify(
                {
                  order_id: order.id,
                  symbol: order.symbol,
                  qty: order.qty,
                  side: order.side,
                  type: order.type,
                  status: order.status,
                },
                null,
                2
              ),
            },
          ],
        };
      } catch (err) {
        logError("close_position", err);
        return {
          content: [
            {
              type: "text" as const,
              text: `Error closing position ${symbol}: ${err instanceof Error ? err.message : String(err)}`,
            },
          ],
        };
      }
    }
  );

  server.tool(
    "close_all_positions",
    "⚠️ DANGER: Liquidate ALL open positions. This is irreversible.",
    {},
    async () => {
      try {
        const client = getAlpacaClient();
        const result = await client.closeAllPositions({ cancel_orders: true });

        return {
          content: [
            {
              type: "text" as const,
              text:
                `Closed all positions. ${Array.isArray(result) ? result.length : 0} closing orders submitted.\n\n` +
                JSON.stringify(result, null, 2),
            },
          ],
        };
      } catch (err) {
        logError("close_all_positions", err);
        return {
          content: [
            {
              type: "text" as const,
              text: `Error closing all positions: ${err instanceof Error ? err.message : String(err)}`,
            },
          ],
        };
      }
    }
  );
}
