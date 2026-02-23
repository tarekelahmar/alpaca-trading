import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { registerAccountTools } from "./tools/account.js";
import { registerPositionTools } from "./tools/positions.js";
import { registerOrderTools } from "./tools/orders.js";
import { isPaperTrading } from "./utils/config.js";
import { log } from "./utils/logger.js";

const server = new McpServer({
  name: "alpaca-broker",
  version: "1.0.0",
});

// Register all tools
registerAccountTools(server);
registerPositionTools(server);
registerOrderTools(server);

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  log(`Alpaca Broker MCP Server running (paper=${isPaperTrading()})`);
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
