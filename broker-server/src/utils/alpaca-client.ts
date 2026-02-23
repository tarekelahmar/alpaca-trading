/**
 * Alpaca client singleton factory.
 * Using `any` type because @alpacahq/alpaca-trade-api has incomplete TS definitions.
 */

// eslint-disable-next-line @typescript-eslint/no-require-imports
import Alpaca from "@alpacahq/alpaca-trade-api";
import { isPaperTrading } from "./config.js";
import { log } from "./logger.js";

let _client: any = null;

export function getAlpacaClient(): any {
  if (_client) return _client;

  const keyId = process.env.ALPACA_API_KEY_ID;
  const secretKey = process.env.ALPACA_API_SECRET_KEY;

  if (!keyId || !secretKey) {
    console.error(
      "FATAL: ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY must be set"
    );
    process.exit(1);
  }

  const paper = isPaperTrading();
  log(`Initializing Alpaca client (paper=${paper})`);

  _client = new (Alpaca as any)({
    keyId,
    secretKey,
    paper,
    usePolygon: false,
  });

  return _client;
}
