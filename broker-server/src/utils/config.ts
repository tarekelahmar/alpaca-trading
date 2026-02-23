/**
 * Configuration loaded from environment variables.
 */

export interface RiskConfig {
  maxPositionSize: number;
  maxDailyLoss: number;
  largeOrderThreshold: number;
  positionConcentrationLimit: number;
  maxOrderQty: number;
  maxPortfolioVolatility: number;
  maxSectorExposure: number;
  drawdownKillSwitch: number;
  kellyFractionCap: number;
  maxOpenPositions: number;
}

export function getRiskConfig(): RiskConfig {
  return {
    maxPositionSize: parseFloat(process.env.MAX_POSITION_SIZE || "10000"),
    maxDailyLoss: parseFloat(process.env.MAX_DAILY_LOSS || "5000"),
    largeOrderThreshold: parseInt(process.env.LARGE_ORDER_THRESHOLD || "100", 10),
    positionConcentrationLimit: parseFloat(process.env.POSITION_CONCENTRATION_LIMIT || "0.25"),
    maxOrderQty: parseInt(process.env.MAX_ORDER_QTY || "500", 10),
    maxPortfolioVolatility: parseFloat(process.env.MAX_PORTFOLIO_VOLATILITY || "0.20"),
    maxSectorExposure: parseFloat(process.env.MAX_SECTOR_EXPOSURE || "0.30"),
    drawdownKillSwitch: parseFloat(process.env.DRAWDOWN_KILL_SWITCH || "0.10"),
    kellyFractionCap: parseFloat(process.env.KELLY_FRACTION_CAP || "0.25"),
    maxOpenPositions: parseInt(process.env.MAX_OPEN_POSITIONS || "20", 10),
  };
}

export function isPaperTrading(): boolean {
  return (process.env.PAPER_TRADING || "true").toLowerCase() === "true";
}
