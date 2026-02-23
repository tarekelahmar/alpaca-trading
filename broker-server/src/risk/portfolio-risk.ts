/**
 * Portfolio-level risk constraints.
 * These are checked in addition to per-order risk checks.
 */

export interface PortfolioRiskSummary {
  totalExposure: number;
  exposurePct: number;
  numPositions: number;
  largestPosition: string;
  largestPositionPct: number;
  sectorConcentration: Record<string, number>;
}

export function summarizePortfolioRisk(
  positions: any[],
  equity: number
): PortfolioRiskSummary {
  let totalExposure = 0;
  let largestSymbol = "";
  let largestValue = 0;

  for (const pos of positions) {
    const value = Math.abs(parseFloat(pos.market_value || "0"));
    totalExposure += value;
    if (value > largestValue) {
      largestValue = value;
      largestSymbol = pos.symbol;
    }
  }

  return {
    totalExposure,
    exposurePct: equity > 0 ? totalExposure / equity : 0,
    numPositions: positions.length,
    largestPosition: largestSymbol,
    largestPositionPct: equity > 0 ? largestValue / equity : 0,
    sectorConcentration: {}, // Would need sector data to populate
  };
}
