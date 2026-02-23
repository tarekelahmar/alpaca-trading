import { log } from "../utils/logger.js";

export class DailyPnLTracker {
  private startOfDayEquity: number | null = null;
  private tradingDate: string | null = null;

  recordStartOfDay(equity: number): void {
    const today = new Date().toISOString().split("T")[0];
    if (this.tradingDate !== today) {
      this.tradingDate = today;
      this.startOfDayEquity = equity;
      log(`Start of day equity recorded: $${equity.toFixed(2)}`);
    }
  }

  getDailyPnL(currentEquity: number): number {
    if (this.startOfDayEquity === null) {
      this.recordStartOfDay(currentEquity);
      return 0;
    }
    return currentEquity - this.startOfDayEquity;
  }

  getStartOfDayEquity(): number | null {
    return this.startOfDayEquity;
  }
}

let _tracker: DailyPnLTracker | null = null;

export function getDailyTracker(): DailyPnLTracker {
  if (!_tracker) _tracker = new DailyPnLTracker();
  return _tracker;
}
