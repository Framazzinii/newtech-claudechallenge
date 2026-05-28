"""
Trend-Following (Breakout) Trading Agent

Strategy brief: Trend follower
Idea: when price breaks out of a recent trading range it tends to keep moving in
that direction.  We track a short exponential moving average (EMA) and a long
EMA.  A bullish crossover (short EMA crosses above long EMA) triggers a buy;
a bearish crossover triggers a sell.  We add a simple stop-loss based on the
recent price range (ATR-like) to limit downside.

Parameters we tuned:
  - fast_span:    span of the fast EMA in observations (default 10)
  - slow_span:    span of the slow EMA in observations (default 30)
  - order_size:   shares per trade (default 40)
  - max_pos:      maximum absolute net position (default 200 shares)
  - stop_ticks:   stop-loss width in cents (default 20)
  - wake_up_freq: polling interval
"""

from typing import List, Optional

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.utils import str_to_ns

from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import Side
from abides_markets.agents.trading_agent import TradingAgent


class TrendFollowerAgent(TradingAgent):
    """
    EMA-crossover trend-following agent with a rudimentary stop-loss.

    At each wake-up the agent polls the spread, records the mid-price, and
    recomputes both EMAs.  A signal is generated when the fast EMA crosses the
    slow EMA.  We only trade once per crossover (we require the signal to
    *change* from the previous bar) to avoid churning.
    """

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        fast_span: int = 10,
        slow_span: int = 30,
        order_size: int = 40,
        max_pos: int = 200,
        stop_ticks: int = 20,
        wake_up_freq: NanosecondTime = str_to_ns("60s"),
        log_orders: bool = False,
    ) -> None:
        super().__init__(id, name, type, random_state, starting_cash, log_orders)

        self.symbol = symbol
        self.fast_span = fast_span
        self.slow_span = slow_span
        self.order_size = order_size
        self.max_pos = max_pos
        self.stop_ticks = stop_ticks
        self.wake_up_freq = wake_up_freq

        self.mid_prices: List[float] = []

        # EMA state (updated incrementally)
        self.fast_ema: Optional[float] = None
        self.slow_ema: Optional[float] = None
        self.fast_alpha = 2.0 / (fast_span + 1)
        self.slow_alpha = 2.0 / (slow_span + 1)

        # Track last signal so we trade only on crossover, not on every bar
        self.last_signal: Optional[str] = None   # "BUY" | "SELL" | None

        # Keep entry price for the stop-loss
        self.entry_price: Optional[float] = None

        self.state = "AWAITING_WAKEUP"

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)

    def wakeup(self, current_time: NanosecondTime) -> None:
        can_trade = super().wakeup(current_time)
        if can_trade:
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"

    def get_wake_frequency(self) -> NanosecondTime:
        return self.wake_up_freq

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        super().receive_message(current_time, sender_id, message)

        if (
            self.state == "AWAITING_SPREAD"
            and isinstance(message, QuerySpreadResponseMsg)
        ):
            bid, _, ask, _ = self.get_known_bid_ask(self.symbol)
            if bid and ask:
                self._update_ema_and_trade(bid, ask)
            self.set_wakeup(current_time + self.wake_up_freq)
            self.state = "AWAITING_WAKEUP"

    # ------------------------------------------------------------------
    # Strategy logic
    # ------------------------------------------------------------------

    def _update_ema(self, price: float) -> None:
        """Incremental EMA update."""
        if self.fast_ema is None:
            self.fast_ema = price
            self.slow_ema = price
        else:
            self.fast_ema = self.fast_alpha * price + (1 - self.fast_alpha) * self.fast_ema
            self.slow_ema = self.slow_alpha * price + (1 - self.slow_alpha) * self.slow_ema

    def _update_ema_and_trade(self, bid: int, ask: int) -> None:
        mid = (bid + ask) / 2.0
        self.mid_prices.append(mid)
        self._update_ema(mid)

        # Need enough history for both EMAs to warm up
        if len(self.mid_prices) < self.slow_span:
            return

        current_pos = self.holdings.get(self.symbol, 0)

        # --- Stop-loss check ---
        if self.entry_price is not None and current_pos != 0:
            if current_pos > 0 and mid < self.entry_price - self.stop_ticks:
                # Long position hit stop → close
                self.place_limit_order(
                    self.symbol,
                    quantity=abs(current_pos),
                    side=Side.ASK,
                    limit_price=bid,
                )
                self.entry_price = None
                self.last_signal = None
                return
            elif current_pos < 0 and mid > self.entry_price + self.stop_ticks:
                # Short position hit stop → close
                self.place_limit_order(
                    self.symbol,
                    quantity=abs(current_pos),
                    side=Side.BID,
                    limit_price=ask,
                )
                self.entry_price = None
                self.last_signal = None
                return

        # --- Crossover signal ---
        if self.fast_ema > self.slow_ema:
            signal = "BUY"
        else:
            signal = "SELL"

        # Only act on a new crossover
        if signal == self.last_signal:
            return
        self.last_signal = signal

        if signal == "BUY" and current_pos < self.max_pos:
            qty = min(self.order_size, self.max_pos - current_pos)
            if qty > 0:
                self.place_limit_order(
                    self.symbol,
                    quantity=qty,
                    side=Side.BID,
                    limit_price=ask,
                )
                self.entry_price = mid

        elif signal == "SELL" and current_pos > -self.max_pos:
            qty = min(self.order_size, self.max_pos + current_pos)
            if qty > 0:
                self.place_limit_order(
                    self.symbol,
                    quantity=qty,
                    side=Side.ASK,
                    limit_price=bid,
                )
                self.entry_price = mid
