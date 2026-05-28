"""
Mean-Reversion Trading Agent

Strategy brief: Mean-reversion trader
Idea: prices tend to revert to their recent average. When the current mid-price
deviates significantly from its rolling mean we trade against the deviation —
buy when the price is too low, sell when it is too high — and close the position
once the price normalises.

Parameters we tuned:
  - window:      lookback window for the rolling mean (default 30 observations)
  - threshold:   how many "ticks" of deviation trigger a trade (default 10)
  - max_pos:     maximum net position we are willing to hold (default 200 shares)
  - wake_up_freq: how often to wake up and check the market
"""

from typing import List, Optional

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.utils import str_to_ns

from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import Side
from abides_markets.agents.trading_agent import TradingAgent


class MeanReversionAgent(TradingAgent):
    """
    Polls the best bid/ask at a fixed frequency.  Maintains a rolling window of
    mid-price observations and computes a simple arithmetic mean.  When the
    current mid deviates from that mean by more than `threshold` it places a
    single limit order betting on reversion.  A crude position limit prevents
    the agent from accumulating a dangerously large inventory.
    """

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        window: int = 30,
        threshold: int = 10,
        order_size: int = 50,
        max_pos: int = 200,
        wake_up_freq: NanosecondTime = str_to_ns("60s"),
        log_orders: bool = False,
    ) -> None:
        super().__init__(id, name, type, random_state, starting_cash, log_orders)

        self.symbol = symbol
        self.window = window          # rolling mean window length
        self.threshold = threshold    # deviation in cents that triggers a trade
        self.order_size = order_size  # shares per order
        self.max_pos = max_pos        # maximum absolute position
        self.wake_up_freq = wake_up_freq

        self.mid_prices: List[float] = []   # history of mid-price observations
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
                self._update_and_trade(bid, ask)
            # schedule next wake-up
            self.set_wakeup(current_time + self.wake_up_freq)
            self.state = "AWAITING_WAKEUP"

    # ------------------------------------------------------------------
    # Strategy logic
    # ------------------------------------------------------------------

    def _update_and_trade(self, bid: int, ask: int) -> None:
        mid = (bid + ask) / 2.0
        self.mid_prices.append(mid)

        # Need at least `window` observations before we can act
        if len(self.mid_prices) < self.window:
            return

        # Keep the list from growing unboundedly
        if len(self.mid_prices) > self.window * 2:
            self.mid_prices = self.mid_prices[-self.window:]

        rolling_mean = np.mean(self.mid_prices[-self.window:])
        deviation = mid - rolling_mean

        # Current net position in the symbol
        current_pos = self.holdings.get(self.symbol, 0)

        if deviation > self.threshold:
            # Price is above mean → expect it to fall → sell
            if current_pos > -self.max_pos:
                qty = min(self.order_size, self.max_pos + current_pos)
                if qty > 0:
                    self.place_limit_order(
                        self.symbol,
                        quantity=qty,
                        side=Side.ASK,
                        limit_price=bid,  # sell at bid for fast execution
                    )

        elif deviation < -self.threshold:
            # Price is below mean → expect it to rise → buy
            if current_pos < self.max_pos:
                qty = min(self.order_size, self.max_pos - current_pos)
                if qty > 0:
                    self.place_limit_order(
                        self.symbol,
                        quantity=qty,
                        side=Side.BID,
                        limit_price=ask,  # buy at ask for fast execution
                    )

        # If we hold inventory and price is back near the mean, unwind
        elif abs(deviation) < self.threshold / 2:
            if current_pos > 0:
                self.place_limit_order(
                    self.symbol,
                    quantity=min(self.order_size, current_pos),
                    side=Side.ASK,
                    limit_price=bid,
                )
            elif current_pos < 0:
                self.place_limit_order(
                    self.symbol,
                    quantity=min(self.order_size, -current_pos),
                    side=Side.BID,
                    limit_price=ask,
                )
