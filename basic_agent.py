"""
BasicAgent
----------

A minimal trading agent for ABIDES.  It wakes up once at noon and sends a
single market order to buy ``order_size`` shares of ``symbol``, then goes idle
for the rest of the day.

The agent illustrates the bare-minimum lifecycle of a ``TradingAgent``
subclass:

  1.  At the kernel-start wakeup we let the base class learn the market
      hours, then schedule a wakeup for noon.
  2.  At noon we send a ``MarketOrder`` (a buy on the BID side) and mark
      ourselves as "done" so we never trade again.
"""

from typing import Optional

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.utils import str_to_ns

from abides_markets.orders import Side
from abides_markets.agents.trading_agent import TradingAgent


class BasicAgent(TradingAgent):
    """
    A toy agent that performs exactly one buy trade at noon.

    Parameters
    ----------
    id, name, type, random_state, starting_cash, log_orders
        Standard ``TradingAgent`` arguments — forwarded to the base class.
    symbol
        The ticker symbol to trade.
    order_size
        Number of shares to buy when the agent fires its single order.
    trade_time
        Absolute nanosecond timestamp at which to trade.  Defaults to noon
        of the simulation date if not supplied; the caller is expected to
        provide ``DATE + str_to_ns("12:00:00")`` so the absolute time is
        unambiguous.
    """

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        order_size: int = 100,
        trade_time: Optional[NanosecondTime] = None,
        log_orders: bool = False,
    ) -> None:
        super().__init__(id, name, type, random_state, starting_cash, log_orders)

        self.symbol: str = symbol
        self.order_size: int = order_size
        # If no absolute trade time is given, fall back to "12:00:00" of the
        # current simulation day.  The kernel start_time gives us the day.
        self.trade_time: Optional[NanosecondTime] = trade_time

        # Whether the single buy order has been sent yet.
        self.has_traded: bool = False

        self.state: str = "AWAITING_WAKEUP"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)

        # Default trade time = noon of the simulation day, derived from the
        # kernel start_time (which is at midnight of the simulation date).
        if self.trade_time is None:
            day_start = (start_time // str_to_ns("24h")) * str_to_ns("24h")
            self.trade_time = day_start + str_to_ns("12:00:00")

    def wakeup(self, current_time: NanosecondTime) -> None:
        # The parent handles exchange-time discovery and the initial
        # market-open wakeup.
        can_trade = super().wakeup(current_time)

        # We trade at most once per simulation.
        if self.has_traded:
            return

        if not can_trade:
            return

        # If it's not yet noon, schedule a wakeup for noon and exit.
        if current_time < self.trade_time:
            self.set_wakeup(self.trade_time)
            return

        # It's noon (or past noon on the first wake-up after noon) — fire
        # the single market buy order and mark ourselves done.
        self.place_market_order(
            symbol=self.symbol,
            quantity=self.order_size,
            side=Side.ASK,
        )
        self.has_traded = True
        self.state = "DONE"

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        # We don't need to react to anything; let the base class handle
        # bookkeeping (market hours, order acknowledgements, …).
        super().receive_message(current_time, sender_id, message)

    def get_wake_frequency(self) -> NanosecondTime:
        # Only relevant for agents that want a periodic poll; we explicitly
        # schedule our single wakeup in `wakeup`, so any value is fine.
        return str_to_ns("1h")
