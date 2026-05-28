"""
Liquidity Provider (Simple Market-Maker) Agent

Strategy brief: Liquidity provider
Idea: earn the bid-ask spread by posting a limit buy slightly below and a
limit sell slightly above the current mid-price, then refreshing quotes
periodically.  We manage inventory risk by skewing quotes: when we are long
we lower both legs to attract sellers; when we are short we raise both legs
to attract buyers.  We also cancel outstanding orders before reposting to
avoid accumulating stale quotes.

Parameters we tuned:
  - half_spread:   half-width of our posted spread in cents (default 5)
  - order_size:    shares at each level (default 30)
  - max_inv:       inventory limit — beyond this we stop adding on that side
  - skew_per_lot:  extra cents of skew per lot of inventory (default 0.5)
  - wake_up_freq:  how often to refresh quotes
"""

from typing import List, Optional

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.utils import str_to_ns

from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import Side
from abides_markets.agents.trading_agent import TradingAgent


class LiquidityProviderAgent(TradingAgent):
    """
    A simple symmetric market-maker that posts one bid and one ask around the
    mid-price and refreshes them every `wake_up_freq` nanoseconds.

    Inventory skew: the posted mid is shifted by `skew_per_lot * inventory`
    cents away from zero, so the agent naturally leans against its position.
    """

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        half_spread: int = 5,
        order_size: int = 30,
        max_inv: int = 300,
        skew_per_lot: float = 0.5,
        wake_up_freq: NanosecondTime = str_to_ns("30s"),
        log_orders: bool = False,
    ) -> None:
        super().__init__(id, name, type, random_state, starting_cash, log_orders)

        self.symbol = symbol
        self.half_spread = half_spread
        self.order_size = order_size
        self.max_inv = max_inv
        self.skew_per_lot = skew_per_lot
        self.wake_up_freq = wake_up_freq

        # IDs of our currently outstanding quote orders (so we can cancel them)
        self.bid_order_id: Optional[int] = None
        self.ask_order_id: Optional[int] = None

        self.state = "AWAITING_WAKEUP"

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)

    def wakeup(self, current_time: NanosecondTime) -> None:
        can_trade = super().wakeup(current_time)
        if can_trade:
            # Step 1: cancel any live quotes before reposting
            self._cancel_live_quotes()
            # Step 2: fetch the current spread so we can set new quotes
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
                self._post_quotes(bid, ask)
            self.set_wakeup(current_time + self.wake_up_freq)
            self.state = "AWAITING_WAKEUP"

    # ------------------------------------------------------------------
    # Strategy logic
    # ------------------------------------------------------------------

    def _cancel_live_quotes(self) -> None:
        """Cancel the previously posted bid and ask if they are still open."""
        if self.bid_order_id is not None and self.bid_order_id in self.orders:
            self.cancel_order(self.orders[self.bid_order_id])
        if self.ask_order_id is not None and self.ask_order_id in self.orders:
            self.cancel_order(self.orders[self.ask_order_id])
        self.bid_order_id = None
        self.ask_order_id = None

    def _post_quotes(self, market_bid: int, market_ask: int) -> None:
        """
        Compute skewed mid-price and place a new bid and ask around it.

        The skew shifts our quoted mid in the direction that reduces inventory:
          - long inventory  → quoted mid moves down  (cheaper ask → sell more)
          - short inventory → quoted mid moves up    (higher bid → buy more)
        """
        mid = (market_bid + market_ask) / 2.0
        inventory = self.holdings.get(self.symbol, 0)

        # Inventory skew (in cents)
        skew = -self.skew_per_lot * inventory
        quoted_mid = mid + skew

        our_bid = int(round(quoted_mid - self.half_spread))
        our_ask = int(round(quoted_mid + self.half_spread))

        # Sanity check: our quotes must be inside or at the market spread
        # so we don't cross the book.
        our_bid = min(our_bid, market_bid)
        our_ask = max(our_ask, market_ask)

        # Don't post on the buy side if already too long
        if inventory < self.max_inv:
            bid_qty = min(self.order_size, self.max_inv - inventory)
            if bid_qty > 0:
                self.place_limit_order(
                    self.symbol,
                    quantity=bid_qty,
                    side=Side.BID,
                    limit_price=our_bid,
                )
                # Remember the latest order ID placed
                if self.orders:
                    self.bid_order_id = max(self.orders.keys())

        # Don't post on the sell side if already too short
        if inventory > -self.max_inv:
            ask_qty = min(self.order_size, self.max_inv + inventory)
            if ask_qty > 0:
                self.place_limit_order(
                    self.symbol,
                    quantity=ask_qty,
                    side=Side.ASK,
                    limit_price=our_ask,
                )
                if self.orders:
                    self.ask_order_id = max(self.orders.keys())
