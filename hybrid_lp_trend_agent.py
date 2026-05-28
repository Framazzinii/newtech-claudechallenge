"""
Pure Liquidity Provider Agent — High Volume / Sharpe-Maximised
==============================================================

Strategia
---------
Pure market making senza alcun segnale direzionale (no trend, no mean
reversion, no Glosten-Milgrom premium). L'edge dell'agente e' la
cattura sistematica del bid-ask spread sui flow non-informati. Per
massimizzare contemporaneamente VOLUME e PROFITTO usiamo:

1.  **Microprice** (Stoikov 2018) come fair value di riferimento, anche'
    del mid: e' un predittore migliore del prossimo prezzo perche' pesa
    il fair value verso il lato del book con meno liquidita'.
2.  **Inside-spread quoting**: quando lo spread di mercato e' largo
    abbastanza, postiamo DENTRO il book (market_bid + 1, market_ask - 1)
    per ottenere priorita' di queue → fill quasi certi quando arrivano
    ordini marketable. Quando il mercato e' gia' stretto (1-2 ticks)
    joiniamo la coda al best bid/best ask.
3.  **Adaptive spread**: lo spread quotato e' max(base_half_spread,
    k_vol * sigma). Quando la vol esplode il nostro spread si allarga
    automaticamente, compensando l'adverse selection senza bisogno di
    altri segnali.
4.  **Avellaneda-Stoikov inventory skew**: skew = -alpha_inv * inventory
    spinge le quote in direzione opposta all'inventario per controllare
    la varianza della posizione (riduce sigma del pnl ⇒ Sharpe sale).
5.  **Refresh ad alta frequenza** (4 secondi): piu' refresh = piu'
    chance di fill = piu' volume.
6.  **Size piccola per quote** (20 share/lato): tanti piccoli fill
    invece di pochi grossi → varianza per trade bassa → Sharpe alto.
7.  **Inventory cap + stop-loss + EOD flatten**: niente posizioni che
    esplodono, niente rischio overnight.

Riferimenti
-----------
- Stoikov (2018), "The micro-price".
- Avellaneda & Stoikov (2008), "High-frequency trading in a limit order
  book": forma chiusa per skew anti-inventario in un mercato OU.
- Ho & Stoll (1981), "Optimal dealer pricing under transactions and
  return uncertainty": inventory management nei dealer markets.

Massimizzazione Sharpe
----------------------
Sharpe = E[r] / sigma(r). Le scelte di design:
- Inside quoting ⇒ piu' fill ⇒ E[r] sale.
- Vol scaling sullo spread ⇒ sigma cala in regimi rumorosi.
- Inventory skew aggressivo ⇒ sigma cala (posizione vincolata).
- Stop-loss + EOD flatten ⇒ code della distribuzione di pnl troncate.
- Tante operazioni piccole ⇒ legge dei grandi numeri ⇒ sigma cala.
"""

from typing import List, Optional

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.utils import str_to_ns

from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import Side
from abides_markets.agents.trading_agent import TradingAgent


class HybridLPTrendAgent(TradingAgent):
    """
    Pure liquidity provider con inside-spread quoting e Avellaneda-Stoikov
    inventory skew.

    Nota: il nome della classe e' rimasto 'HybridLPTrendAgent' per
    retrocompatibilita' con il notebook esistente. La strategia interna e'
    market making puro — niente trend following, niente mean reversion,
    niente premio Glosten-Milgrom.  Vedi docstring del modulo per dettagli.

    Parametri principali
    --------------------
    base_half_spread
        Half-spread minimo in cents. Floor sullo spread quotato.
    k_vol
        Premio sullo spread proporzionale alla volatilita' realizzata.
        Lo spread effettivo e' max(base_half_spread, k_vol * sigma).
    alpha_inv
        Skew cents/share contro l'inventario (Avellaneda-Stoikov).
    order_size, max_inv
        Size per quote e inventario massimo assoluto.
    wake_up_freq
        Cadenza di refresh delle quote (default 4s = molto reattivo).
    stop_loss_cents, eod_flatten_offset
        Risk management standard.
    """

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        # spread (adaptive: base + vol component)
        base_half_spread: int = 4,
        k_vol: float = 2.5,
        # inventory skew (Avellaneda-Stoikov)
        alpha_inv: float = 0.4,
        # size & inventory
        order_size: int = 20,
        max_inv: int = 300,
        # vol tracking
        vol_window: int = 60,
        # frequency: 4s for high refresh rate
        wake_up_freq: NanosecondTime = str_to_ns("4s"),
        # risk
        stop_loss_cents: int = 50000,
        eod_flatten_offset: NanosecondTime = str_to_ns("5min"),
        log_orders: bool = False,
    ) -> None:
        super().__init__(id, name, type, random_state, starting_cash, log_orders)

        self.symbol = symbol
        self.starting_cash = starting_cash

        self.base_half_spread = base_half_spread
        self.k_vol = k_vol
        self.alpha_inv = alpha_inv
        self.order_size = order_size
        self.max_inv = max_inv
        self.vol_window = vol_window
        self.wake_up_freq = wake_up_freq
        self.stop_loss_cents = stop_loss_cents
        self.eod_flatten_offset = eod_flatten_offset

        # state
        self.mid_history: List[float] = []
        self.bid_order_id: Optional[int] = None
        self.ask_order_id: Optional[int] = None
        self.eod_flatten_time: Optional[NanosecondTime] = None
        self.stopped_out = False
        self.state = "AWAITING_WAKEUP"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)
        day_start = (start_time // str_to_ns("24h")) * str_to_ns("24h")
        market_close = day_start + str_to_ns("17:30:00")
        self.eod_flatten_time = market_close - self.eod_flatten_offset

    def wakeup(self, current_time: NanosecondTime) -> None:
        can_trade = super().wakeup(current_time)
        if not can_trade or self.stopped_out:
            return

        # EOD: cancel + flatten + stop trading for the day.
        if (
            self.eod_flatten_time is not None
            and current_time >= self.eod_flatten_time
        ):
            self._cancel_live_quotes()
            self._flatten_inventory()
            self.stopped_out = True
            return

        # Routine: cancel old quotes, then query spread.
        self._cancel_live_quotes()
        self.get_current_spread(self.symbol)
        self.state = "AWAITING_SPREAD"

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        super().receive_message(current_time, sender_id, message)

        if self.state != "AWAITING_SPREAD" or not isinstance(
            message, QuerySpreadResponseMsg
        ):
            return

        bid, bid_vol, ask, ask_vol = self.get_known_bid_ask(self.symbol)
        bid_vol = bid_vol or 0
        ask_vol = ask_vol or 0

        if bid and ask and (bid_vol + ask_vol) > 0:
            mid = (bid + ask) / 2.0
            # Microprice: peso il fair value verso il lato meno liquido.
            microprice = (bid * ask_vol + ask * bid_vol) / (bid_vol + ask_vol)

            self.mid_history.append(mid)
            if len(self.mid_history) > self.vol_window * 3:
                self.mid_history = self.mid_history[-self.vol_window * 2:]

            # Stop-loss giornaliero: trunca le code della distribuzione di pnl.
            if self._check_stop_loss():
                self._cancel_live_quotes()
                self._flatten_inventory()
                self.stopped_out = True
                self.state = "AWAITING_WAKEUP"
                self.set_wakeup(current_time + self.wake_up_freq)
                return

            self._post_quotes(microprice, bid, ask)

        self.set_wakeup(current_time + self.wake_up_freq)
        self.state = "AWAITING_WAKEUP"

    def get_wake_frequency(self) -> NanosecondTime:
        return self.wake_up_freq

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _realized_vol(self) -> float:
        """Vol realizzata recente del mid (std, in cents)."""
        if len(self.mid_history) < 5:
            return 0.0
        n = min(self.vol_window, len(self.mid_history))
        return float(np.std(self.mid_history[-n:]))

    def _check_stop_loss(self) -> bool:
        try:
            mtm = self.mark_to_market(self.holdings)
        except Exception:
            return False
        return (mtm - self.starting_cash) < -self.stop_loss_cents

    def _cancel_live_quotes(self) -> None:
        if self.bid_order_id is not None and self.bid_order_id in self.orders:
            self.cancel_order(self.orders[self.bid_order_id])
        if self.ask_order_id is not None and self.ask_order_id in self.orders:
            self.cancel_order(self.orders[self.ask_order_id])
        self.bid_order_id = None
        self.ask_order_id = None

    def _flatten_inventory(self) -> None:
        inv = self.holdings.get(self.symbol, 0)
        if inv > 0:
            self.place_market_order(self.symbol, quantity=inv, side=Side.ASK)
        elif inv < 0:
            self.place_market_order(self.symbol, quantity=-inv, side=Side.BID)

    # ------------------------------------------------------------------
    # Quoting logic — pure LP con inside-spread + inventory skew
    # ------------------------------------------------------------------

    def _post_quotes(
        self, microprice: float, market_bid: int, market_ask: int
    ) -> None:
        inventory = self.holdings.get(self.symbol, 0)
        market_spread = market_ask - market_bid

        # Spread quotato: adattivo sulla volatilita' realizzata.
        vol = self._realized_vol()
        desired_half_spread = max(float(self.base_half_spread), self.k_vol * vol)

        # Inventory skew (Avellaneda-Stoikov): shift sul mid quotato per
        # leaning against position. Long ⇒ skew negativo ⇒ quotes giu'
        # ⇒ piu' facile vendere, piu' difficile comprare ancora.
        skew = -self.alpha_inv * inventory
        quoted_mid = microprice + skew

        target_bid = int(round(quoted_mid - desired_half_spread))
        target_ask = int(round(quoted_mid + desired_half_spread))

        # === inside-spread quoting ===
        # Se lo spread di mercato e' largo (>= 2 ticks oltre il nostro
        # desired spread + 2), abbiamo spazio per postare DENTRO il book
        # ottenendo priorita' di queue (best bid / best ask).
        # Altrimenti joiniamo la coda al best bid/best ask attuali.
        if market_spread >= 2 * self.base_half_spread + 2:
            # mercato largo: posta inside
            our_bid = max(target_bid, market_bid + 1)
            our_ask = min(target_ask, market_ask - 1)
        else:
            # mercato gia' stretto: joina la coda al best
            our_bid = market_bid
            our_ask = market_ask

        # Safety: garantisci bid < ask.
        if our_ask <= our_bid:
            our_ask = our_bid + 1

        # === posta bid ===
        if inventory < self.max_inv:
            bid_qty = min(self.order_size, self.max_inv - inventory)
            if bid_qty > 0:
                self.place_limit_order(
                    self.symbol,
                    quantity=bid_qty,
                    side=Side.BID,
                    limit_price=our_bid,
                )
                if self.orders:
                    self.bid_order_id = max(self.orders.keys())

        # === posta ask ===
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
