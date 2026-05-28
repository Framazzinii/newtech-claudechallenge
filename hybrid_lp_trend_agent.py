"""
Hybrid Liquidity Provider + Trend-Aware Inventory Management Agent
==================================================================

L'agente unisce market making (liquidity provision) con gestione
dell'inventario informata da segnali di trend (EMA crossover) e da metriche
di microstruttura del limit order book.

Cornice teorica
---------------
1.  Glosten & Milgrom (1985) — "Bid, Ask and Transaction Prices in a
    Specialist Market with Heterogeneously Informed Traders".
    Lo spread bid-ask quotato dal market maker non compensa solo costi di
    processing/inventario, ma soprattutto le perdite attese contro i trader
    informati (asimmetria informativa). In un mondo dove convivono insider
    (osservano il fair value V) e noise traders (trading random), il MM
    deve porre ask = E[V | arriva un buy] e bid = E[V | arriva un sell].
    Implicazione operativa: lo spread quotato e' ADATTIVO — piu largo
    quando aumenta la probabilita di trovare flusso informato.

2.  Easley, Kiefer, O'Hara, Paperman (1996) — PIN (Probability of Informed
    Trading).  Easley, Lopez de Prado, O'Hara (2012) — VPIN
    (Volume-synchronized PIN). L'imbalance del flusso di ordini e' una
    proxy della PIN: forte imbalance ⇒ alta probabilita di flusso informato
    ⇒ flusso "tossico" per il MM, che dovrebbe ritirare le quote o
    allargare lo spread.

3.  Stoikov (2018) — "The micro-price: A high frequency estimator of
    future prices".  Il microprice
        microprice = (bid * V_ask + ask * V_bid) / (V_bid + V_ask)
    e' un fair value pesato per i volumi al best level che predice il
    prossimo prezzo meglio del semplice mid.  Lo usiamo come reference
    per le nostre quote.

4.  Avellaneda & Stoikov (2008) — il MM ottimo skewa le quote contro il
    proprio inventario per controllarne la volatilita (inventory risk).

Strategia in 11 passi
---------------------
A ogni wake-up (~ogni 8 secondi):
 1. Cancella le quote vecchie ancora vive sul book.
 2. Interroga lo spread corrente; in receive_message su QuerySpreadResponseMsg:
 3. Legge (bid, V_bid, ask, V_ask) dal book.
 4. Calcola microprice (Stoikov) e OBI = (V_bid - V_ask)/(V_bid + V_ask).
 5. Aggiorna EMA fast/slow del microprice per il segnale di trend.
 6. Calcola la volatilita realizzata recente del mid.
 7. Risk checks:
      - Stop-loss giornaliero (M2M).
      - Toxic-flow circuit breaker (rolling |OBI| alto ⇒ pull quotes).
 8. Spread quotato adattivo (Glosten-Milgrom):
      half_spread_q = max(base, k_vol * vol, k_adv * |OBI| * 2*base)
 9. Skew composito (cents):
      skew = -alpha_inv * inventory
             + alpha_trend * trend_signal * sign(inventory)
             + alpha_obi * OBI * base_half_spread
10. Posta bid/ask attorno a (microprice + skew), quantita cappata da max_inv.
11. 5 minuti prima del close: cancella tutto e liquida inventario residuo
    via market order (no rischio overnight).
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
    Trend-aware liquidity provider con spread Glosten-Milgrom e microprice.

    Parameters
    ----------
    id, name, type, random_state, starting_cash, log_orders
        Argomenti standard di TradingAgent.
    symbol
        Ticker su cui operare (default ABM in RMSC04).
    base_half_spread
        Half-spread minimo in cents.  Lo spread effettivo puo crescere
        sopra questo per volatilita o adverse selection.
    k_vol
        Moltiplicatore per la componente "volatilita realizzata" dello
        spread.
    k_adv
        Moltiplicatore per la componente Glosten-Milgrom (|OBI|) dello
        spread.
    alpha_inv
        Skew (cents per share di inventario) — leans against position.
    alpha_trend
        Skew per segnale di trend (cents).  Quando trend e inventario
        sono allineati l'agente lascia correre i profitti.
    alpha_obi
        Skew per OBI (proporzionale a base_half_spread) — leans away
        from informed flow.
    order_size
        Shares per quote (lato).
    max_inv
        Inventario massimo (assoluto).  Sopra non posta piu sul lato che
        accumula.
    fast_span, slow_span, trend_tol
        Parametri delle due EMA del microprice e tolleranza per attivare
        il segnale di trend.
    vol_window
        Lookback (osservazioni) per la volatilita realizzata.
    obi_toxic_window, obi_toxic_threshold
        Finestra e soglia per il circuit breaker VPIN-like.  Se
        media(|OBI|) supera la soglia, niente quote in questo round.
    wake_up_freq
        Cadenza di refresh delle quote.
    stop_loss_cents
        Drawdown M2M massimo (cents) prima di flattenare la posizione e
        smettere di tradare per la giornata.
    eod_flatten_offset
        Anticipo rispetto al close del mercato per liquidare l'inventario
        ed evitare overnight risk.
    """

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        # spread / quote — spread piu' largo per coprire adverse selection
        base_half_spread: int = 12,       # ↑ da 6 (combatte adverse selection)
        k_vol: float = 2.5,               # ↑ da 1.5 (piu' premio per vol)
        k_adv: float = 1.8,               # ↑ da 0.8 (piu' premio Glosten-Milgrom)
        # skew
        alpha_inv: float = 0.5,           # ↑ da 0.4
        alpha_trend: float = 3.0,
        alpha_obi: float = 2.5,           # ↑ da 1.2 (lean piu' forte da informed flow)
        # size & inventory
        order_size: int = 20,             # ↓ da 25 (size piu' piccola = meno adverse selection)
        max_inv: int = 200,               # ↓ da 300 (turnover piu' veloce)
        # EMA per trend
        fast_span: int = 12,
        slow_span: int = 40,
        trend_tol: float = 0.0002,
        # rolling vol & OBI
        vol_window: int = 30,
        obi_toxic_window: int = 8,        # ↓ da 10 (reagisce piu' in fretta)
        obi_toxic_threshold: float = 0.45,  # ↓ da 0.6 (pull quotes prima)
        # frequency
        wake_up_freq: NanosecondTime = str_to_ns("8s"),
        # risk
        stop_loss_cents: int = 40000,
        eod_flatten_offset: NanosecondTime = str_to_ns("5min"),
        log_orders: bool = False,
    ) -> None:
        super().__init__(id, name, type, random_state, starting_cash, log_orders)

        self.symbol = symbol
        self.starting_cash = starting_cash

        # spread / quote
        self.base_half_spread = base_half_spread
        self.k_vol = k_vol
        self.k_adv = k_adv

        # skew
        self.alpha_inv = alpha_inv
        self.alpha_trend = alpha_trend
        self.alpha_obi = alpha_obi

        # size & inventory
        self.order_size = order_size
        self.max_inv = max_inv

        # EMA
        self.fast_span = fast_span
        self.slow_span = slow_span
        self.fast_alpha = 2.0 / (fast_span + 1)
        self.slow_alpha = 2.0 / (slow_span + 1)
        self.fast_ema: Optional[float] = None
        self.slow_ema: Optional[float] = None
        self.trend_tol = trend_tol

        # rolling vol & OBI
        self.vol_window = vol_window
        self.obi_toxic_window = obi_toxic_window
        self.obi_toxic_threshold = obi_toxic_threshold
        self.mid_history: List[float] = []
        self.obi_history: List[float] = []

        # ordini live (per cancel prima di reposting)
        self.bid_order_id: Optional[int] = None
        self.ask_order_id: Optional[int] = None

        # frequency & risk
        self.wake_up_freq = wake_up_freq
        self.stop_loss_cents = stop_loss_cents
        self.eod_flatten_offset = eod_flatten_offset

        # runtime state
        self.eod_flatten_time: Optional[NanosecondTime] = None
        self.stopped_out = False
        self.state = "AWAITING_WAKEUP"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)
        # EOD flatten time = 17:30:00 - eod_flatten_offset, dello stesso
        # giorno di simulazione.  start_time e' la mezzanotte del giorno.
        day_start = (start_time // str_to_ns("24h")) * str_to_ns("24h")
        market_close = day_start + str_to_ns("17:30:00")
        self.eod_flatten_time = market_close - self.eod_flatten_offset

    def wakeup(self, current_time: NanosecondTime) -> None:
        can_trade = super().wakeup(current_time)
        if not can_trade or self.stopped_out:
            return

        # Se siamo entrati nella finestra EOD: flatten e stop per la giornata.
        if (
            self.eod_flatten_time is not None
            and current_time >= self.eod_flatten_time
        ):
            self._cancel_live_quotes()
            self._flatten_inventory()
            self.stopped_out = True
            return

        # Routine normale: cancel + query spread.
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

        # Procediamo solo se entrambi i lati del book sono validi.
        if bid and ask and (bid_vol + ask_vol) > 0:
            mid = (bid + ask) / 2.0
            # Microprice (Stoikov): peso il fair value verso il lato meno
            # liquido — l'altro lato sta "vincendo lo squeeze" e probabilmente
            # il prossimo trade lo eseguira'.
            microprice = (bid * ask_vol + ask * bid_vol) / (bid_vol + ask_vol)
            # OBI in [-1, +1]: > 0 ⇒ pressione di acquisto (informed buyers?).
            obi = (bid_vol - ask_vol) / (bid_vol + ask_vol)

            self._update_ema(microprice)

            self.mid_history.append(mid)
            if len(self.mid_history) > self.vol_window * 2:
                self.mid_history = self.mid_history[-self.vol_window:]
            self.obi_history.append(obi)
            if len(self.obi_history) > self.obi_toxic_window * 2:
                self.obi_history = self.obi_history[-self.obi_toxic_window:]

            # Stop-loss giornaliero: se il drawdown M2M e' troppo grande
            # chiudiamo tutto e usciamo dal mercato per il resto del giorno.
            if self._check_stop_loss():
                self._cancel_live_quotes()
                self._flatten_inventory()
                self.stopped_out = True
                self.state = "AWAITING_WAKEUP"
                # rischediamo lo stesso per essere svegliati di nuovo
                self.set_wakeup(current_time + self.wake_up_freq)
                return

            # Toxic-flow check (VPIN-like): se |OBI| medio sopra soglia il
            # flusso e' troppo direzionale per fare MM in sicurezza —
            # saltiamo questo round.
            if not self._is_toxic_flow():
                self._post_quotes(microprice, obi, bid, ask)

        # Schedule prossimo refresh.
        self.set_wakeup(current_time + self.wake_up_freq)
        self.state = "AWAITING_WAKEUP"

    def get_wake_frequency(self) -> NanosecondTime:
        return self.wake_up_freq

    # ------------------------------------------------------------------
    # Helpers — micro-struttura
    # ------------------------------------------------------------------

    def _update_ema(self, price: float) -> None:
        """EMA incrementale (fast & slow) sul microprice."""
        if self.fast_ema is None:
            self.fast_ema = price
            self.slow_ema = price
        else:
            self.fast_ema = (
                self.fast_alpha * price + (1 - self.fast_alpha) * self.fast_ema
            )
            self.slow_ema = (
                self.slow_alpha * price + (1 - self.slow_alpha) * self.slow_ema
            )

    def _trend_signal(self) -> int:
        """+1 uptrend, -1 downtrend, 0 flat — basato su gap delle EMA."""
        if self.fast_ema is None or self.slow_ema is None:
            return 0
        gap = self.fast_ema - self.slow_ema
        threshold = self.slow_ema * self.trend_tol
        if gap > threshold:
            return 1
        if gap < -threshold:
            return -1
        return 0

    def _realized_vol(self) -> float:
        """Volatilita realizzata recente del mid (std, in cents)."""
        if len(self.mid_history) < 5:
            return 0.0
        recent = self.mid_history[-self.vol_window:]
        return float(np.std(recent))

    def _is_toxic_flow(self) -> bool:
        """VPIN-like: rolling mean(|OBI|) sopra soglia ⇒ flusso informato."""
        if len(self.obi_history) < self.obi_toxic_window:
            return False
        recent = self.obi_history[-self.obi_toxic_window:]
        mean_abs_obi = float(np.mean(np.abs(recent)))
        return mean_abs_obi > self.obi_toxic_threshold

    # ------------------------------------------------------------------
    # Helpers — risk & ordini
    # ------------------------------------------------------------------

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
        """Chiude la posizione residua via market order."""
        inv = self.holdings.get(self.symbol, 0)
        if inv > 0:
            self.place_market_order(self.symbol, quantity=inv, side=Side.ASK)
        elif inv < 0:
            self.place_market_order(self.symbol, quantity=-inv, side=Side.BID)

    def _post_quotes(
        self,
        microprice: float,
        obi: float,
        market_bid: int,
        market_ask: int,
    ) -> None:
        inventory = self.holdings.get(self.symbol, 0)
        trend = self._trend_signal()
        inv_sign = 1 if inventory > 0 else (-1 if inventory < 0 else 0)

        # === spread quotato (Glosten-Milgrom adattivo) ===
        # Tre componenti, prendiamo il max:
        #   1) base: floor minimo
        #   2) volatilita realizzata: piu rumore ⇒ piu adverse selection
        #   3) |OBI|: imbalance forte ⇒ asimmetria informativa attesa alta
        vol = self._realized_vol()
        half_spread = max(
            float(self.base_half_spread),
            self.k_vol * vol,
            self.k_adv * abs(obi) * (2 * self.base_half_spread),
        )

        # === skew composito ===
        # 1) -alpha_inv * inventory: lean against (Avellaneda-Stoikov).
        # 2) +alpha_trend * trend * inv_sign: quando trend e inventario sono
        #    allineati alziamo il mid quotato cosi' vendiamo piu caro / compriamo
        #    meno caro (lasciamo correre i profitti); quando sono opposti lo
        #    abbassiamo per liquidare rapidamente.
        # 3) +alpha_obi * OBI * base_half_spread: leaning away from informed
        #    flow (se OBI > 0 alziamo le quote, costringendo i compratori
        #    informati a pagare di piu').
        skew = (
            -self.alpha_inv * inventory
            + self.alpha_trend * trend * inv_sign
            + self.alpha_obi * obi * self.base_half_spread
        )

        quoted_mid = microprice + skew
        our_bid = int(round(quoted_mid - half_spread))
        our_ask = int(round(quoted_mid + half_spread))

        # Safety: non crossare mai il book opposto e mantieni spread minimo.
        our_bid = min(our_bid, market_bid)
        our_ask = max(our_ask, market_ask)
        if our_ask <= our_bid:
            our_ask = our_bid + 1

        # Posta il bid solo se l'inventario e' sotto il cap.
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

        # Posta l'ask solo se l'inventario e' sopra il cap negativo.
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
