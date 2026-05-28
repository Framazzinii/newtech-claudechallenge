"""
EMA Crossover Trend Follower
============================

Agente di trend-following basato sull'incrocio di due medie mobili
esponenziali (EMA) del mid-price. Strategia semplice e robusta,
direttamente ispirata all'esempio TrendFollowerAgent fornito a lezione,
estesa con simmetria long/short e EOD flatten.

Strategia
---------
1.  Mantenere due EMA del mid-price: una "fast" (10 osservazioni) e una
    "slow" (30 osservazioni).
2.  Calcolare il segnale a ogni wake-up:
        fast > slow  ⇒  BUY  (uptrend, posizione long)
        fast < slow  ⇒  SELL (downtrend, posizione short)
3.  Agire solo quando il segnale CAMBIA rispetto al precedente
    (cross-over): inversione completa della posizione tramite market
    order.
4.  Stop-loss fisso a 30 cents per share dal prezzo di ingresso: se
    viene toccato chiude la posizione e attende il prossimo crossover.
5.  EOD flatten 5 minuti prima del close per evitare rischio overnight.

Differenze vs. l'esempio TrendFollowerAgent della classe
-------------------------------------------------------
- L'esempio apriva solo posizioni LONG (sui crossover bullish) e le
  chiudeva sui crossover bearish, restando flat in mezzo.
- Qui invertiamo in maniera SIMMETRICA: dopo un crossover bearish
  apriamo una posizione SHORT, raddoppiando le opportunita' di profitto.
- Aggiunto EOD flatten per non rimanere esposti nelle ultime decine di
  secondi prima del close.
- Size piu' grande (100 share/segnale vs 40) per amplificare il P&L
  per ogni trend catturato.

Razionale
---------
Il mercato RMSC04 contiene NoiseAgents (1000) e ValueAgents (102) i cui
ordini generano microtrend riconoscibili dall'EMA crossover. Il fast
EMA reagisce velocemente al cambio di pressione, il slow EMA filtra il
rumore di breve. Il loro incrocio identifica un cambio di regime
direzionale.

Reference
---------
Murphy, J. (1999), "Technical Analysis of the Financial Markets",
New York Institute of Finance. Capitoli sui moving average crossover
systems.
"""

from typing import Optional

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.utils import str_to_ns

from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import Side
from abides_markets.agents.trading_agent import TradingAgent


class HybridLPTrendAgent(TradingAgent):
    """
    EMA crossover trend follower, long/short simmetrico, con stop-loss.

    Nome classe legacy per non rompere l'import esistente nel notebook.

    Parametri
    ---------
    fast_span, slow_span
        Span (in osservazioni) delle due EMA. Default 10 e 30.
    order_size
        Size assoluta della posizione mantenuta (long o short).
    stop_loss_cents
        Massima loss per share tollerata dal prezzo d'ingresso.
    wake_up_freq
        Cadenza di valutazione del segnale.
    eod_flatten_offset
        Anticipo rispetto al close per liquidare l'inventario.
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
        order_size: int = 100,
        stop_loss_cents: int = 30,
        wake_up_freq: NanosecondTime = str_to_ns("30s"),
        eod_flatten_offset: NanosecondTime = str_to_ns("5min"),
        log_orders: bool = False,
    ) -> None:
        super().__init__(id, name, type, random_state, starting_cash, log_orders)

        self.symbol = symbol
        self.fast_span = fast_span
        self.slow_span = slow_span
        self.fast_alpha = 2.0 / (fast_span + 1)
        self.slow_alpha = 2.0 / (slow_span + 1)
        self.order_size = order_size
        self.stop_loss_cents = stop_loss_cents
        self.wake_up_freq = wake_up_freq
        self.eod_flatten_offset = eod_flatten_offset

        # state delle EMA (update incrementale)
        self.fast_ema: Optional[float] = None
        self.slow_ema: Optional[float] = None
        self.n_obs: int = 0

        # state del segnale e dell'entry corrente
        self.last_signal: Optional[str] = None       # "BUY" / "SELL" / None
        self.entry_price: Optional[float] = None     # mid al momento dell'entry

        # state EOD / risk
        self.eod_flatten_time: Optional[NanosecondTime] = None
        self.stopped_out: bool = False
        self.state: str = "AWAITING_WAKEUP"

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

        # EOD: liquida tutto e basta per il giorno.
        if (
            self.eod_flatten_time is not None
            and current_time >= self.eod_flatten_time
        ):
            self._flatten()
            self.stopped_out = True
            return

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

        bid, _, ask, _ = self.get_known_bid_ask(self.symbol)

        if bid and ask:
            mid = (bid + ask) / 2.0
            self._update_ema(mid)
            self.n_obs += 1

            # Warmup: aspetta che ci siano abbastanza obs per la EMA slow.
            if self.n_obs >= self.slow_span:
                self._trade(mid)

        self.set_wakeup(current_time + self.wake_up_freq)
        self.state = "AWAITING_WAKEUP"

    def get_wake_frequency(self) -> NanosecondTime:
        return self.wake_up_freq

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_ema(self, price: float) -> None:
        """Aggiornamento incrementale delle due EMA."""
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

    def _trade(self, mid: float) -> None:
        current_pos = self.holdings.get(self.symbol, 0)

        # === 1. Stop-loss check ===
        # Se siamo in posizione e l'entry e' registrato, verifica se il
        # prezzo ha violato la soglia di stop.
        if self.entry_price is not None and current_pos != 0:
            if current_pos > 0 and mid < self.entry_price - self.stop_loss_cents:
                # Long stoppato: chiudi a flat.
                self.place_market_order(
                    self.symbol, quantity=current_pos, side=Side.ASK
                )
                self.entry_price = None
                return
            if current_pos < 0 and mid > self.entry_price + self.stop_loss_cents:
                # Short stoppato: chiudi a flat.
                self.place_market_order(
                    self.symbol, quantity=-current_pos, side=Side.BID
                )
                self.entry_price = None
                return

        # === 2. Crossover signal ===
        signal = "BUY" if self.fast_ema > self.slow_ema else "SELL"

        # Agiamo solo sul CAMBIO di segnale (non a ogni wakeup).
        if signal == self.last_signal:
            return
        self.last_signal = signal

        # === 3. Inversione della posizione tramite market order ===
        if signal == "BUY":
            # Target: long order_size shares.
            target = self.order_size
            delta = target - current_pos
            if delta > 0:
                self.place_market_order(
                    self.symbol, quantity=delta, side=Side.BID
                )
                self.entry_price = mid
        else:  # SELL
            # Target: short order_size shares.
            target = -self.order_size
            delta = current_pos - target  # sempre positivo se target<current
            if delta > 0:
                self.place_market_order(
                    self.symbol, quantity=delta, side=Side.ASK
                )
                self.entry_price = mid

    def _flatten(self) -> None:
        """Chiude tutta la posizione via market order (per EOD)."""
        inv = self.holdings.get(self.symbol, 0)
        if inv > 0:
            self.place_market_order(self.symbol, quantity=inv, side=Side.ASK)
        elif inv < 0:
            self.place_market_order(self.symbol, quantity=-inv, side=Side.BID)
        self.entry_price = None
