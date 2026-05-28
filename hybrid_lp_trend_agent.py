"""
Multi-Signal Trend Follower with ATR-Adaptive Risk and Pyramiding
==================================================================

Sistema di trend-following istituzionale che combina cinque tecniche
classiche della letteratura per massimizzare il profit-per-trend
catturato:

1.  EMA crossover (fast/slow) come segnale di direzione primario.
2.  Trend-strength filter stile ADX (gap normalizzato sull'ATR) per
    evitare i whipsaw nei range trading.
3.  Position pyramiding turtle-style: aggiunge unita' incrementali
    mentre il trend si sviluppa favorevolmente.
4.  Chandelier exit (Chande-Kroll): trailing stop a 3*ATR sotto al
    Highest High della posizione corrente. Lascia correre i profitti
    e si attiva automaticamente sull'inversione.
5.  Take-profit parziale al raggiungimento di un multiplo dell'ATR
    favorevole: cristallizza una frazione del guadagno preservando
    l'esposizione al residuo del trend.
6.  Hard-stop fisso a 2*ATR dall'entry medio: limite assoluto di
    drawdown per posizione.
7.  EOD flatten 5 min prima del close per azzerare il rischio
    overnight.

Cornice teorica
---------------
- Donchian (anni '60): pioneer del canale di breakout e del trend
  systematic.
- Wilder (1978), "New Concepts in Technical Trading Systems":
  introduzione di ATR (Average True Range), RSI, ADX.
- Dennis & Eckhardt (1983), Turtle Trading System: combinazione di
  breakout, ATR-based sizing e pyramiding modulare.
- Chande & Kroll (1994), "The New Technical Trader": chandelier exit
  come trailing stop volatility-adjusted.
- Murphy (1999), "Technical Analysis of the Financial Markets":
  moving average crossover systems standardizzati.
- Covel (2007), "The Complete Turtle Trader": sintesi storica delle
  performance dei sistemi trend-following sui mercati reali.

Razionale operativo
-------------------
Il mercato simulato ABIDES RMSC04 contiene 1000 NoiseAgents + 102
ValueAgents + 2 MarketMakerAgents.  Gli ordini aggregati generano
micro-trend riconoscibili dall'EMA crossover (lo stesso esempio
TrendFollowerAgent della classe vince il tournament in 5/6 seed
testati).  La nostra versione potenzia questo edge con:

- Sizing ATR-adattivo (size grande in mercati calmi, ridotta in
  regimi volatili) ⇒ varianza per trade contenuta ⇒ Sharpe alto.
- Pyramiding ⇒ partecipazione non-lineare nei trend prolungati ⇒
  cattura asimmetrica del massimo move favorevole.
- Chandelier exit + take-profit parziale ⇒ lock-in dei profitti su
  ogni movimento, senza tagliare prematuramente i trend duraturi.
- Trend-strength filter ⇒ riduce drasticamente i falsi segnali in
  regime di range trading.

Tutto in market orders ⇒ execution istantanea, niente queue waiting,
niente missed fills.
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
    Multi-signal trend follower con pyramiding, chandelier exit, partial
    take-profit, hard stop e EOD flatten.  Nome classe legacy.

    Parametri
    ---------
    fast_span, slow_span
        Spans (in osservazioni) delle due EMA.
    atr_window
        Lookback per il calcolo del ATR (Wilder).
    trend_strength_min
        Soglia minima di gap normalizzato (|fast-slow|/ATR) per attivare
        un'entrata.  Filtra i whipsaw.
    base_size, unit_size, max_units
        Size iniziale, size di ogni pyramid add, e numero massimo di
        unita' totali (1 base + max_units-1 pyramid).
    atr_step_for_add
        Ogni quante unita' di ATR favorevoli viene aggiunto un pyramid
        unit (Turtle classic = 0.5; noi 1.0 per essere prudenti).
    chandelier_k
        ATR multiplier per il trailing stop.
    hard_stop_k
        ATR multiplier per lo stop assoluto dal prezzo medio d'ingresso.
    take_profit_k, take_profit_fraction
        ATR multiplier per il partial take-profit e frazione della
        posizione da chiudere quando si attiva.
    wake_up_freq
        Cadenza di refresh del segnale (default 15s = doppio rispetto
        all'esempio di classe ⇒ piu' reattivo).
    eod_flatten_offset
        Anticipo rispetto al close per il flatten.
    """

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        # EMA
        fast_span: int = 10,
        slow_span: int = 30,
        # ATR
        atr_window: int = 14,
        # trend strength filter
        trend_strength_min: float = 1.0,
        # sizing & pyramiding
        base_size: int = 100,
        unit_size: int = 50,
        max_units: int = 4,
        atr_step_for_add: float = 1.0,
        # stops
        chandelier_k: float = 3.0,
        hard_stop_k: float = 2.0,
        # take-profit
        take_profit_k: float = 3.0,
        take_profit_fraction: float = 0.33,
        # frequency
        wake_up_freq: NanosecondTime = str_to_ns("15s"),
        eod_flatten_offset: NanosecondTime = str_to_ns("5min"),
        log_orders: bool = False,
    ) -> None:
        super().__init__(id, name, type, random_state, starting_cash, log_orders)

        self.symbol = symbol

        # EMA
        self.fast_span = fast_span
        self.slow_span = slow_span
        self.fast_alpha = 2.0 / (fast_span + 1)
        self.slow_alpha = 2.0 / (slow_span + 1)

        # ATR
        self.atr_window = atr_window

        # filter
        self.trend_strength_min = trend_strength_min

        # sizing
        self.base_size = base_size
        self.unit_size = unit_size
        self.max_units = max_units
        self.atr_step_for_add = atr_step_for_add

        # stops & take-profit
        self.chandelier_k = chandelier_k
        self.hard_stop_k = hard_stop_k
        self.take_profit_k = take_profit_k
        self.take_profit_fraction = take_profit_fraction

        # frequency
        self.wake_up_freq = wake_up_freq
        self.eod_flatten_offset = eod_flatten_offset

        # ---- runtime state ----
        # EMA
        self.fast_ema: Optional[float] = None
        self.slow_ema: Optional[float] = None
        # ATR (true range history)
        self.tr_history: List[float] = []
        self.prev_mid: Optional[float] = None
        # signal
        self.last_signal: Optional[str] = None
        # position tracking
        self.initial_entry_price: Optional[float] = None
        self.last_add_price: Optional[float] = None
        self.units_added: int = 0
        self.highest_high: Optional[float] = None
        self.lowest_low: Optional[float] = None
        self.partial_tp_taken: bool = False
        # warmup & lifecycle
        self.n_obs: int = 0
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

        if (
            self.eod_flatten_time is not None
            and current_time >= self.eod_flatten_time
        ):
            self._flatten_all()
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
            self._update_indicators(mid)
            self.n_obs += 1

            # Warmup: aspetta abbastanza obs per stabilizzare EMA e ATR.
            if self.n_obs >= max(self.slow_span, self.atr_window) + 5:
                self._trade(mid)

        self.set_wakeup(current_time + self.wake_up_freq)
        self.state = "AWAITING_WAKEUP"

    def get_wake_frequency(self) -> NanosecondTime:
        return self.wake_up_freq

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    def _update_indicators(self, mid: float) -> None:
        # EMA fast & slow (incremental)
        if self.fast_ema is None:
            self.fast_ema = mid
            self.slow_ema = mid
        else:
            self.fast_ema = (
                self.fast_alpha * mid + (1 - self.fast_alpha) * self.fast_ema
            )
            self.slow_ema = (
                self.slow_alpha * mid + (1 - self.slow_alpha) * self.slow_ema
            )

        # True Range = |mid_t - mid_{t-1}|.  Per l'ATR a la Wilder
        # standard si userebbe high-low, ma qui abbiamo solo il mid;
        # |delta_mid| e' una proxy comune e adeguata per agenti
        # algoritmici intraday.
        if self.prev_mid is not None:
            tr = abs(mid - self.prev_mid)
            self.tr_history.append(tr)
            if len(self.tr_history) > self.atr_window * 3:
                self.tr_history = self.tr_history[-self.atr_window * 2:]
        self.prev_mid = mid

    def _atr(self) -> float:
        if len(self.tr_history) < self.atr_window:
            return 0.0
        return float(np.mean(self.tr_history[-self.atr_window:]))

    def _trend_strength(self) -> float:
        """Gap normalizzato fra le EMA in unita' di ATR (ADX-like)."""
        if self.fast_ema is None or self.slow_ema is None:
            return 0.0
        atr = self._atr()
        if atr < 1e-6:
            return 0.0
        return abs(self.fast_ema - self.slow_ema) / atr

    # ------------------------------------------------------------------
    # Trading logic
    # ------------------------------------------------------------------

    def _trade(self, mid: float) -> None:
        inv = self.holdings.get(self.symbol, 0)
        direction = 1 if inv > 0 else (-1 if inv < 0 else 0)
        atr = self._atr()

        # Aggiorna highest_high / lowest_low della posizione corrente
        # (servono al chandelier exit).
        if direction > 0:
            self.highest_high = (
                mid if self.highest_high is None else max(self.highest_high, mid)
            )
        elif direction < 0:
            self.lowest_low = (
                mid if self.lowest_low is None else min(self.lowest_low, mid)
            )

        # ====== STOPS sulla posizione esistente ======
        if direction != 0 and self.initial_entry_price is not None and atr > 0:

            # 1. Hard stop: 2 * ATR dall'entry iniziale.
            if (
                direction > 0
                and mid < self.initial_entry_price - self.hard_stop_k * atr
            ):
                self._flatten_all()
                return
            if (
                direction < 0
                and mid > self.initial_entry_price + self.hard_stop_k * atr
            ):
                self._flatten_all()
                return

            # 2. Chandelier exit: trailing stop a chandelier_k * ATR
            #    dal punto piu' favorevole (high se long, low se short).
            if (
                direction > 0
                and self.highest_high is not None
                and mid < self.highest_high - self.chandelier_k * atr
            ):
                self._flatten_all()
                return
            if (
                direction < 0
                and self.lowest_low is not None
                and mid > self.lowest_low + self.chandelier_k * atr
            ):
                self._flatten_all()
                return

            # 3. Partial take-profit: alla soglia take_profit_k * ATR
            #    favorevole, vendi take_profit_fraction della posizione.
            if not self.partial_tp_taken:
                if (
                    direction > 0
                    and mid > self.initial_entry_price + self.take_profit_k * atr
                ):
                    qty = max(1, int(inv * self.take_profit_fraction))
                    self.place_market_order(
                        self.symbol, quantity=qty, side=Side.ASK
                    )
                    self.partial_tp_taken = True
                elif (
                    direction < 0
                    and mid < self.initial_entry_price - self.take_profit_k * atr
                ):
                    qty = max(1, int(-inv * self.take_profit_fraction))
                    self.place_market_order(
                        self.symbol, quantity=qty, side=Side.BID
                    )
                    self.partial_tp_taken = True

        # ====== SIGNAL ======
        signal = "BUY" if self.fast_ema > self.slow_ema else "SELL"
        signal_changed = signal != self.last_signal

        # ====== ENTRY / REVERSAL su signal change ======
        if signal_changed:
            self.last_signal = signal

            # Filter: trend troppo debole ⇒ chiudi eventuale posizione
            # ma non aprirne una nuova nel verso opposto.
            if self._trend_strength() < self.trend_strength_min:
                if inv != 0:
                    self._flatten_all()
                return

            # Trend abbastanza forte ⇒ apri/inverti la posizione.
            target = self.base_size if signal == "BUY" else -self.base_size
            delta = target - inv
            if delta > 0:
                self.place_market_order(
                    self.symbol, quantity=delta, side=Side.BID
                )
            elif delta < 0:
                self.place_market_order(
                    self.symbol, quantity=-delta, side=Side.ASK
                )

            # Reset position tracking
            self.initial_entry_price = mid
            self.last_add_price = mid
            self.units_added = 0
            self.highest_high = mid if signal == "BUY" else None
            self.lowest_low = mid if signal == "SELL" else None
            self.partial_tp_taken = False
            return

        # ====== PYRAMIDING su segnale persistente ======
        # Aggiungi un'unita' ogni volta che il prezzo si muove di
        # atr_step_for_add * ATR ulteriormente nel verso favorevole.
        if (
            direction != 0
            and self.units_added < (self.max_units - 1)
            and self.last_add_price is not None
            and atr > 0
            # E il trend e' ancora forte abbastanza da giustificare l'add.
            and self._trend_strength() >= self.trend_strength_min
        ):
            step = self.atr_step_for_add * atr
            if direction > 0 and mid > self.last_add_price + step:
                self.place_market_order(
                    self.symbol, quantity=self.unit_size, side=Side.BID
                )
                self.units_added += 1
                self.last_add_price = mid
            elif direction < 0 and mid < self.last_add_price - step:
                self.place_market_order(
                    self.symbol, quantity=self.unit_size, side=Side.ASK
                )
                self.units_added += 1
                self.last_add_price = mid

    # ------------------------------------------------------------------
    # Order helpers
    # ------------------------------------------------------------------

    def _flatten_all(self) -> None:
        """Chiude tutta la posizione via market order e resetta lo stato."""
        inv = self.holdings.get(self.symbol, 0)
        if inv > 0:
            self.place_market_order(self.symbol, quantity=inv, side=Side.ASK)
        elif inv < 0:
            self.place_market_order(self.symbol, quantity=-inv, side=Side.BID)
        # Reset per la prossima posizione (waitha il prossimo segnale).
        self.initial_entry_price = None
        self.last_add_price = None
        self.units_added = 0
        self.highest_high = None
        self.lowest_low = None
        self.partial_tp_taken = False
