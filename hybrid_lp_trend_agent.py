"""
Fast Trend Follower with Pyramiding, Chandelier Exit & Partial Take-Profit
==========================================================================

Trend follower a bassa latenza per ABIDES RMSC04, progettato per CATTURARE
ogni microtrend senza essere bloccato da filtri troppo restrittivi.

Strategia in 4 livelli
----------------------
1.  SEGNALE: EMA fast (5 obs) vs EMA slow (15 obs) del mid-price.  Con
    wake-up ogni 5 secondi questo equivale a EMA ~25s e ~75s ⇒
    reattivita' alta ma rumore filtrato.

2.  TARGETING CON HYSTERESIS: l'agente vuole essere LONG quando
    (fast - slow) > +hysteresis e SHORT quando (fast - slow) < -hysteresis.
    Nella zona morta |diff| < hysteresis mantiene la posizione corrente.
    Niente "signal-change tracking" che potrebbe far perdere il primo
    crossover: ogni wake-up confronta la posizione effettiva con quella
    desiderata e aggiusta via market order.

3.  PYRAMIDING: per ogni movimento favorevole di pyramid_step cents si
    aggiunge un'unita' (turtle-style), fino a max_units totali.

4.  TRIPLE RISK MANAGEMENT su ogni posizione:
        a) Hard stop fisso a hard_stop cents dall'entry iniziale.
        b) Trailing stop (Chandelier) a trail_stop cents dal punto piu'
           favorevole raggiunto (highest_high se long, lowest_low se short).
        c) Partial take-profit: chiude take_profit_frac della posizione a
           +take_profit cents favorevoli (cristallizza profitto residuo
           lascia correre).
    + Daily stop globale (M2M).
    + EOD flatten 5 min prima del close.

Anti-whipsaw: cooldown
----------------------
Dopo un flatten innescato da uno stop, l'agente NON ri-entra immediatamente
sullo stesso lato fino a quando il segnale EMA non cambia direzione.
Questo evita il "stop-and-reverse-and-stop-again" classico nei range
trading.

Cornice teorica
---------------
- Murphy (1999), Technical Analysis of the Financial Markets: moving
  average crossover come segnale di trend.
- Wilder (1978), New Concepts in Technical Trading Systems: ATR e
  trailing stop, base per la nostra logica di chandelier exit.
- Dennis & Eckhardt (1983), Turtle Trading System: pyramiding lot-by-lot
  mentre il trend si sviluppa.
- Chande & Kroll (1994), The New Technical Trader: chandelier exit.
- Covel (2007), The Complete Turtle Trader: sintesi sui sistemi
  trend-following istituzionali.

L'agente fa SOLO trading direzionale, NIENTE market making.
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
    Fast EMA-crossover trend follower con hysteresis, pyramiding,
    chandelier exit, partial take-profit, hard stop e daily stop.

    Nome classe legacy.
    """

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        # EMA spans (osservazioni)
        fast_span: int = 5,
        slow_span: int = 15,
        # hysteresis (cents) per evitare flip-flop in deadband
        hysteresis_cents: int = 3,
        # sizing & pyramiding
        base_size: int = 100,
        unit_size: int = 50,
        max_units: int = 4,                # 1 base + 3 pyramid adds = max 250
        pyramid_step_cents: int = 20,
        # stops (cents per share)
        hard_stop_cents: int = 30,
        trail_stop_cents: int = 40,
        # take-profit parziale
        take_profit_cents: int = 60,
        take_profit_fraction: float = 0.33,
        # daily risk
        max_daily_loss_cents: int = 50000,  # $500
        # frequency
        wake_up_freq: NanosecondTime = str_to_ns("5s"),
        eod_flatten_offset: NanosecondTime = str_to_ns("5min"),
        log_orders: bool = False,
    ) -> None:
        super().__init__(id, name, type, random_state, starting_cash, log_orders)

        self.symbol = symbol
        self.starting_cash = starting_cash

        # EMA
        self.fast_alpha = 2.0 / (fast_span + 1)
        self.slow_alpha = 2.0 / (slow_span + 1)
        self.slow_span = slow_span

        # hysteresis
        self.hysteresis_cents = hysteresis_cents

        # sizing & pyramiding
        self.base_size = base_size
        self.unit_size = unit_size
        self.max_units = max_units
        self.pyramid_step_cents = pyramid_step_cents

        # stops
        self.hard_stop_cents = hard_stop_cents
        self.trail_stop_cents = trail_stop_cents

        # take-profit
        self.take_profit_cents = take_profit_cents
        self.take_profit_fraction = take_profit_fraction

        # daily risk
        self.max_daily_loss_cents = max_daily_loss_cents

        # frequency
        self.wake_up_freq = wake_up_freq
        self.eod_flatten_offset = eod_flatten_offset

        # ---- runtime state ----
        self.fast_ema: Optional[float] = None
        self.slow_ema: Optional[float] = None
        self.n_obs: int = 0

        # tracking della posizione corrente
        self.initial_entry_price: Optional[float] = None
        self.last_add_price: Optional[float] = None
        self.units_added: int = 0
        self.highest_high: Optional[float] = None
        self.lowest_low: Optional[float] = None
        self.partial_tp_taken: bool = False

        # anti-whipsaw cooldown: dopo uno stop forzato, salviamo la
        # direzione segnale dell'EMA in quel momento; ri-entrare in
        # quella stessa direzione e' bloccato finche' il segnale non
        # gira (cambio di sign del diff).
        self.cooldown_direction: int = 0

        # lifecycle
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
            self._update_ema(mid)
            self.n_obs += 1

            # Warmup minimo per stabilizzare la slow EMA.
            if self.n_obs >= self.slow_span:
                # Daily stop check (M2M).
                if self._check_daily_dd():
                    self._flatten_all()
                    self.stopped_out = True
                else:
                    self._trade(mid)

        self.set_wakeup(current_time + self.wake_up_freq)
        self.state = "AWAITING_WAKEUP"

    def get_wake_frequency(self) -> NanosecondTime:
        return self.wake_up_freq

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_ema(self, price: float) -> None:
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

    def _check_daily_dd(self) -> bool:
        try:
            mtm = self.mark_to_market(self.holdings)
        except Exception:
            return False
        return (mtm - self.starting_cash) < -self.max_daily_loss_cents

    def _desired_direction(self) -> int:
        """+1 long, -1 short, 0 deadband (mantieni la posizione corrente)."""
        diff = self.fast_ema - self.slow_ema
        if diff > self.hysteresis_cents:
            return 1
        if diff < -self.hysteresis_cents:
            return -1
        return 0

    # ------------------------------------------------------------------
    # Trading logic
    # ------------------------------------------------------------------

    def _trade(self, mid: float) -> None:
        inv = self.holdings.get(self.symbol, 0)
        direction = 1 if inv > 0 else (-1 if inv < 0 else 0)

        # Update highest_high / lowest_low per il chandelier exit.
        if direction > 0:
            self.highest_high = (
                mid if self.highest_high is None else max(self.highest_high, mid)
            )
        elif direction < 0:
            self.lowest_low = (
                mid if self.lowest_low is None else min(self.lowest_low, mid)
            )

        # ====== STOPS sulla posizione esistente ======
        if direction != 0 and self.initial_entry_price is not None:

            # 1. Hard stop.
            if (
                direction > 0
                and mid < self.initial_entry_price - self.hard_stop_cents
            ):
                self._flatten_all()
                return
            if (
                direction < 0
                and mid > self.initial_entry_price + self.hard_stop_cents
            ):
                self._flatten_all()
                return

            # 2. Trailing stop (chandelier).
            if (
                direction > 0
                and self.highest_high is not None
                and mid < self.highest_high - self.trail_stop_cents
            ):
                self._flatten_all()
                return
            if (
                direction < 0
                and self.lowest_low is not None
                and mid > self.lowest_low + self.trail_stop_cents
            ):
                self._flatten_all()
                return

            # 3. Partial take-profit (una sola volta per posizione).
            if not self.partial_tp_taken:
                if (
                    direction > 0
                    and mid > self.initial_entry_price + self.take_profit_cents
                ):
                    qty = max(1, int(inv * self.take_profit_fraction))
                    self.place_market_order(
                        self.symbol, quantity=qty, side=Side.ASK
                    )
                    self.partial_tp_taken = True
                elif (
                    direction < 0
                    and mid < self.initial_entry_price - self.take_profit_cents
                ):
                    qty = max(1, int(-inv * self.take_profit_fraction))
                    self.place_market_order(
                        self.symbol, quantity=qty, side=Side.BID
                    )
                    self.partial_tp_taken = True

        # ====== SIGNAL + TARGETING ======
        desired = self._desired_direction()

        # Cooldown anti-whipsaw: dopo un flatten forzato, blocca ri-entry
        # sullo stesso lato finche' il segnale non gira.
        if self.cooldown_direction != 0:
            if desired == self.cooldown_direction:
                # Stesso lato del flatten precedente → blocca entry.
                # Continua a controllare per pyramid se siamo gia' in pos.
                if inv == 0:
                    return
            else:
                # Il segnale e' girato (o e' deadband): cooldown rilasciato.
                self.cooldown_direction = 0

        # Aggiusta la posizione effettiva alla direzione desiderata.
        # (deadband: desired = 0 ⇒ teniamo la posizione corrente)
        if desired == 1 and direction <= 0:
            # Vogliamo essere long.
            target = self.base_size
            delta = target - inv
            if delta > 0:
                self.place_market_order(
                    self.symbol, quantity=delta, side=Side.BID
                )
                self._reset_position_tracking(mid, going_long=True)
                return  # entry done; pyramiding al prossimo wake-up
        elif desired == -1 and direction >= 0:
            target = -self.base_size
            delta = inv - target  # sempre positivo
            if delta > 0:
                self.place_market_order(
                    self.symbol, quantity=delta, side=Side.ASK
                )
                self._reset_position_tracking(mid, going_long=False)
                return

        # ====== PYRAMIDING sulla posizione esistente ======
        if (
            direction != 0
            and self.units_added < (self.max_units - 1)
            and self.last_add_price is not None
        ):
            if (
                direction > 0
                and mid > self.last_add_price + self.pyramid_step_cents
            ):
                self.place_market_order(
                    self.symbol, quantity=self.unit_size, side=Side.BID
                )
                self.units_added += 1
                self.last_add_price = mid
            elif (
                direction < 0
                and mid < self.last_add_price - self.pyramid_step_cents
            ):
                self.place_market_order(
                    self.symbol, quantity=self.unit_size, side=Side.ASK
                )
                self.units_added += 1
                self.last_add_price = mid

    # ------------------------------------------------------------------
    # Position tracking helpers
    # ------------------------------------------------------------------

    def _reset_position_tracking(self, mid: float, going_long: bool) -> None:
        """Resetta highest/lowest/entry quando apriamo una nuova posizione."""
        self.initial_entry_price = mid
        self.last_add_price = mid
        self.units_added = 0
        self.highest_high = mid if going_long else None
        self.lowest_low = mid if not going_long else None
        self.partial_tp_taken = False
        self.cooldown_direction = 0

    def _flatten_all(self) -> None:
        """Chiude tutta la posizione via market order. Imposta cooldown."""
        inv = self.holdings.get(self.symbol, 0)
        # Memorizza la direzione corrente del segnale per il cooldown.
        if self.fast_ema is not None and self.slow_ema is not None:
            diff = self.fast_ema - self.slow_ema
            if diff > 0:
                self.cooldown_direction = 1
            elif diff < 0:
                self.cooldown_direction = -1
        if inv > 0:
            self.place_market_order(self.symbol, quantity=inv, side=Side.ASK)
        elif inv < 0:
            self.place_market_order(self.symbol, quantity=-inv, side=Side.BID)
        self.initial_entry_price = None
        self.last_add_price = None
        self.units_added = 0
        self.highest_high = None
        self.lowest_low = None
        self.partial_tp_taken = False
