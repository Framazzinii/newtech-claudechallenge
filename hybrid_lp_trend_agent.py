"""
Adaptive Statistical Arbitrage Agent
====================================

Aggressive directional trading agent per ABIDES RMSC04. NO market making.
Pure stat-arb mean-reversion sull'Ornstein-Uhlenbeck del fundamental,
con position pyramiding, momentum filter, e triple stop-loss system
(trailing, hard, daily).

Foundamenti accademici
----------------------
1.  Uhlenbeck & Ornstein (1930), "On the theory of the Brownian motion":
    processo stocastico mean-reverting dX_t = -theta*(X_t - mu)*dt +
    sigma*dW_t. RMSC04 (Byrd et al. 2020) usa esattamente questo
    processo per il fundamental: ogni deviazione dalla media e'
    riassorbita con velocita' theta.

2.  Bertsimas & Lo (1998), "Optimal control of execution costs". La
    strategia ottima in un processo mean-reverting con costi di
    transazione e' tradare in proporzione al z-score del prezzo
    rispetto alla media stimata.

3.  Avellaneda & Lee (2010), "Statistical Arbitrage in the U.S.
    Equities Market" (Quantitative Finance). Framework operativo:
    decomposizione segnale-rumore, entry su |z|>soglia, exit a z=0,
    con S-score normalization e mean reversion velocity check.

4.  Kalman (1960), "A new approach to linear filtering and prediction
    problems". Stima ottima lineare di stato. Qui usiamo un EWMA come
    approssimazione computazionalmente leggera del Kalman filter per
    stimare la media corrente del processo OU.

5.  Tharp (1998), "Trade Your Way to Financial Freedom". Trailing
    stop-loss come tecnica universale per "cut your losses, let your
    profits run": cattura la maggior parte del trend favorevole ed
    esce automaticamente sulla prima inversione significativa.

6.  Vince (1992), "The Mathematics of Money Management". Position
    pyramiding: incrementare la posizione mentre il segnale persiste,
    sfruttando l'edge in modo non-lineare nel range di alta convinzione.

Strategia operativa
-------------------
Ogni 5 secondi:
 1.  Update mu (EWMA, alpha=0.02 ~ Kalman) e sigma (rolling std).
 2.  Calcola z = (mid - mu) / sigma.
 3.  Update EMA fast/slow per filtro momentum.
 4.  Risk checks (in ordine):
       a) Daily drawdown stop  ⇒ flatten + STOP per il giorno.
       b) Hard stop per trade  ⇒ flatten posizione corrente.
       c) Trailing stop        ⇒ flatten posizione corrente.
       d) Panic exit (z > z_panic in posizione sbagliata) ⇒ flatten.
 5.  Take-profit: se la posizione e' aperta e z e' rientrato verso
     zero (|z| < z_exit), flatten posizione (reversione completa).
 6.  Entry / pyramiding:
       z < -z_enter ⇒ LONG via market order (se momentum non blocca).
       z >  z_enter ⇒ SHORT via market order (se momentum non blocca).
     Lot size base = base_size; accumula fino a max_position con
     pyramiding lot-by-lot.
 7.  EOD flatten 5 min prima del close (no overnight risk).

Massimizzazione del profitto
----------------------------
- Pyramiding: cattura non-linearmente i segnali forti.
- Trailing stop al 50% del max favorable P&L: lascia correre i profitti
  ma cristallizza meta' del guadagno alla prima inversione.
- Market orders: zero missed-fill, execution istantanea.
- Triple stop layered: trailing (profit lock-in) + hard (max loss per
  trade) + daily (max daily loss) ⇒ tail risk eliminato.
- Momentum filter: non entra contro un trend forte (anche se z e'
  estremo, aspetta che il momentum si esaurisca prima di entrare).

L'agente NON quota, NON posta limit orders, NON fa market making.
Trade solo quando il segnale e' chiaro, con posizioni dimensionate dal
livello di convinzione.
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
    Adaptive statistical arbitrage agent — pyramiding stat-arb su z-score
    con triple stop-loss.

    Nome classe legacy. Strategia interna e' pure directional MR trading.
    Vedi docstring del modulo per dettagli + bibliografia.

    Parametri
    ---------
    mu_alpha
        Tasso di update EWMA della media (proxy del Kalman). 0.02 =
        finestra effettiva ~50 obs.
    vol_window
        Lookback per stima rolling di sigma (osservazioni).
    z_enter, z_exit, z_panic
        Soglie sul z-score per entry, take-profit, panic exit.
    fast_ema_span, slow_ema_span, momentum_block_threshold
        Filtro momentum: blocca l'entry MR contro un trend forte.
    base_size, max_position
        Lot size per pyramiding e cap assoluto sulla posizione.
    trail_pct
        Frazione del max favorable P&L data via prima di triggerare il
        trailing stop (0.5 = lock-in 50% del best).
    hard_stop_cents
        Loss massima per share su una singola posizione.
    max_daily_loss_cents
        Loss giornaliera massima totale (M2M) prima di smettere di
        tradare per il resto del giorno.
    """

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        # mean estimate (Kalman ~ EWMA)
        mu_alpha: float = 0.02,
        # vol estimate
        vol_window: int = 100,
        # signal thresholds
        z_enter: float = 1.5,
        z_exit: float = 0.3,
        z_panic: float = 3.5,
        # momentum filter
        fast_ema_span: int = 5,
        slow_ema_span: int = 20,
        momentum_block_threshold: float = 1.5,
        # sizing & pyramiding
        base_size: int = 75,
        max_position: int = 300,
        # stops
        trail_pct: float = 0.5,
        hard_stop_cents: float = 25.0,
        max_daily_loss_cents: int = 40000,
        # frequency & EOD
        wake_up_freq: NanosecondTime = str_to_ns("5s"),
        eod_flatten_offset: NanosecondTime = str_to_ns("5min"),
        log_orders: bool = False,
    ) -> None:
        super().__init__(id, name, type, random_state, starting_cash, log_orders)

        self.symbol = symbol
        self.starting_cash = starting_cash

        self.mu_alpha = mu_alpha
        self.vol_window = vol_window
        self.z_enter = z_enter
        self.z_exit = z_exit
        self.z_panic = z_panic

        self.fast_alpha = 2.0 / (fast_ema_span + 1)
        self.slow_alpha = 2.0 / (slow_ema_span + 1)
        self.slow_ema_span = slow_ema_span
        self.momentum_block_threshold = momentum_block_threshold

        self.base_size = base_size
        self.max_position = max_position

        self.trail_pct = trail_pct
        self.hard_stop_cents = hard_stop_cents
        self.max_daily_loss_cents = max_daily_loss_cents

        self.wake_up_freq = wake_up_freq
        self.eod_flatten_offset = eod_flatten_offset

        # state
        self.mu: Optional[float] = None
        self.fast_ema: Optional[float] = None
        self.slow_ema: Optional[float] = None
        self.mid_history: List[float] = []

        # position tracking (per trailing stop)
        self.position_open_mtm: Optional[float] = None
        self.position_max_pnl: float = 0.0

        # risk state
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
            self.mid_history.append(mid)
            if len(self.mid_history) > self.vol_window * 3:
                self.mid_history = self.mid_history[-self.vol_window * 2:]

            self._update_estimates(mid)

            # Daily drawdown stop
            if self._check_daily_dd():
                self._flatten_all()
                self.stopped_out = True
                self.state = "AWAITING_WAKEUP"
                self.set_wakeup(current_time + self.wake_up_freq)
                return

            self._trade(mid)

        self.set_wakeup(current_time + self.wake_up_freq)
        self.state = "AWAITING_WAKEUP"

    def get_wake_frequency(self) -> NanosecondTime:
        return self.wake_up_freq

    # ------------------------------------------------------------------
    # Estimation (mu via EWMA-Kalman, vol via rolling std, momentum EMAs)
    # ------------------------------------------------------------------

    def _update_estimates(self, mid: float) -> None:
        # EWMA della media (proxy del Kalman filter).
        if self.mu is None:
            self.mu = mid
        else:
            self.mu = (1 - self.mu_alpha) * self.mu + self.mu_alpha * mid

        # EMAs per momentum filter.
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

    def _sigma(self) -> float:
        if len(self.mid_history) < 20:
            return 0.0
        n = min(self.vol_window, len(self.mid_history))
        return float(np.std(self.mid_history[-n:]))

    def _z_score(self, mid: float) -> float:
        if self.mu is None:
            return 0.0
        sigma = self._sigma()
        if sigma < 1e-6:
            return 0.0
        return (mid - self.mu) / sigma

    def _momentum_normalised(self) -> float:
        """Momentum in unita' di sigma (signed)."""
        if self.fast_ema is None or self.slow_ema is None:
            return 0.0
        if len(self.mid_history) < self.slow_ema_span:
            return 0.0
        sigma = self._sigma()
        if sigma < 1e-6:
            return 0.0
        return (self.fast_ema - self.slow_ema) / sigma

    # ------------------------------------------------------------------
    # Risk: triple stop (daily / hard / trailing)
    # ------------------------------------------------------------------

    def _check_daily_dd(self) -> bool:
        try:
            mtm = self.mark_to_market(self.holdings)
        except Exception:
            return False
        return (mtm - self.starting_cash) < -self.max_daily_loss_cents

    def _update_position_tracking(self) -> None:
        """Tiene traccia di open_mtm e max_pnl per il trailing stop."""
        inv = self.holdings.get(self.symbol, 0)
        if inv == 0:
            self.position_open_mtm = None
            self.position_max_pnl = 0.0
            return
        if self.position_open_mtm is None:
            try:
                self.position_open_mtm = self.mark_to_market(self.holdings)
            except Exception:
                return
            self.position_max_pnl = 0.0
            return
        try:
            current_mtm = self.mark_to_market(self.holdings)
        except Exception:
            return
        pnl = current_mtm - self.position_open_mtm
        if pnl > self.position_max_pnl:
            self.position_max_pnl = pnl

    def _check_trailing_stop(self) -> bool:
        inv = self.holdings.get(self.symbol, 0)
        if inv == 0 or self.position_open_mtm is None:
            return False
        # Trailing stop si attiva solo se siamo gia' stati profittevoli.
        if self.position_max_pnl <= 0:
            return False
        try:
            current_mtm = self.mark_to_market(self.holdings)
        except Exception:
            return False
        pnl = current_mtm - self.position_open_mtm
        drawdown = self.position_max_pnl - pnl
        return drawdown > self.trail_pct * self.position_max_pnl

    def _check_hard_stop(self) -> bool:
        inv = self.holdings.get(self.symbol, 0)
        if inv == 0 or self.position_open_mtm is None:
            return False
        try:
            current_mtm = self.mark_to_market(self.holdings)
        except Exception:
            return False
        pnl = current_mtm - self.position_open_mtm
        loss_per_share = -pnl / abs(inv)
        return loss_per_share > self.hard_stop_cents

    # ------------------------------------------------------------------
    # Trading logic
    # ------------------------------------------------------------------

    def _trade(self, mid: float) -> None:
        inv = self.holdings.get(self.symbol, 0)

        # 0. Update tracking per trailing stop.
        self._update_position_tracking()

        z = self._z_score(mid)
        mom = self._momentum_normalised()

        # 1. Stops on existing position.
        if self._check_hard_stop() or self._check_trailing_stop():
            self._flatten_all()
            return

        # 2. Panic: posizione nel verso sbagliato + z estremo.
        if inv > 0 and z > self.z_panic:
            self._flatten_all()
            return
        if inv < 0 and z < -self.z_panic:
            self._flatten_all()
            return

        # 3. Take-profit: il prezzo e' rientrato verso la media.
        # Long: exit quando z sale sopra -z_exit (es. da -2.0 a -0.3).
        if inv > 0 and z >= -self.z_exit:
            self._flatten_all()
            return
        # Short: exit quando z scende sotto +z_exit.
        if inv < 0 and z <= self.z_exit:
            self._flatten_all()
            return

        # 4. Entry / pyramiding sull'edge MR.
        # LONG signal: z < -z_enter (prezzo molto sotto la media).
        if z < -self.z_enter and inv >= 0:
            # Momentum filter: non comprare in un downtrend forte.
            if mom < -self.momentum_block_threshold:
                return
            if inv < self.max_position:
                new_size = min(self.base_size, self.max_position - inv)
                if new_size > 0:
                    self.place_market_order(
                        self.symbol, quantity=new_size, side=Side.BID
                    )

        # SHORT signal: z > z_enter (prezzo molto sopra la media).
        elif z > self.z_enter and inv <= 0:
            # Momentum filter: non shortare in un uptrend forte.
            if mom > self.momentum_block_threshold:
                return
            if inv > -self.max_position:
                new_size = min(self.base_size, self.max_position + inv)
                if new_size > 0:
                    self.place_market_order(
                        self.symbol, quantity=new_size, side=Side.ASK
                    )

    def _flatten_all(self) -> None:
        inv = self.holdings.get(self.symbol, 0)
        if inv > 0:
            self.place_market_order(self.symbol, quantity=inv, side=Side.ASK)
        elif inv < 0:
            self.place_market_order(self.symbol, quantity=-inv, side=Side.BID)
        self.position_open_mtm = None
        self.position_max_pnl = 0.0
