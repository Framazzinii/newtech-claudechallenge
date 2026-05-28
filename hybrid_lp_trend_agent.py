"""
Hybrid Liquidity Provider + Mean-Reversion Inventory Management Agent
=====================================================================

Cornice teorica
---------------
ABIDES RMSC04 e' un mercato SIMULATO popolato da agenti regolati: Noise,
Value, MarketMaker.  Non ci sono insider ne' informazione privilegiata
asimmetrica strutturalmente sfruttabile: di conseguenza NON applichiamo
premio di adverse selection a la Glosten-Milgrom sullo spread.

Il fundamental e' un processo Ornstein-Uhlenbeck (OU) MEAN-REVERTING:
ogni deviazione dalla media tende a essere riassorbita con velocita'
theta.  Questa proprieta' e' lo stylized fact centrale che la strategia
sfrutta.  Sui timeframe intraday il mid-price del ticker eredita la
stessa proprieta' tramite gli ordini dei ValueAgents.

Riferimenti
-----------
- Byrd et al. (2020), ABIDES RMSC04: fundamental OU mean-reverting.
- Stoikov (2018), microprice: fair value pesato per i volumi al best
  level, miglior predittore del prossimo prezzo rispetto al mid.
- Avellaneda & Stoikov (2008), HFT market making: il MM ottimo skewa
  le quote contro il proprio inventario per controllarne la volatilita.
- Statistical arbitrage classico (Pole 2007, Avellaneda & Lee 2010):
  trade su z-score di una serie mean-reverting, posizioni di segno
  opposto alla deviazione.

Strategia: Mean-Reverting Market Maker
--------------------------------------
A ogni wake-up (~6 secondi):
 1.  Cancella le quote vecchie ancora vive sul book.
 2.  Interroga lo spread; in receive_message su QuerySpreadResponseMsg:
 3.  Legge (bid, V_bid, ask, V_ask) dal book → calcola microprice e mid.
 4.  Aggiorna rolling history del mid (window di ~6 min) → rolling_mean
     e rolling_std → z-score = (mid - mean) / std.
 5.  Stop-loss giornaliero (M2M).
 6.  Spread quotato adattivo:
         half_spread_q = max(base_half_spread, k_vol * realized_vol)
     Niente termine Glosten-Milgrom: nessun adverse selection premium.
 7.  Skew composito (cents):
         skew = - alpha_inv * inventory          (Avellaneda-Stoikov)
                - alpha_mr  * z_score            (mean-reversion alpha)
     Se z > 0 (prezzo alto, atteso ribasso): abbassa il mid quotato.
     Se z < 0 (prezzo basso, atteso rialzo): alza il mid quotato.
     L'effetto netto sull'inventario:
       - long & z>0: skew NEGATIVO ⇒ ask scende ⇒ liquidazione veloce.
       - long & z<0: skew POSITIVO ⇒ ask sale ⇒ tieni il long, sale.
       - short & z>0: MR domina, skew NEG ⇒ ask scende ⇒ ride lo short.
       - short & z<0: skew POSITIVO ⇒ bid sale ⇒ copri lo short.
 8.  ONE-SIDED quoting per segnali MR forti (|z| > z_strong):
         z >  z_strong ⇒ posta solo ASK (prezzo alto, build short).
         z < -z_strong ⇒ posta solo BID (prezzo basso, build long).
     Questo evita di accumulare inventario nella direzione SBAGLIATA
     proprio quando il segnale di reversione e' piu' affidabile.
 9.  Vol-scaled order size: in regime di alta volatilita' la size si
     riduce, abbattendo la varianza per trade (numeratore del Sharpe
     penalizzato meno dal denominatore che esplode in regimi rumorosi).
10.  Inventory cap (max_inv).
11.  EOD flatten 5 min prima del close per azzerare il rischio
     overnight.

Massimizzazione dello Sharpe ratio
----------------------------------
Sharpe = E[r] / sigma(r).  Per massimizzarlo:
- One-sided quoting nei regimi |z|>strong evita accumulo nella
  direzione sbagliata ⇒ riduce drawdown tail ⇒ sigma cala.
- Vol scaling sull'order_size riduce la variance per trade ⇒ sigma cala.
- Skew aggressivo accelera il take-profit alla reversione ⇒ E[r] sale.
- Stop-loss giornaliero tronca la coda sinistra della distribuzione di
  pnl ⇒ kurtosis e sigma calano.
- EOD flatten elimina rischio overnight ⇒ jump risk eliminato.

L'agente NON e' un puro market maker (avrebbe edge negativo contro i
ValueAgents) ne' un puro stat-arb (perderebbe il guadagno dello spread):
e' un MM con view direzionale che fa entrambe le cose insieme.
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
    Liquidity provider con inventory management mean-reversion-driven.

    Nota: il nome della classe e' rimasto 'HybridLPTrendAgent' per
    retrocompatibilita' con il notebook esistente, ma la strategia
    interna e' completamente mean-reversion (non trend following).
    Vedi il docstring del modulo per il razionale e le referenze.

    Parametri
    ---------
    base_half_spread
        Half-spread minimo in cents.
    k_vol
        Premio sullo spread proporzionale alla volatilita' realizzata.
    mr_window
        Lookback (osservazioni) per rolling mean e std del mid.
    z_enter
        |z| minimo per attivare la componente MR dello skew (no-edge
        zone sotto questa soglia).
    z_strong
        |z| sopra cui si fa ONE-SIDED quoting: si posta solo il lato
        coerente col segnale di mean reversion.
    alpha_inv
        Skew cents/share per leaning against inventory.
    alpha_mr
        Skew cents per unita' di z-score.
    order_size, max_inv
        Size base per quote e inventario massimo assoluto.
    vol_size_scale
        Aggressivita' del vol-scaling sull'order_size (0 = off).
    stop_loss_cents
        Drawdown M2M massimo (cents) prima di flattenare la posizione e
        smettere di tradare per la giornata.
    eod_flatten_offset
        Anticipo rispetto al close per liquidare l'inventario residuo.
    """

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        # spread (no Glosten-Milgrom term)
        base_half_spread: int = 4,
        k_vol: float = 2.0,
        # mean reversion signal
        mr_window: int = 60,                # ~6 min at 6s wake-up
        z_enter: float = 0.5,
        z_strong: float = 1.8,
        # skew
        alpha_inv: float = 0.3,
        alpha_mr: float = 7.0,
        # size
        order_size: int = 30,
        max_inv: int = 200,
        vol_size_scale: float = 0.3,
        # frequency
        wake_up_freq: NanosecondTime = str_to_ns("6s"),
        # risk
        stop_loss_cents: int = 40000,
        eod_flatten_offset: NanosecondTime = str_to_ns("5min"),
        log_orders: bool = False,
    ) -> None:
        super().__init__(id, name, type, random_state, starting_cash, log_orders)

        self.symbol = symbol
        self.starting_cash = starting_cash

        # spread
        self.base_half_spread = base_half_spread
        self.k_vol = k_vol

        # MR signal
        self.mr_window = mr_window
        self.z_enter = z_enter
        self.z_strong = z_strong

        # skew
        self.alpha_inv = alpha_inv
        self.alpha_mr = alpha_mr

        # size
        self.order_size = order_size
        self.max_inv = max_inv
        self.vol_size_scale = vol_size_scale

        # frequency & risk
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

        # EOD: flatten + stop trading.
        if (
            self.eod_flatten_time is not None
            and current_time >= self.eod_flatten_time
        ):
            self._cancel_live_quotes()
            self._flatten_inventory()
            self.stopped_out = True
            return

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
            # Microprice (Stoikov): peso il fair value verso il lato meno
            # liquido. Predittore migliore del mid sul brevissimo termine.
            microprice = (bid * ask_vol + ask * bid_vol) / (bid_vol + ask_vol)

            self.mid_history.append(mid)
            if len(self.mid_history) > self.mr_window * 3:
                self.mid_history = self.mid_history[-self.mr_window * 2:]

            # Stop-loss giornaliero.
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
    # Helpers — segnale mean reversion
    # ------------------------------------------------------------------

    def _rolling_stats(self):
        """Ritorna (mean, std) sugli ultimi mr_window mid, o (None, None)."""
        if len(self.mid_history) < self.mr_window:
            return None, None
        recent = self.mid_history[-self.mr_window:]
        return float(np.mean(recent)), float(np.std(recent))

    def _z_score(self, mid: float) -> float:
        """z-score del mid rispetto al rolling mean. 0 finche' non c'e' storia."""
        mean, std = self._rolling_stats()
        if mean is None or std is None or std < 1e-6:
            return 0.0
        return (mid - mean) / std

    def _realized_vol(self) -> float:
        """Vol realizzata recente del mid (std, in cents)."""
        if len(self.mid_history) < 5:
            return 0.0
        n = min(self.mr_window, len(self.mid_history))
        return float(np.std(self.mid_history[-n:]))

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
        inv = self.holdings.get(self.symbol, 0)
        if inv > 0:
            self.place_market_order(self.symbol, quantity=inv, side=Side.ASK)
        elif inv < 0:
            self.place_market_order(self.symbol, quantity=-inv, side=Side.BID)

    # ------------------------------------------------------------------
    # Posting logic
    # ------------------------------------------------------------------

    def _post_quotes(
        self, microprice: float, market_bid: int, market_ask: int
    ) -> None:
        inventory = self.holdings.get(self.symbol, 0)
        mid = (market_bid + market_ask) / 2.0
        z = self._z_score(mid)
        vol = self._realized_vol()

        # === spread quotato (adattivo solo per volatilita') ===
        half_spread = max(float(self.base_half_spread), self.k_vol * vol)

        # === skew composito ===
        # Componente 1 (Avellaneda-Stoikov): lean against inventory.
        # Componente 2 (mean reversion): segno opposto al z-score.  Solo
        # quando |z| > z_enter (no-edge zone altrimenti per evitare di
        # tradare rumore).
        mr_component = -self.alpha_mr * z if abs(z) > self.z_enter else 0.0
        skew = -self.alpha_inv * inventory + mr_component

        quoted_mid = microprice + skew
        our_bid = int(round(quoted_mid - half_spread))
        our_ask = int(round(quoted_mid + half_spread))

        # Safety: non crossare il book opposto, spread minimo 1 cent.
        our_bid = min(our_bid, market_bid)
        our_ask = max(our_ask, market_ask)
        if our_ask <= our_bid:
            our_ask = our_bid + 1

        # === vol-scaled order size ===
        # In regimi rumorosi riduce la size per abbassare la variance.
        scale = 1.0
        if self.vol_size_scale > 0 and vol > 0:
            scale = 1.0 - self.vol_size_scale * (vol / (vol + 3.0))
            scale = max(0.3, scale)
        effective_size = max(1, int(self.order_size * scale))

        # === ONE-SIDED quoting su |z| forte ===
        # z >  z_strong → prezzo molto alto, vogliamo solo VENDERE (build
        # short / mantieni short). Niente bid.
        # z < -z_strong → prezzo molto basso, vogliamo solo COMPRARE
        # (build long / mantieni long). Niente ask.
        post_bid_allowed = (z <= self.z_strong)
        post_ask_allowed = (z >= -self.z_strong)

        # === posta bid ===
        if inventory < self.max_inv and post_bid_allowed:
            bid_qty = min(effective_size, self.max_inv - inventory)
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
        if inventory > -self.max_inv and post_ask_allowed:
            ask_qty = min(effective_size, self.max_inv + inventory)
            if ask_qty > 0:
                self.place_limit_order(
                    self.symbol,
                    quantity=ask_qty,
                    side=Side.ASK,
                    limit_price=our_ask,
                )
                if self.orders:
                    self.ask_order_id = max(self.orders.keys())
