"""Smoke test: 5-minute RMSC04 simulation.

Used by `make smoke` and the dev container's postCreate to confirm the
environment is functional. Should finish in ~10-30 seconds on a laptop.

Exits non-zero if:
  - any sub-package fails to import,
  - the simulation raises,
  - the order book is empty at the end of the run.
"""

from __future__ import annotations

import sys
import time
import traceback


def main() -> int:
    t0 = time.perf_counter()

    try:
        import numpy as np  # noqa: F401  (sanity-check the heavy deps load)

        from abides_core import abides
        from abides_markets.configs.rmsc04 import build_config
    except Exception:
        print("[smoke] FAIL: imports broken", file=sys.stderr)
        traceback.print_exc()
        return 1

    print("[smoke] Building RMSC04 config (5-minute simulation)...")
    config = build_config(
        seed=1,
        end_time="09:35:00",   # 5 minutes after the default 09:30 open
        log_orders=False,
        exchange_log_orders=False,
        book_logging=True,
        stdout_log_level="WARNING",
    )

    print("[smoke] Running simulation...")
    try:
        end_state = abides.run(config)
    except Exception:
        print("[smoke] FAIL: simulation raised", file=sys.stderr)
        traceback.print_exc()
        return 2

    # The exchange agent is the first agent and exposes the order book.
    agents = end_state.get("agents", [])
    if not agents:
        print("[smoke] FAIL: end_state has no agents", file=sys.stderr)
        return 3

    exchange = agents[0]
    order_books = getattr(exchange, "order_books", None)
    if not order_books:
        print("[smoke] FAIL: exchange has no order books", file=sys.stderr)
        return 4

    # Just confirm at least one symbol's book has activity.
    any_trades = False
    for symbol, book in order_books.items():
        history_len = len(getattr(book, "history", []) or [])
        bids = len(getattr(book, "bids", []) or [])
        asks = len(getattr(book, "asks", []) or [])
        print(f"[smoke] {symbol}: bids={bids} asks={asks} history={history_len}")
        if history_len or bids or asks:
            any_trades = True

    elapsed = time.perf_counter() - t0
    if not any_trades:
        print(
            f"[smoke] FAIL: order book is empty after {elapsed:.1f}s",
            file=sys.stderr,
        )
        return 5

    print(f"[smoke] OK ({elapsed:.1f}s) — environment looks healthy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
