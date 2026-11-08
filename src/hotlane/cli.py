"""``hotlane study`` (train + evaluate + drift) and ``hotlane serve``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hotlane")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("study")
    s.add_argument("--out", default="results/hot_pricing")
    s.add_argument("--episodes", type=int, default=300)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--q-table", default="results/hot_pricing/q_table.npy")
    a = ap.parse_args(argv)
    if a.cmd == "study":
        from .report import write_report
        from .study import run

        summ = run(a.out, train_episodes=a.episodes)
        print(write_report(summ, Path(a.out)).read_text(encoding="utf-8"))
    else:
        import uvicorn

        from .api.app import create_app

        uvicorn.run(create_app(Path(a.q_table)), host="127.0.0.1", port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
