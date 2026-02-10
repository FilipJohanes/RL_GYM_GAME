"""CLI entry point."""

from __future__ import annotations

import argparse

from .simulation import run_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="DemoWorld simulation")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--initial-pop", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--csv", type=str, default=None, help="Write per-step stats to CSV")
    parser.add_argument("--no-summary", action="store_true", help="Disable summary output")
    parser.add_argument("--render-every", type=int, default=0, help="Render every N steps")
    parser.add_argument("--render-path", type=str, default=None, help="Output path or dir for PPM frames")
    parser.add_argument("--render-ascii", action="store_true", help="Print ASCII map at render steps")
    parser.add_argument("--render-scale", type=int, default=4, help="PPM scale factor")
    args = parser.parse_args()

    run_simulation(
        steps=args.steps,
        initial_population=args.initial_pop,
        seed=args.seed,
        log_every=args.log_every,
        csv_path=args.csv,
        summary=not args.no_summary,
        render_every=args.render_every,
        render_path=args.render_path,
        render_ascii_enabled=args.render_ascii,
        render_scale=args.render_scale,
    )


if __name__ == "__main__":
    main()
