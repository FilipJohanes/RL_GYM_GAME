"""3D viewer for the DemoWorld environment (matplotlib)."""

from __future__ import annotations

import argparse
import random
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "Viewer requires matplotlib and numpy. Install with: pip install matplotlib numpy"
    ) from exc

from . import config
from .environment import Environment


ZONE_COLORS = {
    config.ZONE_WATER: (0.15, 0.35, 0.7, 1.0),
    config.ZONE_SHORE: (0.85, 0.8, 0.55, 1.0),
    config.ZONE_GRASSLAND: (0.2, 0.6, 0.3, 1.0),
    config.ZONE_FOREST: (0.1, 0.45, 0.2, 1.0),
    config.ZONE_ROCKY: (0.5, 0.5, 0.5, 1.0),
    config.ZONE_HIGHLAND: (0.7, 0.7, 0.7, 1.0),
    config.ZONE_VOLCANO: (0.55, 0.25, 0.2, 1.0),
}


def _build_surface(env: Environment) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = np.arange(env.width)
    ys = np.arange(env.height)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.array(env.height_map)

    colors = np.zeros((env.height, env.width, 4), dtype=float)
    for y in range(env.height):
        for x in range(env.width):
            zone = env.zone_map[y][x]
            colors[y, x] = ZONE_COLORS.get(zone, (0.2, 0.2, 0.2, 1.0))
    return xx, yy, zz, colors


def run_viewer(
    steps: int,
    initial_population: int,
    seed: int | None = None,
    interval_ms: int = 100,
    elev: float = 35.0,
    azim: float = -60.0,
    agent_size: int = 15,
) -> None:
    rng = random.Random(seed)
    env = Environment(rng)
    env.seed_initial_population(initial_population)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    xx, yy, zz, colors = _build_surface(env)
    ax.plot_surface(xx, yy, zz, facecolors=colors, linewidth=0, antialiased=False, shade=False)

    def agent_scatter():
        xs = [a.x for a in env.agents if a.alive]
        ys = [a.y for a in env.agents if a.alive]
        zs = [a.z + 0.3 for a in env.agents if a.alive]
        cs = [
            (0.9, 0.2, 0.2) if a.sex == config.SEX_A else (0.2, 0.3, 0.9)
            for a in env.agents
            if a.alive
        ]
        return xs, ys, zs, cs

    xs, ys, zs, cs = agent_scatter()
    scatter = ax.scatter(xs, ys, zs, c=cs, s=agent_size)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_zlim(0, max(config.VOLCANO_PEAK_HEIGHT + config.ISLAND_HEIGHT_PEAK + 2, 10))

    def update(_frame: int):
        env.step()
        xs, ys, zs, cs = agent_scatter()
        scatter._offsets3d = (xs, ys, zs)
        scatter.set_color(cs)
        ax.set_title(f"Step {env.step_count}  Pop {len(env.agents)}")
        return (scatter,)

    FuncAnimation(fig, update, frames=steps, interval=interval_ms, blit=False)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="DemoWorld 3D viewer")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--initial-pop", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--interval", type=int, default=80, help="Milliseconds between frames")
    parser.add_argument("--elev", type=float, default=35.0)
    parser.add_argument("--azim", type=float, default=-60.0)
    parser.add_argument("--agent-size", type=int, default=15)
    args = parser.parse_args()

    run_viewer(
        steps=args.steps,
        initial_population=args.initial_pop,
        seed=args.seed,
        interval_ms=args.interval,
        elev=args.elev,
        azim=args.azim,
        agent_size=args.agent_size,
    )


if __name__ == "__main__":
    main()
