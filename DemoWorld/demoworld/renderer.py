"""Rendering utilities for the DemoWorld environment."""

from __future__ import annotations

from typing import Dict, Tuple
import math

from . import config
from .environment import Environment


TERRAIN_CHARS: Dict[int, str] = {
    config.TERRAIN_VOID: " ",
    config.TERRAIN_PLATFORM: ".",
    config.TERRAIN_HOUSING: "H",
    config.TERRAIN_RESOURCE: "R",
}

TERRAIN_COLORS: Dict[int, Tuple[int, int, int]] = {
    config.TERRAIN_VOID: (20, 20, 20),
    config.TERRAIN_PLATFORM: (120, 120, 120),
    config.TERRAIN_HOUSING: (180, 130, 60),
    config.TERRAIN_RESOURCE: (50, 200, 200),
}

AGENT_COLOR_A = (220, 60, 60)
AGENT_COLOR_B = (60, 100, 220)
AGENT_COLOR_MIX = (170, 70, 170)


def render_ascii(env: Environment, max_width: int = 120, max_height: int = 60) -> str:
    scale_x = max(1, int(math.ceil(env.width / max_width)))
    scale_y = max(1, int(math.ceil(env.height / max_height)))
    out_width = int(math.ceil(env.width / scale_x))
    out_height = int(math.ceil(env.height / scale_y))

    agent_map: Dict[Tuple[int, int], str] = {}
    for agent in env.agents:
        if not agent.alive:
            continue
        sx = agent.x // scale_x
        sy = agent.y // scale_y
        key = (sx, sy)
        if key in agent_map:
            agent_map[key] = "*"
        else:
            agent_map[key] = config.SEX_A if agent.sex == config.SEX_A else config.SEX_B

    lines = []
    for sy in range(out_height):
        row = []
        y = min(env.height - 1, sy * scale_y)
        for sx in range(out_width):
            x = min(env.width - 1, sx * scale_x)
            char = TERRAIN_CHARS.get(env.tiles[y][x], "?")
            row.append(agent_map.get((sx, sy), char))
        lines.append("".join(row))
    return "\n".join(lines)


def render_ppm(env: Environment, path: str, scale: int = 4) -> None:
    width = env.width
    height = env.height
    scale = max(1, int(scale))
    img_w = width * scale
    img_h = height * scale

    agent_colors: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    for agent in env.agents:
        if not agent.alive:
            continue
        key = (agent.x, agent.y)
        if key in agent_colors:
            agent_colors[key] = AGENT_COLOR_MIX
        else:
            agent_colors[key] = AGENT_COLOR_A if agent.sex == config.SEX_A else AGENT_COLOR_B

    with open(path, "w", encoding="ascii") as handle:
        handle.write(f"P3\n{img_w} {img_h}\n255\n")
        for y in range(height):
            row_colors = []
            for x in range(width):
                color = agent_colors.get((x, y), TERRAIN_COLORS.get(env.tiles[y][x], (0, 0, 0)))
                row_colors.append(color)
            for _ in range(scale):
                line_parts = []
                for color in row_colors:
                    r, g, b = color
                    line_parts.extend([f"{r} {g} {b} "] * scale)
                handle.write("".join(line_parts).rstrip() + "\n")
