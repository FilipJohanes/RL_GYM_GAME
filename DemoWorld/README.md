# DemoWorld

A lightweight, pure-Python simulation inspired by Calhoun''s Universe 25: agents have needs (food, water), can reproduce, and can die. Agents differ in traits (shape, color, size) sampled from your specified distributions.

## Quick start

From `C:\Projects\RL_GYM_GAME\DemoWorld`:

```powershell
python -m demoworld.run --steps 5000 --initial-pop 50
```

## Logs and stats

You can print periodic logs with `--log-every` and export per-step stats to CSV:

```powershell
python -m demoworld.run --steps 5000 --initial-pop 50 --log-every 200 --csv stats.csv
```

## Rendering

Render ASCII maps or write PPM frames (no extra dependencies):

```powershell
python -m demoworld.run --steps 500 --initial-pop 80 --render-every 50 --render-ascii
python -m demoworld.run --steps 500 --initial-pop 80 --render-every 50 --render-path frames
```

PPM files can be opened by most image viewers or converted later.

## 3D Viewer

Interactive 3D viewer (requires `matplotlib` + `numpy`):

```powershell
python -m demoworld.viewer --steps 2000 --initial-pop 200
```

## FPS Viewer (WASD + Mouse)

First-person 3D viewer using pyglet 2.x:

```powershell
pip install pyglet
python -m demoworld.viewer_fps --initial-pop 200 --sim-rate 20
```

Controls:
- `W/A/S/D` move
- `Q/E` down/up
- Mouse to look
- `Space` pause
- `Esc` release mouse, click to lock again

Mini-map:
- Shown in the top-left of the FPS viewer (terrain + agents).

Toggles:
- `T` terrain on/off
- `G` agents on/off
- `M` minimap on/off
- `H` cycle heatmap: terrain/food/water/material/height
- `1..5` direct heatmap select

## Panda3D Viewer

Panda3D-based 3D viewer (recommended):

```powershell
pip install panda3d
python -m demoworld.viewer_panda3d --initial-pop 500 --sim-rate 20
```

## Notes
- Shape (0-1) represents "pretty".
- Color (0-1) represents cleverness.
- Size (0-1) represents strength.
- 90% of agents sample each trait from [0.6, 0.8], otherwise from [0, 1].
