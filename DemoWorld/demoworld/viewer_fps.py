"""First-person 3D viewer with WASD + mouse (pyglet 2.x, modern OpenGL)."""

from __future__ import annotations

import argparse
import math
import random
from typing import List, Tuple

try:
    import pyglet
    import pyglet.gl as gl
    from pyglet.graphics.shader import Shader, ShaderProgram
    from pyglet.math import Mat4, Vec3
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("Viewer requires pyglet 2.x. Install with: pip install pyglet") from exc

from . import config
from .environment import Environment


ZONE_COLORS = {
    config.ZONE_VOID: (0.1, 0.1, 0.1),
    config.ZONE_PLATFORM: (0.55, 0.55, 0.55),
    config.ZONE_HOUSING: (0.8, 0.55, 0.2),
    config.ZONE_RESOURCE: (0.2, 0.7, 0.9),
}


VERTEX_SRC = """
#version 330 core
in vec3 position;
in vec3 color;
uniform mat4 u_mvp;
uniform float u_point_size;
out vec3 v_color;
void main() {
    gl_Position = u_mvp * vec4(position, 1.0);
    gl_PointSize = u_point_size;
    v_color = color;
}
"""

FRAGMENT_SRC = """
#version 330 core
in vec3 v_color;
out vec4 out_color;
void main() {
    out_color = vec4(v_color, 1.0);
}
"""


class ViewerWindow(pyglet.window.Window):
    def __init__(
        self,
        env: Environment,
        sim_rate: float,
        move_speed: float,
        mouse_sensitivity: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.env = env
        self.sim_rate = sim_rate
        self.move_speed = move_speed
        self.mouse_sensitivity = mouse_sensitivity
        self.keys = pyglet.window.key.KeyStateHandler()
        self.push_handlers(self.keys)

        self.yaw = -45.0
        self.pitch = -35.0
        self.pos_x = env.width * 0.5
        self.pos_z = env.height * 0.5
        max_h = max(max(row) for row in env.height_map) if env.height_map else 10.0
        self.pos_y = max(12.0, max_h + 8.0)

        self._sim_accum = 0.0
        self.paused = False
        self.show_terrain = True
        self.show_agents = True
        self.show_minimap = True
        self.heatmap_mode = "terrain"  # terrain, food, water, material, height
        self.show_debug = True
        self.force_no_depth = False
        self.show_debug_triangle = False
        self.terrain_mode = "triangles"  # triangles, points, wire
        self.high_contrast = False

        self._build_shaders()
        self._build_meshes()
        self._build_minimap()
        self._build_debug()

    def _build_debug(self) -> None:
        verts = [-0.6, -0.6, 0.0, 0.6, -0.6, 0.0, 0.0, 0.6, 0.0]
        colors = [0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.2, 0.9]
        self.debug_vlist = self.shader.vertex_list(
            3,
            gl.GL_TRIANGLES,
            position=("f", verts),
            color=("f", colors),
        )
        self.hud_label = pyglet.text.Label(
            "",
            x=10,
            y=self.height - 10,
            anchor_x="left",
            anchor_y="top",
            color=(230, 230, 230, 255),
        )

        self.set_exclusive_mouse(True)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glClearColor(0.05, 0.05, 0.08, 1.0)

    def _build_shaders(self) -> None:
        self.shader = ShaderProgram(
            Shader(VERTEX_SRC, "vertex"),
            Shader(FRAGMENT_SRC, "fragment"),
        )

    def _build_meshes(self) -> None:
        self._terrain_vertices, self._terrain_colors = self._terrain_buffers()
        self.terrain_vlist = self.shader.vertex_list(
            len(self._terrain_vertices) // 3,
            gl.GL_TRIANGLES,
            position=("f", self._terrain_vertices),
            color=("f", self._terrain_colors),
        )
        self._rebuild_agents()

    def _build_minimap(self) -> None:
        self._minimap_vertices, self._minimap_colors = self._minimap_buffers()
        self.minimap_vlist = self.shader.vertex_list(
            len(self._minimap_vertices) // 3,
            gl.GL_TRIANGLES,
            position=("f", self._minimap_vertices),
            color=("f", self._minimap_colors),
        )
        self._rebuild_minimap_agents()

    def on_resize(self, width: int, height: int) -> None:
        super().on_resize(width, height)
        self._minimap_vertices, self._minimap_colors = self._minimap_buffers()
        self.minimap_vlist.delete()
        self.minimap_vlist = self.shader.vertex_list(
            len(self._minimap_vertices) // 3,
            gl.GL_TRIANGLES,
            position=("f", self._minimap_vertices),
            color=("f", self._minimap_colors),
        )
        self._rebuild_minimap_agents()
        self.hud_label.y = height - 10

    def on_draw(self) -> None:
        self.clear()
        gl.glViewport(0, 0, self.width, self.height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glDisable(gl.GL_CULL_FACE)

        proj = Mat4.perspective_projection(60.0, self.width / max(1, self.height), 0.1, 500.0)
        view = self._calc_view_matrix()
        mvp = proj @ view

        self.shader.use()
        self.shader["u_mvp"] = mvp
        self.shader["u_point_size"] = 1.0

        self._update_terrain_colors()
        if self.show_terrain:
            if self.force_no_depth:
                gl.glDisable(gl.GL_DEPTH_TEST)
            if self.terrain_mode == "points":
                self.shader["u_point_size"] = 3.0
                self.terrain_vlist.draw(gl.GL_POINTS)
                self.shader["u_point_size"] = 1.0
            elif self.terrain_mode == "wire":
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
                self.terrain_vlist.draw(gl.GL_TRIANGLES)
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            else:
                self.terrain_vlist.draw(gl.GL_TRIANGLES)
            if self.force_no_depth:
                gl.glEnable(gl.GL_DEPTH_TEST)
        if self.show_agents:
            self._update_agents()
            self.shader["u_point_size"] = 6.0
            self.agent_vlist.draw(gl.GL_POINTS)
            self.shader["u_point_size"] = 1.0

        if self.show_minimap:
            self._draw_minimap()

        if self.show_debug_triangle:
            self.shader["u_mvp"] = Mat4()
            self.debug_vlist.draw(gl.GL_TRIANGLES)
            self.shader["u_mvp"] = mvp

        if self.show_debug:
            self._draw_hud()

    def on_mouse_motion(self, _x: int, _y: int, dx: int, dy: int) -> None:
        self.yaw += dx * self.mouse_sensitivity
        self.pitch -= dy * self.mouse_sensitivity
        self.pitch = max(-89.0, min(89.0, self.pitch))

    def on_key_press(self, symbol: int, _modifiers: int) -> None:
        if symbol == pyglet.window.key.ESCAPE:
            self.set_exclusive_mouse(False)
        elif symbol == pyglet.window.key.SPACE:
            self.paused = not self.paused
        elif symbol == pyglet.window.key.T:
            self.show_terrain = not self.show_terrain
        elif symbol == pyglet.window.key.G:
            self.show_agents = not self.show_agents
        elif symbol == pyglet.window.key.M:
            self.show_minimap = not self.show_minimap
        elif symbol == pyglet.window.key.H:
            self._cycle_heatmap()
        elif symbol == pyglet.window.key.F:
            self.force_no_depth = not self.force_no_depth
        elif symbol == pyglet.window.key.B:
            self.show_debug_triangle = not self.show_debug_triangle
        elif symbol == pyglet.window.key.I:
            self.show_debug = not self.show_debug
        elif symbol == pyglet.window.key.P:
            self._cycle_terrain_mode()
        elif symbol == pyglet.window.key.C:
            self.high_contrast = not self.high_contrast
        elif symbol == pyglet.window.key._1:
            self.heatmap_mode = "terrain"
        elif symbol == pyglet.window.key._2:
            self.heatmap_mode = "food"
        elif symbol == pyglet.window.key._3:
            self.heatmap_mode = "water"
        elif symbol == pyglet.window.key._4:
            self.heatmap_mode = "material"
        elif symbol == pyglet.window.key._5:
            self.heatmap_mode = "height"

    def on_mouse_press(self, _x: int, _y: int, _button: int, _modifiers: int) -> None:
        self.set_exclusive_mouse(True)

    def update(self, dt: float) -> None:
        self._update_camera(dt)
        if self.paused:
            return
        self._sim_accum += dt * self.sim_rate
        steps = int(self._sim_accum)
        self._sim_accum -= steps
        for _ in range(steps):
            self.env.step()

    def _update_camera(self, dt: float) -> None:
        speed = self.move_speed * dt
        if self.keys[pyglet.window.key.LSHIFT]:
            speed *= 2.0

        yaw_rad = math.radians(self.yaw)
        forward_x = math.cos(yaw_rad)
        forward_z = math.sin(yaw_rad)
        right_x = -forward_z
        right_z = forward_x

        if self.keys[pyglet.window.key.W]:
            self.pos_x += forward_x * speed
            self.pos_z += forward_z * speed
        if self.keys[pyglet.window.key.S]:
            self.pos_x -= forward_x * speed
            self.pos_z -= forward_z * speed
        if self.keys[pyglet.window.key.A]:
            self.pos_x -= right_x * speed
            self.pos_z -= right_z * speed
        if self.keys[pyglet.window.key.D]:
            self.pos_x += right_x * speed
            self.pos_z += right_z * speed
        if self.keys[pyglet.window.key.Q]:
            self.pos_y -= speed
        if self.keys[pyglet.window.key.E]:
            self.pos_y += speed

        self.pos_x = max(0.0, min(self.env.width, self.pos_x))
        self.pos_z = max(0.0, min(self.env.height, self.pos_z))
        self.pos_y = max(1.0, self.pos_y)

    def _calc_view_matrix(self) -> Mat4:
        pitch_rad = math.radians(self.pitch)
        yaw_rad = math.radians(self.yaw)
        direction = Vec3(
            math.cos(pitch_rad) * math.cos(yaw_rad),
            math.sin(pitch_rad),
            math.cos(pitch_rad) * math.sin(yaw_rad),
        )
        pos = Vec3(self.pos_x, self.pos_y, self.pos_z)
        target = pos + direction
        return Mat4.look_at(pos, target, Vec3(0, 1, 0))

    def _terrain_buffers(self) -> Tuple[List[float], List[float]]:
        verts: List[float] = []
        colors: List[float] = []
        for y in range(self.env.height - 1):
            for x in range(self.env.width - 1):
                h00 = self.env.height_map[y][x]
                h10 = self.env.height_map[y][x + 1]
                h11 = self.env.height_map[y + 1][x + 1]
                h01 = self.env.height_map[y + 1][x]
                # two triangles
                verts.extend([x, h00, y, x + 1, h10, y, x + 1, h11, y + 1])
                verts.extend([x, h00, y, x + 1, h11, y + 1, x, h01, y + 1])
                color = self._tile_color(x, y)
                colors.extend(color * 6)
        return verts, colors

    def _minimap_buffers(self) -> Tuple[List[float], List[float]]:
        map_width = min(240, int(self.width * 0.35))
        map_height = int(map_width * (self.env.height / max(1, self.env.width)))
        x0 = 10
        y0 = 10
        tile_w = map_width / max(1, self.env.width)
        tile_h = map_height / max(1, self.env.height)

        verts: List[float] = []
        colors: List[float] = []
        for y in range(self.env.height - 1):
            for x in range(self.env.width - 1):
                sx = x0 + x * tile_w
                sy = y0 + y * tile_h
                verts.extend([sx, sy, 0.0, sx + tile_w, sy, 0.0, sx + tile_w, sy + tile_h, 0.0])
                verts.extend([sx, sy, 0.0, sx + tile_w, sy + tile_h, 0.0, sx, sy + tile_h, 0.0])
                color = self._tile_color(x, y)
                colors.extend(color * 6)
        return verts, colors

    def _tile_color(self, x: int, y: int) -> Tuple[float, float, float]:
        if self.high_contrast:
            return (0.1, 0.9, 0.2)
        mode = self.heatmap_mode
        if mode == "terrain":
            return ZONE_COLORS.get(self.env.zone_map[y][x], (0.2, 0.2, 0.2))
        if mode == "food":
            return self._heat_color(self.env.food_map[y][x], config.TILE_FOOD_MAX, (0.1, 0.2, 0.1), (0.2, 0.8, 0.2))
        if mode == "water":
            return self._heat_color(self.env.water_map[y][x], config.TILE_WATER_MAX, (0.1, 0.1, 0.3), (0.2, 0.5, 0.9))
        if mode == "material":
            return self._heat_color(
                self.env.material_map[y][x],
                config.TILE_MATERIAL_MAX,
                (0.2, 0.2, 0.2),
                (0.9, 0.7, 0.4),
            )
        if mode == "height":
            max_h = config.VOLCANO_PEAK_HEIGHT + config.ISLAND_HEIGHT_PEAK + 2
            return self._heat_color(self.env.height_map[y][x], max_h, (0.1, 0.1, 0.1), (0.9, 0.9, 0.9))
        return (0.2, 0.2, 0.2)

    def _heat_color(
        self,
        value: float,
        max_value: float,
        low: Tuple[float, float, float],
        high: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        if max_value <= 0:
            return low
        t = max(0.0, min(1.0, value / max_value))
        return (
            low[0] + (high[0] - low[0]) * t,
            low[1] + (high[1] - low[1]) * t,
            low[2] + (high[2] - low[2]) * t,
        )

    def _update_terrain_colors(self) -> None:
        colors: List[float] = []
        for y in range(self.env.height - 1):
            for x in range(self.env.width - 1):
                color = self._tile_color(x, y)
                colors.extend(color * 6)
        self.terrain_vlist.color[:] = colors

        if self.show_minimap:
            mini_colors: List[float] = []
            for y in range(self.env.height - 1):
                for x in range(self.env.width - 1):
                    color = self._tile_color(x, y)
                    mini_colors.extend(color * 6)
            self.minimap_vlist.color[:] = mini_colors

    def _rebuild_agents(self) -> None:
        positions: List[float] = []
        colors: List[float] = []
        for agent in self.env.agents:
            if not agent.alive:
                continue
            positions.extend([agent.x + 0.5, agent.z + 0.5, agent.y + 0.5])
            if agent.sex == config.SEX_A:
                colors.extend([0.9, 0.2, 0.2])
            else:
                colors.extend([0.2, 0.3, 0.9])
        self.agent_vlist = self.shader.vertex_list(
            max(1, len(positions) // 3),
            gl.GL_POINTS,
            position=("f", positions if positions else [0.0, 0.0, 0.0]),
            color=("f", colors if colors else [0.0, 0.0, 0.0]),
        )

    def _update_agents(self) -> None:
        positions: List[float] = []
        colors: List[float] = []
        for agent in self.env.agents:
            if not agent.alive:
                continue
            positions.extend([agent.x + 0.5, agent.z + 0.5, agent.y + 0.5])
            if agent.sex == config.SEX_A:
                colors.extend([0.9, 0.2, 0.2])
            else:
                colors.extend([0.2, 0.3, 0.9])

        count = len(positions) // 3
        if count == 0:
            positions = [0.0, 0.0, 0.0]
            colors = [0.0, 0.0, 0.0]
            count = 1

        if count != self.agent_vlist.count:
            self.agent_vlist.delete()
            self.agent_vlist = self.shader.vertex_list(
                count,
                gl.GL_POINTS,
                position=("f", positions),
                color=("f", colors),
            )
        else:
            self.agent_vlist.position[:] = positions
            self.agent_vlist.color[:] = colors

    def _rebuild_minimap_agents(self) -> None:
        positions, colors = self._minimap_agent_buffers()
        self.minimap_agent_vlist = self.shader.vertex_list(
            max(1, len(positions) // 3),
            gl.GL_POINTS,
            position=("f", positions if positions else [0.0, 0.0, 0.0]),
            color=("f", colors if colors else [0.0, 0.0, 0.0]),
        )

    def _update_minimap_agents(self) -> None:
        positions, colors = self._minimap_agent_buffers()
        count = len(positions) // 3
        if count == 0:
            positions = [0.0, 0.0, 0.0]
            colors = [0.0, 0.0, 0.0]
            count = 1
        if count != self.minimap_agent_vlist.count:
            self.minimap_agent_vlist.delete()
            self.minimap_agent_vlist = self.shader.vertex_list(
                count,
                gl.GL_POINTS,
                position=("f", positions),
                color=("f", colors),
            )
        else:
            self.minimap_agent_vlist.position[:] = positions
            self.minimap_agent_vlist.color[:] = colors

    def _minimap_agent_buffers(self) -> Tuple[List[float], List[float]]:
        map_width = min(240, int(self.width * 0.35))
        map_height = int(map_width * (self.env.height / max(1, self.env.width)))
        x0 = 10
        y0 = 10
        tile_w = map_width / max(1, self.env.width)
        tile_h = map_height / max(1, self.env.height)

        positions: List[float] = []
        colors: List[float] = []
        for agent in self.env.agents:
            if not agent.alive:
                continue
            sx = x0 + agent.x * tile_w + tile_w * 0.5
            sy = y0 + agent.y * tile_h + tile_h * 0.5
            positions.extend([sx, sy, 0.0])
            if agent.sex == config.SEX_A:
                colors.extend([0.9, 0.2, 0.2])
            else:
                colors.extend([0.2, 0.3, 0.9])
        return positions, colors

    def _draw_minimap(self) -> None:
        gl.glDisable(gl.GL_DEPTH_TEST)
        proj = Mat4.orthogonal_projection(0, self.width, 0, self.height, -1, 1)
        self.shader["u_mvp"] = proj
        self.shader["u_point_size"] = 2.0
        self.minimap_vlist.draw(gl.GL_TRIANGLES)
        if self.show_agents:
            self._update_minimap_agents()
            self.shader["u_point_size"] = 4.0
            self.minimap_agent_vlist.draw(gl.GL_POINTS)

        # restore 3D matrix for next frame
        gl.glEnable(gl.GL_DEPTH_TEST)
        proj3d = Mat4.perspective_projection(60.0, self.width / max(1, self.height), 0.1, 500.0)
        view = self._calc_view_matrix()
        self.shader["u_mvp"] = proj3d @ view

    def _draw_hud(self) -> None:
        self.hud_label.text = (
            f"pos=({self.pos_x:.1f},{self.pos_y:.1f},{self.pos_z:.1f}) "
            f"yaw={self.yaw:.1f} pitch={self.pitch:.1f} "
            f"mode={self.heatmap_mode} depth_off={self.force_no_depth} "
            f"terrain={self.terrain_mode} verts={self.terrain_vlist.count}"
        )
        self.hud_label.draw()

    def _cycle_heatmap(self) -> None:
        modes = ["terrain", "food", "water", "material", "height"]
        idx = modes.index(self.heatmap_mode)
        self.heatmap_mode = modes[(idx + 1) % len(modes)]

    def _cycle_terrain_mode(self) -> None:
        modes = ["triangles", "points", "wire"]
        idx = modes.index(self.terrain_mode)
        self.terrain_mode = modes[(idx + 1) % len(modes)]


def main() -> None:
    parser = argparse.ArgumentParser(description="DemoWorld FPS viewer (WASD + mouse)")
    parser.add_argument("--steps", type=int, default=0, help="0 = run indefinitely")
    parser.add_argument("--initial-pop", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sim-rate", type=float, default=20.0, help="Simulation steps per second")
    parser.add_argument("--move-speed", type=float, default=8.0, help="Camera speed")
    parser.add_argument("--mouse-sensitivity", type=float, default=0.15)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    env = Environment(rng)
    env.seed_initial_population(args.initial_pop)

    window = ViewerWindow(
        env=env,
        sim_rate=args.sim_rate,
        move_speed=args.move_speed,
        mouse_sensitivity=args.mouse_sensitivity,
        width=args.width,
        height=args.height,
        caption="DemoWorld Viewer",
        resizable=True,
    )

    if args.steps > 0:
        target_steps = args.steps

        def stop_after(_dt: float) -> None:
            if env.step_count >= target_steps:
                pyglet.app.exit()

        pyglet.clock.schedule_interval(stop_after, 0.1)

    pyglet.clock.schedule_interval(window.update, 1.0 / 60.0)
    pyglet.app.run()


if __name__ == "__main__":
    main()
