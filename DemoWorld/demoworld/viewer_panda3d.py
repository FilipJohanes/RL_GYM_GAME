"""Panda3D viewer for DemoWorld (WASD + mouse look)."""

from __future__ import annotations

import argparse
import math
import random
from typing import List

try:
    from direct.showbase.ShowBase import ShowBase
    from direct.gui.OnscreenText import OnscreenText
    from panda3d.core import (
        Geom,
        GeomNode,
        GeomPoints,
        GeomTriangles,
        GeomVertexData,
        GeomVertexFormat,
        GeomVertexWriter,
        LColor,
        NodePath,
        LineSegs,
        TextNode,
        WindowProperties,
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Viewer requires panda3d. Install with: pip install panda3d") from exc

from . import config
from .environment import Environment


ZONE_COLORS = {
    config.ZONE_VOID: LColor(0.1, 0.1, 0.1, 1.0),
    config.ZONE_PLATFORM: LColor(0.55, 0.55, 0.55, 1.0),
    config.ZONE_HOUSING: LColor(0.8, 0.55, 0.2, 1.0),
    config.ZONE_RESOURCE: LColor(0.2, 0.7, 0.9, 1.0),
}


class DemoWorldApp(ShowBase):
    def __init__(self, env: Environment, sim_rate: float, initial_pop: int, headless: bool = False) -> None:
        if headless:
            from panda3d.core import loadPrcFileData

            loadPrcFileData("", "window-type none")

        super().__init__()
        self.disableMouse()
        self.render.setLightOff()
        self.render.setShaderOff()

        self.env = env
        self.sim_rate = sim_rate
        self._sim_accum = 0.0
        self.paused = False
        self._last_stats = None

        self.key_map = {"w": False, "a": False, "s": False, "d": False, "q": False, "e": False}
        for key in self.key_map:
            self.accept(key, self._set_key, [key, True])
            self.accept(f"{key}-up", self._set_key, [key, False])

        self.time_text = OnscreenText(
            text="",
            pos=(-1.32, 0.92),
            scale=0.045,
            align=TextNode.ALeft,
            fg=(1, 1, 1, 1),
            shadow=(0, 0, 0, 1),
        )
        self.accept("space", self._toggle_pause)
        self.accept("escape", self._toggle_mouse)
        self.accept("mouse3", self._toggle_mouse)
        self.accept("wheel_up", self._speed_up)
        self.accept("wheel_down", self._speed_down)
        self.accept("window-event", self._on_window_event)

        self.heading = 0.0
        self.pitch = -89.0
        max_h = max(max(row) for row in env.height_map) if env.height_map else 10.0
        scale_xy = config.RENDER_SCALE_XY
        scale_z = config.RENDER_SCALE_Z
        lens = self.cam.node().getLens()
        far_plane = max(env.width, env.height) * scale_xy * 20.0
        lens.setNearFar(0.1, far_plane)
        center_x = env.width * 0.5 * scale_xy
        center_y = env.height * 0.5 * scale_xy
        extent = max(env.width, env.height) * scale_xy
        self.camera.setPos(
            center_x,
            center_y,
            (max_h * scale_z) + extent * 1.2,
        )

        self._mouse_enabled = False
        # Scale speed to world size; user can further adjust with mouse wheel.
        self.move_speed = max(2000.0, config.RENDER_SCALE_XY * 0.5)
        self._setup_mouse()

        self.terrain_node = self._build_terrain()
        self.agent_node = self._build_agents()
        self._build_axes()
        self._build_housing_grid()
        self._build_debug_cube()

        self.taskMgr.add(self._update_task, "update")

    def _set_key(self, key: str, value: bool) -> None:
        self.key_map[key] = value

    def _toggle_pause(self) -> None:
        self.paused = not self.paused

    def _toggle_mouse(self) -> None:
        self._mouse_enabled = not self._mouse_enabled
        props = WindowProperties()
        props.setCursorHidden(self._mouse_enabled)
        self.win.requestProperties(props)
        if self._mouse_enabled:
            self._center_mouse()

    def _setup_mouse(self) -> None:
        self._mouse_enabled = True
        props = WindowProperties()
        props.setCursorHidden(True)
        self.win.requestProperties(props)
        self._center_mouse()

    def _on_window_event(self, window):
        if not window:
            return
        if window.isActive():
            if self._mouse_enabled:
                self._center_mouse()
        else:
            self._mouse_enabled = False
            props = WindowProperties()
            props.setCursorHidden(False)
            self.win.requestProperties(props)

    def _speed_up(self) -> None:
        self.move_speed = min(self.move_speed * 1.75, config.RENDER_SCALE_XY * 50.0)

    def _speed_down(self) -> None:
        self.move_speed = max(self.move_speed / 1.75, 50.0)

    def _center_mouse(self) -> None:
        if not self.win:
            return
        x = int(self.win.getXSize() / 2)
        y = int(self.win.getYSize() / 2)
        self.win.movePointer(0, x, y)

    def _update_task(self, task):
        dt = globalClock.getDt()
        self._update_camera(dt)
        if not self.paused:
            self._sim_accum += dt * self.sim_rate
            steps = int(self._sim_accum)
            self._sim_accum -= steps
            for _ in range(steps):
                step_stats = self.env.step()
                self._last_stats = step_stats
                if step_stats.step % 20 == 0:
                    if self.env.agents:
                        min_h = min(a.hydration for a in self.env.agents)
                        max_h = max(a.hydration for a in self.env.agents)
                    else:
                        min_h = 0.0
                        max_h = 0.0
                    print(
                        f"step={step_stats.step} pop={step_stats.population} "
                        f"births={step_stats.births} deaths={step_stats.deaths} "
                        f"starv={step_stats.deaths_starvation} dehydr={step_stats.deaths_dehydration} "
                        f"heart={step_stats.deaths_heart} age={step_stats.deaths_age} "
                        f"acc={step_stats.deaths_accident} "
                        f"avgE={step_stats.avg_energy:.3f} avgH={step_stats.avg_hydration:.3f} "
                        f"minH={min_h:.3f} maxH={max_h:.3f} avgHeart={step_stats.avg_heart:.3f}"
                    )
            if self._last_stats is not None:
                phase = "day" if self._last_stats.is_day else "night"
                self.time_text.setText(
                    f"step={self._last_stats.step} year={self._last_stats.year_index} "
                    f"day={self._last_stats.day_index} {phase}"
                )
            self._update_agents()
        return task.cont

    def _update_camera(self, dt: float) -> None:
        if self._mouse_enabled and self.mouseWatcherNode.hasMouse():
            md = self.win.getPointer(0)
            x = md.getX()
            y = md.getY()
            cx = self.win.getXSize() / 2
            cy = self.win.getYSize() / 2
            dx = (x - cx) * 0.1
            dy = (y - cy) * 0.1
            self.heading -= dx
            self.pitch = max(-89.0, min(89.0, self.pitch - dy))
            self._center_mouse()

        self.camera.setHpr(self.heading, self.pitch, 0)
        speed = self.move_speed * dt
        if self.key_map["w"]:
            self.camera.setY(self.camera, speed)
        if self.key_map["s"]:
            self.camera.setY(self.camera, -speed)
        if self.key_map["a"]:
            self.camera.setX(self.camera, -speed)
        if self.key_map["d"]:
            self.camera.setX(self.camera, speed)
        if self.key_map["q"]:
            self.camera.setZ(self.camera, self.camera.getZ() - speed)
        if self.key_map["e"]:
            self.camera.setZ(self.camera, self.camera.getZ() + speed)

    def _build_terrain(self) -> NodePath:
        format_ = GeomVertexFormat.getV3c4()
        vdata = GeomVertexData("terrain", format_, Geom.UHStatic)
        vwriter = GeomVertexWriter(vdata, "vertex")
        cwriter = GeomVertexWriter(vdata, "color")
        scale_xy = config.RENDER_SCALE_XY
        scale_z = config.RENDER_SCALE_Z

        for y in range(self.env.height - 1):
            for x in range(self.env.width - 1):
                h00 = self.env.height_map[y][x]
                h10 = self.env.height_map[y][x + 1]
                h11 = self.env.height_map[y + 1][x + 1]
                h01 = self.env.height_map[y + 1][x]
                if self.env.tiles[y][x] == config.TERRAIN_HOUSING:
                    h00 += config.HOUSING_HEIGHT_OFFSET
                    h10 += config.HOUSING_HEIGHT_OFFSET
                    h11 += config.HOUSING_HEIGHT_OFFSET
                    h01 += config.HOUSING_HEIGHT_OFFSET
                color = ZONE_COLORS.get(self.env.zone_map[y][x], LColor(0.2, 0.2, 0.2, 1.0))

                vwriter.addData3f(x * scale_xy, y * scale_xy, h00 * scale_z)
                cwriter.addData4f(color)
                vwriter.addData3f((x + 1) * scale_xy, y * scale_xy, h10 * scale_z)
                cwriter.addData4f(color)
                vwriter.addData3f((x + 1) * scale_xy, (y + 1) * scale_xy, h11 * scale_z)
                cwriter.addData4f(color)

                vwriter.addData3f(x * scale_xy, y * scale_xy, h00 * scale_z)
                cwriter.addData4f(color)
                vwriter.addData3f((x + 1) * scale_xy, (y + 1) * scale_xy, h11 * scale_z)
                cwriter.addData4f(color)
                vwriter.addData3f(x * scale_xy, (y + 1) * scale_xy, h01 * scale_z)
                cwriter.addData4f(color)

        tris = GeomTriangles(Geom.UHStatic)
        for i in range(0, vdata.getNumRows(), 3):
            tris.addVertices(i, i + 1, i + 2)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("terrain")
        node.addGeom(geom)
        np = self.render.attachNewNode(node)
        np.setTwoSided(True)
        return np

    def _build_axes(self) -> None:
        scale_xy = config.RENDER_SCALE_XY
        axes = LineSegs()
        axes.setThickness(2.0)
        axes.setColor(1, 0, 0, 1)
        axes.moveTo(0, 0, 0)
        axes.drawTo(20 * scale_xy, 0, 0)
        axes.setColor(0, 1, 0, 1)
        axes.moveTo(0, 0, 0)
        axes.drawTo(0, 20 * scale_xy, 0)
        axes.setColor(0, 0, 1, 1)
        axes.moveTo(0, 0, 0)
        axes.drawTo(0, 0, 20 * config.RENDER_SCALE_Z)
        self.render.attachNewNode(axes.create())

    def _build_housing_grid(self) -> None:
        grid = LineSegs()
        grid.setThickness(1.0)
        grid.setColor(0.1, 0.1, 0.1, 1.0)
        scale_xy = config.RENDER_SCALE_XY
        for floor in range(config.FLOOR_COUNT):
            y_start = floor * config.FLOOR_HEIGHT
            y_end = y_start + config.FLOOR_HEIGHT
            z = floor * config.PLATFORM_Z_STEP * config.RENDER_SCALE_Z + config.HOUSING_HEIGHT_OFFSET * config.RENDER_SCALE_Z
            for y in range(y_start, y_end + 1):
                grid.moveTo(0, y * scale_xy, z)
                grid.drawTo(config.HOUSING_DEPTH * scale_xy, y * scale_xy, z)
                grid.moveTo((self.env.width - config.HOUSING_DEPTH) * scale_xy, y * scale_xy, z)
                grid.drawTo(self.env.width * scale_xy, y * scale_xy, z)
            for x in range(config.HOUSING_DEPTH + 1):
                grid.moveTo(x * scale_xy, y_start * scale_xy, z)
                grid.drawTo(x * scale_xy, y_end * scale_xy, z)
            for x in range(self.env.width - config.HOUSING_DEPTH, self.env.width + 1):
                grid.moveTo(x * scale_xy, y_start * scale_xy, z)
                grid.drawTo(x * scale_xy, y_end * scale_xy, z)

        self.render.attachNewNode(grid.create())

    def _build_debug_cube(self) -> None:
        from panda3d.core import CardMaker

        size = 50 * config.RENDER_SCALE_XY
        cm = CardMaker("debug")
        cm.setFrame(-size, size, -size, size)
        card = self.render.attachNewNode(cm.generate())
        card.setPos(self.env.width * 0.5 * config.RENDER_SCALE_XY, 0, 0)
        card.setHpr(0, 90, 0)
        card.setColor(1, 0, 0, 1)

    def _build_agents(self) -> NodePath:
        format_ = GeomVertexFormat.getV3c4()
        vdata = GeomVertexData("agents", format_, Geom.UHDynamic)
        vdata.setNumRows(3)
        self.agent_vwriter = GeomVertexWriter(vdata, "vertex")
        self.agent_cwriter = GeomVertexWriter(vdata, "color")

        self.agent_tris = GeomTriangles(Geom.UHDynamic)
        geom = Geom(vdata)
        geom.addPrimitive(self.agent_tris)
        node = GeomNode("agents")
        node.addGeom(geom)
        np = self.render.attachNewNode(node)
        np.setTwoSided(True)
        self.agent_geom = geom
        return np

    def _update_agents(self) -> None:
        alive = [a for a in self.env.agents if a.alive]
        if not alive:
            return

        radius = config.AGENT_RENDER_RADIUS * config.RENDER_SCALE_XY
        circle_segments = 10
        verts_per_circle = circle_segments
        tris_per_circle = circle_segments - 2
        square_verts = 4
        square_tris = 2

        total_verts = 0
        total_tris = 0
        for agent in alive:
            if agent.sex == config.SEX_A:
                total_verts += square_verts
                total_tris += square_tris
            else:
                total_verts += verts_per_circle
                total_tris += tris_per_circle

        vdata = self.agent_vwriter.getVertexData()
        if vdata.getNumRows() != total_verts:
            vdata.setNumRows(total_verts)

        scale_xy = config.RENDER_SCALE_XY
        scale_z = config.RENDER_SCALE_Z
        self.agent_vwriter.setRow(0)
        self.agent_cwriter.setRow(0)
        self.agent_tris.clearVertices()

        vert_index = 0
        for agent in alive:
            base_x = (agent.x + 0.5) * scale_xy
            base_y = (agent.y + 0.5) * scale_xy
            base_z = (agent.z + 0.5) * scale_z
            if agent.sex == config.SEX_A:
                color = (0.9, 0.2, 0.2, 1.0)
                corners = [
                    (base_x - radius, base_y - radius, base_z),
                    (base_x + radius, base_y - radius, base_z),
                    (base_x + radius, base_y + radius, base_z),
                    (base_x - radius, base_y + radius, base_z),
                ]
                for x, y, z in corners:
                    self.agent_vwriter.addData3f(x, y, z)
                    self.agent_cwriter.addData4f(*color)
                self.agent_tris.addVertices(vert_index, vert_index + 1, vert_index + 2)
                self.agent_tris.addVertices(vert_index, vert_index + 2, vert_index + 3)
                vert_index += 4
            else:
                color = (0.2, 0.3, 0.9, 1.0)
                for i in range(circle_segments):
                    angle = (2 * math.pi * i) / circle_segments
                    x = base_x + math.cos(angle) * radius
                    y = base_y + math.sin(angle) * radius
                    self.agent_vwriter.addData3f(x, y, base_z)
                    self.agent_cwriter.addData4f(*color)
                for i in range(1, circle_segments - 1):
                    self.agent_tris.addVertices(vert_index, vert_index + i, vert_index + i + 1)
                vert_index += circle_segments

        self.agent_geom.clearPrimitives()
        self.agent_geom.addPrimitive(self.agent_tris)


def main() -> None:
    parser = argparse.ArgumentParser(description="DemoWorld Panda3D viewer")
    parser.add_argument("--initial-pop", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sim-rate", type=float, default=20.0, help="Simulation steps per second")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    env = Environment(rng)
    env.seed_initial_population(args.initial_pop)

    app = DemoWorldApp(env, sim_rate=args.sim_rate, initial_pop=args.initial_pop)
    app.run()


if __name__ == "__main__":
    main()
