import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

def make_env(render_mode=None) -> gym.Env:
    return TagIsoEnv(render_mode=render_mode)


class TagIsoEnv(gym.Env):
    """
    2-player Tag game (self-play with shared policy).
    - Two cubes in 2D (x,y) with simple jump (z) for each.
    - One is Runner, the other is Tagger. If tagged -> roles swap.
    - Round lasts fixed time (max_steps).
    - At end: whoever is Runner gets the point.

    Observation (for the "controlled" agent = player_id):
      [ self_x, self_y, self_z, self_vz,
        other_x, other_y, other_z, other_vz,
        is_runner (0/1), time_left (0..1) ] normalized roughly to [0,1]/[-1,1].

    Action: MultiDiscrete([5,2])
      move: 0 stay, 1 left, 2 right, 3 up, 4 down
      jump: 0 no, 1 jump
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        arena_size=10.0,
        max_seconds=30,
        fps=30,
        tag_radius=0.8,
        move_speed=0.35,
        jump_speed=1.8,
        gravity=0.25,
        seed=None,
    ):
        super().__init__()
        self.render_mode = render_mode

        # World params
        self.arena_size = float(arena_size)
        self.max_steps = int(max_seconds * fps)
        self.fps = int(fps)

        self.tag_radius = float(tag_radius)
        self.move_speed = float(move_speed)
        self.jump_speed = float(jump_speed)
        self.gravity = float(gravity)

        # Multi-agent internal state
        self.pos = np.zeros((2, 2), dtype=np.float32)   # x,y for players 0/1
        self.z = np.zeros(2, dtype=np.float32)
        self.vz = np.zeros(2, dtype=np.float32)
        self.is_runner = 0  # which player id is runner (0 or 1)

        self.step_count = 0

        # Spaces
        self.action_space = spaces.MultiDiscrete([5, 2])

        # Observation is 10 floats
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )

        # Rendering
        self._screen = None
        self._clock = None
        self._pygame = None

        # For rendering UI score across episodes (not used in training)
        self.score = [0, 0]

        if seed is not None:
            self.reset(seed=seed)

    def _clamp_arena(self):
        self.pos[:, 0] = np.clip(self.pos[:, 0], 0.0, self.arena_size)
        self.pos[:, 1] = np.clip(self.pos[:, 1], 0.0, self.arena_size)

    def _norm_pos(self, xy):
        # map [0, arena] -> [-1, 1]
        return (xy / self.arena_size) * 2.0 - 1.0

    def _get_obs_for(self, player_id: int):
        other_id = 1 - player_id

        sx, sy = self._norm_pos(self.pos[player_id])
        ox, oy = self._norm_pos(self.pos[other_id])

        # z/vz: keep in [-1,1] with simple scaling
        sz = np.clip(self.z[player_id] / 3.0, -1.0, 1.0)
        svz = np.clip(self.vz[player_id] / 3.0, -1.0, 1.0)
        oz = np.clip(self.z[other_id] / 3.0, -1.0, 1.0)
        ovz = np.clip(self.vz[other_id] / 3.0, -1.0, 1.0)

        is_runner_flag = 1.0 if self.is_runner == player_id else 0.0
        time_left = 1.0 - (self.step_count / max(1, self.max_steps))

        return np.array(
            [sx, sy, sz, svz, ox, oy, oz, ovz, is_runner_flag, time_left],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random positions separated a bit
        self.pos[0] = self.np_random.uniform(1.0, self.arena_size - 1.0, size=(2,))
        self.pos[1] = self.np_random.uniform(1.0, self.arena_size - 1.0, size=(2,))
        # Ensure not too close
        for _ in range(10):
            if np.linalg.norm(self.pos[0] - self.pos[1]) > 2.0:
                break
            self.pos[1] = self.np_random.uniform(1.0, self.arena_size - 1.0, size=(2,))

        self.z[:] = 0.0
        self.vz[:] = 0.0

        # Random runner
        self.is_runner = int(self.np_random.integers(0, 2))
        self.step_count = 0

        # Training will typically control one "agent" at a time; for simple testing we return obs for player 0.
        obs = self._get_obs_for(0)
        info = {"runner_id": self.is_runner}
        return obs, info

    def _apply_action(self, pid: int, action):
        move, jump = int(action[0]), int(action[1])

        # movement
        dx = dy = 0.0
        if move == 1:
            dx = -self.move_speed
        elif move == 2:
            dx = self.move_speed
        elif move == 3:
            dy = -self.move_speed
        elif move == 4:
            dy = self.move_speed

        self.pos[pid, 0] += dx
        self.pos[pid, 1] += dy

        # jump
        if jump == 1 and self.z[pid] <= 1e-6:
            self.vz[pid] = self.jump_speed

    def _step_jump_physics(self):
        # simple gravity and ground collision
        for pid in (0, 1):
            self.vz[pid] -= self.gravity
            self.z[pid] += self.vz[pid]
            if self.z[pid] < 0.0:
                self.z[pid] = 0.0
                self.vz[pid] = 0.0

    def _check_tag(self):
        # tag based on xy proximity; ignore z for now (simple)
        d = float(np.linalg.norm(self.pos[0] - self.pos[1]))
        return d <= self.tag_radius

    def step(self, action):
        """
        Single-agent Gym interface: we will treat 'action' as action for player 0,
        and use a simple scripted policy for player 1 in test mode (or later, we will wrap self-play).
        For now: player 1 does a naive chase/escape heuristic (to make the world dynamic).
        """
        self.step_count += 1

        # Determine roles
        runner = self.is_runner
        tagger = 1 - runner

        # Action for player 0 comes from caller
        self._apply_action(0, action)

        # Simple heuristic for player 1 (only for now; training will replace this)
        # If player 1 is tagger: chase runner; if runner: run away.
        target_dir = self.pos[runner] - self.pos[1]
        dist = float(np.linalg.norm(target_dir) + 1e-8)
        unit = target_dir / dist

        # convert to a move action
        if self.is_runner == 1:
            # player1 is runner: run away from tagger (player0)
            unit = -unit

        # choose axis-aligned move
        ax, ay = unit[0], unit[1]
        if abs(ax) > abs(ay):
            move1 = 2 if ax > 0 else 1
        else:
            move1 = 4 if ay > 0 else 3

        jump1 = 1 if (self.z[1] <= 1e-6 and self.np_random.random() < 0.02) else 0
        self._apply_action(1, (move1, jump1))

        self._clamp_arena()
        self._step_jump_physics()

        # Rewards from player0 perspective
        reward = 0.0
        terminated = False

        tagged = self._check_tag()
        if tagged:
            # swap roles
            old_runner = self.is_runner
            self.is_runner = 1 - self.is_runner

            # If player0 was tagger and got tag -> good. If player0 was runner and got tagged -> bad.
            if old_runner == 0:
                # player0 was runner, got tagged
                reward -= 1.0
            else:
                # player0 was tagger, tagged runner
                reward += 1.0

        # Small shaping: survive bonus if player0 is runner
        if self.is_runner == 0:
            reward += 0.01
        else:
            reward -= 0.01

        # time limit
        if self.step_count >= self.max_steps:
            terminated = True
            # end-of-round reward: if player0 ends as runner -> big positive
            if self.is_runner == 0:
                reward += 2.0
                self.score[0] += 1
            else:
                reward -= 2.0
                self.score[1] += 1

        obs = self._get_obs_for(0)
        info = {
            "runner_id": self.is_runner,
            "time_left": 1.0 - (self.step_count / max(1, self.max_steps)),
            "score_p0": self.score[0],
            "score_p1": self.score[1],
        }

        if self.render_mode == "human":
            self.render(info)

        return obs, reward, terminated, False, info

    # ---------- Rendering (Isometric-ish) ----------
    def _iso(self, x, y, z=0.0):
        # Isometric projection
        iso_scale = 28
        ox = 220
        oy = 110
        sx = (x - y) * iso_scale + ox
        sy = (x + y) * iso_scale * 0.5 + oy - (z * 18)
        return int(sx), int(sy)

    def render(self, info=None):
        if self._screen is None:
            import pygame
            self._pygame = pygame
            pygame.init()
            self._screen = pygame.display.set_mode((640, 480))
            pygame.display.set_caption("Tag (Isometric) RL")
            self._clock = pygame.time.Clock()

        pygame = self._pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self._screen.fill((18, 18, 22))

        # Draw arena grid
        grid_n = int(self.arena_size)
        for i in range(grid_n + 1):
            # horizontal lines (y constant)
            x0, y0 = self._iso(0, i, 0)
            x1, y1 = self._iso(self.arena_size, i, 0)
            pygame.draw.line(self._screen, (40, 40, 48), (x0, y0), (x1, y1), 1)

            # vertical lines (x constant)
            x0, y0 = self._iso(i, 0, 0)
            x1, y1 = self._iso(i, self.arena_size, 0)
            pygame.draw.line(self._screen, (40, 40, 48), (x0, y0), (x1, y1), 1)

        # Draw players as "cubes" (simple rectangles offset by z)
        def draw_cube(pid, color_top, color_side):
            x, y = self.pos[pid]
            z = self.z[pid]
            sx, sy = self._iso(x, y, z)
            # cube size in screen space
            w, h = 22, 22
            # shadow / base
            bx, by = self._iso(x, y, 0)
            pygame.draw.ellipse(self._screen, (10, 10, 10), (bx - 10, by + 8, 20, 8))
            # body
            pygame.draw.rect(self._screen, color_side, (sx - w//2, sy - h//2, w, h))
            # top highlight
            pygame.draw.rect(self._screen, color_top, (sx - w//2, sy - h//2, w, h//2))

            # Role marker
            if self.is_runner == pid:
                pygame.draw.circle(self._screen, (255, 220, 80), (sx, sy - 18), 6)
            else:
                pygame.draw.circle(self._screen, (120, 200, 255), (sx, sy - 18), 6)

        # Draw in y-order for nicer overlap
        order = sorted([0, 1], key=lambda i: (self.pos[i][0] + self.pos[i][1]))
        for pid in order:
            if pid == 0:
                draw_cube(pid, (200, 120, 120), (150, 70, 70))
            else:
                draw_cube(pid, (120, 160, 220), (70, 110, 170))

        # HUD
        font = pygame.font.SysFont(None, 26)
        time_left = info.get("time_left") if info else 0.0
        runner = info.get("runner_id") if info else self.is_runner
        s0 = info.get("score_p0") if info else self.score[0]
        s1 = info.get("score_p1") if info else self.score[1]

        txt1 = font.render(f"Time: {time_left*30:05.1f}s", True, (230, 230, 240))
        txt2 = font.render(f"Score P0: {s0}   P1: {s1}", True, (230, 230, 240))
        txt3 = font.render(f"Runner: P{runner}", True, (230, 230, 240))

        self._screen.blit(txt1, (12, 12))
        self._screen.blit(txt2, (12, 40))
        self._screen.blit(txt3, (12, 68))

        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None

        if self.render_mode == "rgb_array":
            frame = pygame.surfarray.array3d(self._screen)
            # pygame returns (W, H, C); convert to (H, W, C)
            frame = np.transpose(frame, (1, 0, 2))
            return frame

        return None

    def close(self):
        if self._screen is not None:
            self._pygame.quit()
            self._screen = None
            self._clock = None
            self._pygame = None
