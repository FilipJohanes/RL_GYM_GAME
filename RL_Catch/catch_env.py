import numpy as np
import gymnasium as gym
from gymnasium import spaces

class CatchEnv(gym.Env):
    """
    Simple Catch game:
    - paddle at bottom moves left/right
    - ball falls from top
    - reward +1 if caught, -1 if missed
    Observation: [ball_x, ball_y, paddle_x] normalized to [0,1]
    Actions: 0=stay, 1=left, 2=right
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, width=400, height=400):
        super().__init__()
        self.render_mode = render_mode
        self.width = width
        self.height = height

        # Game objects
        self.paddle_w = 80
        self.paddle_h = 12
        self.paddle_y = self.height - 30
        self.paddle_speed = 12

        self.ball_r = 10
        self.ball_speed = 6

        # Actions: stay, left, right
        self.action_space = spaces.Discrete(3)

        # Observations: ball_x, ball_y, paddle_x in [0,1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Internal state
        self.ball_x = None
        self.ball_y = None
        self.paddle_x = None

        # Pygame
        self._screen = None
        self._clock = None

    def _get_obs(self):
        return np.array(
            [
                self.ball_x / self.width,
                self.ball_y / self.height,
                self.paddle_x / self.width,
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Paddle starts centered
        self.paddle_x = (self.width - self.paddle_w) / 2

        # Ball spawns at random x, near top
        self.ball_x = self.np_random.integers(self.ball_r, self.width - self.ball_r)
        self.ball_y = self.ball_r + 5

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        # Move paddle
        if action == 1:
            self.paddle_x -= self.paddle_speed
        elif action == 2:
            self.paddle_x += self.paddle_speed

        self.paddle_x = float(np.clip(self.paddle_x, 0, self.width - self.paddle_w))

        # Move ball down
        self.ball_y += self.ball_speed

        terminated = False
        reward = 0.0

        # Check if ball reached paddle line (bottom zone)
        if self.ball_y + self.ball_r >= self.paddle_y:
            terminated = True
            paddle_left = self.paddle_x
            paddle_right = self.paddle_x + self.paddle_w

            caught = (paddle_left <= self.ball_x <= paddle_right)
            reward = 1.0 if caught else -1.0

        obs = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, False, info

    def render(self):
        if self._screen is None:
            import pygame
            pygame.init()
            self._screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Catch RL")
            self._clock = pygame.time.Clock()

        import pygame

        # Handle window events (prevents "not responding")
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self._screen = None
                return

        self._screen.fill((20, 20, 30))

        # Draw paddle
        paddle_rect = pygame.Rect(
            int(self.paddle_x), int(self.paddle_y), int(self.paddle_w), int(self.paddle_h)
        )
        pygame.draw.rect(self._screen, (200, 200, 240), paddle_rect)

        # Draw ball
        pygame.draw.circle(
            self._screen,
            (120, 200, 120),
            (int(self.ball_x), int(self.ball_y)),
            int(self.ball_r),
        )

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None
            self._clock = None
