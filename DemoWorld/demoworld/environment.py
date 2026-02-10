"""Environment and simulation loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math
import random

from . import config
from .agents import Agent, BirthRequest, sample_trait, sample_genetic_trait, clamp01


@dataclass
class StepStats:
    step: int
    day_index: int
    year_index: int
    is_day: bool
    population: int
    births: int
    deaths: int
    accidental_deaths: int
    deaths_starvation: int
    deaths_dehydration: int
    deaths_heart: int
    deaths_age: int
    deaths_accident: int
    avg_age_years: float
    avg_energy: float
    avg_hydration: float
    avg_heart: float
    avg_wisdom: float
    avg_empathy: float
    plant_biomass: float
    materials: float


class Environment:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.agents: List[Agent] = []
        self.next_id = 1
        self.width = config.GRID_WIDTH
        self.height = config.GRID_HEIGHT
        self.tiles, self.land_mask = self._generate_terrain()
        self.height_map = self._generate_height_map()
        self.zone_map = self._generate_zones()
        self.food_map, self.water_map, self.material_map = self._init_resources()
        self.food = sum(sum(row) for row in self.food_map)
        self.water = sum(sum(row) for row in self.water_map)
        self.plant_biomass = self.food
        self.materials = sum(sum(row) for row in self.material_map)
        self.step_count = 0
        self.last_ration_day = -1
        self.resource_coords = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self.tiles[y][x] == config.TERRAIN_RESOURCE
        ]
        self.housing_coords = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self.tiles[y][x] == config.TERRAIN_HOUSING
        ]
        self.best_food_pos = None
        self.best_water_pos = None

    def seed_initial_population(self, count: int) -> None:
        for _ in range(count):
            self.agents.append(self._create_random_agent())

    def _generate_terrain(self) -> Tuple[List[List[int]], List[List[bool]]]:
        tiles = [[config.TERRAIN_VOID for _ in range(self.width)] for _ in range(self.height)]
        land_mask = [[False for _ in range(self.width)] for _ in range(self.height)]

        for floor in range(config.FLOOR_COUNT):
            y_start = floor * config.FLOOR_HEIGHT
            y_end = y_start + config.FLOOR_HEIGHT
            for y in range(y_start, y_end):
                for x in range(self.width):
                    tiles[y][x] = config.TERRAIN_PLATFORM
                    land_mask[y][x] = True

                for x in range(config.HOUSING_DEPTH):
                    tiles[y][x] = config.TERRAIN_HOUSING
                for x in range(self.width - config.HOUSING_DEPTH, self.width):
                    tiles[y][x] = config.TERRAIN_HOUSING

        # Food & water in the middle of each floor
        cx = self.width // 2
        for floor in range(config.FLOOR_COUNT):
            start = floor * config.FLOOR_HEIGHT
            cy = start + config.FLOOR_HEIGHT // 2
            for y in range(
                max(start, cy - config.RESOURCE_PATCH_RADIUS),
                min(start + config.FLOOR_HEIGHT, cy + config.RESOURCE_PATCH_RADIUS + 1),
            ):
                for x in range(
                    max(0, cx - config.RESOURCE_PATCH_RADIUS),
                    min(self.width, cx + config.RESOURCE_PATCH_RADIUS + 1),
                ):
                    tiles[y][x] = config.TERRAIN_RESOURCE

        return tiles, land_mask

    def _generate_height_map(self) -> List[List[float]]:
        heights = [[0.0 for _ in range(self.width)] for _ in range(self.height)]
        for y in range(self.height):
            floor = y // config.FLOOR_HEIGHT
            base = floor * config.PLATFORM_Z_STEP
            for x in range(self.width):
                if self.tiles[y][x] == config.TERRAIN_VOID:
                    heights[y][x] = 0.0
                else:
                    heights[y][x] = base
        return heights

    def _generate_zones(self) -> List[List[int]]:
        zones = [[config.ZONE_VOID for _ in range(self.width)] for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                tile = self.tiles[y][x]
                if tile == config.TERRAIN_HOUSING:
                    zones[y][x] = config.ZONE_HOUSING
                elif tile == config.TERRAIN_RESOURCE:
                    zones[y][x] = config.ZONE_RESOURCE
                elif tile == config.TERRAIN_PLATFORM:
                    zones[y][x] = config.ZONE_PLATFORM
                else:
                    zones[y][x] = config.ZONE_VOID
        return zones

    def _adjacent_to_water(self, x: int, y: int) -> bool:
        for nx, ny in self._neighbor_positions(x, y):
            if self.tiles[ny][nx] == config.TERRAIN_WATER:
                return True
        return False

    def _init_resources(self) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        food = [[0.0 for _ in range(self.width)] for _ in range(self.height)]
        water = [[0.0 for _ in range(self.width)] for _ in range(self.height)]
        materials = [[0.0 for _ in range(self.width)] for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                if self.tiles[y][x] == config.TERRAIN_RESOURCE:
                    food[y][x] = config.TILE_FOOD_MAX
                    water[y][x] = config.TILE_WATER_MAX
        return food, water, materials

    def _random_land_position(self) -> Tuple[int, int]:
        for _ in range(1000):
            x = self.rng.randrange(self.width)
            y = self.rng.randrange(self.height)
            if self.land_mask[y][x] and self.tiles[y][x] != config.TERRAIN_VOID:
                return x, y
        for y in range(self.height):
            for x in range(self.width):
                if self.land_mask[y][x]:
                    return x, y
        return 0, 0

    def _find_nearby_land(self, x: int, y: int, radius: int) -> Tuple[int, int] | None:
        for r in range(1, radius + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx = x + dx
                    ny = y + dy
                    if nx < 0 or ny < 0 or nx >= self.width or ny >= self.height:
                        continue
                    if self.land_mask[ny][nx] and self.tiles[ny][nx] != config.TERRAIN_VOID:
                        return nx, ny
        return None

    def _spawn_position(self, parent_a: Agent | None, parent_b: Agent | None) -> Tuple[int, int]:
        base = parent_a or parent_b
        if base is not None:
            found = self._find_nearby_land(base.x, base.y, radius=2)
            if found is not None:
                return found
        return self._random_land_position()

    def _set_agent_position(self, agent: Agent, x: int, y: int) -> None:
        agent.x = x
        agent.y = y
        agent.z = self.height_map[y][x]

    def _create_random_agent(self) -> Agent:
        x, y = self._random_land_position()
        agent = Agent(
            id=self.next_id,
            sex=config.SEX_A if self.rng.random() < 0.5 else config.SEX_B,
            age=self.rng.randint(0, config.MATURITY_AGE_STEPS),
            parent_a_id=None,
            parent_b_id=None,
            generation=0,
            x=x,
            y=y,
            z=self.height_map[y][x],
            shape_base=sample_trait(self.rng),
            color_base=sample_trait(self.rng),
            size_base=sample_trait(self.rng),
            curiosity_base=sample_trait(self.rng),
            sociality_base=sample_trait(self.rng),
            risk_base=sample_trait(self.rng),
            aggression_base=sample_trait(self.rng),
            creative_bias_base=sample_trait(self.rng),
            extraversion_base=sample_trait(self.rng),
            openness_base=sample_trait(self.rng),
            conscientiousness_base=sample_trait(self.rng),
            agreeableness_base=sample_trait(self.rng),
            neuroticism_base=sample_trait(self.rng),
        )
        self.next_id += 1
        agent.memory["parents"] = None
        agent.memory["generation"] = 0
        return agent

    def step(self) -> StepStats:
        self.step_count += 1
        births: List[BirthRequest] = []
        deaths = 0
        accidental_deaths = 0
        deaths_starvation = 0
        deaths_dehydration = 0
        deaths_heart = 0
        deaths_age = 0
        deaths_accident = 0
        mating_a: List[Agent] = []
        mating_b: List[Agent] = []

        is_day, day_index, year_index, step_in_cycle = self._time_state()
        crowding = self._crowding_factor()
        if is_day and step_in_cycle == 0 and day_index != self.last_ration_day:
            self._apply_daily_rations(day_index)
        self._replenish_resources(is_day)
        self._update_best_resources()
        self._update_scarcity_state()

        # Metabolism and resource consumption
        for agent in self.agents:
            if not agent.alive:
                continue
            agent.apply_decay()
            action = self._choose_action(agent)
            self._maybe_move(agent, action)
            if action == "eat":
                self._feed(agent, eat=True, drink=False)
            elif action == "drink":
                self._feed(agent, eat=False, drink=True)
            elif action == "mate":
                if agent.sex == config.SEX_A:
                    mating_a.append(agent)
                else:
                    mating_b.append(agent)
            elif action == "self_realization":
                self._self_realization(agent)
            elif action == "idea":
                self._generate_idea(agent)
            self._apply_heart(agent, crowding)
            agent.step_age()
            agent.update_age_effects()
            self._apply_trait_drift(agent)
            self._decay_bonds(agent)

        self._social_interactions()

        # Reproduction
        births.extend(self._reproduction_phase(mating_a, mating_b, crowding))

        # Resolve deaths
        for agent in self.agents:
            if not agent.alive:
                continue
            if agent.energy <= 0.0:
                agent.alive = False
                deaths += 1
                deaths_starvation += 1
                continue
            if agent.hydration <= 0.0:
                agent.alive = False
                deaths += 1
                deaths_dehydration += 1
                continue
            if agent.heart <= 0.0:
                agent.alive = False
                deaths += 1
                deaths_heart += 1
                continue
            if agent.age > config.MAX_AGE_STEPS:
                agent.alive = False
                deaths += 1
                deaths_age += 1
                continue
            if self._age_mortality(agent):
                agent.alive = False
                deaths += 1
                deaths_age += 1
                continue
            if self._accident_mortality(agent):
                agent.alive = False
                deaths += 1
                accidental_deaths += 1
                deaths_accident += 1
                continue

        # Add newborns
        for req in births:
            parent_a = self._agent_by_id(req.parent_a_id)
            parent_b = self._agent_by_id(req.parent_b_id)
            parent_gen = 0
            if parent_a is not None:
                parent_gen = max(parent_gen, parent_a.generation)
            if parent_b is not None:
                parent_gen = max(parent_gen, parent_b.generation)
            x, y = self._spawn_position(parent_a, parent_b)
            self.agents.append(
                Agent(
                    id=self.next_id,
                    sex=config.SEX_A if self.rng.random() < 0.5 else config.SEX_B,
                    age=0,
                    parent_a_id=req.parent_a_id,
                    parent_b_id=req.parent_b_id,
                    generation=parent_gen + 1,
                    x=x,
                    y=y,
                    z=self.height_map[y][x],
                    shape_base=req.shape_base,
                    color_base=req.color_base,
                    size_base=req.size_base,
                    curiosity_base=req.curiosity_base,
                    sociality_base=req.sociality_base,
                    risk_base=req.risk_base,
                    aggression_base=req.aggression_base,
                    creative_bias_base=req.creative_bias_base,
                    extraversion_base=req.extraversion_base,
                    openness_base=req.openness_base,
                    conscientiousness_base=req.conscientiousness_base,
                    agreeableness_base=req.agreeableness_base,
                    neuroticism_base=req.neuroticism_base,
                )
            )
            self.agents[-1].memory["parents"] = (req.parent_a_id, req.parent_b_id)
            self.agents[-1].memory["generation"] = parent_gen + 1
            self.next_id += 1

        # Compact list
        self.agents = [a for a in self.agents if a.alive]

        return self._collect_stats(
            births=len(births),
            deaths=deaths,
            accidental_deaths=accidental_deaths,
            deaths_starvation=deaths_starvation,
            deaths_dehydration=deaths_dehydration,
            deaths_heart=deaths_heart,
            deaths_age=deaths_age,
            deaths_accident=deaths_accident,
            is_day=is_day,
            day_index=day_index,
            year_index=year_index,
        )

    def _replenish_resources(self, is_day: bool) -> None:
        total_food = 0.0
        total_water = 0.0
        total_materials = 0.0

        for y in range(self.height):
            for x in range(self.width):
                if self.tiles[y][x] == config.TERRAIN_RESOURCE:
                    if self.food_map[y][x] < config.TILE_FOOD_MAX:
                        self.food_map[y][x] = min(
                            config.TILE_FOOD_MAX, self.food_map[y][x] + config.SOURCE_FOOD_PER_STEP
                        )
                    if self.water_map[y][x] < config.TILE_WATER_MAX:
                        self.water_map[y][x] = min(
                            config.TILE_WATER_MAX, self.water_map[y][x] + config.SOURCE_WATER_PER_STEP
                        )

                total_food += self.food_map[y][x]
                total_water += self.water_map[y][x]
                total_materials += self.material_map[y][x]

        self.food = total_food
        self.water = total_water
        self.plant_biomass = total_food
        self.materials = total_materials

    def _update_best_resources(self) -> None:
        best_food = None
        best_food_val = -1.0
        best_water = None
        best_water_val = -1.0
        for x, y in self.resource_coords:
            food_val = self.food_map[y][x]
            if food_val > best_food_val:
                best_food_val = food_val
                best_food = (x, y)
            water_val = self.water_map[y][x]
            if water_val > best_water_val:
                best_water_val = water_val
                best_water = (x, y)
        self.best_food_pos = best_food
        self.best_water_pos = best_water

    def _feed(self, agent: Agent, eat: bool, drink: bool) -> None:
        x = agent.x
        y = agent.y
        if eat and self.food_map[y][x] > 0:
            take = min(self.food_map[y][x], 1.0)
            self.food_map[y][x] -= take
            self.food = max(0.0, self.food - take)
            agent.energy = min(1.0, agent.energy + take * config.FOOD_TO_ENERGY)
            if self.tiles[y][x] == config.TERRAIN_RESOURCE and (x, y) not in agent.known_food:
                agent.known_food.append((x, y))
        if drink and self.water_map[y][x] > 0:
            take = min(self.water_map[y][x], 1.0)
            self.water_map[y][x] -= take
            self.water = max(0.0, self.water - take)
            agent.hydration = min(1.0, agent.hydration + take * config.WATER_TO_HYDRATION)
            if self.tiles[y][x] == config.TERRAIN_RESOURCE and (x, y) not in agent.known_water:
                agent.known_water.append((x, y))
            if agent.carry_water < config.WATER_CARRY_MAX and self.water_map[y][x] > 0:
                extra = min(
                    self.water_map[y][x],
                    config.WATER_CARRY_FILL_PER_STEP,
                    config.WATER_CARRY_MAX - agent.carry_water,
                )
                if extra > 0:
                    self.water_map[y][x] -= extra
                    self.water = max(0.0, self.water - extra)
                    agent.carry_water += extra

    def _maybe_move(self, agent: Agent, action: str) -> None:
        move_prob = (
            config.MOVE_BASE_PROB
            + agent.curiosity * config.MOVE_CURIOSITY_WEIGHT
            + agent.extraversion * config.MOVE_EXTRAVERSION_WEIGHT
            + agent.openness * config.MOVE_OPENNESS_WEIGHT
        )
        if action == "eat":
            threshold = config.RESOURCE_SEEK_THRESHOLD + agent.conscientiousness * config.RESOURCE_SEEK_CONSCIENTIOUSNESS_BOOST
            if self.food_map[agent.y][agent.x] < threshold:
                self._move_toward_resource(agent, "food")
            elif self.rng.random() < move_prob * 0.1:
                self._random_move(agent)
            return
        if action == "drink":
            threshold = config.RESOURCE_SEEK_THRESHOLD + agent.conscientiousness * config.RESOURCE_SEEK_CONSCIENTIOUSNESS_BOOST
            if self.water_map[agent.y][agent.x] < threshold:
                self._move_toward_resource(agent, "water")
            elif self.rng.random() < move_prob * 0.1:
                self._random_move(agent)
            return
        if action == "mate":
            target = self._nearest_agent(agent, lambda other: other.sex != agent.sex and other.is_fertile())
            if target is not None:
                step = self._next_step_toward(agent.x, agent.y, target[0], target[1])
                if step is not None:
                    self._set_agent_position(agent, step[0], step[1])
                    return
            if self.rng.random() < move_prob:
                self._random_move(agent)
            return
        if action == "idea":
            scarcity = agent.memory.get("scarcity")
            idea = agent.memory.get("idea")
            task = agent.memory.get("task")
            if scarcity == "water":
                if task == "trade_water":
                    target = self._nearest_agent(
                        agent, lambda other: other.id != agent.id and other.hydration < config.WATER_SHARE_THRESHOLD
                    )
                    if target is not None:
                        step = self._next_step_toward(agent.x, agent.y, target[0], target[1])
                        if step is not None:
                            self._set_agent_position(agent, step[0], step[1])
                            return
                if task == "transport_water":
                    if agent.carry_water < config.WATER_CARRY_MAX:
                        target = self.best_water_pos
                    else:
                        target = self._nearest_agent(
                            agent, lambda other: other.id != agent.id and other.hydration < config.WATER_SHARE_THRESHOLD
                        )
                    if target is not None:
                        step = self._next_step_toward(agent.x, agent.y, target[0], target[1])
                        if step is not None:
                            self._set_agent_position(agent, step[0], step[1])
                            return
                if task == "engineer_water" and self.housing_coords:
                    target = self._nearest_tile(agent, self.housing_coords)
                    if target is not None:
                        step = self._next_step_toward(agent.x, agent.y, target[0], target[1])
                        if step is not None:
                            self._set_agent_position(agent, step[0], step[1])
                            return
                    self._maybe_engineer_source(agent, "water")
                if task == "memory_water" and agent.known_water:
                    target = self._nearest_tile(agent, agent.known_water)
                    if target is not None:
                        step = self._next_step_toward(agent.x, agent.y, target[0], target[1])
                        if step is not None:
                            self._set_agent_position(agent, step[0], step[1])
                            return
                target = self.best_water_pos
                if target is not None:
                    step = self._next_step_toward(agent.x, agent.y, target[0], target[1])
                    if step is not None:
                        self._set_agent_position(agent, step[0], step[1])
                        return
            if scarcity == "food":
                if task == "engineer_food" and self.housing_coords:
                    target = self._nearest_tile(agent, self.housing_coords)
                    if target is not None:
                        step = self._next_step_toward(agent.x, agent.y, target[0], target[1])
                        if step is not None:
                            self._set_agent_position(agent, step[0], step[1])
                            return
                    self._maybe_engineer_source(agent, "food")
                if task == "memory_food" and agent.known_food:
                    target = self._nearest_tile(agent, agent.known_food)
                    if target is not None:
                        step = self._next_step_toward(agent.x, agent.y, target[0], target[1])
                        if step is not None:
                            self._set_agent_position(agent, step[0], step[1])
                            return
                target = self.best_food_pos
                if target is not None:
                    step = self._next_step_toward(agent.x, agent.y, target[0], target[1])
                    if step is not None:
                        self._set_agent_position(agent, step[0], step[1])
                        return
            if self.rng.random() < move_prob:
                self._random_move(agent)
            return
        if action == "explore":
            target = agent.memory.get("explore_target")
            if not isinstance(target, tuple) or len(target) != 2:
                target = self._random_land_position()
                agent.memory["explore_target"] = target
            if target[0] == agent.x and target[1] == agent.y:
                agent.memory.pop("explore_target", None)
                return
            step = self._next_step_toward(agent.x, agent.y, target[0], target[1])
            if step is not None:
                self._set_agent_position(agent, step[0], step[1])
                return
            if self.rng.random() < move_prob:
                self._random_move(agent)
            return
        if action == "self_realization":
            if agent.creative_bias >= 0.5 and self.housing_coords:
                target = self._nearest_tile(agent, self.housing_coords)
            else:
                target = self._nearest_agent(agent, lambda other: other.id != agent.id)
            if target is not None:
                step = self._next_step_toward(agent.x, agent.y, target[0], target[1])
                if step is not None:
                    self._set_agent_position(agent, step[0], step[1])
                    return
        if self.rng.random() < move_prob:
            self._random_move(agent)

    def _move_toward_resource(self, agent: Agent, resource: str) -> None:
        if config.GLOBAL_RESOURCE_SEEK:
            target = self.best_food_pos if resource == "food" else self.best_water_pos
            if target is None:
                self._random_move(agent)
                return
            step = self._next_step_toward(agent.x, agent.y, target[0], target[1])
            if step is not None:
                self._set_agent_position(agent, step[0], step[1])
                return
            self._random_move(agent)
            return

        best = None
        best_value = -1.0
        for nx, ny in self._neighbor_positions(agent.x, agent.y):
            if not self.land_mask[ny][nx]:
                continue
            value = self.food_map[ny][nx] if resource == "food" else self.water_map[ny][nx]
            if value > best_value:
                best_value = value
                best = (nx, ny)
        if best is not None and best_value > 0.0:
            self._set_agent_position(agent, best[0], best[1])
        else:
            self._random_move(agent)

    def _next_step_toward(self, sx: int, sy: int, tx: int, ty: int) -> tuple[int, int] | None:
        if sx == tx and sy == ty:
            return None
        if not (0 <= tx < self.width and 0 <= ty < self.height):
            return None
        if not self.land_mask[ty][tx]:
            return None

        greedy = self._greedy_step_toward(sx, sy, tx, ty)
        if greedy is not None:
            return greedy

        from collections import deque

        start = (sx, sy)
        goal = (tx, ty)
        queue = deque([start])
        came_from = {start: None}
        visited = 0

        while queue:
            x, y = queue.popleft()
            if (x, y) == goal:
                break
            for nx, ny in self._neighbor_positions(x, y):
                if not self.land_mask[ny][nx]:
                    continue
                if (nx, ny) in came_from:
                    continue
                came_from[(nx, ny)] = (x, y)
                queue.append((nx, ny))
                visited += 1
                if visited >= config.PATHFINDING_MAX_NODES:
                    queue.clear()
                    break

        if goal not in came_from:
            return None

        current = goal
        while came_from[current] != start:
            current = came_from[current]
            if current is None:
                return None
        return current

    def _greedy_step_toward(self, sx: int, sy: int, tx: int, ty: int) -> tuple[int, int] | None:
        dx = 0 if tx == sx else (1 if tx > sx else -1)
        dy = 0 if ty == sy else (1 if ty > sy else -1)
        candidates = []
        if abs(tx - sx) >= abs(ty - sy):
            candidates.append((sx + dx, sy))
            candidates.append((sx, sy + dy))
        else:
            candidates.append((sx, sy + dy))
            candidates.append((sx + dx, sy))
        for nx, ny in candidates:
            if 0 <= nx < self.width and 0 <= ny < self.height and self.land_mask[ny][nx]:
                return (nx, ny)
        return None

    def _nearest_agent(self, agent: Agent, predicate) -> tuple[int, int] | None:
        best = None
        best_dist = None
        for other in self.agents:
            if not other.alive:
                continue
            if not predicate(other):
                continue
            dist = abs(other.x - agent.x) + abs(other.y - agent.y)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = (other.x, other.y)
        return best

    def _nearest_tile(self, agent: Agent, coords: list[tuple[int, int]]) -> tuple[int, int] | None:
        best = None
        best_dist = None
        for x, y in coords:
            dist = abs(x - agent.x) + abs(y - agent.y)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = (x, y)
        return best

    def _random_move(self, agent: Agent) -> None:
        neighbors = [
            pos
            for pos in self._neighbor_positions(agent.x, agent.y)
            if self.land_mask[pos[1]][pos[0]]
        ]
        if not neighbors:
            return
        nx, ny = self.rng.choice(neighbors)
        self._set_agent_position(agent, nx, ny)

    def _neighbor_positions(self, x: int, y: int) -> List[Tuple[int, int]]:
        positions: List[Tuple[int, int]] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = x + dx
                ny = y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    positions.append((nx, ny))
        return positions

    def _time_state(self) -> Tuple[bool, int, int, int]:
        cycle_steps = max(1, config.CYCLE_STEPS)
        step_in_cycle = (self.step_count - 1) % cycle_steps
        is_day = step_in_cycle < config.DAY_STEPS
        day_index = (self.step_count - 1) // cycle_steps
        year_index = int(day_index // config.YEAR_DAYS)
        return is_day, day_index, year_index, step_in_cycle

    def _apply_daily_rations(self, day_index: int) -> None:
        if not self.resource_coords:
            return
        alive = sum(1 for agent in self.agents if agent.alive)
        if alive <= 0:
            self.last_ration_day = day_index
            return

        steps_per_day = max(1, config.CYCLE_STEPS)
        food_per_agent = 0.0
        water_per_agent = 0.0
        if config.FOOD_TO_ENERGY > 0:
            food_per_agent = (config.ENERGY_DECAY * steps_per_day) / config.FOOD_TO_ENERGY
        if config.WATER_TO_HYDRATION > 0:
            water_per_agent = (config.HYDRATION_DECAY * steps_per_day) / config.WATER_TO_HYDRATION

        food_total = food_per_agent * alive * 1.10
        water_total = water_per_agent * alive * 1.10
        piles = max(1, config.RATION_PILES_PER_DAY)
        weights = [self.rng.random() for _ in range(piles)]
        weight_sum = sum(weights) or 1.0
        weights = [w / weight_sum for w in weights]

        if piles <= len(self.resource_coords):
            coords = self.rng.sample(self.resource_coords, piles)
        else:
            coords = [self.rng.choice(self.resource_coords) for _ in range(piles)]

        for (x, y), w in zip(coords, weights):
            self.food_map[y][x] += food_total * w
            self.water_map[y][x] += water_total * w

        self.last_ration_day = day_index

    def _crowding_factor(self) -> float:
        return max(0.0, len(self.agents) - config.ENV_CAPACITY) / max(1.0, config.ENV_CAPACITY)

    def _apply_heart(self, agent: Agent, crowding: float) -> None:
        if agent.energy > 0.8 and agent.hydration > 0.8 and crowding < 0.2:
            agent.heart = min(1.0, agent.heart + config.HEART_RECOVERY)
            return
        decay = config.HEART_DECAY_BASE
        if agent.energy < config.HUNGER_THRESHOLD:
            decay += config.HEART_DECAY_HUNGER
        if agent.hydration < config.THIRST_THRESHOLD:
            decay += config.HEART_DECAY_THIRST
        decay += crowding * config.HEART_DECAY_CROWDING
        agent.heart = max(0.0, agent.heart - decay)

    def _apply_trait_drift(self, agent: Agent) -> None:
        if config.TRAIT_DRIFT_STD <= 0:
            return
        agent.curiosity = clamp01(agent.curiosity + self.rng.gauss(0.0, config.TRAIT_DRIFT_STD))
        agent.sociality = clamp01(agent.sociality + self.rng.gauss(0.0, config.TRAIT_DRIFT_STD))
        agent.risk_tolerance = clamp01(agent.risk_tolerance + self.rng.gauss(0.0, config.TRAIT_DRIFT_STD))
        agent.aggression = clamp01(agent.aggression + self.rng.gauss(0.0, config.TRAIT_DRIFT_STD))
        agent.creative_bias = clamp01(agent.creative_bias + self.rng.gauss(0.0, config.TRAIT_DRIFT_STD))
        agent.extraversion = clamp01(agent.extraversion + self.rng.gauss(0.0, config.TRAIT_DRIFT_STD))
        agent.openness = clamp01(agent.openness + self.rng.gauss(0.0, config.TRAIT_DRIFT_STD))
        agent.conscientiousness = clamp01(agent.conscientiousness + self.rng.gauss(0.0, config.TRAIT_DRIFT_STD))
        agent.agreeableness = clamp01(agent.agreeableness + self.rng.gauss(0.0, config.TRAIT_DRIFT_STD))
        agent.neuroticism = clamp01(agent.neuroticism + self.rng.gauss(0.0, config.TRAIT_DRIFT_STD))

    def _decay_bonds(self, agent: Agent) -> None:
        if not agent.bonds:
            return
        to_delete = []
        for other_id, strength in agent.bonds.items():
            strength *= 1.0 - config.BOND_DECAY
            if strength < config.BOND_MIN:
                to_delete.append(other_id)
            else:
                agent.bonds[other_id] = strength
        for other_id in to_delete:
            del agent.bonds[other_id]

    def _self_realization(self, agent: Agent) -> None:
        if agent.energy < config.SELF_REALIZATION_ENERGY_MIN:
            return
        agent.energy = max(0.0, agent.energy - config.SELF_REALIZATION_ENERGY_COST)
        if self.rng.random() < agent.creative_bias:
            agent.wisdom = clamp01(agent.wisdom + config.CREATIVE_REALIZATION_WISDOM_GAIN)
            if self.rng.random() < 0.1:
                agent.notes.append("creative_insight")
            return

        candidates = [
            other
            for other in self.agents
            if other.alive and other.id != agent.id and abs(other.x - agent.x) <= 1 and abs(other.y - agent.y) <= 1
        ]
        if not candidates:
            return
        partner = self.rng.choice(candidates)
        agent.empathy = clamp01(agent.empathy + config.SOCIAL_REALIZATION_EMPATHY_GAIN)
        partner.empathy = clamp01(partner.empathy + config.SOCIAL_REALIZATION_EMPATHY_GAIN)
        agent.bonds[partner.id] = min(
            1.0, agent.bonds.get(partner.id, 0.0) + config.SOCIAL_REALIZATION_BOND_GAIN
        )
        partner.bonds[agent.id] = min(
            1.0, partner.bonds.get(agent.id, 0.0) + config.SOCIAL_REALIZATION_BOND_GAIN
        )
        if agent.carry_water > 0 and partner.hydration < config.WATER_SHARE_THRESHOLD:
            give = min(
                agent.carry_water,
                config.WATER_SHARE_AMOUNT,
                max(0.0, 1.0 - partner.hydration),
            )
            if give > 0:
                agent.carry_water -= give
                partner.hydration = min(1.0, partner.hydration + give)
        if partner.carry_water > 0 and agent.hydration < config.WATER_SHARE_THRESHOLD:
            give = min(
                partner.carry_water,
                config.WATER_SHARE_AMOUNT,
                max(0.0, 1.0 - agent.hydration),
            )
            if give > 0:
                partner.carry_water -= give
                agent.hydration = min(1.0, agent.hydration + give)

    def _maybe_engineer_source(self, agent: Agent, kind: str) -> None:
        if self.tiles[agent.y][agent.x] == config.TERRAIN_RESOURCE:
            return
        chance = 0.05 + 0.2 * agent.creative_bias + 0.2 * agent.openness + 0.1 * agent.wisdom
        if self.rng.random() > chance:
            return
        self.tiles[agent.y][agent.x] = config.TERRAIN_RESOURCE
        if (agent.x, agent.y) not in self.resource_coords:
            self.resource_coords.append((agent.x, agent.y))
        if kind == "water":
            self.water_map[agent.y][agent.x] += config.ENGINEER_SOURCE_WATER
            self.water += config.ENGINEER_SOURCE_WATER
        else:
            self.food_map[agent.y][agent.x] += config.ENGINEER_SOURCE_FOOD
            self.food += config.ENGINEER_SOURCE_FOOD

    def _social_interactions(self) -> None:
        alive = [a for a in self.agents if a.alive]
        if len(alive) < 2:
            return
        for agent in alive:
            prob = (
                config.SOCIAL_INTERACTION_BASE
                + agent.sociality * config.SOCIAL_INTERACTION_SOCIALITY_WEIGHT
                + agent.extraversion * config.SOCIAL_INTERACTION_EXTRAVERSION_WEIGHT
            )
            if self.rng.random() > prob:
                continue
            other = self.rng.choice(alive)
            if other.id == agent.id:
                continue
            agree = 0.5 + 0.5 * min(agent.agreeableness, other.agreeableness)
            boost = config.BOND_INCREASE * (0.5 + 0.5 * min(agent.empathy, other.empathy))
            boost += agree * config.BOND_AGREEABLENESS_WEIGHT
            agent.bonds[other.id] = clamp01(agent.bonds.get(other.id, 0.0) + boost)
            other.bonds[agent.id] = clamp01(other.bonds.get(agent.id, 0.0) + boost * 0.8)
            if agent.known_water:
                for coord in agent.known_water:
                    if coord not in other.known_water:
                        other.known_water.append(coord)
            if other.known_water:
                for coord in other.known_water:
                    if coord not in agent.known_water:
                        agent.known_water.append(coord)
            if agent.known_food:
                for coord in agent.known_food:
                    if coord not in other.known_food:
                        other.known_food.append(coord)
            if other.known_food:
                for coord in other.known_food:
                    if coord not in agent.known_food:
                        agent.known_food.append(coord)

            if agent.memory.get("task") and not other.memory.get("task"):
                if self.rng.random() < 0.5 * (agent.agreeableness + agent.extraversion):
                    other.memory["task"] = agent.memory.get("task")
                    other.memory["task_target"] = agent.memory.get("task_target")
            if other.memory.get("task") and not agent.memory.get("task"):
                if self.rng.random() < 0.5 * (other.agreeableness + other.extraversion):
                    agent.memory["task"] = other.memory.get("task")
                    agent.memory["task_target"] = other.memory.get("task_target")

            if agent.carry_water > 0 and other.hydration < config.WATER_SHARE_THRESHOLD:
                give = min(
                    agent.carry_water,
                    config.WATER_SHARE_AMOUNT,
                    max(0.0, 1.0 - other.hydration),
                )
                if give > 0:
                    agent.carry_water -= give
                    other.hydration = min(1.0, other.hydration + give)
                    agent.memory["favors"] = agent.memory.get("favors", {})
                    other.memory["favors"] = other.memory.get("favors", {})
                    other.memory["favors"][agent.id] = other.memory["favors"].get(agent.id, 0) + 1
            if other.carry_water > 0 and agent.hydration < config.WATER_SHARE_THRESHOLD:
                give = min(
                    other.carry_water,
                    config.WATER_SHARE_AMOUNT,
                    max(0.0, 1.0 - agent.hydration),
                )
                if give > 0:
                    other.carry_water -= give
                    agent.hydration = min(1.0, agent.hydration + give)
                    other.memory["favors"] = other.memory.get("favors", {})
                    agent.memory["favors"] = agent.memory.get("favors", {})
                    agent.memory["favors"][other.id] = agent.memory["favors"].get(other.id, 0) + 1

    def _age_mortality(self, agent: Agent) -> bool:
        if agent.age < config.AGE_MORTALITY_START_STEPS:
            return False
        base_rate = config.AGE_MORTALITY_RATE
        if agent.age >= config.MEDIAN_AGE_STEPS:
            span = max(1, config.MAX_AGE_STEPS - config.MEDIAN_AGE_STEPS)
            factor = min(1.0, (agent.age - config.MEDIAN_AGE_STEPS) / span)
            base_rate *= 1.0 + factor * config.AGE_MORTALITY_ACCEL
        return self.rng.random() < base_rate

    def _accident_mortality(self, agent: Agent) -> bool:
        base = config.ACCIDENT_BASE_RATE
        risk = (agent.risk_tolerance - 0.5) * config.ACCIDENT_RISK_WEIGHT
        aggression = (agent.aggression - 0.5) * config.ACCIDENT_AGGRESSION_WEIGHT
        wisdom = (agent.wisdom - 0.5) * config.ACCIDENT_WISDOM_PROTECT
        conscientious = (agent.conscientiousness - 0.5) * config.ACCIDENT_CONSCIENTIOUSNESS_PROTECT
        neurotic = (agent.neuroticism - 0.5) * config.ACCIDENT_NEUROTICISM_RISK
        rate = base * max(0.0, 1.0 + risk + aggression - wisdom - conscientious + neurotic)
        return self.rng.random() < rate

    def _agent_by_id(self, agent_id: int | None) -> Agent | None:
        if agent_id is None:
            return None
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None

    def _choose_action(self, agent: Agent) -> str:
        hunger_threshold = config.HUNGER_THRESHOLD + (agent.neuroticism - 0.5) * config.NEED_THRESHOLD_NEUROTICISM_BIAS
        thirst_threshold = config.THIRST_THRESHOLD + (agent.neuroticism - 0.5) * config.NEED_THRESHOLD_NEUROTICISM_BIAS
        if agent.energy < hunger_threshold:
            return "eat"
        if agent.hydration < thirst_threshold:
            return "drink"
        scarcity = agent.memory.get("scarcity")
        if scarcity and (self.step_count - agent.last_idea_step) >= config.IDEA_COOLDOWN_STEPS:
            if self.rng.random() < config.IDEA_PROB:
                return "idea"
        survival = agent.goals.get("survival", 0.5)
        social = agent.goals.get("social", 0.2)
        creative = agent.goals.get("creative", 0.2)
        reproduction = agent.goals.get("reproduction", 0.2)
        exploration = agent.goals.get("exploration", 0.2)

        eat_score = survival * 0.1
        drink_score = survival * 0.1

        mate_score = 0.0
        if (
            agent.is_fertile()
            and agent.energy >= config.MATE_ENERGY_THRESHOLD
            and agent.hydration >= config.MATE_HYDRATION_THRESHOLD
        ):
            mate_score = reproduction * ((agent.energy + agent.hydration) * 0.5)

        self_score = 0.0
        if agent.energy >= config.SELF_REALIZATION_ENERGY_MIN:
            energy_factor = (agent.energy - config.SELF_REALIZATION_ENERGY_MIN) / max(
                0.0001, 1.0 - config.SELF_REALIZATION_ENERGY_MIN
            )
            creative_score = creative * agent.creative_bias
            social_score = social * (1.0 - agent.creative_bias)
            self_score = max(creative_score, social_score) * energy_factor

        explore_score = exploration * (0.5 + 0.5 * agent.curiosity)

        scores = {
            "eat": eat_score,
            "drink": drink_score,
            "mate": mate_score,
            "self_realization": self_score,
            "explore": explore_score,
        }
        for key in scores:
            scores[key] *= 0.9 + 0.2 * self.rng.random()

        best_action = max(scores.items(), key=lambda kv: kv[1])[0]
        return best_action

    def _update_scarcity_state(self) -> None:
        alive = [a for a in self.agents if a.alive]
        if not alive:
            return
        water_per_agent = self.water / max(1, len(alive))
        food_per_agent = self.food / max(1, len(alive))
        scarcity_water = water_per_agent < config.SCARCITY_WATER_PER_AGENT
        scarcity_food = food_per_agent < config.SCARCITY_FOOD_PER_AGENT
        for agent in alive:
            if scarcity_water and agent.hydration < config.THIRST_THRESHOLD:
                agent.memory["scarcity"] = "water"
            elif scarcity_food and agent.energy < config.HUNGER_THRESHOLD:
                agent.memory["scarcity"] = "food"
            else:
                agent.memory.pop("scarcity", None)

    def _generate_idea(self, agent: Agent) -> None:
        scarcity = agent.memory.get("scarcity")
        if not scarcity:
            return
        weights = []
        ideas = []
        ideas.append("engineer")
        weights.append(config.IDEA_ENGINEER_PROB * (0.5 + 0.5 * (agent.openness + agent.creative_bias)))
        ideas.append("transport")
        weights.append(config.IDEA_TRANSPORT_PROB * (0.5 + 0.5 * agent.conscientiousness))
        ideas.append("coordinate")
        weights.append(config.IDEA_COORDINATE_PROB * (0.5 + 0.5 * agent.extraversion))
        ideas.append("memory")
        weights.append(config.IDEA_MEMORY_PROB * (0.5 + 0.5 * agent.curiosity))
        ideas.append("trade")
        weights.append(config.IDEA_TRADE_PROB * (0.5 + 0.5 * agent.agreeableness))

        idea = self.rng.choices(ideas, weights=weights, k=1)[0]
        agent.memory["idea"] = idea
        agent.notes.append(f"idea:{idea}:{scarcity}")

        if idea == "coordinate":
            agent.memory["task"] = f"coordinate_{scarcity}"
        elif idea == "transport":
            agent.memory["task"] = f"transport_{scarcity}"
        elif idea == "engineer":
            agent.memory["task"] = f"engineer_{scarcity}"
        elif idea == "memory":
            agent.memory["task"] = f"memory_{scarcity}"
        elif idea == "trade":
            agent.memory["task"] = f"trade_{scarcity}"

        agent.last_idea_step = self.step_count

    def _reproduction_phase(
        self,
        sex_a: List[Agent],
        sex_b: List[Agent],
        crowding: float,
    ) -> List[BirthRequest]:
        births: List[BirthRequest] = []
        sex_a = [a for a in sex_a if a.alive and a.is_fertile()]
        sex_b = [a for a in sex_b if a.alive and a.is_fertile()]
        if not sex_a or not sex_b:
            return births

        encounter_prob = max(0.0, config.ENCOUNTER_PROB * (1.0 - crowding))

        for parent_a in sex_a:
            if self.rng.random() > encounter_prob:
                continue
            candidates = [
                other
                for other in sex_b
                if abs(other.x - parent_a.x) <= 1 and abs(other.y - parent_a.y) <= 1
            ]
            if not candidates:
                continue
            parent_b = self.rng.choice(candidates)
            if self.rng.random() > config.CONCEPTION_PROB:
                continue
            parent_a.gestation = config.GESTATION_STEPS
            parent_a.repro_cooldown = config.REPRO_COOLDOWN
            parent_b.repro_cooldown = config.REPRO_COOLDOWN

            births.append(
                BirthRequest(
                    parent_a_id=parent_a.id,
                    parent_b_id=parent_b.id,
                    shape_base=sample_genetic_trait(self.rng, parent_a.shape_base, parent_b.shape_base),
                    color_base=sample_genetic_trait(self.rng, parent_a.color_base, parent_b.color_base),
                    size_base=sample_genetic_trait(self.rng, parent_a.size_base, parent_b.size_base),
                    curiosity_base=sample_genetic_trait(
                        self.rng, parent_a.curiosity_base, parent_b.curiosity_base
                    ),
                    sociality_base=sample_genetic_trait(
                        self.rng, parent_a.sociality_base, parent_b.sociality_base
                    ),
                    risk_base=sample_genetic_trait(self.rng, parent_a.risk_base, parent_b.risk_base),
                    aggression_base=sample_genetic_trait(
                        self.rng, parent_a.aggression_base, parent_b.aggression_base
                    ),
                    creative_bias_base=sample_genetic_trait(
                        self.rng, parent_a.creative_bias_base, parent_b.creative_bias_base
                    ),
                    extraversion_base=sample_genetic_trait(
                        self.rng, parent_a.extraversion_base, parent_b.extraversion_base
                    ),
                    openness_base=sample_genetic_trait(
                        self.rng, parent_a.openness_base, parent_b.openness_base
                    ),
                    conscientiousness_base=sample_genetic_trait(
                        self.rng, parent_a.conscientiousness_base, parent_b.conscientiousness_base
                    ),
                    agreeableness_base=sample_genetic_trait(
                        self.rng, parent_a.agreeableness_base, parent_b.agreeableness_base
                    ),
                    neuroticism_base=sample_genetic_trait(
                        self.rng, parent_a.neuroticism_base, parent_b.neuroticism_base
                    ),
                )
            )
        return births

    def _collect_stats(
        self,
        births: int,
        deaths: int,
        accidental_deaths: int,
        deaths_starvation: int,
        deaths_dehydration: int,
        deaths_heart: int,
        deaths_age: int,
        deaths_accident: int,
        is_day: bool,
        day_index: int,
        year_index: int,
    ) -> StepStats:
        if not self.agents:
            return StepStats(
                step=self.step_count,
                day_index=day_index,
                year_index=year_index,
                is_day=is_day,
                population=0,
                births=births,
                deaths=deaths,
                accidental_deaths=accidental_deaths,
                deaths_starvation=deaths_starvation,
                deaths_dehydration=deaths_dehydration,
                deaths_heart=deaths_heart,
                deaths_age=deaths_age,
                deaths_accident=deaths_accident,
                avg_age_years=0.0,
                avg_energy=0.0,
                avg_hydration=0.0,
                avg_heart=0.0,
                avg_wisdom=0.0,
                avg_empathy=0.0,
                plant_biomass=self.plant_biomass,
                materials=self.materials,
            )
        avg_age_years = (sum(a.age for a in self.agents) / len(self.agents)) / max(1, config.YEAR_STEPS)
        avg_energy = sum(a.energy for a in self.agents) / len(self.agents)
        avg_hydration = sum(a.hydration for a in self.agents) / len(self.agents)
        avg_heart = sum(a.heart for a in self.agents) / len(self.agents)
        avg_wisdom = sum(a.wisdom for a in self.agents) / len(self.agents)
        avg_empathy = sum(a.empathy for a in self.agents) / len(self.agents)
        return StepStats(
            step=self.step_count,
            day_index=day_index,
            year_index=year_index,
            is_day=is_day,
            population=len(self.agents),
            births=births,
            deaths=deaths,
            accidental_deaths=accidental_deaths,
            deaths_starvation=deaths_starvation,
            deaths_dehydration=deaths_dehydration,
            deaths_heart=deaths_heart,
            deaths_age=deaths_age,
            deaths_accident=deaths_accident,
            avg_age_years=avg_age_years,
            avg_energy=avg_energy,
            avg_hydration=avg_hydration,
            avg_heart=avg_heart,
            avg_wisdom=avg_wisdom,
            avg_empathy=avg_empathy,
            plant_biomass=self.plant_biomass,
            materials=self.materials,
        )
