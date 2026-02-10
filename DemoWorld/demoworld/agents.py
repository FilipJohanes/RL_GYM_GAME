"""Agent model and trait sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import math
import random

from . import config


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def sample_trait(rng: random.Random) -> float:
    if rng.random() < config.TRAIT_CORE_PROB:
        return rng.uniform(config.TRAIT_CORE_MIN, config.TRAIT_CORE_MAX)
    return rng.random()


def mutate_trait(rng: random.Random, value: float) -> float:
    return clamp01(rng.gauss(value, config.MUTATION_STD))

def sample_genetic_trait(rng: random.Random, parent_a: float, parent_b: float) -> float:
    lo = min(parent_a, parent_b)
    hi = max(parent_a, parent_b)
    if hi <= lo:
        return clamp01(lo)

    bin_size = max(0.0001, config.GENETIC_BIN)
    steps = max(1, int(math.ceil((hi - lo) / bin_size)))
    mid = (hi + lo) / 2.0

    bins = []
    weights = []
    for i in range(steps + 1):
        center = lo + i * bin_size
        if center > hi:
            center = hi
        bins.append(center)
        dist = abs(center - mid)
        step_dist = int(round(dist / bin_size))
        weight = max(config.GENETIC_MIN_WEIGHT, 1.0 - config.GENETIC_DROP * step_dist)
        weights.append(weight)

    center = rng.choices(bins, weights=weights, k=1)[0]
    half = bin_size / 2.0
    low = max(lo, center - half)
    high = min(hi, center + half)
    return clamp01(rng.uniform(low, high))


@dataclass
class Agent:
    id: int
    sex: str
    age: int
    parent_a_id: int | None
    parent_b_id: int | None
    generation: int
    x: int
    y: int
    z: float
    shape_base: float
    color_base: float
    size_base: float
    curiosity_base: float
    sociality_base: float
    risk_base: float
    aggression_base: float
    creative_bias_base: float
    extraversion_base: float
    openness_base: float
    conscientiousness_base: float
    agreeableness_base: float
    neuroticism_base: float
    shape: float = field(init=False)
    color: float = field(init=False)
    size: float = field(init=False)
    wisdom: float = field(init=False)
    empathy: float = field(init=False)
    curiosity: float = field(init=False)
    sociality: float = field(init=False)
    risk_tolerance: float = field(init=False)
    aggression: float = field(init=False)
    creative_bias: float = field(init=False)
    extraversion: float = field(init=False)
    openness: float = field(init=False)
    conscientiousness: float = field(init=False)
    agreeableness: float = field(init=False)
    neuroticism: float = field(init=False)
    goals: Dict[str, float] = field(default_factory=dict)
    name: str = ""
    memory: Dict[str, Any] = field(default_factory=dict)
    lexicon: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    bonds: Dict[int, float] = field(default_factory=dict)
    known_water: List[tuple[int, int]] = field(default_factory=list)
    known_food: List[tuple[int, int]] = field(default_factory=list)
    heart: float = 1.0
    energy: float = 1.0
    hydration: float = 1.0
    carry_water: float = 0.0
    alive: bool = True
    gestation: int = 0
    repro_cooldown: int = 0
    last_idea_step: int = 0

    def __post_init__(self) -> None:
        self.shape = self.shape_base
        self.color = self.color_base
        self.size = self.size_base
        self.wisdom = clamp01(config.WISDOM_BASE + self.color_base * config.WISDOM_COLOR_WEIGHT)
        self.empathy = clamp01(config.EMPATHY_BASE + self.shape_base * config.EMPATHY_SHAPE_WEIGHT)
        self.curiosity = clamp01(self.curiosity_base * config.DEVELOPMENT_START)
        self.sociality = clamp01(self.sociality_base * config.DEVELOPMENT_START)
        self.risk_tolerance = clamp01(self.risk_base * config.DEVELOPMENT_START)
        self.aggression = clamp01(self.aggression_base * config.DEVELOPMENT_START)
        self.creative_bias = clamp01(self.creative_bias_base * config.DEVELOPMENT_START)
        self.extraversion = clamp01(self.extraversion_base * config.DEVELOPMENT_START)
        self.openness = clamp01(self.openness_base * config.DEVELOPMENT_START)
        self.conscientiousness = clamp01(self.conscientiousness_base * config.DEVELOPMENT_START)
        self.agreeableness = clamp01(self.agreeableness_base * config.DEVELOPMENT_START)
        self.neuroticism = clamp01(self.neuroticism_base * config.DEVELOPMENT_START)
        self._update_goals()

    def is_mature(self) -> bool:
        return self.age >= config.MATURITY_AGE_STEPS

    def is_fertile(self) -> bool:
        return self.is_mature() and self.gestation == 0 and self.repro_cooldown == 0

    def step_age(self) -> None:
        self.age += 1
        if self.gestation > 0:
            self.gestation -= 1
        if self.repro_cooldown > 0:
            self.repro_cooldown -= 1

    def apply_decay(self) -> None:
        self.energy -= config.ENERGY_DECAY
        self.hydration -= config.HYDRATION_DECAY

    def update_age_effects(self) -> None:
        age_years = self.age / max(1, config.YEAR_STEPS)
        if age_years > config.PHYSICAL_DECLINE_START_YEARS:
            span = max(1.0, config.MAX_LIFESPAN_YEARS - config.PHYSICAL_DECLINE_START_YEARS)
            decline = min(1.0, (age_years - config.PHYSICAL_DECLINE_START_YEARS) / span)
            factor = 1.0 - decline * config.PHYSICAL_DECLINE_MAX
            self.shape = clamp01(self.shape_base * factor)
            self.size = clamp01(self.size_base * factor)
        else:
            self.shape = self.shape_base
            self.size = self.size_base
        self.color = self.color_base

        target_wisdom = clamp01(config.WISDOM_BASE + self.color_base * config.WISDOM_COLOR_WEIGHT)
        self.wisdom = clamp01(self.wisdom + config.WISDOM_GROWTH_RATE * (target_wisdom - self.wisdom))
        target_empathy = clamp01(config.EMPATHY_BASE + self.shape_base * config.EMPATHY_SHAPE_WEIGHT)
        self.empathy = clamp01(self.empathy + config.EMPATHY_GROWTH_RATE * (target_empathy - self.empathy))
        self.curiosity = clamp01(
            self.curiosity + config.CURIOSITY_GROWTH_RATE * (self.curiosity_base - self.curiosity)
        )
        self.sociality = clamp01(
            self.sociality + config.SOCIALITY_GROWTH_RATE * (self.sociality_base - self.sociality)
        )
        self.risk_tolerance = clamp01(
            self.risk_tolerance + config.RISK_GROWTH_RATE * (self.risk_base - self.risk_tolerance)
        )
        self.aggression = clamp01(
            self.aggression + config.AGGRESSION_GROWTH_RATE * (self.aggression_base - self.aggression)
        )
        self.creative_bias = clamp01(
            self.creative_bias + config.CREATIVE_GROWTH_RATE * (self.creative_bias_base - self.creative_bias)
        )
        self.extraversion = clamp01(
            self.extraversion + config.EXTRAVERSION_GROWTH_RATE * (self.extraversion_base - self.extraversion)
        )
        self.openness = clamp01(
            self.openness + config.OPENNESS_GROWTH_RATE * (self.openness_base - self.openness)
        )
        self.conscientiousness = clamp01(
            self.conscientiousness
            + config.CONSCIENTIOUSNESS_GROWTH_RATE * (self.conscientiousness_base - self.conscientiousness)
        )
        self.agreeableness = clamp01(
            self.agreeableness + config.AGREEABLENESS_GROWTH_RATE * (self.agreeableness_base - self.agreeableness)
        )
        self.neuroticism = clamp01(
            self.neuroticism + config.NEUROTICISM_GROWTH_RATE * (self.neuroticism_base - self.neuroticism)
        )
        self._update_goals()

    def _update_goals(self) -> None:
        survival = 0.5 + self.conscientiousness * 0.3 + self.neuroticism * 0.2
        social = 0.2 + self.extraversion * 0.4 + self.agreeableness * 0.3 + (1.0 - self.creative_bias) * 0.1
        creative = 0.2 + self.openness * 0.4 + self.creative_bias * 0.3
        reproduction = 0.2 + self.sociality * 0.3 + self.extraversion * 0.2 - self.neuroticism * 0.1
        exploration = 0.2 + self.curiosity * 0.4 + self.openness * 0.2 - self.conscientiousness * 0.1

        goals = {
            "survival": clamp01(survival),
            "social": clamp01(social),
            "creative": clamp01(creative),
            "reproduction": clamp01(reproduction),
            "exploration": clamp01(exploration),
        }
        total = sum(goals.values()) or 1.0
        self.goals = {k: v / total for k, v in goals.items()}

    def is_dead(self) -> bool:
        if self.energy <= 0.0 or self.hydration <= 0.0:
            return True
        if self.heart <= 0.0:
            return True
        if self.age > config.MAX_AGE_STEPS:
            return True
        return False


@dataclass
class BirthRequest:
    parent_a_id: int
    parent_b_id: int
    shape_base: float
    color_base: float
    size_base: float
    curiosity_base: float
    sociality_base: float
    risk_base: float
    aggression_base: float
    creative_bias_base: float
    extraversion_base: float
    openness_base: float
    conscientiousness_base: float
    agreeableness_base: float
    neuroticism_base: float
