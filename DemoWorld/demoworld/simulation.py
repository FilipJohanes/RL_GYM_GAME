"""Simulation driver."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional
import csv
import os
import random

from .environment import Environment, StepStats
from .renderer import render_ascii, render_ppm


def _write_csv(stats: List[StepStats], csv_path: str) -> None:
    if not stats:
        with open(csv_path, "w", newline="") as handle:
            handle.write("")
        return
    fieldnames = list(asdict(stats[0]).keys())
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in stats:
            writer.writerow(asdict(row))


def _summarize(stats: List[StepStats]) -> Dict[str, float]:
    if not stats:
        return {
            "steps": 0,
            "final_pop": 0,
            "peak_pop": 0,
            "avg_pop": 0.0,
            "total_births": 0,
            "total_deaths": 0,
            "total_accidents": 0,
            "final_year": 0,
            "final_day": 0,
            "avg_age_years": 0.0,
            "avg_energy": 0.0,
            "avg_hydration": 0.0,
            "avg_heart": 0.0,
            "avg_wisdom": 0.0,
            "avg_empathy": 0.0,
        }
    total_births = sum(s.births for s in stats)
    total_deaths = sum(s.deaths for s in stats)
    total_accidents = sum(s.accidental_deaths for s in stats)
    peak_pop = max(s.population for s in stats)
    avg_pop = sum(s.population for s in stats) / len(stats)
    last = stats[-1]
    return {
        "steps": len(stats),
        "final_pop": last.population,
        "peak_pop": peak_pop,
        "avg_pop": avg_pop,
        "total_births": total_births,
        "total_deaths": total_deaths,
        "total_accidents": total_accidents,
        "final_year": last.year_index,
        "final_day": last.day_index,
        "avg_age_years": sum(s.avg_age_years for s in stats) / len(stats),
        "avg_energy": sum(s.avg_energy for s in stats) / len(stats),
        "avg_hydration": sum(s.avg_hydration for s in stats) / len(stats),
        "avg_heart": sum(s.avg_heart for s in stats) / len(stats),
        "avg_wisdom": sum(s.avg_wisdom for s in stats) / len(stats),
        "avg_empathy": sum(s.avg_empathy for s in stats) / len(stats),
    }


def _print_summary(summary: Dict[str, float]) -> None:
    print("summary:")
    print(f"  steps={int(summary['steps'])} final_pop={int(summary['final_pop'])}")
    print(
        f"  total_births={int(summary['total_births'])} "
        f"total_deaths={int(summary['total_deaths'])} "
        f"accidents={int(summary['total_accidents'])} peak_pop={int(summary['peak_pop'])}"
    )
    print(
        f"  avg_pop={summary['avg_pop']:.1f} avg_age_years={summary['avg_age_years']:.2f} "
        f"avg_energy={summary['avg_energy']:.3f} avg_hydration={summary['avg_hydration']:.3f}"
    )
    print(
        f"  avg_heart={summary['avg_heart']:.3f} avg_wisdom={summary['avg_wisdom']:.3f} "
        f"avg_empathy={summary['avg_empathy']:.3f}"
    )
    print(f"  final_year={int(summary['final_year'])} final_day={int(summary['final_day'])}")


def _resolve_render_path(base: str, step: int) -> str:
    if "{step}" in base:
        return base.format(step=step)
    if base.lower().endswith(".ppm"):
        return base
    return os.path.join(base, f"frame_{step}.ppm")


def run_simulation(
    steps: int,
    initial_population: int,
    seed: Optional[int] = None,
    log_every: int = 100,
    csv_path: Optional[str] = None,
    summary: bool = True,
    render_every: int = 0,
    render_path: Optional[str] = None,
    render_ascii_enabled: bool = False,
    render_scale: int = 4,
) -> List[StepStats]:
    rng = random.Random(seed)
    env = Environment(rng)
    env.seed_initial_population(initial_population)

    stats: List[StepStats] = []
    for _ in range(steps):
        step_stats = env.step()
        stats.append(step_stats)
        if log_every and step_stats.step % log_every == 0:
            phase = "day" if step_stats.is_day else "night"
            print(
                f"step={step_stats.step} year={step_stats.year_index} day={step_stats.day_index} "
                f"phase={phase} pop={step_stats.population} "
                f"births={step_stats.births} deaths={step_stats.deaths} acc={step_stats.accidental_deaths} "
                f"starv={step_stats.deaths_starvation} dehydr={step_stats.deaths_dehydration} "
                f"heart={step_stats.deaths_heart} age={step_stats.deaths_age} accD={step_stats.deaths_accident} "
                f"avgE={step_stats.avg_energy:.3f} avgH={step_stats.avg_hydration:.3f} "
                f"avgHeart={step_stats.avg_heart:.3f}"
            )
        if render_every and step_stats.step % render_every == 0:
            if render_ascii_enabled:
                print(render_ascii(env))
            if render_path:
                target = _resolve_render_path(render_path, step_stats.step)
                os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
                render_ppm(env, target, scale=render_scale)
    if csv_path:
        _write_csv(stats, csv_path)
    if summary:
        _print_summary(_summarize(stats))
    return stats
