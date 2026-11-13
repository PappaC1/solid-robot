import io
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from pypdf import PdfReader


# ==========================================
# CONFIGURATION
# ==========================================

REQUIRED_FORM_COLUMNS = {
    "horse",
    "date",
    "course",
    "distance_m",
    "finish_position",
    "runners",
}

OPTIONAL_FORM_COLUMNS = {
    "going",
    "weight_kg",
    "draw",
    "jockey",
    "trainer",
    "market_odds",
    "won",
    "placed",
}

COURSE_CODES = {
    "N": "Greyville Polytrack",
    "D": "Greyville Turf",
    "G": "Greyville",
    "S": "Scottsville",
    "P": "Scottsville Inside",
    "K": "Kenilworth",
    "W": "Kenilworth Winter",
    "UB": "Durbanville",
    "E": "Fairview Polytrack",
    "F": "Fairview Turf",
    "T": "Turffontein",
    "V": "Vaal",
    "B": "Vaal Classic",
}

GOING_CODES = {
    "n": "Normal",
    "r": "Rain Affected",
    "g": "Good",
    "y": "Yielding",
    "s": "Soft",
    "f": "Firm",
    "h": "Heavy",
}


# ==========================================
# HISTORICAL FORM ENGINE
# ==========================================


class HistoricalFormEngine:
    """
    Calculates confidence and opportunity from a horse's historical runs.
    """

    def __init__(
        self,
        recent_runs: int = 5,
        freshness_ideal_days: int = 35,
        freshness_max_days: int = 180,
    ):
        self.recent_runs = recent_runs
        self.freshness_ideal_days = freshness_ideal_days
        self.freshness_max_days = freshness_max_days

    def calculate_runner_profile(
        self,
        horse: str,
        course: str,
        distance_m: int,
        going: str,
        jockey: str,
        trainer: str,
        market_odds: Optional[float],
        form_history: pd.DataFrame,
    ) -> Dict[str, float]:
        if form_history.empty or "horse" not in form_history.columns:
            return self._no_history_profile(market_odds)

        history = form_history[
            form_history["horse"].str.casefold() == horse.casefold()
        ].copy()

        if history.empty:
            return self._no_history_profile(market_odds)

        history["date"] = pd.to_datetime(history["date"], errors="coerce")
        history = history.sort_values("date", ascending=False)

        recent = history.head(self.recent_runs)
        recent_form = self._recent_form_score(recent)
        consistency = self._consistency_score(history)
        course_distance = self._course_distance_score(history, course, distance_m)
        going_score = self._going_score(history, going)
        freshness = self._freshness_score(history)
        jockey_score = self._jockey_score(history, jockey)
        trainer_score = self._trainer_score(history, trainer)
        market_score = self._market_score(market_odds)

        confidence = (
            recent_form * 0.35
            + consistency * 0.20
            + jockey_score * 0.15
            + trainer_score * 0.15
            + freshness * 0.15
        )

        opportunity = (
            course_distance * 0.35
            + going_score * 0.20
            + market_score * 0.25
            + recent_form * 0.20
        )

        return {
            "recent_form": round(recent_form, 1),
            "consistency": round(consistency, 1),
            "course_distance": round(course_distance, 1),
            "going": round(going_score, 1),
            "freshness": round(freshness, 1),
            "jockey": round(jockey_score, 1),
            "trainer": round(trainer_score, 1),
            "market": round(market_score, 1),
            "confidence": round(confidence, 1),
            "opportunity": round(opportunity, 1),
            "historical_runs": int(len(history)),
        }

    def _no_history_profile(self, market_odds: Optional[float]) -> Dict[str, float]:
        market_score = self._market_score(market_odds)
        return {
            "recent_form": 65.0,
            "consistency": 65.0,
            "course_distance": 65.0,
            "going": 65.0,
            "freshness": 65.0,
            "jockey": 65.0,
            "trainer": 65.0,
            "market": market_score,
            "confidence": 65.0,
            "opportunity": market_score,
            "historical_runs": 0,
        }

    def _recent_form_score(self, recent: pd.DataFrame) -> float:
        if recent.empty:
            return 65.0

        scores = []
        for _, run in recent.iterrows():
            position = self._safe_number(run.get("finish_position"))
            runners = self._safe_number(run.get("runners"))

            if position is None or runners is None or runners < 2:
                continue

            percentile = 1 - ((position - 1) / max(runners - 1, 1))
            scores.append(percentile * 100)

        if not scores:
            return 65.0

        weights = np.linspace(1.0, 0.5, len(scores))
        return float(np.average(scores, weights=weights))

    def _consistency_score(self, history: pd.DataFrame) -> float:
        if history.empty:
            return 65.0

        valid = history[
            history["finish_position"].apply(self._safe_number).notna()
            & history["runners"].apply(self._safe_number).notna()
        ].copy()

        if valid.empty:
            return 65.0

        percentiles = []
        for _, run in valid.iterrows():
            position = self._safe_number(run["finish_position"])
            field_size = self._safe_number(run["runners"])
            if position is None or field_size is None or field_size < 2:
                continue
            percentiles.append(1 - ((position - 1) / max(field_size - 1, 1)))

        if not percentiles:
            return 65.0

        return float(np.mean(percentiles) * 100)

    def _course_distance_score(
        self, history: pd.DataFrame, course: str, distance_m: int
    ) -> float:
        if not course and not distance_m:
            return 65.0

        same_course = (
            history[
                history["course"]
                .astype(str)
                .str.casefold()
                .str.contains(course.casefold(), na=False)
            ]
            if course
            else pd.DataFrame()
        )
        same_distance = (
            history[history["distance_m"].apply(self._safe_number) == distance_m]
            if distance_m
            else pd.DataFrame()
        )

        course_score = self._result_score(same_course)
        distance_score = self._result_score(same_distance)

        return (course_score * 0.4) + (distance_score * 0.6)

    def _going_score(self, history: pd.DataFrame, going: str) -> float:
        if not going or "going" not in history.columns:
            return 65.0

        same_going = history[
            history["going"]
            .astype(str)
            .str.casefold()
            .str.contains(going.casefold(), na=False)
        ]
        return self._result_score(same_going)

    def _freshness_score(self, history: pd.DataFrame) -> float:
        if history.empty:
            return 65.0

        last_run = history["date"].dropna().max()
        if pd.isna(last_run):
            return 65.0

        days_since_run = abs((datetime.now() - last_run.to_pydatetime()).days)

        if days_since_run < 7:
            return 75.0
        if days_since_run <= self.freshness_ideal_days:
            return 95.0
        if days_since_run > self.freshness_max_days:
            return 55.0

        decline = (days_since_run - self.freshness_ideal_days) / (
            self.freshness_max_days - self.freshness_ideal_days
        )
        return float(95.0 - (decline * 40.0))

    def _jockey_score(self, history: pd.DataFrame, jockey: str) -> float:
        return self._participant_score(history, "jockey", jockey)

    def _trainer_score(self, history: pd.DataFrame, trainer: str) -> float:
        return self._participant_score(history, "trainer", trainer)

    def _participant_score(
        self, history: pd.DataFrame, column: str, participant: str
    ) -> float:
        if not participant or column not in history.columns:
            return 65.0

        clean_name = re.sub(r"[^a-zA-Z\s]", "", participant).strip()
        if not clean_name:
            return 65.0

        last_name = clean_name.split()[-1].casefold()
        participant_runs = history[
            history[column]
            .astype(str)
            .str.casefold()
            .str.contains(last_name, na=False)
        ]

        if participant_runs.empty:
            return 65.0

        return self._result_score(participant_runs)

    def _result_score(self, runs: pd.DataFrame) -> float:
        if runs.empty:
            return 65.0

        positions = runs["finish_position"].apply(self._safe_number).dropna()
        if positions.empty:
            return 65.0

        wins = (positions == 1).mean()
        places = (positions <= 3).mean()
        return float(max(45.0, min(100.0, (wins * 65) + (places * 35) + 35)))

    def _market_score(self, market_odds: Optional[float]) -> float:
        if not market_odds or market_odds <= 1:
            return 70.0
        return float(max(50.0, min(95.0, 100.0 - (market_odds * 2.0))))

    @staticmethod
    def _safe_number(value) -> Optional[float]:
        try:
            if pd.isna(value):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None


# ==========================================
# UHIS AND ODDS ENGINES
# ==========================================


class UHISEngine:
    def calculate(self, metrics: Dict[str, float]) -> float:
        if not metrics:
            return 0.0
        return round(sum(metrics.values()) / len(metrics), 2)


class ProbabilityEngine:
    def calculate(self, uhis: float, total_race_uhis: float) -> float:
        if total_race_uhis <= 0 or uhis <= 0:
            return 0.0
