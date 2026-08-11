"""STAMP-style multi-objective rewards and hope/fear selection."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_MIN_SCORE = 1e-12


@dataclass(frozen=True, slots=True)
class RewardExponents:
    """Integer temperatures for the adapted STAMP weighted product.

    ``likelihood`` is still tracked and reported because reversals on it inform
    the other temperatures, but it does not weight the reward itself: hope/fear
    selection already scores reachability from the same quantity.
    """

    style: int = 1
    semantic: int = 1
    likelihood: int = 1
    length: int = 1

    def __post_init__(self) -> None:
        if min(self.style, self.semantic, self.likelihood, self.length) < 1:
            raise ValueError("Reward exponents must be positive integers")

    def as_dict(self) -> dict[str, int]:
        return {
            "style": self.style,
            "semantic": self.semantic,
            "likelihood": self.likelihood,
            "length": self.length,
        }


def normalized_base_model_likelihood(
    log_likelihood: float,
    token_count: int | None = None,
) -> float:
    """Map total or mean log likelihood to average token probability in [0, 1]."""

    if token_count is not None:
        if token_count < 1:
            raise ValueError("token_count must be positive")
        log_likelihood /= token_count
    if not math.isfinite(log_likelihood):
        return 0.0 if log_likelihood < 0 else 1.0
    return math.exp(min(0.0, float(log_likelihood)))


def bounded_length_score(length_ratio: float) -> float:
    """Symmetric [0, 1] score: 1 at equal length and smaller toward either extreme."""

    if not math.isfinite(length_ratio) or length_ratio <= 0:
        return 0.0
    return min(length_ratio, 1.0 / length_ratio)


@dataclass(frozen=True, slots=True)
class CandidateReward:
    """All objective scores needed for preference-pair construction."""

    style_probability: float
    semantic_similarity: float
    base_model_likelihood: float
    length_ratio: float
    candidate_id: str = ""
    text: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("style_probability", self.style_probability),
            ("semantic_similarity", self.semantic_similarity),
            ("base_model_likelihood", self.base_model_likelihood),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if not math.isfinite(self.length_ratio) or self.length_ratio <= 0:
            raise ValueError("length_ratio must be finite and positive")

    @property
    def length_score(self) -> float:
        return bounded_length_score(self.length_ratio)

    @property
    def normalized_likelihood(self) -> float:
        return self.base_model_likelihood

    def aggregate(self, exponents: RewardExponents | None = None) -> float:
        """Score the quality objectives as a weighted geometric mean in [0, 1].

        A raw weighted product shrinks toward zero as the exponents grow, which
        left the reward orders of magnitude under the reachability term that
        hope/fear selection adds it to, reducing that selection to a pure
        base-model-likelihood ranking. Averaging in log space keeps the reward
        on the same scale as reachability, which also already scores
        likelihood, so likelihood is left out here rather than counted twice.
        """

        weights = exponents or RewardExponents()
        terms = (
            (weights.style, self.style_probability),
            (weights.semantic, self.semantic_similarity),
            (weights.length, self.length_score),
        )
        total = sum(weight for weight, _ in terms)
        logged = sum(
            weight * math.log(max(score, _MIN_SCORE)) for weight, score in terms
        )
        return math.exp(logged / total)

    def objective_scores(self) -> dict[str, float]:
        return {
            "style": self.style_probability,
            "semantic": self.semantic_similarity,
            "likelihood": self.base_model_likelihood,
            "length": self.length_score,
        }

    def as_dict(self, exponents: RewardExponents | None = None) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "text": self.text,
            "style_probability": self.style_probability,
            "semantic_similarity": self.semantic_similarity,
            "base_model_likelihood": self.base_model_likelihood,
            "length_ratio": self.length_ratio,
            "length_score": self.length_score,
            "aggregate_reward": self.aggregate(exponents),
        }


def reversal_counts(
    preference_pairs: Sequence[tuple[CandidateReward, CandidateReward]],
) -> dict[str, int]:
    """Count objectives where preferred candidates score below rejected ones."""

    counts = {"style": 0, "semantic": 0, "likelihood": 0, "length": 0}
    for preferred, rejected in preference_pairs:
        preferred_scores = preferred.objective_scores()
        rejected_scores = rejected.objective_scores()
        for objective in counts:
            counts[objective] += preferred_scores[objective] < rejected_scores[objective]
    return counts


def dynamic_reward_exponents(
    preference_pairs: Sequence[tuple[CandidateReward, CandidateReward]],
    *,
    max_exponent: int = 5,
) -> RewardExponents:
    """Adapt STAMP's reversal-based temperatures to the four local objectives.

    STAMP raises an objective's product exponent when current preference pairs
    reverse that objective. This bounded generalization maps the largest observed
    reversal count to ``max_exponent`` and keeps objectives with no reversals at 1.
    Exponents should be recomputed after each preference-optimization iteration.
    """

    if max_exponent < 1:
        raise ValueError("max_exponent must be positive")
    counts = reversal_counts(preference_pairs)
    largest = max(counts.values(), default=0)
    if largest == 0 or max_exponent == 1:
        return RewardExponents()

    def exponent(objective: str) -> int:
        return 1 + round((max_exponent - 1) * counts[objective] / largest)

    return RewardExponents(
        style=exponent("style"),
        semantic=exponent("semantic"),
        likelihood=exponent("likelihood"),
        length=exponent("length"),
    )


@dataclass(frozen=True, slots=True)
class HopeFearSelection:
    hope: CandidateReward
    fear: CandidateReward
    hope_score: float
    fear_score: float

    def as_dict(self, exponents: RewardExponents | None = None) -> dict[str, Any]:
        return {
            "hope": self.hope.as_dict(exponents),
            "fear": self.fear.as_dict(exponents),
            "hope_score": self.hope_score,
            "fear_score": self.fear_score,
        }


def _stable_candidate_key(candidate: CandidateReward) -> tuple[str, str]:
    return (candidate.candidate_id or candidate.text, candidate.text)


def select_hope_and_fear(
    candidates: Sequence[CandidateReward],
    *,
    exponents: RewardExponents | None = None,
    model_temperature: float = 0.1,
) -> HopeFearSelection:
    """Select reachable high/low reward candidates with deterministic tie breaks.

    Following STAMP, hope maximizes ``M**tau + R`` and fear maximizes
    ``M**tau - R``. The selected candidates are forced to be distinct so the
    result is always usable as a preference pair.
    """

    if len(candidates) < 2:
        raise ValueError("Hope/fear selection requires at least two candidates")
    if model_temperature <= 0:
        raise ValueError("model_temperature must be positive")
    keys = [_stable_candidate_key(candidate) for candidate in candidates]
    if any(not key[0] for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("Candidates must have unique nonempty IDs or texts")
    weights = exponents or RewardExponents()
    ordered = sorted(candidates, key=_stable_candidate_key)

    def reachability(candidate: CandidateReward) -> float:
        return candidate.base_model_likelihood**model_temperature

    def hope_objective(candidate: CandidateReward) -> float:
        return reachability(candidate) + candidate.aggregate(weights)

    hope = max(ordered, key=hope_objective)
    remaining = [candidate for candidate in ordered if candidate is not hope]

    def fear_objective(candidate: CandidateReward) -> float:
        return reachability(candidate) - candidate.aggregate(weights)

    fear = max(remaining, key=fear_objective)
    return HopeFearSelection(
        hope=hope,
        fear=fear,
        hope_score=hope_objective(hope),
        fear_score=fear_objective(fear),
    )


def candidate_reward_from_scores(
    *,
    candidate_id: str,
    text: str,
    style_probability: float,
    semantic_similarity: float,
    base_log_likelihood: float,
    token_count: int,
    source_length: int,
) -> CandidateReward:
    """Construct a candidate reward from common raw scorer outputs."""

    if source_length < 1:
        raise ValueError("source_length must be positive")
    return CandidateReward(
        candidate_id=candidate_id,
        text=text,
        style_probability=style_probability,
        semantic_similarity=max(0.0, min(1.0, semantic_similarity)),
        base_model_likelihood=normalized_base_model_likelihood(
            base_log_likelihood,
            token_count,
        ),
        length_ratio=max(1, len(text.strip())) / source_length,
    )


def mean_objectives(candidates: Sequence[CandidateReward]) -> Mapping[str, float]:
    if not candidates:
        raise ValueError("At least one candidate is required")
    objectives = ("style", "semantic", "likelihood", "length")
    scores = [candidate.objective_scores() for candidate in candidates]
    return {
        objective: sum(score[objective] for score in scores) / len(scores)
        for objective in objectives
    }
