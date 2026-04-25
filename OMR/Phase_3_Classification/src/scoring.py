"""
scoring.py - Row-level decision engine for Phase 3.

Adds explicit ambiguity and multi-mark handling on top of relative row scoring.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RowDecision:
    selected: str | None
    status: str
    is_blank: bool
    is_tie: bool
    is_multi_mark: bool
    needs_review: bool
    confidence: float
    margin_top2: float
    scores: dict[str, float]


class RelativeRowDecisionEngine:
    """Decision layer that converts per-bubble probabilities into row outcomes."""

    def __init__(
        self,
        min_fill_prob: float = 0.30,
        tie_threshold: float = 0.15,
        multi_mark_threshold: float = 0.50,
        review_band: float = 0.06,
    ) -> None:
        self.min_fill_prob = min_fill_prob
        self.tie_threshold = tie_threshold
        self.multi_mark_threshold = multi_mark_threshold
        self.review_band = review_band

    def decide_row(self, bubble_probs: list[float], bubble_labels: list[str]) -> RowDecision:
        if len(bubble_probs) != len(bubble_labels):
            raise ValueError(
                f"bubble_probs length ({len(bubble_probs)}) must match "
                f"bubble_labels length ({len(bubble_labels)})."
            )
        if not bubble_probs:
            raise ValueError("bubble_probs must not be empty.")

        scores = {label: float(prob) for label, prob in zip(bubble_labels, bubble_probs)}
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        top_label, top_prob = sorted_scores[0]
        second_prob = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
        margin_top2 = float(top_prob - second_prob)

        is_blank = top_prob < self.min_fill_prob
        is_tie = margin_top2 < self.tie_threshold if len(sorted_scores) > 1 else False
        is_multi_mark = (
            len(sorted_scores) > 1
            and top_prob >= self.multi_mark_threshold
            and second_prob >= self.multi_mark_threshold
            and margin_top2 <= self.review_band
        )

        if len(sorted_scores) > 1:
            mean_others = sum(prob for _, prob in sorted_scores[1:]) / (len(sorted_scores) - 1)
            confidence = float(top_prob - mean_others)
        else:
            confidence = float(top_prob)

        if is_blank:
            status = "blank"
            selected = None
        elif is_multi_mark:
            status = "multi_mark_review"
            selected = top_label
        elif is_tie:
            status = "tie_review"
            selected = top_label
        else:
            status = "selected"
            selected = top_label

        needs_review = status in {"multi_mark_review", "tie_review"}

        return RowDecision(
            selected=selected,
            status=status,
            is_blank=is_blank,
            is_tie=is_tie,
            is_multi_mark=is_multi_mark,
            needs_review=needs_review,
            confidence=confidence,
            margin_top2=margin_top2,
            scores=scores,
        )

    def decide_sheet(self, row_probs: list[list[float]], row_labels: list[list[str]]) -> list[RowDecision]:
        if len(row_probs) != len(row_labels):
            raise ValueError(
                f"row_probs length ({len(row_probs)}) must match "
                f"row_labels length ({len(row_labels)})."
            )
        return [self.decide_row(probs, labels) for probs, labels in zip(row_probs, row_labels)]
