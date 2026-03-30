"""
relative_row_scorer.py – Relative Row Scoring for OMR answer detection.

**Relative Row Scoring** determines the selected bubble in each answer row by
comparing fill probabilities *relative to each other*, rather than applying a
flat darkness threshold.  This makes the algorithm robust to global exposure
variation: even if all bubbles look slightly dark (e.g., heavy shadow), only
the comparatively darkest one is selected.

Algorithm (per row)
-------------------
1. Identify the bubble with the highest fill probability → candidate answer.
2. If its probability is below ``min_fill_prob``, flag the row as *blank*.
3. If the difference between the top-2 probabilities is below
   ``tie_threshold``, flag the row as an *erasure tie* (ambiguous answer).
4. Otherwise, report the candidate as the selected answer with a relative
   confidence score.
"""

from __future__ import annotations


class RelativeRowScorer:
    """Scores OMR answer rows using relative comparison of fill probabilities.

    Parameters
    ----------
    tie_threshold:
        If the difference between the highest and second-highest fill
        probabilities is strictly less than this value, the row is flagged as
        a tie (potential erasure or double-mark).  Defaults to ``0.15``.
    min_fill_prob:
        If the highest fill probability in a row is below this threshold, the
        row is considered blank (no bubble selected).  Defaults to ``0.3``.
    """

    def __init__(
        self,
        tie_threshold: float = 0.15,
        min_fill_prob: float = 0.3,
    ) -> None:
        self.tie_threshold = tie_threshold
        self.min_fill_prob = min_fill_prob

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_row(
        self,
        bubble_probs: list[float],
        bubble_labels: list[str],
    ) -> dict:
        """Determine the selected answer for a single answer row.

        The decision logic is purely *relative*: no fixed pixel-darkness
        threshold is used.  Instead, the algorithm asks "which bubble is
        *most filled relative to its neighbours*?".

        Parameters
        ----------
        bubble_probs:
            Fill probabilities for each bubble in the row, in order
            (e.g., ``[p_A, p_B, p_C, p_D, p_E]``).  Values should be in
            ``[0, 1]``.
        bubble_labels:
            Label for each bubble, parallel to *bubble_probs*
            (e.g., ``["A", "B", "C", "D", "E"]``).

        Returns
        -------
        dict with keys:

        * ``selected``   – label of the selected bubble, or ``None`` if blank.
        * ``is_tie``     – ``True`` when top-2 probabilities are within
                           ``tie_threshold`` of each other.
        * ``is_blank``   – ``True`` when no bubble exceeds ``min_fill_prob``.
        * ``confidence`` – relative confidence of the selection (difference
                           between top probability and the mean of the rest).
        * ``scores``     – mapping of ``label → probability``.

        Raises
        ------
        ValueError
            If *bubble_probs* and *bubble_labels* have different lengths, or
            if either list is empty.
        """
        if len(bubble_probs) != len(bubble_labels):
            raise ValueError(
                f"bubble_probs length ({len(bubble_probs)}) must match "
                f"bubble_labels length ({len(bubble_labels)})."
            )
        if not bubble_probs:
            raise ValueError("bubble_probs must not be empty.")

        scores: dict[str, float] = {
            label: float(prob)
            for label, prob in zip(bubble_labels, bubble_probs)
        }

        sorted_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_label, top_prob = sorted_items[0]

        # ---- Blank detection (relative: even the best is too weak) --------
        is_blank = top_prob < self.min_fill_prob

        # ---- Tie detection (relative: top-2 are too close) ----------------
        is_tie = False
        if len(sorted_items) >= 2:
            second_prob = sorted_items[1][1]
            is_tie = (top_prob - second_prob) < self.tie_threshold

        # ---- Relative confidence ------------------------------------------
        # Defined as the gap between the top probability and the mean of all
        # other bubbles – captures how much the winner stands out.
        if len(bubble_probs) > 1:
            other_probs = [p for lbl, p in sorted_items[1:]]
            mean_others = sum(other_probs) / len(other_probs)
            confidence = float(top_prob - mean_others)
        else:
            confidence = float(top_prob)

        selected = None if is_blank else top_label

        return {
            "selected": selected,
            "is_tie": is_tie,
            "is_blank": is_blank,
            "confidence": confidence,
            "scores": scores,
        }

    def score_sheet(
        self,
        row_probs: list[list[float]],
        row_labels: list[list[str]],
    ) -> list[dict]:
        """Apply :meth:`score_row` to every row on an answer sheet.

        Parameters
        ----------
        row_probs:
            List of per-row probability lists.  Each inner list corresponds to
            one question row (e.g., five bubbles for A–E options).
        row_labels:
            List of per-row label lists, parallel to *row_probs*.

        Returns
        -------
        list[dict]
            One result dict per row, in the same order as the input.

        Raises
        ------
        ValueError
            If *row_probs* and *row_labels* have different lengths.
        """
        if len(row_probs) != len(row_labels):
            raise ValueError(
                f"row_probs length ({len(row_probs)}) must match "
                f"row_labels length ({len(row_labels)})."
            )
        return [
            self.score_row(probs, labels)
            for probs, labels in zip(row_probs, row_labels)
        ]
