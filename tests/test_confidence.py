from __future__ import annotations

import math

from rag.evaluate.confidence import confidence


class TestConfidenceFlag:
    def test_high_scores_grounded(self):
        result = confidence([2.0, 1.5, 1.0])
        assert result["flag"] == "grounded"

    def test_negative_scores_low_confidence(self):
        result = confidence([-1.0, -2.0])
        assert result["flag"] == "low-confidence / possible hallucination"

    def test_empty_scores_low_confidence(self):
        result = confidence([])
        assert result["flag"] == "low-confidence / possible hallucination"
        assert result["confidence"] == 0.0
        assert result["top_score"] == 0.0
        assert result["margin"] == 0.0


class TestFaithfulnessBlending:
    def test_blending_formula(self):
        scores = [2.0]
        faithfulness = 0.9
        result = confidence(scores, faithfulness=faithfulness)

        retrieval_conf = 1.0 / (1.0 + math.exp(-2.0))
        expected = 0.6 * retrieval_conf + 0.4 * 0.9
        assert result["confidence"] == round(expected, 4)

    def test_no_faithfulness_uses_retrieval_only(self):
        scores = [2.0]
        result = confidence(scores)

        retrieval_conf = 1.0 / (1.0 + math.exp(-2.0))
        assert result["confidence"] == round(retrieval_conf, 4)

    def test_low_faithfulness_flags_hallucination(self):
        result = confidence([2.0], faithfulness=0.3)
        assert result["flag"] == "low-confidence / possible hallucination"


class TestMargin:
    def test_margin_with_two_scores(self):
        result = confidence([3.0, 1.0])
        assert result["margin"] == 2.0

    def test_margin_single_score(self):
        result = confidence([5.0])
        assert result["margin"] == 5.0

    def test_top_score_selected(self):
        result = confidence([1.0, 3.0, 2.0])
        assert result["top_score"] == 3.0
