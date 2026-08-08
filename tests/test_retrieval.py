from __future__ import annotations

import pytest

from rag.evaluate.retrieval import hit_rate, mrr


class TestHitRate:
    def test_all_hits(self):
        retrieved = [["a", "b"], ["c", "d"]]
        relevant = ["a", "c"]
        assert hit_rate(retrieved, relevant) == 1.0

    def test_no_hits(self):
        retrieved = [["x", "y"], ["x", "y"]]
        relevant = ["a", "b"]
        assert hit_rate(retrieved, relevant) == 0.0

    def test_partial_hits(self):
        retrieved = [["a", "b"], ["x", "y"]]
        relevant = ["a", "b"]
        assert hit_rate(retrieved, relevant) == 0.5

    def test_empty_relevant(self):
        assert hit_rate([], []) == 0.0


class TestMRR:
    def test_all_at_rank_one(self):
        retrieved = [["a", "b"], ["c", "d"]]
        relevant = ["a", "c"]
        assert mrr(retrieved, relevant) == 1.0

    def test_relevant_at_rank_two(self):
        retrieved = [["x", "a"]]
        relevant = ["a"]
        assert mrr(retrieved, relevant) == 0.5

    def test_relevant_at_rank_three(self):
        retrieved = [["x", "y", "a"]]
        relevant = ["a"]
        assert mrr(retrieved, relevant) == pytest.approx(1.0 / 3.0)

    def test_no_match_returns_zero(self):
        retrieved = [["x", "y"]]
        relevant = ["a"]
        assert mrr(retrieved, relevant) == 0.0

    def test_mixed_positions(self):
        retrieved = [["a", "b"], ["x", "c"]]
        relevant = ["a", "c"]
        expected = (1.0 + 0.5) / 2
        assert mrr(retrieved, relevant) == expected
