"""Tests for the Bayesian strategies."""

import axelrod as axl
from .test_player import TestPlayer

C, D = axl.Action.C, axl.Action.D


class TestBayesian(TestPlayer):
    name = "Bayesian"
    player = axl.Bayesian
    expected_classifier = {
        "memory_depth": float("inf"),
        "stochastic": False,
        "makes_use_of": {"game"},
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def test_vs_cooperator(self):
        actions = [(C, C), (C, C), (C, C), (C, C), (C, C)]
        self.versus_test(axl.Cooperator(), expected_actions=actions)

    def test_vs_defector(self):
        actions = [(C, D), (C, D), (D, D), (C, D), (D, D)]
        self.versus_test(axl.Defector(), expected_actions=actions)

    def test_vs_alternator(self):
        actions = [(C, C), (C, D), (C, C), (D, D), (C, C)]
        self.versus_test(axl.Alternator(), expected_actions=actions)


class TestCooperativeBayesian(TestPlayer):
    name = "Cooperative Bayesian"
    player = axl.CooperativeBayesian
    expected_classifier = {
        "memory_depth": float("inf"),
        "stochastic": False,
        "makes_use_of": {"game"},
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def test_vs_cooperator(self):
        actions = [(C, C), (C, C), (C, C), (C, C), (C, C)]
        self.versus_test(axl.Cooperator(), expected_actions=actions)

    def test_vs_defector(self):
        actions = [(C, D), (C, D), (C, D), (C, D), (C, D)]
        self.versus_test(axl.Defector(), expected_actions=actions)

    def test_vs_alternator(self):
        actions = [(C, C), (C, D), (C, C), (C, D), (C, C)]
        self.versus_test(axl.Alternator(), expected_actions=actions)


class TestDefectingBayesian(TestPlayer):
    name = "Defecting Bayesian"
    player = axl.DefectingBayesian
    expected_classifier = {
        "memory_depth": float("inf"),
        "stochastic": False,
        "makes_use_of": {"game"},
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def test_vs_cooperator(self):
        actions = [(C, C), (D, C), (D, C), (D, C), (D, C)]
        self.versus_test(axl.Cooperator(), expected_actions=actions)

    def test_vs_defector(self):
        actions = [(C, D), (D, D), (D, D), (D, D), (D, D)]
        self.versus_test(axl.Defector(), expected_actions=actions)

    def test_vs_alternator(self):
        actions = [(C, C), (D, D), (D, C), (D, D), (D, C)]
        self.versus_test(axl.Alternator(), expected_actions=actions)
