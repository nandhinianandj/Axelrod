import unittest
import axelrod as axl
from axelrod.action import Action
from bdi_agents import CooperativeAgent, SelfishAgent, GrudgerAgent

C, D = Action.C, Action.D

class TestBDIAgents(unittest.TestCase):
    def test_cooperative_agent(self):
        opponent = axl.Defector()
        agent = CooperativeAgent()
        self.assertEqual(agent.strategy(opponent), C)

    def test_selfish_agent(self):
        opponent = axl.Cooperator()
        agent = SelfishAgent()
        self.assertEqual(agent.strategy(opponent), D)

    def test_grudger_agent(self):
        class DummyOpponent:
            def __init__(self):
                self.history = []

        opponent = DummyOpponent()
        agent = GrudgerAgent()

        # Should cooperate initially
        self.assertEqual(agent.strategy(opponent), C)

        # Should cooperate as long as the opponent cooperates
        opponent.history.append(C)
        self.assertEqual(agent.strategy(opponent), C)

        # Should defect after the opponent defects
        opponent.history.append(D)
        self.assertEqual(agent.strategy(opponent), D)

        # Should continue to defect
        opponent.history.append(C)
        self.assertEqual(agent.strategy(opponent), D)
