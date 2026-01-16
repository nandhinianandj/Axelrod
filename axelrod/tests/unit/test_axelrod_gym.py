import unittest
import gymnasium as gym
import axelrod as axl
from axelrod.action import Action
from axelrod_gym import AxelrodEnv

C, D = Action.C, Action.D

class TestAxelrodEnv(unittest.TestCase):
    def test_init(self):
        opponent = axl.Cooperator()
        env = AxelrodEnv(opponent)
        self.assertIsInstance(env, gym.Env)
        self.assertEqual(env.action_space.n, 2)
        self.assertEqual(env.observation_space.n, 3)

    def test_reset(self):
        opponent = axl.Cooperator()
        env = AxelrodEnv(opponent)
        obs, info = env.reset()
        self.assertEqual(obs, 2)
        self.assertEqual(env.current_round, 0)
        self.assertEqual(len(env.player_history), 0)
        self.assertEqual(len(env.opponent_history), 0)

    def test_step(self):
        opponent = axl.Alternator()
        env = AxelrodEnv(opponent)
        env.reset()

        # Test a cooperation step
        obs, reward, done, _, _ = env.step(0)  # Cooperate
        self.assertEqual(obs, 0)  # Opponent cooperates
        self.assertEqual(reward, 3) # R (both cooperate)
        self.assertFalse(done)

        # Test a defection step
        obs, reward, done, _, _ = env.step(1) # Defect
        self.assertEqual(obs, 1) # Opponent defects
        self.assertEqual(reward, 1) # P (both defect)
        self.assertFalse(done)
