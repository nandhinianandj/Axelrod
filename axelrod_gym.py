import gymnasium as gym
from gymnasium import spaces
import axelrod as axl
from axelrod.action import Action
import numpy as np

C, D = Action.C, Action.D

class AxelrodEnv(gym.Env):
    """
    A Gymnasium environment for the Iterated Prisoner's Dilemma.
    """
    def __init__(self, opponent: axl.Player, rounds: int = 100):
        super(AxelrodEnv, self).__init__()
        self.opponent = opponent
        self.rounds = rounds
        self.game = axl.Game()

        # Action space: 0 for Cooperate, 1 for Defect
        self.action_space = spaces.Discrete(2)

        # Observation space: Opponent's last move (0 for C, 1 for D)
        # We'll also include an indicator for the first round (2)
        self.observation_space = spaces.Discrete(3)

        self.player_history = []
        self.opponent_history = []
        self.current_round = 0

    def step(self, action):
        player_action = C if action == 0 else D

        # The Axelrod library does not expose a public setter for the history,
        # so we directly manipulate the internal _history attribute. This is a
        # workaround and may be brittle if the library's internal API changes.
        class DummyPlayer(axl.Player):
            pass

        dummy_player = DummyPlayer()
        dummy_player._history = self.player_history

        # Get opponent's action
        self.opponent._history = self.opponent_history
        opponent_action = self.opponent.strategy(dummy_player)


        # Update histories
        self.player_history.append(player_action)
        self.opponent_history.append(opponent_action)


        # Calculate reward
        payoffs = self.game.RPST()
        if player_action == C and opponent_action == C:
            reward = payoffs[0]  # Both cooperate (R)
        elif player_action == C and opponent_action == D:
            reward = payoffs[2]  # Player is sucker (S)
        elif player_action == D and opponent_action == C:
            reward = payoffs[3]  # Player is tempted (T)
        else:  # Both defect
            reward = payoffs[1]  # Both defect (P)


        self.current_round += 1
        done = self.current_round >= self.rounds

        # Observation for the next state is the opponent's last move
        obs = 1 if opponent_action == D else 0

        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_history = []
        self.opponent_history = []
        self.opponent.reset()
        self.current_round = 0

        # Initial observation: 2 to signify the first round
        return 2, {}

    def render(self, mode='human'):
        pass
