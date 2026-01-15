from axelrod.action import Action
from axelrod.player import Player

C, D = Action.C, Action.D


class Bayesian(Player):
    """A player who uses Bayesian inference to predict the opponent's next move and
    mimics that move, assuming the opponent is playing a memory-one strategy."""

    name = "Bayesian"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": False,
        "makes_use_of": {"game"},
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self, priors=None):
        super().__init__()
        if priors is None:
            self.priors = {
                (C, C): [1, 1],
                (C, D): [1, 1],
                (D, C): [1, 1],
                (D, D): [1, 1],
            }
        else:
            self.priors = priors

    def strategy(self, opponent: Player) -> Action:
        if not self.history:
            return C

        posteriors = {k: list(v) for k, v in self.priors.items()}

        for i in range(len(self.history) - 1):
            context = (self.history[i], opponent.history[i])
            next_opponent_move = opponent.history[i + 1]
            if next_opponent_move == C:
                posteriors[context][0] += 1
            else:
                posteriors[context][1] += 1

        context = (self.history[-1], opponent.history[-1])

        alpha, beta = posteriors[context]
        prob_opponent_cooperates = alpha / (alpha + beta)

        if prob_opponent_cooperates >= 0.5:
            return C
        return D


class CooperativeBayesian(Bayesian):
    """A Bayesian agent with priors that assume the opponent is likely to cooperate."""

    name = "Cooperative Bayesian"

    def __init__(self):
        priors = {
            (C, C): [10, 1],
            (C, D): [10, 1],
            (D, C): [10, 1],
            (D, D): [10, 1],
        }
        super().__init__(priors=priors)


class DefectingBayesian(Bayesian):
    """A Bayesian agent with priors that assume the opponent is likely to defect."""

    name = "Defecting Bayesian"

    def __init__(self):
        priors = {
            (C, C): [1, 10],
            (C, D): [1, 10],
            (D, C): [1, 10],
            (D, D): [1, 10],
        }
        super().__init__(priors=priors)
