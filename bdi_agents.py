from axelrod.player import Player
from axelrod.action import Action

C, D = Action.C, Action.D

class BDIAgent(Player):
    """A base class for agents with Beliefs, Desires, and Intentions."""
    name = "BDI Agent"

    def __init__(self):
        super().__init__()
        self.beliefs = {}
        self.desires = []
        self.intentions = []

    def strategy(self, opponent: Player) -> Action:
        self.update_beliefs(opponent)
        self.update_desires()
        self.update_intentions()
        return self.execute_intentions()

    def update_beliefs(self, opponent: Player):
        """Update the agent's beliefs based on the opponent's history."""
        pass

    def update_desires(self):
        """Update the agent's desires based on its beliefs."""
        pass

    def update_intentions(self):
        """Update the agent's intentions based on its desires."""
        pass

    def execute_intentions(self) -> Action:
        """Execute the agent's intentions to choose an action."""
        pass

class CooperativeAgent(BDIAgent):
    """An agent that desires mutual cooperation."""
    name = "Cooperative BDI Agent"

    def __init__(self):
        super().__init__()
        self.desires = ["COOPERATE"]

    def strategy(self, opponent: Player) -> Action:
        # This agent's intention is always to cooperate.
        return C

class SelfishAgent(BDIAgent):
    """An agent that desires to maximize its own score."""
    name = "Selfish BDI Agent"

    def __init__(self):
        super().__init__()
        self.desires = ["MAXIMIZE_SCORE"]

    def strategy(self, opponent: Player) -> Action:
        # This agent believes that defecting is the best way to maximize its score.
        return D

class GrudgerAgent(BDIAgent):
    """An agent that cooperates until the opponent defects."""
    name = "Grudger BDI Agent"

    def __init__(self):
        super().__init__()
        self.desires = ["COOPERATE_UNTIL_BETRAYED"]
        self.betrayed = False

    def strategy(self, opponent: Player) -> Action:
        if self.betrayed:
            return D

        if D in opponent.history:
            self.betrayed = True
            return D

        return C

    def reset(self):
        super().reset()
        self.betrayed = False
