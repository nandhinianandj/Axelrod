import axelrod as axl
from axelrod.action import Action
from axelrod.player import Player
from stable_baselines3 import PPO
from axelrod_gym import AxelrodEnv
from bdi_agents import CooperativeAgent, SelfishAgent, GrudgerAgent
import numpy as np

C, D = Action.C, Action.D

class RLAgent(Player):
    """A player that uses a trained reinforcement learning model to make decisions."""
    name = "RL Agent"

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.obs = np.array(2)  # Initial observation

    def strategy(self, opponent: Player) -> Action:
        # The environment expects the opponent's last move as an observation
        if not opponent.history:
            self.obs = np.array(2)
        else:
            self.obs = np.array(1 if opponent.history[-1] == D else 0)

        action, _states = self.model.predict(self.obs, deterministic=True)
        return C if action == 0 else D

    def reset(self):
        super().reset()
        self.obs = np.array(2)

def train_rl_agent(opponent, training_steps=10000):
    """Train an RL agent against a specific opponent."""
    env = AxelrodEnv(opponent=opponent)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=training_steps)
    return model

def main():
    # Create instances of the BDI agents
    cooperative_agent = CooperativeAgent()
    selfish_agent = SelfishAgent()
    grudger_agent = GrudgerAgent()

    # Train an RL agent against a Grudger
    print("Training RL agent against Grudger...")
    rl_model = train_rl_agent(grudger_agent)
    rl_agent = RLAgent(rl_model)

    # Define the players for the tournament
    players = [
        cooperative_agent,
        selfish_agent,
        grudger_agent,
        rl_agent,
        axl.TitForTat(),
        axl.Defector(),
        axl.Cooperator(),
        axl.Random(),
    ]

    # Create and run the tournament
    tournament = axl.Tournament(players, turns=200, repetitions=10)
    results = tournament.play()

    # Print the results
    print("\nTournament Results:")
    for rank, name in enumerate(results.ranked_names):
        print(f"{rank + 1}: {name}")

    # Show the payoff matrix
    plot = axl.Plot(results)
    fig = plot.boxplot()
    fig.savefig("tournament_results.png")
    print("\nTournament results saved to tournament_results.png")

if __name__ == "__main__":
    main()
