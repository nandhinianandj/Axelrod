import random
import networkx as nx
import axelrod as axl

# Define a game for "big trades" with higher stakes
big_trade_game = axl.Game(r=30, s=0, t=50, p=10)

class City:
    def __init__(self, population, trade_graph, big_trade_probability=0.1, trade_probability=0.5, stagnation_factor=0.9, stagnation_threshold=5, increase_factor=1.5, min_trade_probability=0.1):
        self.population = population
        self.gdp = 0
        self.trade_graph = trade_graph
        self.big_trade_probability = big_trade_probability
        self.last_big_trade_day = 0
        self.trade_probabilities = {node: trade_probability for node in self.trade_graph.nodes()}
        self.stagnation_factor = stagnation_factor
        self.stagnation_threshold = stagnation_threshold
        self.increase_factor = increase_factor
        self.min_trade_probability = min_trade_probability

    def trade(self, agent1, agent2, turns=10, game=axl.DefaultGame):
        """Simulates a trade (a match) between two agents."""
        match = axl.Match((agent1, agent2), turns=turns, game=game)
        match.play()
        payoffs = match.final_score()
        # A trade contributes to the GDP by the sum of payoffs
        self.gdp += sum(payoffs)
        return game is not axl.DefaultGame  # Return true if it was a big trade

    def run_day(self, turns_per_trade=10, day_number=0):
        """Simulates a day of trading."""
        big_trade_occurred_today = False
        agents_in_big_trades = set()

        for agent1_idx, agent2_idx in self.trade_graph.edges():
            # Check if agents will trade today
            if random.random() < self.trade_probabilities[agent1_idx] and \
               random.random() < self.trade_probabilities[agent2_idx]:

                agent1 = self.population[agent1_idx]
                agent2 = self.population[agent2_idx]

                # Decide if this is a big trade
                if random.random() < self.big_trade_probability:
                    if self.trade(agent1, agent2, turns=turns_per_trade, game=big_trade_game):
                        big_trade_occurred_today = True
                        agents_in_big_trades.add(agent1_idx)
                        agents_in_big_trades.add(agent2_idx)
                else:
                    self.trade(agent1, agent2, turns=turns_per_trade)

        if big_trade_occurred_today:
            self.last_big_trade_day = day_number
            self._increase_neighbor_trade_probability(agents_in_big_trades)
        else:
            self._apply_stagnation(day_number)

    def _increase_neighbor_trade_probability(self, agents_in_big_trades):
        """Increases the trade probability for neighbors of agents in big trades."""
        for agent_idx in agents_in_big_trades:
            for neighbor_idx in self.trade_graph.neighbors(agent_idx):
                if neighbor_idx not in agents_in_big_trades:
                    self.trade_probabilities[neighbor_idx] = min(1.0, self.trade_probabilities[neighbor_idx] * self.increase_factor)

    def _apply_stagnation(self, day_number):
        """Reduces trade probability if no big trades have occurred recently."""
        if day_number - self.last_big_trade_day > self.stagnation_threshold:
            for node in self.trade_graph.nodes():
                self.trade_probabilities[node] = max(self.min_trade_probability, self.trade_probabilities[node] * self.stagnation_factor)

    def get_gdp(self):
        return self.gdp
