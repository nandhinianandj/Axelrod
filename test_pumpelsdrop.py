import unittest
import networkx as nx
from pumpelsdrop import City
import axelrod as axl
import random
import numpy as np

class TestCity(unittest.TestCase):
    def setUp(self):
        # Set a seed for reproducibility
        random.seed(0)
        np.random.seed(0)

        self.strategies = [axl.Cooperator(), axl.Defector(), axl.TitForTat()]
        self.trade_graph = nx.complete_graph(len(self.strategies))
        self.city = City(self.strategies, self.trade_graph, trade_probability=1.0)

    def test_city_initialization(self):
        self.assertEqual(len(self.city.population), 3)
        self.assertEqual(self.city.gdp, 0)
        self.assertEqual(len(self.city.trade_graph.nodes), 3)

    def test_trade(self):
        initial_gdp = self.city.gdp
        self.city.trade(self.strategies[0], self.strategies[1])
        self.assertGreater(self.city.gdp, initial_gdp)

    def test_run_day(self):
        initial_gdp = self.city.gdp
        self.city.run_day()
        self.assertGreater(self.city.gdp, initial_gdp)

    def test_big_trade_ripple_effect(self):
        # Set up a city with a known structure
        strategies = [axl.Cooperator() for _ in range(4)]
        trade_graph = nx.path_graph(4)  # 0-1-2-3
        city = City(strategies, trade_graph, trade_probability=0.5, increase_factor=2.0)

        # Manually define the agents who had a big trade
        agents_in_big_trade = {1, 2}
        # Directly call the method to test its logic
        city._increase_neighbor_trade_probability(agents_in_big_trade)

        # Neighbors of {1, 2} in the path graph are {0, 3}.
        # Their probabilities should double.
        self.assertEqual(city.trade_probabilities[0], 1.0)
        self.assertEqual(city.trade_probabilities[3], 1.0)
        # The agents who traded should be unaffected.
        self.assertEqual(city.trade_probabilities[1], 0.5)
        self.assertEqual(city.trade_probabilities[2], 0.5)

    def test_stagnation(self):
        self.city.stagnation_threshold = 3
        self.city.stagnation_factor = 0.9
        self.city.min_trade_probability = 0.1
        self.city.trade_probabilities = {node: 0.5 for node in self.city.trade_graph.nodes()}
        self.city.big_trade_probability = 0.0

        # Simulate a few days without a big trade
        self.city.run_day(day_number=1)
        self.city.run_day(day_number=2)
        self.city.run_day(day_number=3)
        self.city.run_day(day_number=4) # Stagnation should be applied here

        # Check that the trade probabilities have decreased
        for i in range(len(self.strategies)):
            self.assertAlmostEqual(self.city.trade_probabilities[i], 0.45) # 0.5 * 0.9

if __name__ == '__main__':
    unittest.main()
