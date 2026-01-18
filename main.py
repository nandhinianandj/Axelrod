import networkx as nx
import axelrod as axl
from pumpelsdrop import City

if __name__ == "__main__":
    # Define a list of strategies for our population
    strategies = [
        axl.Cooperator,
        axl.Defector,
        axl.TitForTat,
        axl.Grudger,
        axl.Random,
    ]
    population = [s() for s in strategies]

    # Create a graph where everyone trades with everyone else
    num_agents = len(population)
    trade_graph = nx.complete_graph(num_agents)

    # Create the city
    pumpelsdrop = City(population, trade_graph, big_trade_probability=0.1, trade_probability=0.5, stagnation_factor=0.95, stagnation_threshold=3, increase_factor=1.5, min_trade_probability=0.1)

    # Run the simulation for a number of days
    num_days = 50
    print("Running economic simulation with stagnation for Pumpelsdrop...")
    for day in range(num_days):
        pumpelsdrop.run_day(day_number=day)
        print(f"Day {day + 1}: GDP = {pumpelsdrop.get_gdp()}, Last big trade on day {pumpelsdrop.last_big_trade_day}")
        # Print the trade probabilities of the first 5 agents to see the stagnation effect
        for i in range(min(5, num_agents)):
            print(f"  Agent {i} trade probability: {pumpelsdrop.trade_probabilities[i]:.2f}")
