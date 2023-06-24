import numpy as np
from src.random_variable import DiscreteRandomVariable


class AntSystem:
    def __init__(
        self,
        cities_distance: np.ndarray,
        alpha: float,
        beta: float,
        evaporation_rate: float,
        n_ants: int,
        round_trip: bool,
        random_variable: DiscreteRandomVariable,
    ):
        """Ant System algorithm.

        Parameters
        ----------
        cities_distance : np.ndarray
            A square matrix M of shape (n_cities, n_cities) \
            containing the distance between each city. Mij is \
            the distance between city i and city j
        alpha : float
            Factor of pheromone importance, alpha >= 0
        beta : float
            Factor of heuristic importance, beta >= 0
        evaporation_rate : float
            Evaporation rate, 0 <= rho <= 1
        n_ants : int
            Number of ants
        round_trip : bool
            If True, the ants will return to the starting city.
        """
        self.cities_distance = cities_distance
        self.alpha = alpha
        self.beta = beta
        self.n_ants = n_ants
        self.evaporation_rate = evaporation_rate
        self.round_trip = round_trip
        self.random_variable = random_variable
        self.cities = np.arange(cities_distance.shape[0])  # [0, 1, 2, ..., n_cities]
        self.pheromone = np.ones(cities_distance.shape)
        self.best_solution: np.ndarray = None
        self.best_solution_cost: float = np.inf
        self._cycle_best_solution: np.ndarray = None
        self._cycle_best_solution_cost: np.ndarray = None

    def initialization(self):
        # Initialize tabu list
        self.tabu_list = np.zeros(
            (self.n_ants, self.cities_distance.shape[0]),
            dtype=int,
        )
        # Setup random variable generator
        self.random_variable.values = self.cities
        self.random_variable.weights = np.ones(self.cities.shape)

        # Place ants randomly on the graph
        self.tabu_list[:, 0] = [self.random_variable.next() for _ in range(self.n_ants)]

    def cost(self, solution: np.ndarray) -> float:
        """Return the cost of a solution.

        Parameters
        ----------
        solution : np.ndarray
            Solution to evaluate.
        """
        cost = 0
        for i, city in enumerate(solution[:-1]):
            cost += self.cities_distance[city, solution[i + 1]]
        if self.round_trip:
            cost += self.cities_distance[solution[-1], solution[0]]
        return cost

    def next_city(self, ant: int, city: int):
        """Return the next city to visit by an ant.

        Parameters
        ----------
        ant : int
            Ant index.
        city : int
            The current city index of the tabu list.
        """
        visited_cities = self.tabu_list[ant, :city]
        current_city = visited_cities[-1]
        unvisited_cities = np.setdiff1d(self.cities, visited_cities)
        pheromone = self.pheromone[current_city, unvisited_cities]
        heuristic = self.cities_distance[current_city, unvisited_cities]
        heuristic = 1 / heuristic
        probabilities = (pheromone**self.alpha) * (heuristic**self.beta)
        probabilities = probabilities / probabilities.sum()
        # Setup random variable generator
        self.random_variable.values = unvisited_cities
        self.random_variable.weights = probabilities
        return self.random_variable.next()

    def cycle(self):
        for ant in range(self.n_ants):
            for city in range(1, self.cities_distance.shape[0]):
                self.tabu_list[ant, city] = self.next_city(ant, city)

    def cycle_best_solution(self):
        cycle_best_solution = None
        cycle_best_solution_cost = np.inf
        for solution in self.tabu_list:
            solution_cost = self.cost(solution)
            if solution_cost < cycle_best_solution_cost:
                cycle_best_solution = solution
                cycle_best_solution_cost = solution_cost
        # Update cycle best solution
        self._cycle_best_solution = cycle_best_solution
        self._cycle_best_solution_cost = cycle_best_solution_cost
        # Update overall best solution
        if cycle_best_solution_cost < self.best_solution_cost:
            self.best_solution = cycle_best_solution
            self.best_solution_cost = cycle_best_solution_cost

    def update_pheromone(self):
        # Evaporation
        self.pheromone *= 1 - self.evaporation_rate
        # Reinforcement
        for i in range(len(self.cities) - 1):
            current_city = self._cycle_best_solution[i]
            next_city = self._cycle_best_solution[i + 1]
            self.pheromone[next_city, current_city] += (
                1 / self._cycle_best_solution_cost
            )

    def run(self, max_cycles: int, verbose: bool = False):
        for i in range(max_cycles):
            self.initialization()
            self.cycle()
            self.cycle_best_solution()
            self.update_pheromone()
            if verbose:
                print(f"Iteration {i + 1}: {self.best_solution_cost}")
        return self.best_solution
