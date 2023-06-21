from abc import ABC, abstractmethod
from src.random_number import Generator


class RandomVariable(ABC):
    @abstractmethod
    def get_random_variables(self):
        pass


class DiscreteRandomVariable(RandomVariable):
    def __init__(
        self, generator: Generator, values: list[object], weights: list[float]
    ):
        self.generator = generator
        self.values = values
        self.probabilities = [w / sum(weights) for w in weights]

    def _get_random_variable(self, random_number: float):
        cumulative_probability = 0
        for value, probability in zip(self.values, self.probabilities):
            cumulative_probability += probability
            if random_number <= cumulative_probability:
                return value

    def get_random_variables(self):
        return [
            self._get_random_variable(rn) for rn in self.generator.get_random_numbers()
        ]


class UniformDiscreteRandomVariable(DiscreteRandomVariable):
    def __init__(self, generator: Generator, values: list[object]):
        probabilities = [1 / len(values) for _ in values]
        super().__init__(generator, values, probabilities)


class UniformContinuousRandomVariable(RandomVariable):
    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max

    def get_random_variables(self, generator: Generator):
        return [
            self.min + (self.max - self.min) * rn
            for rn in generator.get_random_numbers()
        ]
