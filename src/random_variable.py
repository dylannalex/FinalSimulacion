from abc import ABC, abstractmethod
from typing import Callable
from src.random_number import Generator


class RandomVariable(ABC):
    @abstractmethod
    def get_random_variables(self):
        pass

    @abstractmethod
    def next(self):
        pass


class DiscreteRandomVariable(RandomVariable):
    def __init__(
        self, generator: Generator, values: list[object], weights: list[float]
    ):
        self.generator = generator
        self.values = values
        self.weights = weights

    @property
    def probabilities(self):
        return [w / sum(self.weights) for w in self.weights]

    def _get_random_variable(self, random_number: float):
        cumulative_probability = 0
        for value, probability in zip(self.values, self.probabilities):
            cumulative_probability += probability
            if random_number <= cumulative_probability:
                return value
        else:
            return value

    def get_random_variables(self):
        return [
            self._get_random_variable(rn) for rn in self.generator.get_random_numbers()
        ]

    def next(self):
        return self._get_random_variable(self.generator.next())


class UniformDiscreteRandomVariable(DiscreteRandomVariable):
    def __init__(self, generator: Generator, values: list[object]):
        probabilities = [1 / len(values) for _ in values]
        super().__init__(generator, values, probabilities)


class UniformContinuousRandomVariable(RandomVariable):
    def __init__(self, generator: Generator, a: float, b: float):
        self.generator = generator
        self.a = a
        self.b = b

    def get_random_variables(self):
        return [
            self.a + (self.b - self.a) * rn
            for rn in self.generator.get_random_numbers()
        ]

    def next(self):
        return self.a + (self.b - self.a) * self.generator.next()


class AcceptanceRejectionVariable(RandomVariable):
    def __init__(
        self,
        generator: Generator,
        a: float,
        b: float,
        f: Callable[[float], float],
        g: Callable[[float], float],
        M: float,
    ):
        self.generator = generator
        self.a = a
        self.b = b
        self.f = f
        self.g = g
        self.M = M

    def get_random_variables(self):
        i=0
        random_variables = []
        while i < len(self.generator.get_random_numbers()):
            u1 = self.generator.next()
            u2 = self.generator.next()
            x = self.a + (self.b - self.a) * u1
            if u2 <= self.f(x) / (self.M * self.g(x)):
                random_variables.append(x) 
            i+=1
        return random_variables

    def next(self):
        while True:
            u1 = self.generator.next()
            u2 = self.generator.next()
            x = self.a + (self.b - self.a) * u1
            if u2 <= self.f(x) / (self.M * self.g(x)):
                return x
