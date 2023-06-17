from abc import ABC
from abc import abstractmethod
from matplotlib import pyplot as plt


def _greater_common_divisor(n1: int, n2: int):
    while n2 != 0:
        n1, n2 = n2, n1 % n2
    return n1


def _prime_factors(n: int):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return tuple(set(factors))


class Generator(ABC):
    @abstractmethod
    def get_random_numbers(self):
        pass

    @abstractmethod
    def get_xn_sequence(self):
        pass

    @abstractmethod
    def verify_parameters(self):
        pass

    def plot_random_numbers(self, join_points=True):
        _, axes = plt.subplots()
        rand_nums = self.get_random_numbers()
        if join_points:
            axes.plot(range(len(rand_nums)), rand_nums, color="#19A7CE")
        axes.scatter(range(len(rand_nums)), rand_nums, color="#146C94")

        plt.ylabel("Número Pseudoaleatorio ($\mu_i$)", fontsize=10)
        plt.xlabel("Índice ($i$)", fontsize=10)

    def __len__(self):
        return len(self.get_random_numbers())

    def __str__(self):
        str_random_nums = [str(x) for x in self.get_random_numbers()]
        return " ".join(str_random_nums)


class LinearCongruentialGenerator(Generator):
    def __init__(self, seed: int, a: int, b: int, m: int):
        self.x0 = seed
        self.a = a
        self.b = b
        self.m = m
        self.verify_parameters()

    def verify_parameters(self):
        if self.a <= 0 or self.a >= self.m:
            raise ValueError("'a' must be greater than 0 and less than 'm'")
        if self.b < 0 or self.b >= self.m:
            raise ValueError("'b' must be greater or equal than 0 and less than 'm'")
        if self.m <= 0:
            raise ValueError("'m' must be greater than 0")
        if self.x0 < 0 or self.x0 >= self.m:
            raise ValueError("'x0' must be greater or equal than 0 and less than 'm'")

    def get_xn_sequence(self):
        x = self.x0
        xn_sequence = []
        while x not in xn_sequence:
            xn_sequence.append(x)
            x = (self.a * x + self.b) % self.m
        return xn_sequence

    def get_random_numbers(self):
        return [x / self.m for x in self.get_xn_sequence()]

    @abstractmethod
    def has_max_sequence(self):
        pass


class MixedCongruentialGenerator(LinearCongruentialGenerator):
    def __init__(self, seed: int, a: int, b: int, m: int):
        super().__init__(seed, a, b, m)

    def has_max_sequence(self):
        # b and m are coprime:
        if _greater_common_divisor(self.b, self.m) != 1:
            return False

        # a - 1 is divisible by all prime factors of m:
        for factor in _prime_factors(self.m):
            if (self.a - 1) % factor != 0:
                return False

        # a - 1 is divisible by 4 if m is divisible by 4:
        if self.m % 4 == 0 and (self.a - 1) % 4 != 0:
            return False

        return True


class MultiplicativeCongruentialGenerator(LinearCongruentialGenerator):
    def __init__(self, seed: int, a: int, m: int):
        super().__init__(seed, a, 0, m)

    def verify_parameters(self):
        super().verify_parameters()
        if self.b != 0:
            raise ValueError("'b' must be 0")

    def is_m_prime(self):
        return _prime_factors(self.m) == (self.m,)

    def has_max_sequence(self):
        # m is prime
        if not self.is_m_prime():
            return False

        # a^[(m-1)/q] mod m != 1 for all prime factors q of m-1
        for factor in _prime_factors(self.m - 1):
            if self.a ** ((self.m - 1) // factor) % self.m == 1:
                return False

        return True


class MiddleSquare(Generator):
    def __init__(self, k: int, seed: int):
        self.k = k
        self.seed = seed
        self.verify_parameters()

    def verify_parameters(self):
        if self.k <= 0:
            raise ValueError("'k' must be greater than 0")
        if self.seed <= 0:
            raise ValueError("'seed' must be greater than 0")

    def _get_middle(self, number: int):
        n_squared_str = str(number**2)
        start = (len(n_squared_str) - self.k) // 2
        end = start + self.k
        return int(n_squared_str[start:end])

    def _fill_zeros(self, number: int):
        n_squared_str = str(number**2)
        while len(n_squared_str) < self.k:
            n_squared_str = n_squared_str + "0"
        while len(n_squared_str) % 2 != self.k % 2:
            n_squared_str = n_squared_str + "0"
        return int(n_squared_str)

    def get_xn_sequence(self):
        x = self.seed
        xn_sequence = []
        while x not in xn_sequence:
            xn_sequence.append(x)
            x = self._get_middle(self._fill_zeros(x))
        return xn_sequence

    def get_random_numbers(self):
        return [x / 10**self.k for x in self.get_xn_sequence()]
