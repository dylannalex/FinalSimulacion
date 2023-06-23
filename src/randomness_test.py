import numpy as np
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
from src import utils


class RandomnessTest(ABC):
    @abstractmethod
    def run_test(self) -> None:
        pass


class ChiSquaredTest(RandomnessTest):
    def __init__(self, random_numbers: list[float], intervals: int, statistic: float):
        self.random_numbers = np.array(random_numbers)
        self.intervals = intervals
        self.statistic = statistic
        self.x0 = self._get_x0()

    def _get_x0(self):
        ef = len(self.random_numbers) / self.intervals
        observed_freq, _ = np.histogram(
            self.random_numbers, bins=self.intervals, range=(0, 1)
        )
        expected_freq = np.ones(self.intervals) * ef
        chi_squared = np.sum((expected_freq - observed_freq) ** 2) / ef
        return chi_squared

    def run_test(self):
        statistic_text = r"\chi^2_{(\alpha, k=" + f"{self.intervals-1}" + r")}"

        utils.print_markdown(
            r"$\chi^2_0 = \sum_{i=1}^n \frac{({FO}_i - {FE}_i)^2}{{FE}_i} ="
            + f" {self.x0}$"
        )
        utils.print_markdown(f"${statistic_text} = {self.statistic}$")

        if self.x0 < self.statistic:
            utils.print_markdown(
                f"$\\chi^2_0 <  {statistic_text} \\Rightarrow$ La hipótesis se acepta."
            )
        else:
            utils.print_markdown(
                f"$\\chi^2_0 >  {statistic_text} \\Rightarrow$ La hipótesis se rechaza."
            )


class KolmogorovSmirnovTest(RandomnessTest):
    def __init__(self, random_numbers: list[float], statistic: float):
        self.random_numbers = np.array(random_numbers)
        self.sorted_random_numbers = np.sort(random_numbers)
        self.statistic = statistic
        self.distance = self._get_distance(self.sorted_random_numbers)

    def graph(self) -> None:
        n = len(self.sorted_random_numbers)
        line = np.arange(1, n + 1) / n
        _, ax = plt.subplots(figsize=(10, 5))
        (line1,) = ax.plot(
            self.sorted_random_numbers, label="Número Aleatorio ($\mu_i$)"
        )
        (line2,) = ax.plot(line, label=r"Frecuencia Esperada ($\frac{i}{n}$)")
        ax.legend(handles=[line1, line2])

    def _get_distance(self):
        n = len(self.random_numbers)
        d = np.max(np.arange(1, n + 1) / n - self.sorted_random_numbers)
        return d

    def run_test(self):
        # Print results
        distance_text = r"$max|\frac{i}{n} - \mu_i|" + f" = {self.distance}$"
        statistic_text = (
            f"$D(\\alpha, n={len(self.random_numbers)}) = {self.statistic}$"
        )

        if self.distance < self.statistic:
            utils.print_markdown(
                f"{distance_text} < {statistic_text} $\\Rightarrow$ La hipótesis se acepta."
            )
        else:
            utils.print_markdown(
                f"{distance_text} > {statistic_text} $\\Rightarrow$ La hipótesis se rechaza."
            )


class WaldWolfowitzRunsTest(RandomnessTest):
    def __init__(self, random_numbers: list[float], statistic: float):
        self.random_numbers = np.array(random_numbers)
        self.statistic = statistic
        self.runs, self.positive, self.negative = self._get_runs()
        self.z = self._get_z()

    def _get_runs(self) -> tuple[np.ndarray, int, int]:
        mean = self.random_numbers.mean()
        # Positive and negative
        positive = np.where(self.random_numbers > mean, 1, 0).sum()
        negative = np.where(self.random_numbers <= mean, 1, 0).sum()
        # Runs
        runs_array = np.where(self.random_numbers > mean, 1, -1)
        runs_array = np.split(
            self.random_numbers, np.where(np.diff(runs_array) != 0)[0] + 1
        )
        return runs_array, positive, negative

    def _get_z(self) -> float:
        total_runs = len(self.runs)
        positive, negative = self.positive, self.negative
        total_random_numbers = positive + negative
        mean = 2 * positive * negative / total_random_numbers + 1 / 2
        variance = (
            2 * positive * negative * (2 * positive * negative - total_random_numbers)
        )
        variance /= total_random_numbers**2 * (total_random_numbers - 1)
        z = (total_runs - mean) / np.sqrt(variance)
        return z

    def runs_test(self) -> None:
        runs_text = " ".join([str(run) for run in self.runs])
        statistic_text = r"$Z_{\alpha/2}$"

        print(f"Rachas: {runs_text}")
        utils.print_markdown(f"$b = {len(self.runs)}$ (cantidad de rachas)")
        utils.print_markdown(f"$n_1 = {self.positive}$ (cantidad de números positivos)")
        utils.print_markdown(f"$n_2 = {self.negative}$ (cantidad de números negativos)")
        utils.print_markdown(f"{statistic_text} = {self.statistic}")
        utils.print_markdown(r"$Z_0 = \frac{b - \mu_b}{\sigma_b}" + f" = {self.z}$")

        if np.abs(self.z) <= self.statistic:
            utils.print_markdown(
                f"-{statistic_text} $\leq Z_0 \leq$ {statistic_text} $\\Rightarrow$ La hipótesis se acepta."
            )
        elif self.z > 0:
            utils.print_markdown(
                f"$Z_0 >$ {statistic_text} $\\Rightarrow$ La hipótesis se rechaza."
            )
        elif self.z < 0:
            utils.print_markdown(
                f"$Z_0 <$ -{statistic_text} $\\Rightarrow$ La hipótesis se rechaza."
            )
