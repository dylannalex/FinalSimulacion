import numpy as np
from os import system
from src import random_number
from src import random_variable
from src import randomness_test


def _get_option(title: str, options: dict):
    while True:
        system("cls")
        if title:
            print(title, end="\n\n")
        for k, w in options.items():
            print(f"{k}. {w}")
        option = input("\nIngrese una opción: ")
        if option in options.keys():
            system("cls")
            return int(option)


class _RandomNumberGeneratorMenu:
    options = {
        "1": "Generador congruencial mixto",
        "2": "Generador congruencial multiplicativo",
        "3": "Generador de cuadrados medios",
    }

    def menu(self):
        option = _get_option("Generador de números aleatorios", self.options)

        if option == 1:
            a = int(input("Ingresar el valor de 'a': "))
            m = int(input("Ingresar el valor de 'm': "))
            x0 = int(input("Ingresar el valor de 'x0': "))
            generator = random_number.MultiplicativeCongruentialGenerator(
                seed=x0, a=a, m=m
            )

        elif option == 2:
            a = int(input("Ingresar el valor de 'a': "))
            b = int(input("Ingresar el valor de 'b': "))
            m = int(input("Ingresar el valor de 'm': "))
            x0 = int(input("Ingresar el valor de 'x0': "))
            generator = random_number.MixedCongruentialGenerator(seed=x0, a=a, b=b, m=m)

        elif option == 3:
            k = int(input("Ingresar el valor de 'k': "))
            x0 = int(input("Ingresar el valor de 'x0': "))
            generator = random_number.MiddleSquare(seed=x0, k=k)

        print("Resultado:")
        print(f"La secuencia generada es: {generator}")
        print(f"La longitud del periodo es: {len(generator)}")

        generator.plot_random_numbers()


class _RandomnessTestMenu:
    options = {
        "1": "Prueba de Kolmogorov-Smirnov",
        "2": "Prueba de Chi-Cuadrado",
        "3": "Prueba de Corridas",
    }

    def menu(self):
        option = _get_option("Pruebas de aleatoriedad", self.options)

        # Get random numbers
        n = int(input("Ingrese la cantidad de números aleatorios: "))
        seed = int(input("Ingrese la semilla: "))
        rng = np.random.default_rng(seed=seed)
        random_numbers = rng.random(n)

        # Get statistic
        statistic = float(input("Ingrese el valor del estadístico de la tabla: "))

        # Get Generator
        if option == 1:
            test = randomness_test.KolmogorovSmirnovTest(random_numbers, statistic)
        if option == 2:
            intervals = int(input("Ingrese la cantidad de intervalos: "))
            test = randomness_test.ChiSquaredTest(random_numbers, intervals, statistic)
        if option == 3:
            test = randomness_test.WaldWolfowitzRunsTest(random_numbers, statistic)

        # Run test
        print(f"Los números aleatorios generados son:\n{random_numbers}", end="\n\n")
        test.run_test()


class _RandomVariableGeneratorMenu:
    def menu(self):
        raise NotImplementedError


class Menu:
    options = {
        "1": "Generador de números aleatorios",
        "2": "Generador de variables aleatorias",
        "3": "Realizar pruebas estadísticas de aleatoriedad",
    }

    def __init__(self):
        self.random_number_generator_menu = _RandomNumberGeneratorMenu()
        self.random_variable_generator_menu = _RandomVariableGeneratorMenu()
        self.randomness_test_menu = _RandomnessTestMenu()

    def menu(self):
        option = _get_option("Menú principal", self.options)
        if option == 1:
            self.random_number_generator_menu.menu()
        elif option == 2:
            self.random_variable_generator_menu.menu()
        elif option == 3:
            self.randomness_test_menu.menu()
