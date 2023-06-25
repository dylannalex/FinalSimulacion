import numpy as np
from os import system
from src import random_number
from src import random_variable
from src import randomness_test


class GeneratorMenu:
    def next(self):
        return np.random.rand()


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


class RandomNumberGeneratorMenu:
    options = {
        "1": "Generador congruencial mixto",
        "2": "Generador congruencial multiplicativo",
        "3": "Generador de cuadrados medios",
        "4": "Salir",
    }

    def menu(self):
        option = _get_option("Generador de números aleatorios", self.options)

        if option == 4:
            return

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


class RandomnessTestMenu:
    options = {
        "1": "Prueba de Kolmogorov-Smirnov",
        "2": "Prueba de Chi-Cuadrado",
        "3": "Prueba de Corridas",
        "4": "Salir",
    }

    def menu(self):
        option = _get_option("Pruebas de aleatoriedad", self.options)

        if option == 4:
            return

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


class RandomVariableGeneratorMenu:
    options = {
        "1": "Variable aleatoria discreta",
        "2": "Variable aleatoria discreta uniforme",
        "3": "Variable aleatoria continua uniforme",
        "4": "Salir",
    }

    def menu(self):
        option = _get_option("Pruebas de aleatoriedad", self.options)

        if option == 4:
            return

        total_random_variables = int(
            input("Ingrese la cantidad de variables aleatorias a generar: ")
        )

        # Get Generator
        if option == 1:
            print("\nEjemplo de valores ingresados: variable1 variable2 variable3")
            values = input(
                "Ingrese los valores de la variable separados por un espacio: "
            )
            values = values.split(" ")
            print("\nEjemplo de pesos ingresados: 1 2 0.5")
            weights = input(
                "Ingrese los pesos de la variable separados por un espacio: "
            )
            weights = [float(w) for w in weights.split(" ")]
            rv_generator = random_variable.DiscreteRandomVariable(
                GeneratorMenu(), values, weights
            )

        if option == 2:
            print("\nEjemplo de valores ingresados: variable1 variable2 variable3")
            values = input(
                "Ingrese los valores de la variable separados por un espacio: "
            )
            values = values.split(" ")
            rv_generator = random_variable.UniformDiscreteRandomVariable(
                GeneratorMenu(),
                values,
            )

        if option == 3:
            a = int(input("Ingrese el valor de 'a': "))
            b = int(input("Ingrese el valor de 'b': "))
            rv_generator = random_variable.UniformContinuousRandomVariable(
                GeneratorMenu(), a, b
            )

        random_variables = [rv_generator.next() for _ in range(total_random_variables)]
        print(f"Las variables aleatorias generadas son:\n{random_variables}")
