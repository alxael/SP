import csv
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
from collections.abc import Iterable

# Roulette base class


class RouletteBet():
    def __init__(self, identifier: str, values: Iterable[float], payout: float):
        self.identifier = identifier
        self.values = values
        self.payout = payout

    def evaluate_action(self, value):
        if value in self.values:
            return self.payout
        else:
            return 0


class Roulette():
    ALL_VALUES_COUNT = 37
    ALL_VALUES = [value for value in range(37)]

    SINGLE_BETS = [RouletteBet(f"straight_{index}", [index], 36) for index in range(ALL_VALUES_COUNT)]

    ROW_BETS = [RouletteBet(f"row_{index + 1}", [index * 3 + 1, index * 3 + 2, index * 3 + 3], 12) for index in range(12)]

    FIRST_COLUMN_BET = RouletteBet('column_1', [range(1, ALL_VALUES_COUNT, 3)], 3)
    SECOND_COLUMN_BET = RouletteBet('column_2', [range(2, ALL_VALUES_COUNT, 3)], 3)
    THIRD_COLUMN_BET = RouletteBet('column_3', [range(3, ALL_VALUES_COUNT, 3)], 3)
    COLUMN_BETS = [FIRST_COLUMN_BET, SECOND_COLUMN_BET, THIRD_COLUMN_BET]

    FIRST_DOZEN_BET = RouletteBet('dozen_1', [range(1, 13)], 3)
    SECOND_DOZEN_BET = RouletteBet('dozen_2', [range(13, 25)], 3)
    THIRD_DOZEN_BET = RouletteBet('dozen_3', [range(25, ALL_VALUES_COUNT)], 3)
    DOZEN_BETS = [FIRST_DOZEN_BET, SECOND_DOZEN_BET, THIRD_DOZEN_BET]

    RED_BET = RouletteBet('reds', [32, 19, 21, 25, 34, 27, 36, 30, 23, 5, 16, 1, 14, 9, 18, 7, 12, 3], 2)
    BLACK_BET = RouletteBet('black', [15, 4, 2, 17, 6, 13, 11, 8, 10, 24, 33, 20, 31, 22, 29, 28, 35, 26], 2)
    COLOR_BETS = [RED_BET, BLACK_BET]

    EVEN_BET = RouletteBet('even', [range(0, ALL_VALUES_COUNT, 2)], 2)
    ODD_BET = RouletteBet('odd', [range(1, ALL_VALUES_COUNT, 2)], 2)
    PARITY_BETS = [EVEN_BET, ODD_BET]

    FIRST_HALF_BET = RouletteBet('half_1', [range(0, 19)], 2)
    SECOND_HALF_BET = RouletteBet('half_2', [range(19, ALL_VALUES_COUNT)], 2)
    HALF_BETS = [FIRST_HALF_BET, SECOND_HALF_BET]

    ALL_BETS = SINGLE_BETS + ROW_BETS + COLUMN_BETS + DOZEN_BETS + COLOR_BETS + PARITY_BETS + HALF_BETS


def generate_roulette_probability_distribution_normal(mu: float, sigma: float):
    rng = np.random.default_rng()
    values = rng.normal(mu, sigma, Roulette.ALL_VALUES_COUNT)
    values_sum = sum(values)
    values_normalized = [float(value / values_sum) for value in values]
    return values_normalized


def generate_roulette_probability_distribution_data():
    distribution_params = [(1, 0.05), (1, 0.1), (1, 0.15), (1, 0.2)]

    usable_data = [[float(1 / Roulette.ALL_VALUES_COUNT) for _index in range(Roulette.ALL_VALUES_COUNT)]]

    distributions_data = [['Roulette value', 'Correct probability'] + [f"Fixed probability {index + 1}" for index in range(len(distribution_params))]]
    distributions_data += [[] for _index in range(Roulette.ALL_VALUES_COUNT)]

    for index in range(Roulette.ALL_VALUES_COUNT):
        distributions_data[index + 1].append(index)
        distributions_data[index + 1].append(float(1 / Roulette.ALL_VALUES_COUNT))

    for distribution_param in distribution_params:
        distribution = generate_roulette_probability_distribution_normal(distribution_param[0], distribution_param[1])
        usable_data.append(distribution)
        for index, value in enumerate(distribution):
            distributions_data[index + 1].append(value)

    distributions_file = open("../data/distributions.csv", "w", newline='')
    distributions_file_writer = csv.writer(distributions_file, delimiter=',')
    distributions_file_writer.writerows(distributions_data)

    return usable_data

# Evaluate number against distribution


def evaluate_number_against_probability_distribution(value: float, probability_distribution: Iterable[float]):
    probability_sum = 0
    for index, probability in enumerate(probability_distribution):
        probability_sum += probability
        if probability_sum > value:
            return index
    return len(probability_distribution) - 1

# Distribution plot generation


def generate_probability_distribution_plot(index: int, value_count: int, probability_distribution: Iterable[float]):
    xpoints = [value for value in range(Roulette.ALL_VALUES_COUNT)]
    ypoints = [0 for _value in range(Roulette.ALL_VALUES_COUNT)]
    for _value_index in range(value_count):
        value = np.random.rand()
        ypoints[evaluate_number_against_probability_distribution(value, probability_distribution)] += 1

    plt.rcParams["figure.figsize"] = [20, 5]
    plt.rcParams["figure.autolayout"] = True
    plt.bar(xpoints, ypoints)
    plt.xticks(xpoints)
    plt.savefig(f"../img/probability-distribution-{index}")
    plt.clf()


print("Generating distributions!")
distributions_data = generate_roulette_probability_distribution_data()

print("Generating distribution plots!")
for distribution_data_index in range(len(distributions_data)):
    generate_probability_distribution_plot(distribution_data_index + 1, int(1e6), distributions_data[distribution_data_index])

