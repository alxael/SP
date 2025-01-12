import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
from collections.abc import Iterable

# Global constants

error_margin = 1e-3

# Roulette base class


class RouletteBet():
    def __init__(self, identifier: str, values: Iterable[float], payout: float):
        self.identifier = identifier
        self.values = values
        self.payout = payout

    def evaluate_action(self, value):
        return (value in self.values)

    def get_expected_probability(self):
        return float(len(self.values) / 37)


class Roulette():
    ALL_VALUES_COUNT = 37
    ALL_VALUES = [value for value in range(37)]

    SINGLE_BETS = [RouletteBet(f"straight_{index}", [index], 36) for index in range(ALL_VALUES_COUNT)]

    ROW_BETS = [RouletteBet(f"row_{index + 1}", [index * 3 + 1, index * 3 + 2, index * 3 + 3], 12) for index in range(12)]

    FIRST_COLUMN_BET = RouletteBet('column_1', range(1, ALL_VALUES_COUNT, 3), 3)
    SECOND_COLUMN_BET = RouletteBet('column_2', range(2, ALL_VALUES_COUNT, 3), 3)
    THIRD_COLUMN_BET = RouletteBet('column_3', range(3, ALL_VALUES_COUNT, 3), 3)
    COLUMN_BETS = [FIRST_COLUMN_BET, SECOND_COLUMN_BET, THIRD_COLUMN_BET]

    FIRST_DOZEN_BET = RouletteBet('dozen_1', range(1, 13), 3)
    SECOND_DOZEN_BET = RouletteBet('dozen_2', range(13, 25), 3)
    THIRD_DOZEN_BET = RouletteBet('dozen_3', range(25, ALL_VALUES_COUNT), 3)
    DOZEN_BETS = [FIRST_DOZEN_BET, SECOND_DOZEN_BET, THIRD_DOZEN_BET]

    RED_BET = RouletteBet('reds', [32, 19, 21, 25, 34, 27, 36, 30, 23, 5, 16, 1, 14, 9, 18, 7, 12, 3], 2)
    BLACK_BET = RouletteBet('black', [15, 4, 2, 17, 6, 13, 11, 8, 10, 24, 33, 20, 31, 22, 29, 28, 35, 26], 2)
    COLOR_BETS = [RED_BET, BLACK_BET]

    EVEN_BET = RouletteBet('even', range(2, ALL_VALUES_COUNT, 2), 2)
    ODD_BET = RouletteBet('odd', range(1, ALL_VALUES_COUNT, 2), 2)
    PARITY_BETS = [EVEN_BET, ODD_BET]

    FIRST_HALF_BET = RouletteBet('half_1', range(1, 19), 2)
    SECOND_HALF_BET = RouletteBet('half_2', range(19, ALL_VALUES_COUNT), 2)
    HALF_BETS = [FIRST_HALF_BET, SECOND_HALF_BET]

    ALL_BETS = SINGLE_BETS + ROW_BETS + COLUMN_BETS + DOZEN_BETS + COLOR_BETS + PARITY_BETS + HALF_BETS


# Distribution utilities

def generate_roulette_probability_distribution_normal(mu: float, sigma: float):
    rng = np.random.default_rng()
    values = rng.normal(mu, sigma, Roulette.ALL_VALUES_COUNT)
    values_sum = sum(values)
    values_normalized = [float(value / values_sum) for value in values]
    return values_normalized


def get_probability_distribution_data(probability_distribution: Iterable[float]):
    mean = sum([index * value for index, value in enumerate(probability_distribution)])
    squared_mean = sum([(index ** 2) * value for index, value in enumerate(probability_distribution)])
    variance = squared_mean - mean ** 2
    return (mean, variance)


def evaluate_number_against_probability_distribution(value: float, probability_distribution: Iterable[float]):
    probability_sum = 0
    for index, probability in enumerate(probability_distribution):
        probability_sum += probability
        if probability_sum > value:
            return index
    return len(probability_distribution) - 1


def get_expected_probability_for_strategies(betting_strategies: Iterable[RouletteBet]):
    return sum([betting_strategy.get_expected_probability() for betting_strategy in betting_strategies]) / len(betting_strategies)


# Generate probability distribution data

def generate_roulette_probability_distribution_data():
    distribution_params = [(1, 0.05), (1, 0.1), (1, 0.15), (1, 0.2)]

    usable_data = [[float(1 / Roulette.ALL_VALUES_COUNT) for _index in range(Roulette.ALL_VALUES_COUNT)]]

    distributions_raw_data = [['Roulette value', 'Correct probability'] + [f"Fixed probability {index + 1}" for index in range(len(distribution_params))]]
    distributions_raw_data += [[] for _index in range(Roulette.ALL_VALUES_COUNT)]

    for index in range(Roulette.ALL_VALUES_COUNT):
        distributions_raw_data[index + 1].append(index)
        distributions_raw_data[index + 1].append(float(1 / Roulette.ALL_VALUES_COUNT))

    for distribution_param in distribution_params:
        distribution = generate_roulette_probability_distribution_normal(distribution_param[0], distribution_param[1])
        usable_data.append(distribution)
        for index, value in enumerate(distribution):
            distributions_raw_data[index + 1].append(value)

    distributions_raw_file = open("../data/distributions-raw.csv", "w", newline='')
    distributions_raw_file_writer = csv.writer(distributions_raw_file, delimiter=',')
    distributions_raw_file_writer.writerows(distributions_raw_data)

    distributions_data = [['Probability distribution', 'Weighted mean', 'Variance']]
    for distribution_index, distribution in enumerate(usable_data):
        mean, variance = get_probability_distribution_data(distribution)
        distributions_data.append([distribution_index + 1, mean, variance])

    distributions_file = open("../data/distributions.csv", "w", newline='')
    distributions_file_writer = csv.writer(distributions_file, delimiter=',')
    distributions_file_writer.writerows(distributions_data)

    return usable_data

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

# Monte Carlo simulation


def simulate_roulette_game(sample_size: int, betting_strategies: Iterable[RouletteBet], probability_distribution: Iterable[float], results: Iterable, row_index: int, column_index: int):
    favorable_outcomes = 0
    for _sample_index in range(sample_size):
        betting_strategy = np.random.choice(betting_strategies)
        sample_value = np.random.rand()
        roulette_value = evaluate_number_against_probability_distribution(sample_value, probability_distribution)
        if betting_strategy.evaluate_action(roulette_value):
            favorable_outcomes += 1
    results[row_index][column_index] = float(favorable_outcomes / sample_size)

    # Chebyshev
    expected_probability = get_expected_probability_for_strategies(betting_strategies)
    results[row_index][column_index + 1] = (expected_probability * (1 - expected_probability)) / (error_margin * sample_size)

    # Chernoff
    exponent = float((-2) * sample_size * (error_margin ** 2))
    results[row_index][column_index + 2] = 2 * math.exp(exponent)


def initialize_simulation_report_data(sample_sizes: Iterable[int], betting_strategies_list: Iterable, inequalities: Iterable[str]):
    report_data = [['Sample size'], ['Expected']]
    for strategy_name, betting_strategies in betting_strategies_list.items():
        report_data[0].append(strategy_name)
        report_data[1].append(get_expected_probability_for_strategies(betting_strategies))

        for inequality in inequalities:
            report_data[0].append(f"{strategy_name} {inequality}")
            report_data[1].append("-")

    for sample_size in sample_sizes:
        report_data.append([sample_size] + [0] * len(betting_strategies_list) * (len(inequalities) + 1))
    return report_data


# Main part of the program


print("Generating distributions!")
distributions_data = generate_roulette_probability_distribution_data()

DISTRIBUTION_PLOT_SAMPLE_SIZE = int(1e6)

print("Generating distribution plots!")
for distribution_data_index in range(len(distributions_data)):
    generate_probability_distribution_plot(distribution_data_index + 1, DISTRIBUTION_PLOT_SAMPLE_SIZE, distributions_data[distribution_data_index])


sample_sizes = [1000 * index for index in range(1, 101)]
betting_strategies_list = {
    'Single': Roulette.SINGLE_BETS,
    'Row': Roulette.ROW_BETS,
    'Column': Roulette.COLUMN_BETS,
    'Dozen': Roulette.DOZEN_BETS,
    'Color': Roulette.COLOR_BETS,
    'Parity': Roulette.PARITY_BETS,
    'Half': Roulette.HALF_BETS,
    'All': Roulette.ALL_BETS
}

print("Starting Monte Carlo simulation!")
for distribution_data_index, distribution_data in enumerate(distributions_data):
    inequalities = ['Chebyshev', 'Chernoff']
    distribution_report_data = initialize_simulation_report_data(sample_sizes, betting_strategies_list, inequalities)
    distribution_report_data_inequalities = len(inequalities)
    distribution_report_data_header_size = 2
    threads = [[None] * (len(betting_strategies_list) + 1)] * (len(sample_sizes) + 1)

    for sample_size_index, sample_size in enumerate(sample_sizes):
        betting_strategies_index = 0
        for betting_strategies_name, betting_strategies in betting_strategies_list.items():
            print(f"Starting Monte Carlo simulation for distribution {distribution_data_index + 1} - {sample_size} values - {betting_strategies_name} strategy")
            threads[sample_size_index][betting_strategies_index] = Thread(
                target=simulate_roulette_game,
                args=(
                    sample_size,
                    betting_strategies,
                    distribution_data,
                    distribution_report_data,
                    sample_size_index + distribution_report_data_header_size,
                    betting_strategies_index * (distribution_report_data_inequalities + 1) + 1
                )
            )
            threads[sample_size_index][betting_strategies_index].start()
            betting_strategies_index += 1

    for sample_size_index, sample_size in enumerate(sample_sizes):
        betting_strategies_index = 0
        for betting_strategies_name, betting_strategies in betting_strategies_list.items():
            threads[sample_size_index][betting_strategies_index].join()
            betting_strategies_index += 1

    distribution_report_file = open(f"../data/distribution-{distribution_data_index + 1}-report.csv", "w", newline='')
    distribution_report_file_writer = csv.writer(distribution_report_file, delimiter=',')
    distribution_report_file_writer.writerows(distribution_report_data)
