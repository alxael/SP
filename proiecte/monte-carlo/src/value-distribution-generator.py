import csv
import numpy as np

from main import Roulette


def generate_roulette_probability_distribution_normal(mu: float, sigma: float):
    rng = np.random.default_rng()
    values = rng.normal(mu, sigma, Roulette.ALL_VALUES_COUNT)
    values_sum = sum(values)
    values_normalized = [float(value / values_sum) for value in values]
    return values_normalized


distribution_params = [(1, 0.05), (1, 0.1), (1, 0.15), (1, 0.2)]

distributions_data = [['Roulette value', 'Correct probability'] + [f"Fixed probability {index + 1}" for index in range(len(distribution_params))]]
distributions_data += [[] for index in range(Roulette.ALL_VALUES_COUNT)]

for index in range(Roulette.ALL_VALUES_COUNT):
    distributions_data[index + 1].append(index)
    distributions_data[index + 1].append(float(1 / Roulette.ALL_VALUES_COUNT))

for distribution_param in distribution_params:
    distribution = generate_roulette_probability_distribution_normal(distribution_param[0], distribution_param[1])
    for index, value in enumerate(distribution):
        distributions_data[index + 1].append(value)

distributions_file = open("../data/distributions.csv", "w", newline='')
distributions_file_writer = csv.writer(distributions_file, delimiter=',')
distributions_file_writer.writerows(distributions_data)