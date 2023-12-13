import numpy as np
import matplotlib.pyplot as plt
def evaluate_artistic_value(individual):
    return np.sum(individual)
population_size = 20
individual_length = 100
mutation_rate = 0.1
generations = 50
population = np.random.rand(population_size, individual_length)
for generation in range(generations):
    fitness_values = np.array([evaluate_artistic_value(individual) for individual in population])
    selected_indices = np.random.choice(population_size, size=population_size // 2, p=fitness_values / fitness_values.sum())
    crossover_point = np.random.randint(1, individual_length)
    offspring = np.zeros_like(population)

    for i in range(0, population_size, 2):
        parent1 = population[selected_indices[i]]
        parent2 = population[selected_indices[i + 1]]

        offspring[i, :crossover_point] = parent1[:crossover_point]
        offspring[i, crossover_point:] = parent2[crossover_point:]

        offspring[i + 1, :crossover_point] = parent2[:crossover_point]
        offspring[i + 1, crossover_point:] = parent1[crossover_point:]
    mutation_mask = np.random.rand(population_size, individual_length) < mutation_rate
    mutation_values = np.random.rand(population_size, individual_length)
    offspring = np.where(mutation_mask, mutation_values, offspring)
    population = offspring
best_individual = population[np.argmax(fitness_values)]
plt.plot(best_individual)
plt.title('Generated Art')
plt.show()
