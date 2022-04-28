import random

import numpy as np
from utils import fitness


class Genetic:
    def __init__(self, coords, population_size=100, elite_size=10, mutation_rate=0.01):
        self.coords = coords
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate

    def population_fitness(self, population):
        population_fitness = {}
        for i, individual in enumerate(population):
            # 1/fitness -> change to maximization problem
            population_fitness[i] = 1/fitness(self.coords, individual)

        return {k: v for k, v in sorted(population_fitness.items(), key=lambda item: item[1], reverse=True)}

    def best_solution(self, population):
        population_fitness = list(self.population_fitness(population))
        best_ind = population_fitness[0]
        return population[best_ind]

    def initial_population(self):
        population = []
        # Create initial population
        for i in range(self.population_size):
            solution = np.random.permutation(len(self.coords))
            population.append(solution)
        return population

    def selection(self, population):
        selected = []
        population_f = self.population_fitness(population)
        probability = {}
        sum_f = sum(population_f.values())
        probability_prev = 0.0
        for key, value in population_f.items():
            probability[key] = probability_prev + value / sum_f
            probability_prev = probability[key]
        for i in range(len(population)):
            rand = random.random()
            for key, value in probability.items():
                if rand <= value or i < 10:
                    selected.append(population[key])
                    break
        return selected

    def crossover_population(self, population):
        new_population = []
        for i in range(0, 10):
            new_population.append(population[i])
        for i in range(10, 100):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            first = random.randrange(len(parent1))
            last = random.randrange(len(parent1))
            slice = parent1[first:last]
            new_coords = np.array([x for x in parent2 if x not in slice])
            final = np.append(new_coords, slice)
            new_population.append(final)
        return new_population

    def mutate_population(self, population):
        for i in range(10, 100):
            for j in range(len(population[i])):
                rand = random.random()
                if rand <= self.mutation_rate:
                    swap = random.randrange(0, len(population[i]))
                    population[i][swap], population[i][j] = population[i][j], population[i][swap]
        return population

    def next_generation(self, population):
        selection = self.selection(population)
        children = self.crossover_population(selection)
        next_generation = self.mutate_population(children)
        return next_generation
