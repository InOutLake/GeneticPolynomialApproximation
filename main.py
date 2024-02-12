import numpy as np
import random
import matplotlib.pyplot as plt
import time 

class Individual:
    def __init__(self, *args):
        if len(args) == 1:
            self.n = args[0]
            self.values = np.random.uniform(low=-100.0, high=100.0, size=(self.n,))
        elif len(args) == 2:
            individual1, individual2 = args
            self.n = individual1.n
            divider = random.random()
            self.values = individual1.values*divider + individual2.values*(1-divider)
        random.seed(time.time())
        self.mutation_percent = 0.01*random.randint(1, 20)
        
    def calculate_polynom(self, x):
        powers = np.power(x, np.arange(self.n))
        return np.dot(self.values[:self.n], powers)
    
    def mutate(self):
        multipliers = np.random.uniform(low=1-self.mutation_percent,
                                        high=1+self.mutation_percent,
                                        size=self.values.shape)
        self.values = self.values * multipliers
    

class PolynomialApproximationGeneticEngine:
    def __init__(self, fit_data:dict, population_size:int=100, selection_size:int=15, n=None):
        if n is None:
            self.n = len(fit_data)
        else:
            self.n = n
        self.size = population_size
        self.fit_data = fit_data
        self.selection_size = selection_size
        self.populaion_fitness = None
        self.selected_individuals = None
        self.population = np.array([Individual(self.n) for _ in range(population_size)])

    def fit_population(self):
        deviations = np.zeros(self.size)
        for x in self.fit_data.keys():
            individuals_results = np.array([individual.calculate_polynom(x) for individual in self.population])
            deviations += (individuals_results - self.fit_data[x])**2
        mean_deviations = deviations / self.n
        self.population_fitness = mean_deviations

    def select(self):
        sorted_indices = np.argsort(page.population_fitness)
        sorted_population = page.population[sorted_indices]
        self.selected_individuals = sorted_population[:self.selection_size]

    def mutate_crossover(self, individual1, individual2):
        individual1.mutate()
        individual2.mutate()
        new_individual = Individual(individual1, individual2)
        return new_individual

    def migrate(self):
        self.fit_population()
        self.select()
        new_population = np.array([None for _ in range(self.size)])
        for i in range(self.size-(self.size//10)):
            new_population[i] = self.mutate_crossover(self.selected_individuals[random.randint(0, self.selection_size-1)],
                                                        self.selected_individuals[random.randint(0, self.selection_size-1)])
        for i in range(self.size-(self.size//10), self.size):
            new_population[i] = Individual(self.n)
        self.population = new_population
    

if __name__ == '__main__':
    def y(x):
        return 3*x**4 + 2*x**3 + 5*x**2 + x + 12

    r = [1, 20]
    fit_data = {x: y(x) for x in range(r[0], r[1])}
    n = 3
    population_size=1000
    selection_size=50
    generations = 100
    page = PolynomialApproximationGeneticEngine(fit_data=fit_data, n=6,
                                                population_size=population_size,
                                                selection_size=selection_size)
    saved_instances = []
    for generation in range(generations):
        page.migrate()
        print(f'Generation {generation}...')
        if generation%20 == 0:
            print('Saved instance...')
            saved_instances.append((page.selected_individuals[0], generation))

    page.fit_population()
    sorted_indices = np.argsort(page.population_fitness)
    sorted_population = page.population[sorted_indices]
    sorted_fitness = page.population_fitness[sorted_indices]
    print(f'Results:\nBest individual with fitness={sorted_fitness[0]}\nValues:{sorted_population[0]}')

    x_values = np.linspace(r[0], r[1], 100)

    for saved_instance in saved_instances:
        predicted_values = [saved_instance[0].calculate_polynom(x) for x in x_values]
        plt.plot(x_values, predicted_values, label=f'Generation {saved_instance[1]}')

    best_individual = sorted_population[0]
    best_predicted_values = [best_individual.calculate_polynom(x) for x in x_values]
    plt.plot(x_values, best_predicted_values, linewidth=2, label='Best individual')

    true_values = [y(x) for x in x_values]
    plt.plot(x_values, true_values, linewidth=3, label='True function', color='black')

    plt.legend()
    plt.show()
