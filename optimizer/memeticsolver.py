import numpy as np
import random
from multiprocessing import Pool, cpu_count
from optimizer import utils_solver
from optimizer.geneticsolver import GeneticSolver

def _memetic_worker(args):

    individuals, cylinders, params, search_depth = args

    current_path = individuals[:]
    current_score = utils_solver.calculate_fitness_collision(current_path, cylinders, params, margin=0.8)

    T = 10.0 #On optimise localement donc température assez faible
    alpha = 0.90 #On refroidit assez vite, on a pas le temps pour ces conneries

    for _ in range(search_depth):
        neighbor = current_path[:]
        i, j = random.sample(range(len(current_path)), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

        new_score = utils_solver.calculate_fitness_collision(neighbor, cylinders, params, margin=0.8)

        if new_score > current_score:
            current_path = neighbor
            current_score = new_score

        T *= alpha
        if T < 0.1:
            break
    
    return current_path, current_score


class MemeticSolver(GeneticSolver):
    """
    Algo génétique mais on applique à chaque individu une LocalSearch (en vrai un SA à faible température)
    afin d'optimiser chaque indivu de la pop
    """
    def __init__(self, cylinders, robot_params, pop_size=100, mutation_rate=0.1, generations=100, tournament_k=3, local_search_depth=50):
        super().__init__(cylinders, robot_params, pop_size, mutation_rate, generations, tournament_k)
        self.local_search_depth = local_search_depth

    def solve(self, processes=None):
        
        if processes is None:
            processes = cpu_count()

        print(f"Début de l'optimisation sur {processes} coeurs")

        self._initialize_population()
        best_overall = None
        best_score = -float("inf")

        with Pool(processes=processes) as pool:
            
            for gen in range(self.generations):

                tasks = [(ind, self.cylinders, self.params, self.local_search_depth) for ind in self.population]

                results = pool.map(_memetic_worker, tasks)

                optimized_pop = []
                scores = []

                for path, score in results:
                    optimized_pop.append(path)
                    scores.append(score)
                
                self.population = optimized_pop

                scores_np = np.array(scores)
                max_idx = np.argmax(scores_np)

                if scores_np[max_idx] > best_score:
                    best_overall = self.population[max_idx][:]
                    best_score = scores_np[max_idx]
                    print(f"Gen {gen:03d} | Nouveau record : {int(best_score/1e10)} pts (Fitness : {best_score})")

                new_pop = []
                while len(new_pop) < self.pop_size:
                    parent1 = self._tournament_selection(scores)
                    parent2 = self._tournament_selection(scores)

                    child = self._pmx_crossover(parent1, parent2)
                    child = self._mutate(child)

                    new_pop.append(child)

                self.population = new_pop
        
        return best_overall, best_score
    
    def __str__(self):
        return(f"MemeticSolver(pop_size={self.pop_size}, mutation_rate={self.mutation_rate}, generations={self.generations}, k={self.k}, local_search_depth={self.local_search_depth})")