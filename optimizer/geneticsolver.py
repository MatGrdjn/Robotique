import random
import numpy as np
from multiprocessing import Pool, cpu_count
from optimizer import utils_solver


def _ga_worker(args):
    """
    Wrapper pour le multiprocessing
    """
    individual, cylinders, params = args

    return utils_solver.calculate_fitness_collision(individual, cylinders, params, margin=0.9) #Fitness collisons, on peut la changer

class GeneticSolver:

    def __init__(self, cylinders, robot_params, pop_size=100, mutation_rate=0.1, generations=100, tournament_k=3):

        self.cylinders = cylinders
        self.params = robot_params
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.k = tournament_k

        self.num_cylinders = len(cylinders)
        self.population = []

    
    def _initialize_population(self):
        """
        Création d'une population initiale de permutation aléatoire et "intelligente" (trajet)
        10% intelligents, 90% aléatoires
        """

        self.population = []
        bases_indices = list(range(self.num_cylinders))

        nums_smart = int(self.pop_size * 0.1)

        for _ in range(nums_smart):
            w = random.uniform(0.5, 4.0)
            smart_ind = utils_solver.generate_greedy_path(self.cylinders, weight_mass=w)
            self.population.append(smart_ind)

        for _ in range(self.pop_size - nums_smart):
            self.population.append(random.sample(bases_indices, len(bases_indices)))



    def _tournament_selection(self, scores):
        """
        Sélectionne le meilleur individu parmi k choisi au pif
        """

        candidates_indices = random.sample(range(self.pop_size), self.k)
        winner_idx = max(candidates_indices, key=lambda i : scores[i])

        return self.population[winner_idx]

    def _pmx_crossover(self, parent1, parent2):
        size = len(parent1)
        child = [None] * size
        start, end = sorted(random.sample(range(size), 2))

        #Copie du segment
        child[start:end] = parent1[start:end]

        #Placement des éléments restants de parent2
        for i in range(start, end):

            if parent2[i] not in child[start:end]:

                curr_val = parent2[i]
                idx_in_p1 = i

                while start <= idx_in_p1 < end:

                    val_in_p1 = parent1[idx_in_p1]
                    # Où se trouve cette valeur de p1 dans p2
                    idx_in_p1 = parent2.index(val_in_p1)

                child[idx_in_p1] = curr_val

        for i in range(size):
            if child[i] is None:
                child[i] = parent2[i]
        return child
    
    def _mutate(self, individual):
        """
        Échange deux cylindre au pif avec une proba de mutaion_rate
        """
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            #On swap
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

        return individual
    

    def solve(self, processes=None):
        if processes is None:
            processes = cpu_count()

        print(f"Début de l'optimisation sur {processes} coeurs")
        
        self._initialize_population()
        best_overall = None
        best_score = -float("inf")

        with Pool(processes=processes) as pool:
            for gen in range(self.generations):

                tasks = [(ind, self.cylinders, self.params) for ind in self.population]
                scores = pool.map(_ga_worker, tasks)

                scores_np = np.array(scores)
                max_idx = np.argmax(scores_np)

                if scores_np[max_idx] > best_score:
                    best_score = scores_np[max_idx]
                    best_overall = self.population[max_idx][:]
                    print(f"Gen {gen:03d} | Nouveau record : {int(best_score/1e10)} pts (Fitness : {best_score})")

                new_pop = []
                new_pop.append(self.population[max_idx]) #On fonctionne par élitisme

                while len(new_pop) < self.pop_size:

                    parent1 = self._tournament_selection(scores)
                    parent2 = self._tournament_selection(scores)

                    child = self._pmx_crossover(parent1, parent2)
                    child = self._mutate(child)
                    new_pop.append(child)
                
                self.population = new_pop
        
        return best_overall, best_score


    def __str__(self) -> str:
        return f"GeneticSolver(pop_size={self.pop_size}, mutation_rate={self.mutation_rate}, generations={self.generations}, k={self.k})"