import numpy as np
import random
import math
from multiprocessing import Pool, cpu_count
from optimizer import utils_solver

def _sa_worker(args):
    """
    Exécute un recuit simulé sur un coeur du CPU
    """

    seed, cylinders, params, sa_config, initial_solution = args

    random.seed(seed)
    np.random.seed(seed)

    T = sa_config["T_init"]
    T_min = sa_config["T_min"]
    alpha = sa_config["alpha"]
    max_iter = sa_config["max_iter"]

    if initial_solution: #Pour améliorer une solution trouver par un autre solver
        current_path = initial_solution[:] 

        if random.random() < 0.5:
            i, j = random.sample(range(len(cylinders)), 2)
            current_path[i], current_path[j] = current_path[j], current_path[i]

    else:
        if random.random() < 0.5:
            w = random.uniform(0.5, 4)
            current_path = utils_solver.generate_greedy_path(cylinders, weight_mass=w)
        else:
            current_path = list(range(len(cylinders)))
            random.shuffle(current_path)

    current_score = utils_solver.calculate_fitness_collision(current_path, cylinders, params) #On peut cahnger la fitness

    best_path = current_path[:]
    best_score = current_score

    iteration = 0

    while T > T_min and iteration < max_iter:
        
        iteration += 1

        neighbor = current_path[:]

        #Swap simple pour générer un voisin
        if random.random() < 0.5:
            i, j = random.sample(range(len(cylinders)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

        # Ici on fait un Reversal 2-opt, comme ça on décroise les arêtes
        else:
            i, j = sorted(random.sample(range(len(cylinders)), 2))
            neighbor[i:j+1] = reversed(neighbor[i:j+1])
        
        neighbor_score = utils_solver.calculate_fitness_collision(neighbor, cylinders, params) # On peut changer la fitness

        delta = neighbor_score - current_score

        accept = False
        if delta > 0:
            accept = True
        else:
            try:
                #On normalise delta car nos score sont de l'ordre de 10^10
                scaled_delta = delta / 1e9
                probability = math.exp(scaled_delta / T)

                if random.random() < probability:
                    accept = True
            except OverflowError:
                accept = False

        
        if accept:
            current_path = neighbor
            current_score = neighbor_score

            if current_score > best_score:
                # on met à jour le record local
                best_score = current_score
                best_path = current_path[:]

        #On refroidi
        T *= alpha

    return best_path, best_score


class SimulatedAnnealing:

    def __init__(self, cylinders, robot_params, T_init=100.0, T_min=0.01, alpha=0.995, max_iter=50_000):
        """
        :param T_init: Température initiale (C'est la proba d'accepter le désordre), plus c'est haut plus on rebondit
        :param T_min: Température d'arrêt, il fait froid et ça rebondit plus
        :param alpha: la vitesse de refoidissement (0.999 lent, 0.9 rapide)
        :param max_iter: au cas où ça foire et on arrete pas de rebondir partout
        """

        self.cylinders = cylinders
        self.params = robot_params

        self.sa_config = {
            "T_init" : T_init,
            "T_min" : T_min,
            "alpha" : alpha,
            "max_iter" : max_iter
        }

    def solve(self, processes=None, initial_solution=None):
        """
        Lance l'optimisation avec plusieurs départ en parallèle
        """

        if processes is None:
            processes = cpu_count()
        print(f"Début de l'optimisation sur {processes} coeurs")

        tasks = []
        for i in range(processes):
            seed = random.randint(0, 1_000_000) #Une seed par initialisation
            tasks.append((seed, self.cylinders, self.params, self.sa_config, initial_solution))
        
        best_overall = None
        best_score = -float("inf")

        with Pool(processes=processes) as pool:
            
            results = pool.map(_sa_worker, tasks)

            for path, score in results:
                if score > best_score:
                    best_overall = path
                    best_score = score

        print(f"Meilleur score : {int(best_score / 1e10)} pts (Fitness : {best_score})")

        return best_overall, best_score
    
    def __str__(self):
        return f"SimulatedAnnealing(T_init={self.sa_config["T_init"]}, T_min={self.sa_config["T_min"]}, alpha={self.sa_config["alpha"]}, max_iter={self.sa_config["max_iter"]})"