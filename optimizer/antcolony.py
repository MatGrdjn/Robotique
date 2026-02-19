import numpy as np
import random
from multiprocessing import Pool, cpu_count
from optimizer import utils_solver

def _ant_worker(args):
    """
    Simule le parcours d'une seule fourmi
    Construit sa solution en utilisant les phéromène et une heuristique
    """

    seed, cylinders, params, pheromones, aco_params = args

    random.seed(seed)
    np.random.seed(seed)

    alpha = aco_params["alpha"] #Poids des phéromones
    beta = aco_params["beta"] # Poid de l'heuristique

    num_cylinders = len(cylinders)
    path = []
    visited_mask = [False] * num_cylinders

    current_pos = np.array([0.0, 0.0])
    current_mass = 0.0
    current_time = 0.0
    current_fuel = 0.0

    current_idx = -1 

    V0, a = params["V0"], params["a"]
    b, b0 = params["b"], params ["b0"]
    Qmax, Tmax = params["Qmax"], params["Tmax"]

    R_robot, R_cyl = params["R_ROBOT"], params["R_CYL"]
    COLLISIONS_DIST = R_robot + R_cyl - 0.9

    cylinders = np.asarray(cylinders, dtype=np.float64)
    while True:

        candidates = [i for i in range(num_cylinders) if not visited_mask[i]]

        if not candidates:
            break

        probabilities = []
        feasible_candidates = []

        for next_idx in candidates:
            target = cylinders[next_idx]
            target_center = target[:2]
            reward = target[3]
            mass = target[2]

            _, dist = utils_solver.get_contact_position(current_pos, target_center, R_robot, R_cyl)
            if dist < 0.1:
                dist = 0.1

            v_max = V0 * np.exp(-a * current_mass)
            q = b * current_mass + b0
            dt = dist / v_max
            df = q * dist

            if (current_fuel + df > Qmax) or (current_time + dt > Tmax):
                continue

            eta = reward / (dist * (1 + mass * 2))

            if current_idx == -11:
                tau = 1.0
            else:
                tau = pheromones[current_idx, next_idx]
            
            prob = (tau ** alpha) * (eta ** beta)

            probabilities.append(prob)
            feasible_candidates.append(next_idx)

        if not feasible_candidates:
            break 

        total_prob = sum(probabilities)
        if total_prob == 0:
            next_step = random.choice(feasible_candidates)
        else:
            # on normalise
            probabilities = [p / total_prob for p in probabilities]
            next_step = np.random.choice(feasible_candidates, p=probabilities)
        
        target = cylinders[next_step]
        target_center = target[:2]
        stop_pos, dist = utils_solver.get_contact_position(current_pos, target_center, R_robot, R_cyl)

        v_max = V0 * np.exp(-a * current_mass)
        q = b * current_mass + b0

        dt = dist / v_max
        df = q * dist

        start_sweep = current_pos
        end_sweep = stop_pos

        for other_idx in range(num_cylinders):
            if not visited_mask[other_idx] and other_idx != next_step:
                other_pos = cylinders[other_idx][:2]

                if utils_solver.distance_point_segment(other_pos, start_sweep, end_sweep) <= COLLISIONS_DIST:
                    visited_mask[other_idx] = True
                    current_mass += cylinders[other_idx][2]

        current_time += dt
        current_fuel += df
        current_pos = stop_pos

        if not visited_mask[next_step]:
            current_mass += target[2]
            visited_mask[next_step] = True
        
        current_idx = next_step
        path.append(next_step)

    
    final_score = utils_solver.calculate_fitness_collision(path, cylinders, params, 0.9)

    return path, final_score


class AntColonySolver:

    def __init__(self, cylinders, robot_params, n_ants=30, n_iterations=50, alpha=1.0, beta=2.0, evaporation=0.1, Q=100.0):
        """
        :param n_ants: Nombre de fourmis par itération
        :param n_iterations: Nombre de générations
        :param alpha: Importance des phéromones
        :param beta: Importance de l'heuristique
        :param evaporation: Vitesse d'oubli des phéromones
        :param Q: Quantité de phéromone déposée par les fourmis
        """

        self.cylinders = cylinders
        self.params = robot_params
        self.n_ants = n_ants
        self.n_iterations = n_iterations

        self.aco_config = {
            "alpha" : alpha,
            "beta" : beta
        }

        self.evaporation = evaporation
        self.Q = Q

        self.num_cylinders = len(cylinders)

        #matrice de phéromones
        self.pheromones = np.ones((self.num_cylinders, self.num_cylinders)) * 0.1


    def solve(self, processes=None):

        if processes is None:
            processes = cpu_count()
            
        print(f"Début de l'optimisation sur {processes} coeurs")

        best_overall = []
        best_score = -float("inf")

        with Pool(processes=processes) as pool:

            for iteration in range(self.n_iterations):

                current_pheromones = self.pheromones.copy()

                tasks = []

                for _ in range(self.n_ants):
                    seed = random.randint(0, 1_000_000)
                    tasks.append((seed, self.cylinders, self.params, current_pheromones, self.aco_config))

                results = pool.map(_ant_worker, tasks)

                self.pheromones *= (1.0 - self.evaporation)
                self.pheromones = np.clip(self.pheromones, 0.01, 1000.0)

                iteration_best_score = -1

                for path, score in results:

                    if score > iteration_best_score:
                        iteration_best_score = score

                    if score > best_score:
                        best_score = score
                        best_overall = path[:]
                        print(f"Gen {iteration:03d} | Nouveau record : {int(best_score/1e10)} pts (Fitness : {best_score})")

                results.sort(key=lambda x : x[1], reverse=True)
                top_ants = results[:max(1, int(self.n_ants * 0.25))] #Seule les 25% des meilleures fourmis déposent des phéromones

                for path, score in top_ants:
                    deposit = self.Q

                    for k in range(len(path) - 1):
                        u, v = path[k], path[k+1]
                        self.pheromones[u, v] += deposit
                
                if best_overall:
                    for k in range(len(best_overall) - 1):
                        u, v = best_overall[k], best_overall[k+1]
                        self.pheromones[u, v] += self.Q * 2.0

        return best_overall, best_score
    
    def __str__(self):
        return f"AntColonySolver(n_ants={self.n_ants}, n_iterations={self.n_iterations}, alpha={self.aco_config["alpha"]}, beta={self.aco_config["beta"]}, evaporation={self.evaporation}, Q={self.Q})"