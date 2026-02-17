import numpy as np
import random
import os


class Robot:
    def __init__(self, pos=[0, 0], orientation = 0.0):
        self.pos = np.array(pos)
        self.orientation = orientation


    def _find_angle_and_dist(self, target):
        """
        Calcule l'angle duquel on doit tourner et la distance à avancer pour atteindre la cible
        Retourne l'angle alpha en radian et la distance
        """
        robot_to_target = target - self.pos

        target_theta = np.arctan2(target[1] - self.pos[1], target[0] - self.pos[0])
        alpha = target_theta - self.orientation

        distance = np.linalg.norm(robot_to_target)

        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi #On prends l'angle minimal
    
        return alpha, distance


    def instructions_to_script(self, alpha, distance, path):
        """
        Transforme un angle en radiant et une distance en une instruction et l'écrit dans le fichier cible
        """
        rotation_instruct = None
        distance_instruct = None

        alpha = np.degrees(alpha)

        if np.abs(alpha) > 0.01:
            rotation_instruct = f"TURN {alpha:.2f}\n"
        
        if np.abs(distance) > 0.01:
            distance_instruct = f"GO {distance:.5f}\n" # On arrondira plus tard
        
        with open(path, "a", encoding="utf-8") as f:
            if rotation_instruct:
                f.write(rotation_instruct)
            if distance_instruct:
                f.write(distance_instruct)
        
        #print("Done")

    def catch_all_cylinders(self, cylinders, path):

        pos_cylinders = cylinders[:, :2]

        #On vide le script
        if os.path.exists(path):
            os.remove(path)

        for cylinder in pos_cylinders:
            alpha, distance = self._find_angle_and_dist(cylinder)

            self.instructions_to_script(alpha, distance, path)

            #On update notre position
            self.pos = np.array(cylinder, dtype=float)

            #On update l'orientation et on la normalise entre -pi et pi
            self.orientation += alpha
            self.orientation = (self.orientation + np.pi) % (2 * np.pi) - np.pi

        with open(path, "a", encoding="utf-8") as f:
            f.write("FINISH")
        
        print(f"DONE : script généré dans '{path}'")

    
    def _sort_cylinders(self, cylinders):

        target_pos = cylinders[:, :2]
        points = cylinders[:, 2]
        mass = cylinders[:, 3]

        distances = np.linalg.norm(target_pos - self.pos, axis=1)

        scores = points / (distances * mass + 1e-9) # On évite la division par zéro au cas où

        indices_tries = np.argsort(scores)[::-1]

        return cylinders[indices_tries]
    


class GeneticSolver:

    def __init__(self, cylinders, robot_params, pop_size=100, mutation_rate=0.1, generations=100, tournament_selection=3):

        self.cylinders = cylinders
        self.params = robot_params
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.k = tournament_selection

        self.num_cylinders = len(cylinders)
        self.population = []

        
    
    def _initialize_population(self):
        """
        Création d'une population initiale de permutation aléatoire (trajet)
        """

        self.population = []

        bases_indices = list(range(self.num_cylinders))
        for _ in range(self.pop_size):
            individual = random.sample(bases_indices, len(bases_indices))
            self.population.append(individual)

    def _calculate_fitness(self, individual):
        """
        Simule le parcours et retourne le score total
        Un individu est une liste d'indice [1, 4, 5, 7, 8, ...]
        """

        current_pos = np.array([0.0, 0.0])
        current_mass = 0.0
        current_time = 0.0
        current_fuel = 0.0
        total_reward = 0.0

        V0, a = self.params["V0"], self.params["a"]
        b, b0 = self.params["b"], self.params["b0"]
        Qmax, Tmax = self.params["Qmax"], self.params["Tmax"]

        for cyl_idx in individual:
            
            target = self.cylinders[cyl_idx]
            target_pos = target[:2]
            reward = target[3]
            cyl_mass = target[2]

            dist = np.linalg.norm(target_pos - current_pos)

            v_max = V0 * np.exp(-a * current_mass)

            consommation_rate = b * current_mass + b0

            travel_time = dist / v_max
            fuel_cost = consommation_rate * dist

            #On s'arrete si on n'a plus de carburant ou de temps 
            if (current_time + travel_time > Tmax) or (current_fuel + fuel_cost > Qmax):
                break

            current_time += travel_time
            current_fuel += fuel_cost
            current_pos = target_pos

            total_reward += reward
            current_mass += cyl_mass

        return total_reward
    
    def _calculate_multi_fitness(self, individual):
        """
        Priorité au gain puis au fuel puis au temps
        """
        current_pos = np.array([0.0, 0.0])
        current_mass = 0.0
        current_time = 0.0
        current_fuel = 0.0
        total_reward = 0.0

        V0, a = self.params["V0"], self.params["a"]
        b, b0 = self.params["b"], self.params["b0"]
        Qmax, Tmax = self.params["Qmax"], self.params["Tmax"]

        for cyl_idx in individual:
            
            target = self.cylinders[cyl_idx]
            target_pos = target[:2]
            reward = target[3]
            cyl_mass = target[2]

            dist = np.linalg.norm(target_pos - current_pos)

            v_max = V0 * np.exp(-a * current_mass)

            consommation_rate = b * current_mass + b0

            travel_time = dist / v_max
            fuel_cost = consommation_rate * dist

            #On s'arrete si on n'a plus de carburant ou de temps 
            if (current_time + travel_time > Tmax) or (current_fuel + fuel_cost > Qmax):
                break

            current_time += travel_time
            current_fuel += fuel_cost
            current_pos = target_pos

            total_reward += reward
            current_mass += cyl_mass

        fuel_saved = Qmax - current_fuel
        time_saved = Tmax - current_time

        fitness = (total_reward * 1e10) + (fuel_saved * 1e5) + time_saved    

        return fitness

    def _tournament_selection(self):
        """
        Sélectionne le meilleur individu parmi k choisi au pif
        """

        selection = random.sample(self.population, self.k)
        best = max(selection, key=self._calculate_multi_fitness)

        return best
    

    def _pmx_crossover(self, parent1, parent2):
        size = len(parent1)
        child = [None] * size
        start, end = sorted(random.sample(range(size), 2))

        # 1. Copie du segment
        child[start:end] = parent1[start:end]

        # 2. Placement des éléments restants de parent2
        for i in range(start, end):
            if parent2[i] not in child[start:end]:
                curr_val = parent2[i]
                # On cherche où placer cette valeur qui a été "éjectée"
                # par le segment de parent1
                idx_in_p1 = i
                while start <= idx_in_p1 < end:
                    val_in_p1 = parent1[idx_in_p1]
                    # Où se trouve cette valeur de p1 dans p2 ?
                    idx_in_p1 = parent2.index(val_in_p1)
                child[idx_in_p1] = curr_val

        # 3. Remplissage du reste
        for i in range(size):
            if child[i] is None:
                child[i] = parent2[i]
        return child

    # def _pmx_crossover(self, parent1, parent2):
    #     """
    #     Implémentation du Partially Mapped Crossover
    #     """

    #     size = len(parent1)
    #     child = [None] * size

    #     start, end = sorted(random.sample(range(size), 2))

    #     child[start:end] = parent1[start:end]

    #     mapping = {}

    #     for i in range(start, end):
    #         mapping[parent1[i]] = parent2[i]
    #         mapping[parent2[i]] = parent1[i]

    #     for i in range(size):
    #         if start <= i < end:
    #             continue

    #         curr_val = parent2[i]

    #         while curr_val == mapping[curr_val]:
    #             curr_val = mapping[curr_val]
            
    #         child[i] = curr_val
        
    #     return child
    
    def _mutate(self, individual):
        """
        Échange deux cylindre au pif avec une proba de mutaion_rate
        """
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            #On swap
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

        return individual
    

    def solve(self):
        self._initialize_population()

        best_overall = None
        best_score = -1

        for gen in range(self.generations):
            new_population = []

            # On fait de l'élitsime parce que j'ai oublié les autres

            current_best = max(self.population, key=self._calculate_multi_fitness) #Je teste fitness narmol au début
            current_score = self._calculate_multi_fitness(current_best)

            if current_score > best_score:
                best_score = current_score
                best_overall = current_best[:] #On fait une copie 
                print(f"Génération {gen} : Nouveau record = {best_score}")

            new_population.append(current_best) # On garde le meilleur of course

            while len(new_population) < self.pop_size:
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()

                child = self._pmx_crossover(parent1, parent2)
                child = self._mutate(child)

                new_population.append(child)

            self.population = new_population
        
        return best_overall, best_score

