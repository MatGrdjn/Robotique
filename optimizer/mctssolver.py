import numpy as np
import random
import math
from multiprocessing import Pool, cpu_count
from optimizer import utils_solver

def _mcts_worker(args):
    """
    Exécute un MCTS complet sur un coeur
    """

    seed, cylinders, params, mcts_config = args

    random.seed(seed)
    np.random.seed(seed)

    iterations = mcts_config["iterations"]
    exploration_weight = mcts_config["exploration_weight"]
    num_cylinders = len(cylinders)

    root = MCTSNode(path=[], untried_actions=list(range(num_cylinders)), exploration_weight=exploration_weight)

    best_global_path = []
    best_global_score = -1.0

    SCORE_SCALE = 1e11 #Facteur de normalisation pour avoir des scores aux alentours de 1.0, 2.0

    for _ in range(iterations):
        node = root

        while node.is_fully_expanded() and not node.is_terminal():
            node = node.best_child()
        
        if not node.is_terminal():
            node = node.expand()
        
        rollout_path = node.path[:]
        remaining = [i for i in range(num_cylinders) if i not in rollout_path]

        #50% d'aléatoire et 50% d'heuristique
        if random.random() < 0.5:
            random.shuffle(remaining)
            rollout_path += remaining
        else:
            # on utilise l'heuristique greedy mais allégée
            current_p = cylinders[rollout_path][-1][:2] if rollout_path else np.array([0, 0])
            while remaining:
                best_r = min(remaining, key=lambda x : np.linalg.norm(cylinders[x][:2] - current_p))
                rollout_path.append(best_r)
                remaining.remove(best_r)
                current_p = cylinders[best_r][:2]
        
        raw_score = utils_solver.calculate_fitness_collision(rollout_path, cylinders, params, margin=0.8)

        if raw_score > best_global_score:
            best_global_score = raw_score
            best_global_path = rollout_path[:]
        
        normalized_score = raw_score / SCORE_SCALE

        temp_node = node
        while temp_node is not None:
            temp_node.visits += 1
            temp_node.score_sum += normalized_score

            if normalized_score > temp_node.best_normalized_score: # pour la version qui utilise le record absolu
                temp_node.best_normalized_score = normalized_score

            if raw_score > temp_node.best_score_found:
                temp_node.best_score_found = raw_score
            
            temp_node = temp_node.parent
    
    return best_global_path, best_global_score
            

class MCTSNode:

    def __init__(self, path, untried_actions, parent=None, exploration_weight=1.414):
        self.path = path
        self.untried_actions = untried_actions
        self.parent = parent
        self.children = []
        self.exploration_weight = exploration_weight

        self.visits = 0
        self.score_sum = 0.0 # somme des scores mais normalisés
        self.best_normalized_score = -1.0
        self.best_score_found = -1.0

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def is_terminal(self):
        return self.is_fully_expanded() and len(self.children) == 0
    
    def expand(self):

        idx = random.randrange(len(self.untried_actions))
        action = self.untried_actions[idx]

        new_path = self.path + [action]

        new_untried = self.untried_actions[:]

        child_node = MCTSNode(new_path, new_untried, parent=self)
        self.children.append(child_node)

        return child_node
    
    def best_child(self):
        """
        On sélectionne le meilleur enfant avec la formule UCB1
        """

        best_score = -float("inf")
        best_c = None

        for child in self.children:

            if child.visits == 0:
                return child
            
            #exploitation = child.score_sum / child.visits #moyenne
            exploitation = child.best_normalized_score #record absolu
            exploration = self.exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            ucb1_score = exploitation + exploration

            if ucb1_score > best_score:
                best_score = ucb1_score
                best_c = child
        
        return best_c
    

class MCTSSolver:

    def __init__(self, cylinders, robot_params, iterations=5000, exploration_weight=1.414):
        self.cylinders = cylinders
        self.params = robot_params
        self.mcts_config = {
            "iterations" : iterations,
            "exploration_weight" : exploration_weight
        }

    
    def solve(self, processes=None):

        if processes is None:
            processes = cpu_count()

        print(f"Début de l'optimisation sur {processes} coeurs")

        tasks = []
        for _ in range(processes):
            seed = random.randint(0, 1_000_000)
            tasks.append((seed, self.cylinders, self.params, self.mcts_config))

        best_overall = None
        best_score = -float("inf")

        with Pool(processes=processes) as pool:

            results = pool.map(_mcts_worker, tasks)

            for path, score in results:
                if score > best_score:
                    best_score = score
                    best_overall = path
        
        print(f"Meilleur score : {int(best_score / 1e10)} pts (Fitness : {best_score})")

        return best_overall, best_score

    def __str__(self):
        return f"MCTSSolver(iterations={self.mcts_config["iterations"]}, exploration_weight={self.mcts_config["exploration_weight"]})"