import os 
import numpy as np
import utils
from optimizer.geneticsolver import GeneticSolver
from optimizer.simulatedannealing import SimulatedAnnealing
from optimizer.memeticsolver import MemeticSolver
from optimizer.antcolony import AntColonySolver
from optimizer.mctssolver import MCTSSolver
from robot import Robot
import time

CYLINDER_DIAMETER = 1
ROBOT_DIAMETER = 1

base_folder = "C:/challenge"

exe_path = "C:/Users/Utilisateur/Documents/CoursCI2/Challenge/RunTime/challenge-robotique.exe"
exe_path_aerial = "C:/Users/Utilisateur/Documents/CoursCI2/Challenge/Runtime-2026-v2-small-collider/challenge-robotique.exe"

simulation_path = "C:/Users/Utilisateur/Documents/CoursCI2/Challenge/RunTime/challenge-robotique.exe"
script_path = "C:/challenge/script.txt"
map_path = "C:/challenge/donnees-map.txt"
score_path = "C:/challenge/score.txt"

params = {
    "V0" : 1.0,
    "a" : 6.98 * 1e-2,
    "b" : 3.0,
    "b0" : 100.0,
    "Qmax" : 10_000.0,
    "Tmax" : 600,
    "R_ROBOT" : 0.5,
    "R_CYL" : 0.5
}


def optimize(solver, cylinders):
    print("=== Lancement de l'optimisation ===")
    start_time = time.time()
    best_path, best_score = solver.solve()
    execution_time = time.time() - start_time

    print(f"Optimisation terminée en {execution_time}s")
    opt_cylinders = cylinders[best_path]

    return opt_cylinders, execution_time


def multi_solve(solvers, cylinders, robot, exe_path):
    for solver in solvers:
        robot.reset()

        opt_cylinders, execution_time = optimize(solver, cylinders)

        robot.catch_all_cylinders(opt_cylinders, script_path)

        print("=== Lancement simulation Unity ===")
        res = utils.run_full_simulation(exe_path)
        res["temps_execution"] = execution_time
        utils.log_results_to_csv(res, str(solver))



if __name__ == "__main__":
    
    # chargement des données
    cylinders = np.loadtxt(map_path)
    # Ajout colonne [x, y, mass, reward]
    cylinders = np.hstack((cylinders, np.zeros((cylinders.shape[0], 1)))) 
    cylinders[:, 3] = 2 * cylinders[:, 2] - 1 # Calcul reward

    robot = Robot()

    ga = GeneticSolver(
        cylinders, 
        params,
        pop_size=1000,
        mutation_rate=0.2,
        generations=1000,
        tournament_k=4
    )

    sa = SimulatedAnnealing(
        cylinders,
        params,
        T_init=300,
        T_min=0.001,
        alpha=0.99999,
        max_iter=3_000_000
    ) # Assez capricieux, très variancé

    ms = MemeticSolver(
        cylinders,
        params,
        pop_size=500,
        mutation_rate=0.2,
        generations=600,
        tournament_k=3,
        local_search_depth=50
    )

    aco = AntColonySolver(
        cylinders,
        params,
        n_ants=50,
        n_iterations=2000,
        alpha=1.0,
        beta=2.5,
        evaporation=0.05, 
        Q=100
    )

    mcts = MCTSSolver(
        cylinders,
        params,
        iterations=300_000,
        exploration_weight=1.414
    )

    solvers = [sa]

    multi_solve(solvers, cylinders, robot, exe_path_aerial)
