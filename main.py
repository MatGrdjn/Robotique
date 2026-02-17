import os 
import numpy as np
from utils import Robot, GeneticSolver

CYLINDER_DIAMETER = 1
ROBOT_DIAMETER = 1

script_path = "C:/challenge/script.txt"
map_path = "C:/challenge/donnees-map.txt"

params = {
    "V0" : 1.0,
    "a" : 6.98 * 1e-2,
    "b" : 3.0,
    "b0" : 100.0,
    "Qmax" : 10_000.0,
    "Tmax" : 600
}


if __name__ == "__main__":

    cylinders = np.loadtxt(map_path)
    cylinders = np.hstack((cylinders, np.zeros((cylinders.shape[0], 1)))) # On ajoute une colonne de 0 pour le score des cylindres [x, y, mass, point]
    cylinders[:, 3] = 2 * cylinders[:, 2] - 1

    solver = GeneticSolver(
        cylinders, 
        params,
        pop_size=300,
        mutation_rate=0.1,
        generations=300,
        tournament_selection=4
        )
    
    best_path, best_score = solver.solve()
    ordered_cylinders = cylinders[best_path]

    robot = Robot()
    robot.catch_all_cylinders(ordered_cylinders, script_path)
