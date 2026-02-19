import numpy as np
from multiprocessing import Pool, cpu_count
from optimizer import utils_solver

def _beam_worker(args):
    """
    Exécute une recherche en faisceau (Beam Search) sur un coeur.
    La recherche commence avec un cylindre de départ spécifique.
    """
    start_cyl, cylinders, params, beam_width = args
    num_cylinders = len(cylinders)

    # Initialisation du faisceau (beam) avec le point de départ
    beam = [[start_cyl]]

    # On construit la solution étape par étape (profondeur = nombre total de cylindres)
    for step in range(1, num_cylinders):
        candidates = []
        
        # Pour chaque chemin actuel dans le faisceau
        for path in beam:
            # On cherche tous les cylindres qu'on n'a pas encore visités
            unvisited = [i for i in range(num_cylinders) if i not in path]

            for next_cyl in unvisited:
                new_path = path + [next_cyl]
                
                # Évaluation du chemin partiel avec la fonction de fitness existante
                # Comme la fitness récompense largement (1e10) chaque cylindre ramassé 
                # et pénalise le temps/fuel, elle est parfaite comme heuristique d'étape.
                score = utils_solver.calculate_fitness_collision(new_path, cylinders, params, margin=0.9)
                
                candidates.append((new_path, score))

        # On trie les candidats par score décroissant
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # On ne garde que les 'beam_width' meilleurs chemins (élagage)
        beam = [c[0] for c in candidates[:beam_width]]

        # Arrêt prématuré si aucun candidat n'est valide (ex: contraintes Tmax ou Qmax dépassées)
        if not beam:
            break

    # Une fois la recherche terminée, on réévalue le meilleur chemin final
    best_path = beam[0] if beam else []
    best_score = -float('inf')
    
    if best_path:
        best_score = utils_solver.calculate_fitness_collision(best_path, cylinders, params, margin=0.9)

    return best_path, best_score


class BeamSearchSolver:
    def __init__(self, cylinders, robot_params, beam_width=10):
        """
        :param beam_width: La largeur du faisceau (nombre de chemins conservés à chaque étape)
        """
        self.cylinders = cylinders
        self.params = robot_params
        self.beam_width = beam_width

    def solve(self, processes=None):
        """
        Lance l'optimisation avec plusieurs points de départ en parallèle
        """
        if processes is None:
            processes = cpu_count()
            
        print(f"Début de l'optimisation (Beam Search) sur {processes} coeurs")

        # On crée une tâche par cylindre de départ possible
        num_cylinders = len(self.cylinders)
        tasks = [(i, self.cylinders, self.params, self.beam_width) for i in range(num_cylinders)]

        best_overall = None
        best_score = -float("inf")

        with Pool(processes=processes) as pool:
            # On distribue les calculs sur les coeurs du CPU
            results = pool.map(_beam_worker, tasks)

            for path, score in results:
                if score > best_score:
                    best_overall = path
                    best_score = score
                    
        print(f"Meilleur score : {int(best_score / 1e10)} pts (Fitness : {best_score})")

        return best_overall, best_score
    
    def __str__(self):
        return f"BeamSearchSolver(beam_width={self.beam_width})"