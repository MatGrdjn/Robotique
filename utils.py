import numpy as np
from datetime import datetime
import pandas as pd
import os
import subprocess
import time


def parse_score_file(path):
    """
    Parse le fichier de score et renvoie un dictionnaire
    """

    if not os.path.exists(path):
        print("Erreur, le fichier n'existe pas")
        return None
    
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline()
        if not line:
            return None
    
    clean_line = line.replace(",", ".")
    parts = clean_line.split()

    results = {
            "gain": float(parts[2]),
            "fuel_restant": float(parts[5]),
            "temps_restant": float(parts[8])
        }

    return results


def log_results_to_csv(results, algo_name, csv_path="runs_history.csv"):
    """
    Ajoute les résultats d'une run dans un fichier CSV avec la date
    """

    data = {
        "date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "algorithme": [algo_name],
        "gain": [results["gain"]],
        "fuel_restant": [results["fuel_restant"]],
        "temps_restant": [results["temps_restant"]],
        "temps_execution" : [results["temps_execution"]]
    }

    df_new = pd.DataFrame(data)

    file_exists = os.path.exists(csv_path)
    
    df_new.to_csv(csv_path, mode='a', index=False, header=not file_exists, sep=';', encoding='utf-8')
    
    print(f"Run enregistrée dans {csv_path} (Algo: {algo_name})")


def run_full_simulation(exe_path, script_path="C:/challenge/script.txt", score_path="C:/challenge/score.txt"):
    """
    Lance l'exécutable, attend qu'il finisse et récupère le score
    """
    print(f"Lancement de la simulation Unity")
    
    process = subprocess.Popen(exe_path)
    #process = subprocess.Popen([exe_path, "-batchmode", "-nographics"])
    
    # On surveille la date de modification du fichier
    initial_time = os.path.getmtime(score_path) if os.path.exists(score_path) else 0
    
    timeout = 600 # secondes
    start_wait = time.time()
    
    while True:
        if os.path.exists(score_path):
            if os.path.getmtime(score_path) > initial_time:
                break
        
        if time.time() - start_wait > timeout:
            print("Timeout : La simulation est trop longue")
            process.kill()
            return None
        
        time.sleep(1)

    process.terminate()
    
    #  On parser et enregistre
    results = parse_score_file(score_path)
    return results


def run_parallel_unity_sims(solutions, robot, exe_path, base_script_path="C:/challenge/"):
    """
    Lance N simulations Unity en parallèle
    solutions: liste de tuples (indices_path, score_theorique)
    """
    processes = []
    temp_files = []

    print(f"--- Lancement de {len(solutions)} simulations Unity en parallèle ---")

    for i, (path_indices, theo_score) in enumerate(solutions):
        rank = i + 1
        
        unique_script = os.path.join(base_script_path, f"script_rank_{rank}.txt")
        unique_score = os.path.join(base_script_path, f"score_rank_{rank}.txt")
        
        if os.path.exists(unique_score):
            os.remove(unique_score)

        ordered_cylinders = robot.cylinders[path_indices]
        robot.catch_all_cylinders(ordered_cylinders, unique_script)

        cmd = [
            exe_path, 
            "-batchmode", 
            "-nographics", 
            "--input", unique_script, 
            "--output", unique_score
        ]
        
        p = subprocess.Popen(cmd)
        processes.append(p)
        temp_files.append({"score": unique_score, "rank": rank, "theo": theo_score})
        

    print(f"--- Attente de la fin des {len(processes)} simulations ---")
    for p in processes:
        p.wait()
    
    print("--- Toutes les simulations sont terminées. Récupération des scores ---")

    final_results = []
    for item in temp_files:
        res = parse_score_file(item["score"])
        if res:
            res["rank"] = item["rank"]
            res["score_theorique"] = item["theo"]
            final_results.append(res)
        else:
            print(f"Erreur: Pas de résultat pour le rang {item['rank']}")

    return final_results


