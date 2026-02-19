import numpy as np

def distance_point_segment(point, start, end):
    """
    Calcule la distance minimale entre un point et un segment
    """

    line_vec = end - start
    pnt_vec = point - start
    line_len_sq = np.dot(line_vec, line_vec)

    if line_len_sq == 0:
        return np.linalg.norm(pnt_vec)
    
    # Projection du point (cylindre) sur la ligne (trajectoire)
    t = np.dot(pnt_vec, line_vec) / line_len_sq
    t = np.clip(t, 0, 1)

    nearest = start + t * line_vec
    
    return np.linalg.norm(point - nearest)


def _simulate_physics_step(dist, current_mass, params):
    """
    Calcule le temps et le carburant pour une distance donnée
    """

    v_max = params["V0"] * np.exp(-params["a"] * current_mass)
    q = params["b"] * current_mass + params["b0"]

    time = dist / v_max
    fuel = q * dist

    return time, fuel

def calculat_fitness_simple(path_indices, cylinders, params):
    """
    Fitness classique, cylindre par cylindre, sans détecter les collisions
    """
    current_pos = np.array([0.0, 0.0])
    current_mass = 0.0
    current_fuel = 0.0
    current_time = 0.0
    total_reward = 0.0

    Qmax, Tmax = params["Qmax"], params["Tmax"]

    for cyl_idx in path_indices:
        target = cylinders[cyl_idx]
        target_pos = target[:2]

        dist = np.linalg.norm(target_pos - current_pos)
        t_ij, f_ij = _simulate_physics_step(dist, current_mass, params)

        if (current_time + t_ij > Tmax) or (current_fuel + f_ij > Qmax):
            break

        current_time += t_ij
        current_fuel += f_ij
        current_pos = target_pos

        total_reward += target[3]
        current_mass += target[2]

    fuel_saved = Qmax - current_fuel
    time_saved = Tmax - current_time

    return (total_reward * 1e10) + (fuel_saved * 1e5) + time_saved


def calculate_fitness_collision(path_indices, cylinders, params, margin=0.1):
    """
    Fitness avancée, détection de colisions sur le trajet
    """

    current_pos = np.array([0.0, 0.0])
    current_mass = 0.0
    current_time = 0.0
    current_fuel = 0.0
    total_reward = 0.0

    Qmax, Tmax = params["Qmax"], params["Tmax"]

    #Paramètres de collisions
    R_robot = params["R_ROBOT"]
    R_cyl = params["R_CYL"]
    COLLISIONS_DIST = R_robot + R_cyl - margin # petite marge au cas où

    num_cylinders = len(cylinders)
    collected_mask = [False] * num_cylinders

    for cyl_idx in path_indices:
        
        if collected_mask[cyl_idx]:
            continue

        target = cylinders[cyl_idx]
        target_center = target[:2]
        stop_pos, dist = get_contact_position(current_pos, target_center, R_robot, R_cyl, margin=margin)

        t_ij, f_ij = _simulate_physics_step(dist, current_mass, params)

        if (current_time + t_ij > Tmax) or (current_fuel + f_ij > Tmax):
            break

        # Peti balayage pour les collisons 

        start_sweep = current_pos
        end_sweep = stop_pos 

        for other_idx in range(num_cylinders):
            
            if not collected_mask[other_idx] and other_idx != cyl_idx:
                other_pos = cylinders[other_idx][:2]

                if distance_point_segment(other_pos, start_sweep, end_sweep) <= COLLISIONS_DIST:
                    collected_mask[other_idx] = True
                    total_reward += cylinders[other_idx][3]
                    current_mass += cylinders[other_idx][2]


        current_time += t_ij
        current_fuel += f_ij
        current_pos = stop_pos

        if not collected_mask[cyl_idx]:
            total_reward += target[3]
            current_mass += target[2]
            collected_mask[cyl_idx] = True
    
    fuel_saved = Qmax - current_fuel
    time_saved = Tmax - current_time

    return (total_reward * 1e10) + (fuel_saved * 1e5) + time_saved


def generate_greedy_path(cylinders, weight_mass=2.0):
    """
    Génère un chemin "intelligent" avec une heuristique légère : ratio de reward / (dist * mass)
    weight_mass permet de faire varier le poids que l'on prend ou non (si on en a peur ou pas)
    Utile pour initialiser certaines solution pour nos différents algo au lieu de faire du random
    """

    current_pos = np.array([0.0, 0.0])
    remaining_indices = set(range(len(cylinders)))

    path = []

    while remaining_indices:
        best_idx = -1
        best_score = -1

        for idx in remaining_indices:
            cyl = cylinders[idx]
            pos = cyl[:2]
            mass = cyl[2]
            reward = cyl[3]

            dist = np.linalg.norm(pos - current_pos)
            if(dist < 0.1): #au cas où
                dist = 0.1

            score = reward / (dist * (1 + mass * weight_mass)) # Comme ça j'évite la division par 0

            if score > best_score:
                best_score = score
                best_idx = idx
        
        if best_idx != -1:
            path.append(best_idx)
            remaining_indices.remove(best_idx)
            current_pos = cylinders[best_idx][:2]
        else:
            break
    
    return path


def get_contact_position(start_pos, target_center, r_robot, r_cyl, margin=0.1):
    """
    Calcule la position où le robot s'arrête (contact avec le cylindre)
    On laisse quand même une petit marge au cas où
    """

    vec = target_center - start_pos
    dist = np.linalg.norm(vec)

    stop_dist_from_center = r_robot + r_cyl - margin

    if dist <= stop_dist_from_center:
        return start_pos, 0
    
    u = vec / dist

    travel_dist = dist - stop_dist_from_center

    stop_pos = start_pos + u * travel_dist

    return stop_pos, travel_dist
