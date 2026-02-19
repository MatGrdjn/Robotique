import numpy as np
from numba import njit

@njit(cache=True)
def _distance_point_segment_jit(point, start, end):
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

    if t < 0.0: 
        t = 0.0
    elif t > 1.0: 
        t = 1.0

    nearest = start + t * line_vec
    
    return np.linalg.norm(point - nearest)


@njit(cache=True)
def _simulate_physics_step_jit(dist, current_mass, V0, a, b, b0):
    """Calcule le temps et le carburant pour une distance donnée"""

    v_max = V0 * np.exp(-a * current_mass)
    q = b * current_mass + b0
    
    time = dist / v_max
    fuel = q * dist

    return time, fuel


@njit(cache=True)
def _calculate_fitness_collision_jit(path_indices, cylinders, V0, a, b, b0, Qmax, Tmax, R_robot, R_cyl, margin=0.8):
    """
    Fitness avancée, détection de colisions sur le trajet
    """

    current_pos = np.zeros(2, dtype=np.float64)
    current_mass = 0.0
    current_time = 0.0
    current_fuel = 0.0
    total_reward = 0.0

    COLLISIONS_DIST = R_robot + R_cyl - margin # petite marge au cas où

    num_cylinders = len(cylinders)
    collected_mask = [False] * num_cylinders

    for i in range(len(path_indices)):

        cyl_idx = path_indices[i]
        
        if collected_mask[cyl_idx]:
            continue

        target_center = cylinders[cyl_idx][:2]
        stop_pos, dist = _get_contact_position_jit(current_pos, target_center, R_robot, R_cyl, margin=margin)

        t_ij, f_ij = _simulate_physics_step_jit(dist, current_mass, V0, a, b, b0)

        if (current_time + t_ij > Tmax) or (current_fuel + f_ij > Tmax):
            break

        # Peti balayage pour les collisons 

        start_sweep = current_pos.copy()
        end_sweep = stop_pos.copy()

        for other_idx in range(num_cylinders):
            
            if not collected_mask[other_idx] and other_idx != cyl_idx:
                other_pos = cylinders[other_idx][:2]

                if _distance_point_segment_jit(other_pos, start_sweep, end_sweep) <= COLLISIONS_DIST:
                    collected_mask[other_idx] = True
                    total_reward += cylinders[other_idx][3]
                    current_mass += cylinders[other_idx][2]


        current_time += t_ij
        current_fuel += f_ij

        current_pos[0] = stop_pos[0]
        current_pos[1] = stop_pos[1]

        if not collected_mask[cyl_idx]:
            total_reward += cylinders[cyl_idx][3]
            current_mass += cylinders[cyl_idx][2]
            collected_mask[cyl_idx] = True
    
    fuel_saved = Qmax - current_fuel
    time_saved = Tmax - current_time

    return (total_reward * 1e10) + (fuel_saved * 1e5) + time_saved


@njit(cache=True)
def _generate_greedy_path_jit(cylinders, weight_mass=2.0):
    """
    Génère un chemin "intelligent" avec une heuristique légère : ratio de reward / (dist * mass)
    weight_mass permet de faire varier le poids que l'on prend ou non (si on en a peur ou pas)
    Utile pour initialiser certaines solution pour nos différents algo au lieu de faire du random
    """

    num_cylinder = len(cylinders)
    current_pos = np.zeros(2, dtype=np.float64)

    unvisited = np.ones(num_cylinder, dtype=np.bool_)
    path = np.zeros(num_cylinder, dtype=np.int32)
    path_len = 0

    for _ in range(num_cylinder):
        best_idx = -1
        best_score = -1.0

        for idx in range(num_cylinder):
            
            if unvisited[idx]:
                pos = cylinders[idx][:2]
                mass = cylinders[idx][2]
                reward = cylinders[idx][3]

                dist = np.linalg.norm(pos - current_pos)
                if dist < 0.1:
                    dist = 0.1
                
                score = reward / (dist * (1 + mass * weight_mass))

                if score > best_score:
                    best_score = score
                    best_idx = idx
        
        if best_idx != -1:
            path[path_len] = best_idx
            path_len += 1
            unvisited[best_idx] = False
            current_pos[0] = cylinders[best_idx][0]
            current_pos[1] = cylinders[best_idx][1]
        else:
            break
    return path[:path_len]


@njit(cache=True)
def _get_contact_position_jit(start_pos, target_center, r_robot, r_cyl, margin=0.9):
    """
    Calcule la position où le robot s'arrête (contact avec le cylindre)
    On laisse quand même une petit marge au cas où
    """

    vec = target_center - start_pos
    dist = np.linalg.norm(vec)

    stop_dist_from_center = r_robot + r_cyl - margin

    if dist <= stop_dist_from_center:
        stop_pos = np.zeros(2, dtype=np.float64)
        stop_pos[0] = start_pos[0]
        stop_pos[1] = start_pos[1]
        return start_pos, 0.0
    
    u = vec / dist

    travel_dist = dist - stop_dist_from_center

    stop_pos = start_pos + u * travel_dist

    return stop_pos, travel_dist



def distance_point_segment(point, start, end):
    p_arr = np.asarray(point, dtype=np.float64)
    s_arr = np.asarray(start, dtype=np.float64)
    e_arr = np.asarray(end, dtype=np.float64)
    return _distance_point_segment_jit(p_arr, s_arr, e_arr)


def calculate_fitness_collision(path_indices, cylinders, params, margin=0.8):

    path_array = np.asarray(path_indices, dtype=np.int32)
    cyls_array = np.asarray(cylinders, dtype=np.float64)

    return _calculate_fitness_collision_jit(
        path_array,
        cyls_array,
        params["V0"],
        params["a"],
        params["b"],
        params["b0"],
        params["Qmax"],
        params["Tmax"],
        params["R_ROBOT"],
        params["R_CYL"],
        margin=margin
    )


def generate_greedy_path(cylinders, weight_mass=2.0):
    cyls_array = np.asarray(cylinders, dtype=np.float64)
    path_array = _generate_greedy_path_jit(cyls_array, weight_mass)
    return path_array.tolist()


def simulate_physics_step(dist, current_mass, params):
    return _simulate_physics_step_jit(
        float(dist), float(current_mass), 
        params["V0"], params["a"], params["b"], params["b0"]
    )


def get_contact_position(start_pos, target_center, r_robot=0.5, r_cyl=0.5):

    s_pos = np.array(start_pos, dtype=np.float64)
    t_center = np.array(target_center, dtype=np.float64)
    
    stop_pos, travel_dist = _get_contact_position_jit(s_pos, t_center, r_robot, r_cyl)
    
    return stop_pos.tolist(), travel_dist