import numpy as np
from numba import njit
import math
from optimizer import utils_solver

@njit(cache=True)
def _branch_and_bound_jit(cylinders, V0, a, b, b0, Qmax, Tmax, initial_best_score):
    N = len(cylinders)
    MAX_STACK = 1000 
    
    s_path = np.zeros((MAX_STACK, N), dtype=np.int32)
    s_depth = np.zeros(MAX_STACK, dtype=np.int32)
    s_mask = np.zeros(MAX_STACK, dtype=np.int32)
    s_mass = np.zeros(MAX_STACK, dtype=np.float64)
    s_time = np.zeros(MAX_STACK, dtype=np.float64)
    s_fuel = np.zeros(MAX_STACK, dtype=np.float64)
    s_reward = np.zeros(MAX_STACK, dtype=np.float64)
    s_px = np.zeros(MAX_STACK, dtype=np.float64)
    s_py = np.zeros(MAX_STACK, dtype=np.float64)
    
    stack_ptr = 0
    
    best_score = initial_best_score
    best_path = np.zeros(N, dtype=np.int32)
    best_depth = 0
    
    stop_margin = 0.5 + 0.5 - 0.8
    coll_margin = 0.5 + 0. - 0.8
    
    while stack_ptr >= 0:
        depth = s_depth[stack_ptr]
        mask = s_mask[stack_ptr]
        mass = s_mass[stack_ptr]
        time = s_time[stack_ptr]
        fuel = s_fuel[stack_ptr]
        reward = s_reward[stack_ptr]
        px = s_px[stack_ptr]
        py = s_py[stack_ptr]
        current_path = s_path[stack_ptr, :].copy()
        
        stack_ptr -= 1 
        
        current_fitness = (reward * 1e10) + ((Qmax - fuel) * 1e5) + (Tmax - time)
        if current_fitness > best_score:
            best_score = current_fitness
            best_path[:] = current_path[:]
            best_depth = depth
            

        max_future_reward = 0.0
        for i in range(N):
            if not (mask & (1 << i)):
                max_future_reward += cylinders[i, 3]
                
        best_possible_fitness = ((reward + max_future_reward) * 1e10) + (Qmax * 1e5) + Tmax
        
        if best_possible_fitness <= best_score:
            continue
            
        if depth == N:
            continue # On a tout visité
            
        for i in range(N):
            if not (mask & (1 << i)):
                
                cx = cylinders[i, 0]
                cy = cylinders[i, 1]
                
                vec_x = cx - px
                vec_y = cy - py
                dist = math.sqrt(vec_x**2 + vec_y**2)
                
                if dist <= stop_margin:
                    stop_x, stop_y, travel_dist = px, py, 0.0
                else:
                    travel_dist = dist - stop_margin
                    stop_x = px + (vec_x / dist) * travel_dist
                    stop_y = py + (vec_y / dist) * travel_dist
                    
                v_max = V0 * math.exp(-a * mass)
                q = b * mass + b0
                dt = travel_dist / v_max
                df = q * travel_dist
                
                new_time = time + dt
                new_fuel = fuel + df
                
                if new_time > Tmax or new_fuel > Qmax:
                    continue
                    
                new_mask = mask
                new_reward = reward
                new_mass = mass
                
                line_vec_x = stop_x - px
                line_vec_y = stop_y - py
                line_len_sq = line_vec_x**2 + line_vec_y**2
                
                for j in range(N):
                    if not (new_mask & (1 << j)) and j != i:
                        pnt_vec_x = cylinders[j, 0] - px
                        pnt_vec_y = cylinders[j, 1] - py
                        
                        if line_len_sq == 0.0:
                            d_coll = math.sqrt(pnt_vec_x**2 + pnt_vec_y**2)
                        else:
                            t_proj = (pnt_vec_x * line_vec_x + pnt_vec_y * line_vec_y) / line_len_sq
                            if t_proj < 0.0: t_proj = 0.0
                            elif t_proj > 1.0: t_proj = 1.0
                            near_x = px + t_proj * line_vec_x
                            near_y = py + t_proj * line_vec_y
                            d_coll = math.sqrt((cylinders[j, 0] - near_x)**2 + (cylinders[j, 1] - near_y)**2)
                            
                        if d_coll <= coll_margin:
                            new_mask |= (1 << j) 
                            new_reward += cylinders[j, 3]
                            new_mass += cylinders[j, 2]
                            
                if not (new_mask & (1 << i)):
                    new_mask |= (1 << i)
                    new_reward += cylinders[i, 3]
                    new_mass += cylinders[i, 2]
                    
                stack_ptr += 1
                s_depth[stack_ptr] = depth + 1
                s_mask[stack_ptr] = new_mask
                s_mass[stack_ptr] = new_mass
                s_time[stack_ptr] = new_time
                s_fuel[stack_ptr] = new_fuel
                s_reward[stack_ptr] = new_reward
                s_px[stack_ptr] = stop_x
                s_py[stack_ptr] = stop_y
                
                s_path[stack_ptr, :] = current_path[:]
                s_path[stack_ptr, depth] = i
                
    return best_path[:best_depth], best_score


class ExactSolver:
    def __init__(self, cylinders, robot_params):
        self.cylinders = np.asarray(cylinders, dtype=np.float64)
        self.params = robot_params
        
    def solve(self):
        print("Lancement de le recherche exacte")
        
        greedy_path = utils_solver.generate_greedy_path(self.cylinders, weight_mass=2.0)
        initial_score = utils_solver.calculate_fitness_collision(greedy_path, self.cylinders, self.params)
        
        path_array, best_score = _branch_and_bound_jit(
            self.cylinders,
            self.params["V0"], self.params["a"], self.params["b"], self.params["b0"],
            self.params["Qmax"], self.params["Tmax"],
            initial_score
        )
        
        print(f"Solution parfaite trouvée : {int(best_score/1e10)} pts (Fitness : {best_score})")
        return path_array.tolist(), best_score
    
    def __str__(self):
        return "ExactSolver"