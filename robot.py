import os
import numpy as np

class Robot:
    def __init__(self, pos=[0, 0], orientation = 0.0, r_robot=0.5, r_cyl=0.5):
        self.pos = np.array(pos)
        self.orientation = orientation
        self.cylinders = None

        self.R_ROBOT = r_robot
        self.R_CYL = r_cyl

        self.COLLISION_DIST = self.R_ROBOT + self.R_CYL - 0.9 #Petite marge au cas où, on sait jamais

    def reset(self):
        self.pos = np.array([0, 0])
        self.orientation = 0.0
        self.cylinders = None

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

    def catch_all_cylinders(self, cylinders, path, margin=0.9):

        pos_cylinders = cylinders[:, :2]
        num_cylinders = len(pos_cylinders)

        collected_mask = [False] * num_cylinders

        #On vide le script
        if os.path.exists(path):
            os.remove(path)

        stop_margin = self.R_ROBOT + self.R_CYL - margin

        for i in range(num_cylinders):
            if collected_mask[i]: #Si on a déjà collcté le cylindre bah on y va pas 
                continue

            target_center = pos_cylinders[i]

            alpha, _ = self._find_angle_and_dist(target_center)

            vec = target_center - self.pos
            dist_center_to_center = np.linalg.norm(vec)
            dist_to_go = dist_center_to_center - stop_margin

            if dist_to_go < 0:
                dsit_to_go = 0

            self.instructions_to_script(alpha, dist_to_go, path)

            start_pos = self.pos.copy()

            # màj des coordonnées
            self.orientation += alpha
            self.orientation = (self.orientation + np.pi) % (2 * np.pi) - np.pi

            dir_vec = np.array([np.cos(self.orientation), np.sin(self.orientation)])
            self.pos = start_pos + dir_vec * dist_to_go

            collected_mask[i] = True


            collected_mask[i] = True

            for j in range(num_cylinders):
                if not collected_mask[j] and i != j:
                    other_pos = pos_cylinders[j]

                    dist_to_line = self._distance_point_segment(other_pos, start_pos, self.pos)

                    if dist_to_line <= self.COLLISION_DIST:
                        #print(f"Collision mageule on ramasse un en plus, c'est quoi ce canard OMG merci la France à Macron")
                        collected_mask[j] = True

        with open(path, "a", encoding="utf-8") as f:
            f.write("FINISH")
        
        print(f"DONE : script généré dans '{path}'")


    def _distance_point_segment(self, point, start, end):
        """
        Calcule la distance minimale entre un cylindre et un segment 
        """

        line_vec = end -start

        pnt_vec = point - start

        line_len_sq = np.dot(line_vec, line_vec)

        if line_len_sq == 0:
            return np.linalg.norm(pnt_vec)
        
        t = np.dot(pnt_vec, line_vec) / line_len_sq
        t = np.clip(t, 0, 1)

        nearest = start + t * line_vec

        return np.linalg.norm(point - nearest)