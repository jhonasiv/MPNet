import argparse
import os

import math
import numpy as np
from copy import deepcopy
import torch

from MPNet.enet import data_loader as ae_dl
from MPNet.enet.CAE import ContractiveAutoEncoder
from MPNet.pnet.data_loader import loader
from MPNet.pnet.model import PNet
from MPNet.pnet.visualizer import plot_path

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"


def is_in_collision(x, env):
    for obs in env:
        if not np.any(np.abs(x - obs) > 2.5):
            return True
    return False


def steer_to(start, end, env) -> bool:
    discretization_step = 0.01
    dists = end - start
    
    dist_total = np.linalg.norm(dists)
    
    if dist_total > 0:
        increment_total = dist_total / discretization_step
        dists /= increment_total
        
        num_segments = int(math.floor(increment_total))
        
        state_curr = deepcopy(start)
        for i in range(0, num_segments):
            
            if is_in_collision(state_curr, env):
                return False
            
            state_curr += dists
        
        if is_in_collision(end, env):
            return False
    
    return True


# checks the feasibility of entire path including the path edges
def feasibility_check(path, env) -> bool:
    for i in range(0, len(path[:-1])):
        ind = steer_to(path[i], path[i + 1], env)
        if not ind:
            return False
    return True


# checks the feasibility of path nodes only
def collision_check(path, env) -> bool:
    for state in path:
        if is_in_collision(state, env):
            return False
    return True


def is_reaching_target(start1, start2) -> bool:
    s1 = np.zeros(2, dtype=np.float32)
    s1[0] = start1[0]
    s1[1] = start1[1]
    
    s2 = np.zeros(2, dtype=np.float32)
    s2[0] = start2[0]
    s2[1] = start2[1]
    
    for i in range(0, 2):
        if abs(s1[i] - s2[i]) > 1.0:
            return False
    return True


# lazy vertex contraction
def lvc(path, idx):
    # Iterate from the first beacon state to the second to last.
    for i in range(0, len(path) - 1):
        # Iterate from the last beacon state to the ith.
        for j in range(len(path) - 1, i + 1, -1):
            ind = steer_to(path[i], path[j], idx)
            if ind:
                pc = []
                for k in range(0, i + 1):
                    pc.append(path[k])
                for k in range(j, len(path)):
                    pc.append(path[k])
                
                return lvc(pc, idx)
    
    return path


def replan_path(previous_path, env, data_input, pnet):
    path = []
    for state in previous_path:
        if not is_in_collision(state, env):
            path.append(state)
    new_path = []
    # Iterate through each consecutive beacon state
    for i in range(0, len(path) - 1):
        start = torch.from_numpy(path[i])
        goal = torch.from_numpy(path[i + 1])
        steer = steer_to(start.numpy(), goal.numpy(), env)
        if steer:
            new_path.append(start.numpy())
            new_path.append(goal.numpy())
        else:
            itr = 0
            path_1 = [start.numpy()]
            path_2 = [goal.numpy()]
            target_reached = False
            data_input[-4:-2] = start
            data_input[-2:] = goal
            
            data_2 = deepcopy(data_input)
            data_2[-4:-2] = goal
            data_2[-2:] = start
            tree = False
            while not target_reached and itr < 50:
                itr = itr + 1
                if not tree:
                    start = pnet(data_input)
                    start = start.data.detach()
                    path_1.append(start.numpy())
                    tree = True
                    data_input[-4:-2] = start
                    data_2[-2:] = start
                else:
                    goal = pnet(data_2)
                    goal = goal.data.detach()
                    path_2.append(goal.numpy())
                    tree = False
                    data_2[-4:-2] = goal
                    data_input[-2:] = start
                target_reached = steer_to(start.numpy(), goal.numpy(), env)
            # if not target_reached:
            #     return 0
            # else:
            for p1 in path_1:
                new_path.append(p1)
            for p2 in path_2[::-1]:
                new_path.append(p2)
    
    return new_path


def train(pnet_path, perm, data_input, num_trajs):
    pnet = PNet.load_from_checkpoint(pnet_path)
    pnet.freeze()
    for n in range(num_trajs):
        origin = deepcopy(data_input)
        goal = deepcopy(origin)
        goal[-4:-2] = goal[-2:]
        goal[-2:] = origin[-4:-2]
        
        path_1, path_2 = [deepcopy(origin[-4:-2].numpy())], [deepcopy(goal[-4:-2].numpy())]
        tree, target_reached = False, False
        step = 0
        path = []
        result_1 = origin[-4:-2]
        result_2 = goal[-4:-2]
        while not target_reached and step < 80:
            step += 1
            if not tree:
                result_1 = pnet(origin)
                result_1 = result_1.data.detach()
                path_1.append(result_1.numpy())
                origin[-4:-2] = result_1
                goal[-2:] = result_1
                tree = True
            else:
                result_2 = pnet(goal)
                result_2 = result_2.data.detach()
                path_2.append(result_2.numpy())
                goal[-4:-2] = result_2
                origin[-2:] = result_2
                tree = False
            target_reached = steer_to(result_1.numpy(), result_2.numpy(), perm)
        
        if target_reached:
            for p1 in path_1:
                path.append(p1)
            for p2 in path_2[::-1]:
                path.append(p2)
            path = np.array(path)
            path = np.array(lvc(path, perm))
            indicator = feasibility_check(path, perm)
            if not indicator:
                sp = 0
                indicator = 0
                while not indicator and sp < 10:
                    sp = sp + 1
                    path = replan_path(path, perm, origin, pnet)  # replanning at coarse level
                    if path:
                        path = lvc(path, perm)
                        indicator = feasibility_check(path, perm)
                        
                        if not indicator and sp == 10:
                            print("Replanned path invalid")
                            return path
                        elif indicator:
                            return path
                else:
                    print("Replanning failed")
                    return path
            else:
                return path
        else:
            print("Target not reached!")
            return 0


def main(args):
    perms = ae_dl.load_perms(110, 0)
    
    cae = ContractiveAutoEncoder.load_from_checkpoint(args.enet)
    cae.freeze()
    test_data = loader(cae, f"{project_path}/valEnv", 110, 0, 1, num_workers=4, shuffle=False, get_dataset=True)
    random_path_idx = np.random.choice(len(test_data), 1)[0]
    data_input, _, target_path, chosen_env = test_data[random_path_idx, True]
    data_input[-4:-2] = torch.from_numpy(target_path[0])
    
    perm = perms[chosen_env]
    path = train(args.pnet, perm, data_input, args.num_trajs)
    if not isinstance(path, int):
        path = np.array(path)
        plot_path(perm, path, target_path)
    
    breakpoint()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnet", default="", type=str)
    parser.add_argument("--enet", default="", type=str)
    parser.add_argument("--num_envs", default=1, type=int)
    parser.add_argument("--num_trajs", default=1, type=int)
    
    args = parser.parse_args()
    main(args)
