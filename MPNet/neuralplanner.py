import argparse
import os
from copy import deepcopy

import numpy as np
import torch
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

# from MPNet.visualizer.visualizer import plot_path

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"


def is_in_collision(x, env):
    x = Point(x)
    for obstacle in env:
        if obstacle.contains(x):
            return True
    return False


def steer_to(start, end, env):
    start = Point(start)
    end = Point(end)
    line = LineString([start, end])
    for polygon in env:
        if polygon.intersects(line):
            return False
    else:
        return True


def feasibility_check(path, env) -> bool:
    for i in range(0, len(path[:-1])):
        ind = steer_to(path[i], path[i + 1], env)
        if not ind:
            return False
    return True


def lvc(path, env):
    # Iterate from the first beacon state to the second to last.
    for i in range(0, len(path) - 1):
        # Iterate from the last beacon state to the ith.
        for j in range(len(path) - 1, i + 1, -1):
            ind = steer_to(path[i], path[j], env)
            if ind:
                pc = []
                for k in range(0, i + 1):
                    pc.append(path[k])
                for k in range(j, len(path)):
                    pc.append(path[k])
                
                return lvc(pc, env)
    
    return path


def remove_invalid_beacon_states(path, env):
    new_path = []
    for state in path:
        if not is_in_collision(state, env):
            new_path.append(state)
        else:
            try:
                new_path[-1] = np.mean([new_path[-1], new_path[-2]], axis=0)
            except IndexError:
                pass
    for n in range(len(new_path) - 1, 0, -1):
        if is_in_collision(new_path[n], env):
            try:
                new_path[n + 1] = np.mean([new_path[n + 1], new_path[n + 2]], axis=0)
            except IndexError:
                pass
    new_path = np.array(new_path)
    return new_path


def replan_path(previous_path, env, data_input, pnet, num_tries=12):
    path = remove_invalid_beacon_states(previous_path, env)
    feasible = feasibility_check(path, env)
    tries = 0
    target_reached = False
    while not feasible and tries < num_tries:
        tries += 1
        replanned_path = [path[0]]
        # Iterate through each consecutive beacon state
        for i in range(0, len(path) - 1):
            steerable, start, goal = bidirectional_replan_setup(env, path[i], path[i + 1], data_input)
            if steerable:
                replanned_path.append(path[i + 1])
            else:
                target_reached, rpath_1, rpath_2 = bidirectional_planning(pnet, start, goal, env)
                replanned_path = list(np.concatenate([replanned_path, rpath_1, rpath_2[::-1]]))
                if not target_reached:
                    if i < len(path) - 1:
                        replanned_path = np.concatenate([replanned_path, path[i + 1:]])
                    return False, replanned_path
        
        replanned_path = np.array(replanned_path)
        filtered_path, indexes = np.unique(replanned_path, axis=0, return_index=True)
        filtered_path = filtered_path[np.argsort(indexes)]
        lvc_replanned_path = lvc(filtered_path, env)
        lvc_replanned_path = np.array(lvc_replanned_path)
        feasible = feasibility_check(lvc_replanned_path, env)
        if feasible:
            path = lvc_replanned_path
            break
        elif not target_reached:
            return False, lvc_replanned_path
        else:
            path = np.array(filtered_path)
        path = remove_invalid_beacon_states(path, env)
    return feasible, path


# Checks if it's necessary to replan this section
def bidirectional_replan_setup(env, start_point, goal_point, model_input):
    start = deepcopy(model_input)
    start[-4:] = torch.as_tensor([*start_point, *goal_point])
    goal = deepcopy(start)
    goal[-4:] = goal[[-2, -1, -4, -3]]
    steerable = steer_to(start_point, goal_point, env)
    return steerable, start, goal


def bidirectional_planning(pnet, origin, goal, env, steps=100):
    result_1 = deepcopy(origin[-4:-2])
    result_2 = deepcopy(goal[-4:-2])
    path_1, path_2 = [result_1.numpy()], [result_2.numpy()]
    tree, target_reached = False, False
    step = 0
    while not target_reached and step < steps:
        step += 1
        if not tree:
            result_1 = pnet(origin)
            result_1 = result_1.data.detach()
            path_1.append(result_1.numpy())
            origin[-4:-2] = result_1
            goal[-2:] = result_1
        else:
            result_2 = pnet(goal)
            result_2 = result_2.data.detach()
            path_2.append(result_2.numpy())
            goal[-4:-2] = result_2
            origin[-2:] = result_2
        tree = not tree
        target_reached = steer_to(result_1.numpy(), result_2.numpy(), env)
    return target_reached, path_1, path_2


def bidirectional_planner(pnet, env, model_input):
    origin = deepcopy(model_input)
    goal = deepcopy(origin)
    goal[-4:] = goal[[-2, -1, -4, -3]]
    target_reached, path_1, path_2 = bidirectional_planning(pnet, origin, goal, env)
    return target_reached, path_1, path_2


def plan(pnet, env, data_input):
    env = env_npy_to_polygon(env)
    target_reached, path_1, path_2 = bidirectional_planner(pnet, env, data_input)
    
    if target_reached:
        path = np.concatenate([path_1, path_2[::-1]])
        path = np.array(lvc(path, env))
        feasible = feasibility_check(path, env)
        if not feasible:
            result, replanned_path = replan_path(path, env, data_input, pnet)
            return f"Replan {'Success' if result else 'Failure'}", replanned_path
        else:
            return "Success", path
    else:
        return "Failure", None


def env_npy_to_polygon(env):
    obstacles = []
    for obstacle in env:
        x, y = obstacle
        obstacles.extend([[x - 2.5, y - 2.5], [x - 2.5, y + 2.5], [x + 2.5, y + 2.5], [x + 2.5, y - 2.5],
                          [x - 2.5, y - 2.5], [None, None]])
    obstacles = np.array(obstacles)
    obstacle = []
    obstacles_polygon = []
    for point in obstacles:
        if None in point:
            polygon = Polygon(obstacle)
            obstacles_polygon.append(polygon)
            obstacle = []
        else:
            obstacle.append(tuple(point))
    env = MultiPolygon(obstacles_polygon)
    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnet", default="", type=str)
    parser.add_argument("--enet", default="", type=str)
    parser.add_argument("--num_envs", default=1, type=int)
    parser.add_argument("--num_trajs", default=1, type=int)
    
    args = parser.parse_args()
