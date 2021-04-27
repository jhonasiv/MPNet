import numpy as np
import argparse
import os

project_path = f"{os.path.abspath(__file__).split('MPNet')[0]}MPNet"


def gen_centers(obs_perm, map_size, size, resolution):
    xs = np.random.choice(
        np.arange(start=-map_size / 2 + size / 2, stop=map_size / 2 - size / 2, step=resolution), obs_perm,
    )
    ys = np.random.choice(
        np.arange(start=-map_size / 2 + size / 2, stop=map_size / 2 - size / 2, step=resolution), obs_perm,
    )
    centers = np.array(list(zip(xs, ys))).reshape((-1, 40))
    np.savetxt(f"{project_path}/obs/center.csv", centers, delimiter=",")
    return centers


def gen_perm(centers, num_obs):
    centers = centers.reshape((-1, 2))
    perms = set()
    for _ in range(77520):  # (1) Draw N samples from permutations Universe U (#U = k!)
        while True:
            perm = np.random.permutation(centers)[:num_obs]
            key = tuple(map(tuple, perm))
            if key not in perms:
                perms.update((key,))  # (5) Insert into set
                break  # (6) Break the endless loop

    perms = np.array(list(perms)).astype(np.float).reshape((-1, 14))
    np.savetxt(f"{project_path}/obs/perm.csv", perms, delimiter=",")
    return perms


# def gen_graphs(perms, qtd, obs_size, map_size, resolution, points_per_graph):
#     graphs = []
#     perms = perms.reshape((-1, 7, 2))
#     for obstacles in perms[:qtd]:
#         perm_graphs = []
#         while len(perm_graphs) < points_per_graph:
#             x = np.random.choice(np.arange(start=0, stop=map_size, step=resolution))
#             y = np.random.choice(np.arange(start=0, stop=map_size, step=resolution))
#             point = np.array([x, y])
#             if dist_from_obstacle(point, obstacles, obs_size):
#                 if perm_graphs:
#                     if not np.any(np.all(np.isin(perm_graphs, point), axis=1)):
#                         perm_graphs.append(point)
#                 else:
#                     perm_graphs.append(point)
#         graphs.append(np.array(perm_graphs))

#     graphs = np.array(graphs)
#     np.savetxt(f"{project_path}/obs/graphs.csv", graphs, delimiter=',')


def dist_from_obstacle(point, obstacles, obs_size):
    for obstacle in obstacles:
        dist = np.abs(point - obstacle)
        if not np.any(dist > obs_size / 2):
            return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_obs", default=7)
    parser.add_argument("--map_size", default=40)
    parser.add_argument("--obs_size", default=5)
    parser.add_argument("--obs_perm", default=20)
    parser.add_argument("--resolution", default=0.01)

    args = parser.parse_args()

    centers_data = gen_centers(args.obs_perm, args.map_size, args.obs_size, args.resolution)
    perms_data = gen_perm(centers_data, args.num_obs)
    # gen_graphs(perms_data, 110, args.obs_size, args.map_size, args.resolution, 50000)
