import argparse
import json
import os
from time import time

import pandas as pd

from enet.data_loader import load_perms
from neuralplanner import plan
from pnet.data_loader import loader
from pnet.model import PNet

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"


def setup_data(dataset, data, paths_per_env, selected=None):
    mapping = {}
    if not selected:
        env_ids = pd.unique(data["Env ID"])
        for env_id in env_ids:
            df = data[data["Env ID"] == env_id]
            if env_id == 100:
                paths_per_env *= 10
            selected_path_ids = df.sample(n=paths_per_env).index.tolist()
            inputs = []
            references = []
            for path_id in selected_path_ids:
                data_input, _, reference_path, _ = dataset[path_id, True]
                inputs.append(data_input)
                references.append(reference_path)
            mapping[env_id] = {"input": inputs, "reference": references, "selected_ids": selected_path_ids}
    else:
        for env_id, selected_path_ids in selected.items():
            inputs = []
            references = []
            for path_id in selected_path_ids:
                data_input, _, reference_path, _ = dataset[path_id, True]
                inputs.append(data_input)
                references.append(reference_path)
            mapping[int(env_id)] = {"input": inputs, "reference": references, "selected_ids": selected_path_ids}
    
    return mapping


def run(args):
    pnets = []
    datasets = {}
    dataset_mapping = {}
    for n, model in enumerate(args.pnet):
        pnets.append(PNet.load_from_checkpoint(model))
        pnets[-1].freeze()
        enet_ckpt = pnets[-1].training_config['enet']
        enet_key = os.path.basename(enet_ckpt)
        if enet_key not in datasets:
            datasets[enet_key] = loader(enet_ckpt, f"{project_path}/valEnv", 110, 0, 1, True)
        dataset_mapping[n] = datasets[enet_key]
    
    pnets_basenames = [os.path.basename(p).split('.')[0] for p in args.pnet]
    
    envs = load_perms(110, 0)
    try:
        with open(f"{project_path}/{args.output}/selected_results.json", "r") as f:
            selected_results = json.load(f)
    except FileNotFoundError:
        selected_results = {}
    
    for pnet, dataset, name in zip(pnets, dataset_mapping.values(), pnets_basenames):
        data = pd.DataFrame(dataset.path_files, columns=["Env ID", "Path", "State ID"])
        data['index'] = data.index
        data = data[data["State ID"] == 0].drop(columns=["State ID"])
        
        try:
            with open(f"{project_path}/{args.output}/selected.json", "r") as f:
                selected = json.load(f)
            overall_data = setup_data(dataset, data, 50, selected)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            overall_data = setup_data(dataset, data, 50)
            selected = {int(k): list(overall_data[k]["selected_ids"]) for k in overall_data}
            selected_start_n_goals = {int(k): [[list(ref[0]), list(ref[-1])] for ref in overall_data[k]["reference"]]
                                      for k in overall_data}
            with open(f"{project_path}/{args.output}/selected.json", "w") as f:
                json.dump(selected, f)
            with open(f"{project_path}/{args.output}/selected_points.json", "w") as f:
                json.dump(selected_start_n_goals, f)
        try:
            with open(f"{project_path}/{args.output}/{name}.json", "r") as f:
                results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            results = {"seen": {"Success": 0, "Failure": 0, "Replan Success": 0, "Replan Failure": 0
                                },
                       "unseen": {"Success": 0, "Failure": 0, "Replan Success": 0, "Replan Failure": 0,
                                  }, "Time": {"Total"         : [], "Success": [], "Failure": [],
                                              "Replan Success": [], "Replan Failure": []}}
        
        try:
            paths = pd.read_json(f"{project_path}/{args.output}/paths.json", orient='table')
        except ValueError:
            paths = pd.DataFrame([], columns=['seen', 'model', 'id', 'env_id', 'result', 'initial', 'goal', 'bidir',
                                              'lvc', 'replan', 'final'])
            paths = paths.set_index(['model', 'id'])
        for env_id, mapping in overall_data.items():
            start_idx = len(paths.query(f'env_id == {env_id} and model == "{name}"'))
            for data_input, selected_id in zip(mapping["input"][start_idx:], mapping["selected_ids"][start_idx:]):
                if selected_id not in selected_results:
                    selected_results[selected_id] = {"Success"       : [], "Failure": [], "Replan Success": [],
                                                     "Replan Failure": []}
                start = time()
                result, path, lvc_path, replanned, final_path = plan(pnet, envs[env_id], data_input,
                                                                     detailed_results=True)
                duration = time() - start
                results["seen" if env_id < 100 else "unseen"][result] += 1
                results["Time"]["Total"].append(duration)
                results["Time"][result].append(duration)
                
                paths = paths.append(pd.DataFrame(
                        [[env_id < 100, name, selected_id, env_id, result, data_input[-4:-2], data_input[-2:], path,
                          lvc_path, replanned, final_path]],
                        columns=['seen', 'model', 'id', 'env_id', 'result', 'initial', 'goal', 'bidir',
                                 'lvc', 'replan', 'final']).set_index(['model', 'id']))
                
                selected_results[selected_id][result].append(name)
            paths.to_json(f"{project_path}/{args.output}/paths.json", default_handler=str, orient='table')
            with open(f"{project_path}/{args.output}/{name}.json", "w") as f:
                json.dump(results, f)
            with open(f"{project_path}/{args.output}/selected_results.json", "w") as f:
                json.dump(selected_results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnet", default="", nargs="+", type=str)
    parser.add_argument("--output", default="data", type=str)
    args = parser.parse_args()
    
    run(args)
