import argparse
import os

import numpy as np
import pandas as pd
import plotly.express as px
from plotly import graph_objs as go

import frontend as fe
from MPNet.enet.data_loader import load_perms
from MPNet.neuralplanner import bidirectional_planner, env_npy_to_polygon, feasibility_check, lvc, replan_path
from MPNet.pnet.data_loader import loader
from MPNet.pnet.model import PNet

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"


class Visualizer:
    def __init__(self):
        self.model_ckpts = []
        self.model_basenames = []
        self.models = pd.DataFrame([], columns=["ckpt", "model", "index"])
        self.models = self.models.set_index("ckpt")
        self.datasets = {}
        self.df = pd.DataFrame([], columns=["path_id", "model", "model_id", "stage", "x", "y", "feasible"])
        
        self.data_input = pd.DataFrame([], columns=["model", "path_id", "env_id", "input"])
        self.data_input = self.data_input.set_index(['model', 'path_id'])
        
        self.stage_figs = pd.DataFrame([], columns=["stage", "path_id", "figure"])
        self.stage_figs = self.stage_figs.set_index(["stage", "path_id"])
        
        self.env_figs = pd.DataFrame([], columns=["env_id", "figure"])
        self.env_figs = self.env_figs.set_index("env_id")
        
        self.env_centers = load_perms(110, 0)
        
        self.paths = pd.read_json(f"{project_path}/data/paths.json", orient='table')
        
        self.stages_cbs = {0: self.set_env, 1: self.bidirectional, 2: self.connection, 3: self.lvc,
                           4: self.replan, 5: self.final_path}
    
    def get_available_envs(self):
        envs = self.paths["env_id"].unique()
        return envs
    
    def get_available_paths(self, env):
        paths = self.paths.query(f"env_id == {env}").reset_index(1)['id'].unique()
        return paths
    
    def process(self, path_id):
        data = self.paths.query(f"id == {path_id} and model in {self.model_basenames}")
        env_figure = self.set_env(path_id)
        bidir_figure = self.make_fig(data, path_id, "bidir", 1, lines=False, title="Bidirectional Planner")
        connection_figure = self.make_fig(data, path_id, "bidir", 2, lines=True, title="States Connection")
        lvc_figure = self.make_fig(data, path_id, "lvc", 3, lines=True, title="Lazy Vertex Contraction")
        replan_figure = self.make_fig(data, path_id, "replan", 4, lines=True, title="Replanning")
        final_figure = self.make_fig(data, path_id, "final", 5, lines=True, title="Final Path")
        results_idxs = [(model, path_id) for model in self.model_basenames]
        results = data.loc[results_idxs]['result'].to_list()
        return {0: env_figure, 1: bidir_figure, 2: connection_figure, 3: lvc_figure, 4: replan_figure,
                5: final_figure}, results
    
    def make_fig(self, data, path_id, column, stage, lines=False, title=""):
        env_id = data["env_id"][0]
        fig = go.Figure(self.env_figs.loc[env_id][0])
        path_info = data[column]
        temp_df = pd.DataFrame([], columns=["model", "x", "y"])
        for idx, row in path_info.iteritems():
            p = np.array(row)
            if p.size > 1:
                temp_df = temp_df.append(pd.DataFrame([[idx[0], x, y] for x, y in p], columns=["model", "x",
                                                                                               "y"]),
                                         ignore_index=True)
        nan_stage = data[data[column].isna()]
        for idx, row in nan_stage.iterrows():
            result = row['result']
            if result == 'Success':
                p = np.array(row['lvc'])
            else:
                p = np.array(row['bidir'])
            temp_df = temp_df.append(pd.DataFrame([[idx[0], x, y] for x, y in p], columns=["model", "x", "y"]),
                                     ignore_index=True)
        
        self.add_to_figure(fig, path_id, data["bidir"], temp_df, stage, lines=lines, title=title)
        return self.stage_figs.query(f"stage == {stage} and path_id == {path_id}")["figure"].item()
    
    def add_to_figure(self, fig, path_id, start_n_goal_info, temp_df, stage, lines=False, title=""):
        start_n_goal_info = np.array(start_n_goal_info[0])
        scatter = px.scatter(temp_df, x="x", y="y", color="model")
        self.update_figure(fig, scatter, start=start_n_goal_info[0], goal=start_n_goal_info[-1], lines=lines,
                           title=title)
        new_entry = pd.DataFrame([[stage, path_id, fig]], columns=["stage", "path_id", "figure"])
        new_entry = new_entry.set_index(["stage", "path_id"])
        try:
            self.stage_figs = self.stage_figs.append(new_entry, verify_integrity=True)
        except ValueError:
            self.stage_figs.update(pd.DataFrame([fig], index=(stage, path_id)))
    
    def load_dataset(self):
        cached_datasets = {}
        for n, model in enumerate(self.model_ckpts):
            pnet = PNet.load_from_checkpoint(model)
            # pnet.freeze()
            frame = pd.DataFrame([[model, pnet, n]], columns=["ckpt", "model", "index"])
            frame = frame.set_index("ckpt")
            try:
                self.models = self.models.append(frame, verify_integrity=True)
                enet_ckpt = pnet.training_config['enet']
                enet_key = os.path.basename(enet_ckpt)
                if enet_key not in cached_datasets:
                    cached_datasets[enet_key] = loader(enet_ckpt, f"{project_path}/valEnv", 110, 0, 1, True)
                self.datasets[n] = cached_datasets[enet_key]
            except ValueError:
                pass
    
    def set_input(self, path_id):
        for n, model in enumerate(self.models["model"]):
            if (n, path_id) not in self.data_input.index:
                data_input, _, _, env_id = self.datasets[n][path_id, True]
                new_input = pd.DataFrame([[n, path_id, env_id, data_input]],
                                         columns=["model", "path_id", "env_id", "input"])
                new_input = new_input.set_index(["model", "path_id"])
                try:
                    self.data_input = self.data_input.append(new_input, verify_integrity=True)
                except ValueError:
                    pass
    
    def set_env(self, path_id):
        # self.set_input(path_id)
        # env_id = self.data_input.loc[0, path_id]["env_id"]
        env_id = self.paths.query(f"id == {path_id}").iloc[0]['env_id']
        obstacles = []
        for obstacle in self.env_centers[env_id]:
            x, y = obstacle
            obstacles.extend([[x - 2.5, y - 2.5], [x - 2.5, y + 2.5], [x + 2.5, y + 2.5], [x + 2.5, y - 2.5],
                              [x - 2.5, y - 2.5], [None, None]])
        obstacles = np.array(obstacles)
        
        if env_id not in self.env_figs:
            obstacle_fig = go.Figure()
            obstacle_fig.add_trace(go.Scatter(x=obstacles[:, 0], y=obstacles[:, 1], fill="toself", fillcolor="black",
                                              name="obstacle", marker=dict(color="black")))
            obstacle_fig.update_layout(dict(width=1280, height=720, title="Environment",
                                            legend=dict(yanchor="bottom", xanchor="left", y=-0.1, x=-0.2)))
            obstacle_fig.update_xaxes(range=[-20, 20])
            obstacle_fig.update_yaxes(range=[-20, 20])
            new_entry = pd.DataFrame([[env_id, obstacle_fig]], columns=["env_id", "figure"])
            new_entry = new_entry.set_index("env_id")
            try:
                self.env_figs = self.env_figs.append(new_entry, verify_integrity=True)
            except:
                pass
        return self.env_figs.loc[env_id]['figure']
    
    def update_models(self, ckpts):
        self.model_ckpts = ckpts
        self.model_basenames = [os.path.basename(ckpt).split('.')[0] for ckpt in ckpts]
        # self.load_dataset()
    
    def update_stages(self, path_id):
        figs = {}
        feasibility = {}
        for stage in self.stages_cbs:
            cb = self.stages_cbs[stage]
            if cb:
                cb_results = cb(path_id)
                if isinstance(cb_results, tuple):
                    feasibility[stage] = cb_results[1]
                    figs[stage] = cb_results[0]
                else:
                    figs[stage] = cb_results
        
        models_feasibility = {}
        for stage, status in feasibility.items():
            for model in np.unique(status['model_id'].values):
                query = status.query(f"model_id == {model}")['feasible'].values[0]
                if model not in models_feasibility:
                    if stage == 1 and not query:
                        models_feasibility[model] = "Failure"
                    elif stage == 3:
                        if query:
                            models_feasibility[model] = "Success"
                    elif stage == 4:
                        if query:
                            models_feasibility[model] = "Replan Success"
                        else:
                            models_feasibility[model] = "Replan Failure"
        return figs, models_feasibility
    
    def bidirectional(self, path_id):
        fig = go.Figure(self.stage_figs.loc[0, path_id]["figure"])
        for n, model in enumerate(self.models["model"]):
            query = self.df.query(f"model_id == {n} and path_id == {path_id} and stage == 1")
            if query.empty:
                model_name = os.path.basename(self.models.query(f"index == {n}").index[0]).split('.')[0]
                env_id = self.data_input.loc[n, path_id]["env_id"]
                polygon_env = env_npy_to_polygon(self.env_centers[env_id])
                target_reached, path_1, path_2 = bidirectional_planner(model, polygon_env,
                                                                       self.data_input.loc[n, path_id]["input"])
                path = np.concatenate([path_1, path_2[::-1]])
                new_data = pd.DataFrame([[path_id, model_name, n, 1, x, y, target_reached] for x, y in path],
                                        columns=["path_id", "model", "model_id", "stage", "x", "y", "feasible"])
                self.df = self.df.append(new_data, ignore_index=True)
        
        scatter = px.scatter(self.df.query(f'stage == 1 and path_id == {path_id}'), x="x", y="y", color="model")
        self.update_figure(fig, path_id, scatter, title="Bidirectional Planner")
        new_entry = pd.DataFrame([[1, path_id, fig]], columns=["stage", "path_id", "figure"])
        new_entry = new_entry.set_index(["stage", "path_id"])
        try:
            self.stage_figs = self.stage_figs.append(new_entry, verify_integrity=True)
        except ValueError:
            self.stage_figs.update(pd.DataFrame([fig], index=(1, path_id)))
        return self.stage_figs.loc[1, path_id]["figure"], self.df.query(f"stage == 1 and path_id == {path_id}")[[
                "model_id", "feasible"]]
    
    def connection(self, path_id):
        fig = go.Figure(self.stage_figs.loc[0, path_id]["figure"])
        scatter = px.scatter(self.df.query(f'stage == 1 and path_id == {path_id}'), x="x", y="y", color="model")
        self.update_figure(fig, path_id, scatter, True, "States connection")
        new_entry = pd.DataFrame([[2, path_id, fig]], columns=["stage", "path_id", "figure"])
        new_entry = new_entry.set_index(["stage", "path_id"])
        try:
            self.stage_figs = self.stage_figs.append(new_entry, verify_integrity=True)
        except ValueError:
            self.stage_figs.update(pd.DataFrame([fig], index=(2, path_id)))
        return self.stage_figs.loc[2, path_id]["figure"]
    
    def lvc(self, path_id):
        fig = go.Figure(self.stage_figs.loc[0, path_id]["figure"])
        for n, model in enumerate(self.models["model"]):
            query = self.df.query(f"stage == 3 and path_id == {path_id} and model_id == {n}")
            if query.empty:
                model_name = os.path.basename(self.models.query(f"index == {n}").index[0]).split('.')[0]
                env_id = self.data_input.loc[n, path_id]["env_id"]
                polygon_env = env_npy_to_polygon(self.env_centers[env_id])
                path = self.df.query(f"stage == 1 and path_id == {path_id} and model_id == {n}")[["x", "y"]].values
                lvc_path = lvc(path, polygon_env)
                feasible = feasibility_check(lvc_path, polygon_env)
                new_data = pd.DataFrame([[path_id, model_name, n, 3, x, y, feasible] for x, y in lvc_path],
                                        columns=["path_id", "model", "model_id", "stage", "x", "y", "feasible"])
                self.df = self.df.append(new_data, ignore_index=True)
        scatter = px.scatter(self.df.query(f'stage == 3 and path_id == {path_id}'), x="x", y="y", color="model")
        self.update_figure(fig, path_id, scatter, True, title="LVC")
        new_entry = pd.DataFrame([[3, path_id, fig]], columns=["stage", "path_id", "figure"])
        new_entry = new_entry.set_index(["stage", "path_id"])
        try:
            self.stage_figs = self.stage_figs.append(new_entry, verify_integrity=True)
        except ValueError:
            self.stage_figs.update(pd.DataFrame([fig], index=(3, path_id)))
        return self.stage_figs.loc[3, path_id]["figure"], self.df.query(f"stage == 3 and path_id == {path_id}")[[
                "model_id", "feasible"]]
    
    def replan(self, path_id):
        fig = go.Figure(self.stage_figs.loc[0, path_id]["figure"])
        for n, model in enumerate(self.models["model"]):
            query = self.df.query(f"stage == 4 and path_id == {path_id} and model_id == {n}")
            if query.empty:
                model_name = os.path.basename(self.models.query(f"index == {n}").index[0]).split('.')[0]
                env_id = self.data_input.loc[n, path_id]["env_id"]
                polygon_env = env_npy_to_polygon(self.env_centers[env_id])
                path = self.df.query(f"stage == 3 and path_id == {path_id} and model_id == {n}")[["x", "y"]].values
                result, replanned_path = replan_path(path, polygon_env, self.data_input.loc[n, path_id]["input"], model)
                new_data = pd.DataFrame([[path_id, model_name, n, 4, x, y, result] for x, y in replanned_path],
                                        columns=["path_id", "model", "model_id", "stage", "x", "y", "feasible"])
                self.df = self.df.append(new_data, ignore_index=True)
        scatter = px.scatter(self.df.query(f'stage == 4 and path_id == {path_id}'), x="x", y="y", color="model")
        self.update_figure(fig, path_id, scatter, True, title="Replanning")
        new_entry = pd.DataFrame([[4, path_id, fig]], columns=["stage", "path_id", "figure"])
        new_entry = new_entry.set_index(["stage", "path_id"])
        try:
            self.stage_figs = self.stage_figs.append(new_entry, verify_integrity=True)
        except ValueError:
            self.stage_figs.update(pd.DataFrame([fig], index=(4, path_id)))
        return self.stage_figs.loc[4, path_id]["figure"], self.df.query(f"stage == 4 and path_id == {path_id}")[
            ['model_id', 'feasible']]
    
    def final_path(self, path_id):
        fig = go.Figure(self.stage_figs.loc[0, path_id]["figure"])
        for n, model in enumerate(self.models["model"]):
            query = self.df.query(f"stage == 5 and path_id == {path_id} and model_id == {n}")
            if query.empty:
                model_name = os.path.basename(self.models.query(f"index == {n}").index[0]).split('.')[0]
                env_id = self.data_input.loc[n, path_id]["env_id"]
                polygon_env = env_npy_to_polygon(self.env_centers[env_id])
                path = self.df.query(f"stage == 4 and path_id == {path_id} and model_id == {n}")[["x", "y"]].values
                lvc_replanned = lvc(path, polygon_env)
                feasible = feasibility_check(lvc_replanned, polygon_env)
                new_data = pd.DataFrame([[path_id, model_name, n, 5, x, y, feasible] for x, y in lvc_replanned],
                                        columns=["path_id", "model", "model_id", "stage", "x", "y", "feasible"])
                self.df = self.df.append(new_data, ignore_index=True)
        scatter = px.scatter(self.df.query(f'stage == 5 and path_id == {path_id}'), x="x", y="y", color="model")
        self.update_figure(fig, path_id, scatter, True, title="Final Path")
        new_entry = pd.DataFrame([[5, path_id, fig]], columns=["stage", "path_id", "figure"])
        new_entry = new_entry.set_index(["stage", "path_id"])
        try:
            self.stage_figs = self.stage_figs.append(new_entry, verify_integrity=True)
        except ValueError:
            self.stage_figs.update(pd.DataFrame([fig], index=(5, path_id)))
        return self.stage_figs.loc[5, path_id]["figure"], self.df.query(f"stage == 5 and path_id == {path_id}")[[
                "model_id", "feasible"]]
    
    def update_figure(self, fig, scatter, start, goal, lines=False, title=""):
        scatter.update_traces(dict(marker=dict(size=15)), selector=dict(mode="markers"))
        if lines:
            scatter.update_traces(dict(mode="lines+markers"))
        fig.add_traces(scatter.data)
        fig.add_traces(
                [go.Scatter(x=[start[0]], y=[start[1]], mode="markers", marker=dict(color="green", size=20),
                            name="Origin"),
                 go.Scatter(x=[goal[0]], y=[goal[1]], mode="markers", marker=dict(color="red", size=20),
                            name="Goal")])
        fig.update_layout(dict(title=title))
    
    # def update_figure(self, fig, path_id, scatter, lines=False, title=""):
    #     scatter.update_traces(dict(marker=dict(size=15)), selector=dict(mode="markers"))
    #     if lines:
    #         scatter.update_traces(dict(mode="lines+markers"))
    #     fig.add_traces(scatter.data)
    #     path = self.df.query(f"model_id == 0 and path_id == {path_id} and stage == 1")
    #     fig.add_traces([go.Scatter(x=[path['x'].to_list()[0]], y=[path['y'].to_list()[0]], mode="markers",
    #                                marker=dict(color="green", size=20), name="Origin"),
    #                     go.Scatter(x=[path['x'].to_list()[-1]], y=[path['y'].to_list()[-1]], mode="markers",
    #                                marker=dict(color="red", size=20), name="Goal")])
    #     fig.update_layout(dict(title=title))


def main(args):
    fe.app_object = fe.DashApp()
    fe.app_object.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pnet', default="", nargs="+", type=str)
    args = parser.parse_args()
    
    main(args)
