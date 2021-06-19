import argparse
import math
import os

import numpy as np
import pandas as pd
import plotly.express as px
from future.moves.itertools import zip_longest
from plotly import graph_objs as go

import frontend as fe
from MPNet.enet.data_loader import load_perms
from MPNet.pnet.data_loader import loader
from MPNet.pnet.model import PNet

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"


class Visualizer:
    def __init__(self):
        self.model_ckpts = []
        self.model_basenames = []
        self.models = pd.DataFrame([], columns=["ckpt", "model", "index"])
        self.models = self.models.set_index("ckpt")
        self.dataset = None
        
        self.stage_figs = pd.DataFrame([], columns=["stage", "path_id", "figure"])
        self.stage_figs = self.stage_figs.set_index(["stage", "path_id"])
        
        self.env_figs = pd.DataFrame([], columns=["env_id", "figure"])
        self.env_figs = self.env_figs.set_index("env_id")
        
        self.env_centers = load_perms(110, 0)
        
        self.paths = pd.read_json(f"{project_path}/data/paths.json", orient='table')
    
    def get_available_envs(self):
        envs = self.paths["env_id"].unique()
        return envs
    
    def get_available_paths(self, env):
        paths = self.paths.query(f"env_id == {env}").reset_index(1)['id'].unique()
        return paths
    
    def process(self, path_id):
        data = self.paths.query(f"id == {path_id} and model in {self.model_basenames}")
        env_figure = self.set_env(path_id)
        env_figure = go.Figure(env_figure)
        env_figure.add_traces(
                [go.Scatter(x=[data["bidir"][0][0][0]], y=[data["bidir"][0][0][1]], mode="markers",
                            marker=dict(color="green", size=20, symbol='x'), name="Origin"),
                 go.Scatter(x=[data["bidir"][0][-1][0]], y=[data['bidir'][0][-1][1]],
                            mode="markers", marker=dict(color="red", size=20, symbol='x'),
                            name="Goal")])
        bidir_figures, num_frames = self.bidirectional_planning(data, path_id)
        lsc_figure, num_frames_lsc = self.lsc(data, path_id)
        replan_figure, num_frames_replan = self.replan(data, path_id)
        final_figure = self.final(data, path_id)
        results_idxs = [(model, path_id) for model in self.model_basenames]
        results = data.loc[results_idxs]['result'].to_list()
        return {0: ([env_figure], 1), 1: (bidir_figures, num_frames), 2: (lsc_figure, num_frames_lsc),
                3: (replan_figure, num_frames_replan), 4: (final_figure, 1)}, results
    
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
        reference_scatter = self.reference(path_id)
        self.update_figure(fig, scatter, reference_scatter, start=start_n_goal_info[0], goal=start_n_goal_info[-1],
                           lines=lines,
                           title=title)
        new_entry = pd.DataFrame([[stage, path_id, fig]], columns=["stage", "path_id", "figure"])
        new_entry = new_entry.set_index(["stage", "path_id"])
        try:
            self.stage_figs = self.stage_figs.append(new_entry, verify_integrity=True)
        except ValueError:
            self.stage_figs.update(pd.DataFrame([fig], index=(stage, path_id)))
    
    def load_dataset(self):
        model = self.model_ckpts[0]
        pnet = PNet.load_from_checkpoint(model)
        
        enet_ckpt = pnet.training_config['enet']
        self.dataset = loader(enet_ckpt, f"{project_path}/valEnv", 110, 0, 1, True)
    
    def set_env(self, path_id):
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
                                              name="Obstáculos", marker=dict(color="black")))
            obstacle_fig.update_layout(dict(width=1280, height=720, title="Cenário",
                                            legend=dict(yanchor="bottom", xanchor="left", y=-0.1, x=-0.2)))
            obstacle_fig.update_xaxes(range=[-20, 20])
            obstacle_fig.update_yaxes(range=[-20, 20])
            new_entry = pd.DataFrame([[env_id, obstacle_fig, obstacles]], columns=["env_id", "figure", "obstacles"])
            new_entry = new_entry.set_index("env_id")
            try:
                self.env_figs = self.env_figs.append(new_entry, verify_integrity=True)
            except:
                pass
        return self.env_figs.loc[env_id]['figure']
    
    def update_models(self, ckpts):
        self.model_ckpts = ckpts
        self.model_basenames = [os.path.basename(ckpt).split('.')[0] for ckpt in ckpts]
        if not self.dataset:
            self.load_dataset()
    
    def reference(self, path_id):
        _, _, reference, _ = self.dataset[path_id, True]
        ref_df = pd.DataFrame([["RRT*", x, y] for x, y in reference], columns=["model", "x", "y"])
        scatter = px.scatter(ref_df, x="x", y="y", color="model")
        return scatter
    
    @staticmethod
    def update_figure(fig, scatter, reference_scatter, start, goal, lines=False, title=""):
        scatter.update_traces(name="MPNet", marker=dict(size=15, color="dodgerblue"), selector=dict(mode="markers"))
        reference_scatter.update_traces(marker=dict(size=15, color="darkorange"), selector=dict(mode="markers"))
        if lines:
            scatter.update_traces(mode="lines+markers")
            reference_scatter.update_traces(dict(mode="lines+markers"))
        fig.add_traces(reference_scatter.data)
        fig.add_traces(scatter.data)
        fig.add_traces(
                [go.Scatter(x=[start[0]], y=[start[1]], mode="markers", marker=dict(color="green", size=20),
                            name="Origin"),
                 go.Scatter(x=[goal[0]], y=[goal[1]], mode="markers", marker=dict(color="red", size=20),
                            name="Goal")])
        fig.update_layout(dict(title=title))
    
    def animation_figure_completing(self, figs, path_id, env_id, start, goal, stage, title, lines=False, remove=False):
        reference_scatter = self.reference(path_id)
        if lines:
            reference_scatter.update_traces(mode='lines+markers', marker=dict(size=15, color="darkorange"),
                                            selector=dict(mode="markers"))
        else:
            reference_scatter.update_traces(marker=dict(size=15, color="darkorange"), selector=dict(mode="markers"))
        fig = go.Figure(self.env_figs.loc[env_id][0])
        fig.add_traces(
                [go.Scatter(x=[start[0]], y=[start[1]], mode="markers", marker=dict(color="green", size=20, symbol='x'),
                            name="Origin"),
                 go.Scatter(x=[goal[0]], y=[goal[1]], mode="markers", marker=dict(color="red", size=20, symbol='x'),
                            name="Goal")])
        for n, frame in enumerate(figs):
            frame_data = list(frame.data)
            if n == 0 and remove:
                frame_data = []
            for datum in np.array(reference_scatter.data)[::-1]:
                frame_data.insert(0, datum)
            for datum in np.array(fig.data)[::-1]:
                frame_data.insert(0, datum)
            new_fig = go.Figure()
            new_fig.add_traces(frame_data)
            new_fig.update_layout({"title" : title, "width": 1280, "height": 720,
                                   "legend": {"yanchor": "bottom", "xanchor": "left", "y": -0.1, "x": -0.2}})
            new_fig.update_xaxes(range=[-20, 20])
            new_fig.update_yaxes(range=[-20, 20])
            figs[n] = new_fig
        
        new_entry = pd.DataFrame([[stage, path_id, figs]], columns=["stage", "path_id", "figure"])
        new_entry = new_entry.set_index(["stage", "path_id"])
        try:
            self.stage_figs = self.stage_figs.append(new_entry, verify_integrity=True)
        except ValueError:
            self.stage_figs.update(pd.DataFrame([figs], index=(stage, path_id)))
    
    def bidirectional_planning(self, data, path_id):
        env_id = data["env_id"][0]
        path_info = np.array(data["bidir"].values.tolist()).reshape((-1, 2))
        temp_df = pd.DataFrame([], columns=["itt", "x", "y", "path"])
        path_separator_idx = math.ceil(path_info.shape[0] / 2)
        path_1 = path_info[:path_separator_idx]
        path_2 = path_info[path_separator_idx:][::-1]
        
        for n, (p1, p2) in enumerate(zip_longest(path_1, path_2)):
            if p1 is not None:
                for p in path_1[:n + 1]:
                    temp_df = temp_df.append(
                            pd.DataFrame([[2 * n, p[0], p[1], "sigma 1"]], columns=["itt", "x", "y", "path"]))
                for p in path_2[:n]:
                    temp_df = temp_df.append(
                            pd.DataFrame([[2 * n, p[0], p[1], "sigma 2"]], columns=["itt", "x", "y", "path"]))
                if n == 0:
                    temp_df = temp_df.append(
                            pd.DataFrame([[0, p2[0], p2[1], "sigma 2"]], columns=["itt", "x", "y", "path"]))
            
            if p2 is not None and n != 0:
                for p in path_1[:n + 1]:
                    temp_df = temp_df.append(
                            pd.DataFrame([[2 * n + 1, p[0], p[1], "sigma 1"]], columns=["itt", "x", "y", "path"]))
                for p in path_2[:n + 1]:
                    temp_df = temp_df.append(
                            pd.DataFrame([[2 * n + 1, p[0], p[1], "sigma 2"]], columns=["itt", "x", "y", "path"]))
        num_frames = temp_df["itt"].nunique()
        iterations = pd.unique(temp_df["itt"])
        colors = {0: 'dodgerblue', 1: 'deeppink'}
        figs = []
        for itt in iterations:
            new_fig = go.Figure()
            data = temp_df.query(f"itt == {itt}")
            visible_figs = [True, True]
            for n, datum in data.iterrows():
                if datum['path'] == "sigma 1":
                    new_fig.add_trace(
                            go.Scatter(x=[datum['x']], y=[datum['y']], marker={'color': colors[0], 'size': 15},
                                       legendgroup=datum['path'], name=datum['path'], showlegend=visible_figs[0]))
                    visible_figs[0] = False
                else:
                    new_fig.add_trace(
                            go.Scatter(x=[datum['x']], y=[datum['y']], marker={'color': colors[1], 'size': 15},
                                       legendgroup=datum['path'], name=datum['path'], showlegend=visible_figs[1]))
                    visible_figs[1] = False
            
            figs.append(new_fig)
        
        self.animation_figure_completing(figs, path_id, env_id, path_1[0], path_2[0], "bidir",
                                         title="Estratégia de Planejamento Bidirecional", lines=False, remove=True)
        return self.stage_figs.query(f"stage == 'bidir' and path_id == {path_id}")["figure"].item(), num_frames
    
    def lsc(self, data, path_id):
        env_id = data["env_id"][0]
        path_info = np.array(data["lvc"].values.tolist()).reshape((-1, 2))
        path_before = np.array(data["bidir"].values.tolist()).reshape((-1, 2))
        temp_df = pd.DataFrame([], columns=["x", "y", "step"])
        
        for n, p in enumerate(path_before):
            temp_df = temp_df.append(
                    pd.DataFrame([[0, p[0], p[1], "Antes"]], columns=["itt", "x", "y", "step"]))
        
        for n, p in enumerate(path_info):
            temp_df = temp_df.append(
                    pd.DataFrame([[1, p[0], p[1], "LSC"]], columns=["itt", "x", "y", "step"]))
        for n, p in enumerate(path_before):
            temp_df = temp_df.append(
                    pd.DataFrame([[1, p[0], p[1], "Antes"]], columns=["itt", "x", "y", "step"]))
        
        num_frames = temp_df["itt"].nunique()
        iterations = pd.unique(temp_df["itt"])
        colors = {0: 'dodgerblue', 1: 'deeppink'}
        figs = []
        for itt in iterations:
            new_fig = go.Figure()
            before_data = temp_df.query(f"itt == {itt} and step ==  'Antes'")
            lsc_data = temp_df.query(f"itt == {itt} and step ==  'LSC'")
            new_fig.add_trace(
                    go.Scatter(x=before_data['x'], y=before_data['y'], marker={'color': colors[0], 'size': 15},
                               mode='lines+markers', name='Antes'))
            new_fig.add_trace(go.Scatter(x=lsc_data['x'], y=lsc_data['y'], marker={'color': colors[1], 'size': 15},
                                         mode='lines+markers', name='LSC', line=dict(dash='dash')))
            figs.append(new_fig)
        
        self.animation_figure_completing(figs, path_id, env_id, path_info[0], path_info[-1], "lvc",
                                         title="Lazy States Contraction", lines=True)
        return self.stage_figs.query(f"stage == 'lvc' and path_id == {path_id}")["figure"].item(), num_frames
    
    def replan(self, data, path_id):
        env_id = data["env_id"][0]
        if not pd.isna(data['replan']).item():
            path_info = np.array(data["replan"].values.tolist()).reshape((-1, 2))
        else:
            path_info = []
        path_before = np.array(data["lvc"].values.tolist()).reshape((-1, 2))
        temp_df = pd.DataFrame([], columns=["itt", "x", "y", "step"])
        
        for n, p in enumerate(path_before):
            temp_df = temp_df.append(
                    pd.DataFrame([[0, p[0], p[1], "LSC"]], columns=["itt", "x", "y", "step"]))
        
        for n, p in enumerate(path_before):
            temp_df = temp_df.append(
                    pd.DataFrame([[1, p[0], p[1], "LSC"]], columns=["itt", "x", "y", "step"]))
        for n, p in enumerate(path_info):
            temp_df = temp_df.append(
                    pd.DataFrame([[1, p[0], p[1], "Replanejado"]], columns=["itt", "x", "y", "step"]))
        
        num_frames = temp_df["itt"].nunique()
        iterations = pd.unique(temp_df["itt"])
        colors = {0: 'dodgerblue', 1: 'deeppink'}
        figs = []
        for itt in iterations:
            new_fig = go.Figure()
            lsc_data = temp_df.query(f"itt == {itt} and step ==  'LSC'")
            replan_data = temp_df.query(f"itt == {itt} and step ==  'Replanejado'")
            new_fig.add_trace(
                    go.Scatter(x=lsc_data['x'], y=lsc_data['y'], marker={'color': colors[0], 'size': 15},
                               mode='lines+markers', name='LSC'))
            new_fig.add_trace(
                    go.Scatter(x=replan_data['x'], y=replan_data['y'], marker={'color': colors[1], 'size': 15},
                               mode='lines+markers', name='Replanejado'))
            figs.append(new_fig)
        
        self.animation_figure_completing(figs, path_id, env_id, path_before[0], path_before[-1], "replan",
                                         title="Replanejamento da Trajetória", lines=True)
        return self.stage_figs.query(f"stage == 'replan' and path_id == {path_id}")["figure"].item(), num_frames
    
    def final(self, data, path_id):
        env_id = data["env_id"][0]
        if not pd.isna(data['replan']).item():
            path_info = np.array(data["final"].values.tolist()).reshape((-1, 2))
        else:
            path_info = np.array(data["lvc"].values.tolist()).reshape((-1, 2))
        temp_df = pd.DataFrame([], columns=["x", "y", "step"])
        
        for n, p in enumerate(path_info):
            temp_df = temp_df.append(
                    pd.DataFrame([[p[0], p[1], "Final"]], columns=["x", "y", "step"]))
        
        colors = {0: 'dodgerblue', 1: 'deeppink'}
        figs = []
        new_fig = go.Figure()
        new_fig.add_trace(
                go.Scatter(x=temp_df['x'], y=temp_df['y'], marker={'color': colors[1], 'size': 15},
                           mode='lines+markers', name='MPNet'))
        figs.append(new_fig)
        
        self.animation_figure_completing(figs, path_id, env_id, path_info[0], path_info[-1], "final",
                                         title="Trajetória Resultante", lines=True)
        return self.stage_figs.query(f"stage == 'final' and path_id == {path_id}")["figure"].item()


def main(args):
    fe.app_object = fe.DashApp()
    fe.app_object.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pnet', default="", nargs="+", type=str)
    args = parser.parse_args()
    
    main(args)
