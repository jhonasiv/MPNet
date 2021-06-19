import multiprocessing as mp
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import ALL, Input, MATCH, Output, State
from plotly import graph_objs as go

from MPNet.visualizer.visualizer import Visualizer

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"
stylesheet = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash("MPNet", external_stylesheets=stylesheet)


class DashApp:
    def __init__(self, animated_graphs=[]):
        self.vis = Visualizer()
        self.stage_figures = {0: [go.Figure(layout=dict(title="Cenário"))],
                              1: [go.Figure(layout=dict(title="Planejador Bidirecional"))],
                              # 2: go.Figure(layout=dict(title="Conexão de estados críticos")),
                              2: [go.Figure(layout=dict(title="LSC"))],
                              3: [go.Figure(layout=dict(title="Replanejamento"))],
                              4: [go.Figure(layout=dict(title="Trajetória resultante"))]}
        
        self.frame_sliders = {0: dcc.Slider(id={"type": "dynamic_slider", "index": 0}, value=0, min=0, max=1,
                                            disabled=True),
                              1: dcc.Slider(id={"type": "dynamic_slider", "index": 1}, min=1, max=2, value=1, step=1),
                              2: dcc.Slider(id={"type": "dynamic_slider", "index": 2}, min=1, max=2, value=1, step=1),
                              3: dcc.Slider(id={"type": "dynamic_slider", "index": 3}, min=1, max=2, value=1, step=1),
                              4: dcc.Slider(id={"type": "dynamic_slider", "index": 4}, min=1, max=1, value=1,
                                            disabled=True),
                              }
        self.num_frames = {i: 2 for i in self.frame_sliders.keys()}
        
        # self.graphs = {i: dcc.Graph(id={"type": "graph", "index": i}, animate=i in animated_graphs) for i in
        #                self.stage_figures.keys()}
        self.graph = dcc.Graph(id="graph")
        # self.frame_sliders = {i: dcc.Slider(id={"type": "dynamic_slider", "index": i}, min=1, max=2, value=1,
        #                                     step=1) for i in self.stage_figures.keys()}
        self.manual_update = dcc.Interval(id="update_interval", disabled=True, n_intervals=0, interval=1000,
                                          max_intervals=1)
        self.slide_update = dcc.Interval(id="slide_interval", disabled=True, n_intervals=0, interval=1200,
                                         max_intervals=-1)
        self.pause_button = html.Button("PREVIOUS", id="pause_but", n_clicks=0, disabled=True,
                                        style={"width"     : "100px", "textAlign": "center", "marginBottom": "20px",
                                               "marginLeft": '20px', 'padding': '0px'})
        self.path_input = dcc.Input(id="path_input", type="number", placeholder="Set new PATH ID",
                                    style={"marginRight": "20px"}, debounce=True, disabled=True)
        
        envs = self.vis.get_available_envs()
        
        self.env_dropdown = dcc.Dropdown(id="env_dropdown", placeholder="Select an environment",
                                         options=[{"label": e, "value": e} for e in envs],
                                         style={"marginRight": "20px", "width": "100%"}, disabled=True)
        self.path_dropdown = dcc.Dropdown(id="path_dropdown", placeholder="Choose previously analysed path",
                                          style={"marginRight": "20px", "width": "100%"}, disabled=True)
        self.model_dropdown = dcc.Dropdown(id="model_dropdown", placeholder="Set analysed models",
                                           style={"marginRight": "20px", "width": "100%"},
                                           multi=True,
                                           options=[{'label': model, "value": f'{project_path}/models/{model}'} for
                                                    model in sorted(os.listdir(f"{project_path}/models")) if "pnet" in
                                                    model])
        
        # self.feasible_button_style = {"width"     : "125px", "textAlign": "center", "color": "white",
        #                               "background": "red", "cursor": "default", "font-weight": "bold",
        #                               "font-size" : "14px", "padding": "0 15px"}
        # html.Div(id="feasible_div",
        #          style={"width"  : "125px", "marginBottom": "20px", "float": "right",
        #                 "display": "flex",
        #                 "padding": "0 15px", "justify-content": "flex-end"})
        # graph_divs = {i: html.Div(id={"type": "graph_div", "index": i},
        #                           style={"display": "flex", "justify-content": "center"}) for i in
        #               self.stage_figures.keys()}
        self.status_buttons = {}
        self.status = {}
        
        self.status_div = html.Div(id="status_div", style={"display"                   : "flex",
                                                           "flex-direction"            : "column",
                                                           "border"                    : "1px solid #5555",
                                                           "borderRadius"              : "5px",
                                                           "margin"                    : "10px", "padding": "5px",
                                                           "justifyContent"            :
                                                               "space-around", "height": "fit-content"})
        
        app.layout = html.Div([
                html.H2("MPNet Execution Step by Step"),
                html.Div([
                        html.Div([self.env_dropdown, self.path_dropdown, self.model_dropdown],
                                 style={"marginBottom"   : "10px", "display": 'flex', "align-items": "center",
                                        "justify-content": "flex-start"}),
                        html.Div(id="buttons_sliders", children=[
                                html.Button("NEXT", id="play_but", n_clicks=0, disabled=True,
                                            style={"width"       : "100px", "textAlign": "center",
                                                   "marginBottom": "20px"}),
                                self.pause_button], style={"display": "flex"}),
                        dcc.Tabs(id="stage_tabs", value="0", children=[
                                dcc.Tab(id={"type": "tab", "index": 0}, label="Cenário", value="0"),
                                dcc.Tab(id={"type": "tab", "index": 1}, label="Plan. Bidirecional", value="1"),
                                # dcc.Tab(label="Conexão de Estados", value="2"),
                                dcc.Tab(id={"type": "tab", "index": 2}, label="Otimizador", value="2"),
                                dcc.Tab(id={"type": "tab", "index": 3}, label="Replanejamento", value="3"),
                                dcc.Tab(id={"type": "tab", "index": 4}, label="Traj. Final", value="4")
                                ])], style={"margin-bottom": "15px", "margin-left": "40px", "margin-right": "40px"}),
                html.Div([self.graph, self.status_div], style={"display": "flex", "alignItems": "center"}),
                self.slide_update,
                self.manual_update,
                ])
        self.event = mp.Event()
        self.event.set()
    
    def new_path(self, path_id):
        if path_id is not None:
            if not self.event.is_set():
                self.event.wait()
            else:
                self.event.clear()
                results, self.status[path_id] = self.vis.process(path_id)
                for i in results.keys():
                    self.stage_figures[i], self.num_frames[i] = results[i]
                
                # results = self.vis.update_stages(path_id)
                # self.stage_figures, self.status[path_id] = results
                self.event.set()
    
    def run(self):
        app.run_server(debug=True)


app_object: DashApp


@app.callback(Output("graph", "figure"), Input({"type": "dynamic_slider", "index": ALL}, "value"),
              Input("stage_tabs", "value"), Input("path_dropdown", "value"), prevent_initial_call=True)
def update_graph(sliders_value, stage, _):
    if 0 in sliders_value:
        return dash.no_update
    stage = int(stage)
    value = sliders_value[0]
    try:
        return app_object.stage_figures[stage][value - 1]
    except IndexError:
        return app_object.stage_figures[stage][0]


# @app.callback(Output({"type": "dynamic_slider", "index": MATCH}, "value"), Input("slide_interval", "n_intervals"),
#               Input({"type": "tab", "index": MATCH}, "value"), prevent_initial_call=True)
# def update_sliders(value, stage):
#     stage = int(stage)
#     # sliders_value = [f.value for f in app_object.frame_sliders.values()]
#     value = value % app_object.num_frames[stage]
#     if value == app_object.num_frames[stage] - 1:
#         app_object.slide_update.disabled = True
#     return value + 1


@app.callback(Output("buttons_sliders", "children"), Input("stage_tabs", "value"),
              State("buttons_sliders", "children"))
def update_sliders(value, buttons_sliders):
    value = int(value)
    app_object.frame_sliders[value].max = app_object.num_frames[value]
    if value == 1:
        app_object.frame_sliders[value].marks = {i: f'{i}' for i in range(app_object.num_frames[value] + 1)}
    elif value == 2:
        app_object.frame_sliders[value].marks = {1: "Antes", 2: "LSC"}
    elif value == 3:
        app_object.frame_sliders[value].marks = {1: "LSC", 2: "Replanejado"}
    
    if len(buttons_sliders) < 3:
        buttons_sliders.append(
                html.Div(children=[app_object.frame_sliders[value]],
                         style={"width"  : "100%", "padding-top": "12px",
                                "display": "none" if value in (0, 4) else "block"}))
    else:
        buttons_sliders[-1] = html.Div(children=[app_object.frame_sliders[value]],
                                       style={"width"  : "100%", "padding-top": "12px",
                                              "display": "none" if value in (0, 4) else "block"})
    return buttons_sliders


@app.callback(Output("stage_tabs", "value"), Input("path_dropdown", "value"), prevent_initial_call=True)
def dropdown_selected(value):
    app_object.new_path(value)
    return "0"


@app.callback(Output({"type": "status_but", "index": ALL}, "children"),
              Output({"type": "status_but", "index": ALL}, "style"), Input("path_dropdown", "value"),
              State({"type": "status_but", "index": ALL}, "style"), prevent_initial_call=True)
def update_feasibility(value, style):
    app_object.new_path(value)
    status = app_object.status[value]
    for n, st in enumerate(style):
        st["background-color"] = "red" if 'Failure' in status[n] else "green"
        st["color"] = "white"
    return status, style


@app.callback(Output({"type": "dynamic_slider", "index": MATCH}, "value"), Input("play_but", "n_clicks"),
              Input("pause_but", "n_clicks"), Input({"type": "tab", "index": MATCH}, "value"),
              State({"type": "dynamic_slider", "index": MATCH}, "value"),
              State({"type": "dynamic_slider", "index": MATCH}, "max"), prevent_initial_call=True)
def change_slide_state(play, pause, tab, current_value, current_max):
    caller = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'play' in caller and play > 0:
        if current_value + 1 <= current_max:
            return current_value + 1
        else:
            return 1
    elif 'pause' in caller and pause > 0:
        if current_value - 1 >= 1:
            return current_value - 1
        else:
            return current_max
    return dash.no_update



@app.callback(Output("status_div", "children"), Input("model_dropdown", "value"), prevent_initial_call=True)
def add_status(models):
    children = []
    app_object.status_buttons = {}
    for n, model in enumerate(models):
        model_name = os.path.basename(model).split('.')[0]
        button = html.Button("Status", disabled=True, id={"type": "status_but", "index": n},
                             style={'marginLeft': "8px"})
        app_object.status_buttons[n] = button
        new_child = html.Div(children=[model_name, button],
                             style={"display": "flex", "justifyContent": "space-between", "align-items": "center"})
        children.append(new_child)
    return children


@app.callback(Output("play_but", "disabled"), Output("pause_but", "disabled"), Input("path_dropdown", "value"),
              State("stage_tabs", "value"), prevent_initial_call=True)
def enable_buttons(value, tab):
    if value is not None and tab not in (0, 4):
        return False, False
    else:
        return True, True


@app.callback(Output("path_dropdown", "disabled"), Output("path_dropdown", "options"), Input("env_dropdown", "value"),
              prevent_initial_call=True)
def enable_path_selector(value):
    if value is not None:
        path_options = app_object.vis.get_available_paths(value)
        options = [{"label": p, "value": p} for p in path_options]
        return False, options
    else:
        return True, []


@app.callback(Output("env_dropdown", "disabled"), Input("model_dropdown", "value"),
              prevent_initial_call=True)
def enable_path_selector(ckpts):
    if not isinstance(ckpts, list):
        return dash.no_update
    if len(ckpts) > 0:
        app_object.vis.update_models(ckpts)
        return False
    else:
        return True


if __name__ == '__main__':
    app_object = DashApp()
    app_object.run()
