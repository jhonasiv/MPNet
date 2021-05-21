import os

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import ALL, Input, Output, State
from plotly import graph_objs as go

from MPNet.visualizer.visualizer import Visualizer

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"
stylesheet = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash("MPNet", external_stylesheets=stylesheet)


class DashApp:
    def __init__(self, animated_graphs=[]):
        self.vis = Visualizer()
        self.stage_figures = {0: go.Figure(layout=dict(title="Environment")),
                              1: go.Figure(layout=dict(title="Bidirectional Planning")),
                              2: go.Figure(layout=dict(title="States Connection")),
                              3: go.Figure(layout=dict(title="LVC")),
                              4: go.Figure(layout=dict(title="Replanning")),
                              5: go.Figure(layout=dict(title="Final Path"))}
        # self.graphs = {i: dcc.Graph(id={"type": "graph", "index": i}, animate=i in animated_graphs) for i in
        #                self.stage_figures.keys()}
        self.graph = dcc.Graph(id="graph")
        self.frame_sliders = {i: dcc.Slider(id={"type": "dynamic_slider", "index": i}, min=1, max=2, value=1,
                                            step=1) for i in self.stage_figures.keys()}
        # self.feasible_buttons = {
        #         i: html.Button("FEASIBLE", id={"type": "feasible_but", "index": i}, n_clicks=0, disabled=True,
        #                        style={"textAlign"  : "center", "padding": "0 15px", "font-size": "14px",
        #                               'color'      : "white", "background": "red", "cursor": "default",
        #                               "font-weight": "bold"}) for i in self.stage_figures.keys()}
        self.manual_update = dcc.Interval(id="update_interval", disabled=True, n_intervals=0, interval=1000,
                                          max_intervals=1)
        self.slide_update = dcc.Interval(id="slide_interval", disabled=True, n_intervals=0, interval=1200,
                                         max_intervals=-1)
        self.pause_button = html.Button("PAUSE", id="pause_but", n_clicks=0,
                                        style={"width"     : "100px", "textAlign": "center", "marginBottom": "20px",
                                               "marginLeft": '20px'})
        self.path_input = dcc.Input(id="path_input", type="number", placeholder="Set new PATH ID",
                                    style={"marginRight": "20px"}, debounce=True, disabled=True)
        self.path_dropdown = dcc.Dropdown(id="path_dropdown", placeholder="Choose previously analysed path",
                                          style={"marginRight": "20px", "width": "100%"}, disabled=True)
        self.model_dropdown = dcc.Dropdown(id="model_dropdown", placeholder="Set analised models",
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
                        html.Div([self.path_input, self.path_dropdown, self.model_dropdown],
                                 style={"marginBottom"   : "10px", "display": 'flex', "align-items": "center",
                                        "justify-content": "flex-start"}),
                        html.Div(id="buttons+feasible", children=[
                                html.Div(id="buttons", children=[
                                        html.Button("PLAY", id="play_but", n_clicks=0,
                                                    style={"width"       : "100px", "textAlign": "center",
                                                           "marginBottom": "20px"}),
                                        self.pause_button, ], style={"display": "flex"}),
                                ],
                                 style={"display": "flex", "justify-content": "space-between", "paddingRight": "5px"}),
                        dcc.Tabs(id="stage_tabs", value="0", children=[
                                dcc.Tab(label="Environment", value="0"),
                                dcc.Tab(label="Bidirectional", value="1"),
                                dcc.Tab(label="States Connection", value="2"),
                                dcc.Tab(label="LVC", value="3"),
                                dcc.Tab(label="Replanning", value="4"),
                                dcc.Tab(label="Final Path", value="5")
                                ])], style={"margin-bottom": "15px", "margin-left": "40px", "margin-right": "40px"}),
                html.Div([self.graph, self.status_div], style={"display": "flex", "alignItems": "center"}),
                self.slide_update,
                self.manual_update,
                ])
    
    def new_path(self, path_id):
        results = self.vis.update_stages(path_id)
        self.stage_figures, self.status[path_id] = results
    
    def run(self):
        app.run_server(debug=True)


app_object: DashApp


@app.callback(Output("graph", "figure"),
              [Input("stage_tabs", "value"), Input("update_interval", "n_intervals")])
def update_stage(value, enabled):
    value = int(value)
    return app_object.stage_figures[value]


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
    return list(status.values()), style


@app.callback(Output("slide_interval", "disabled"), Input("play_but", "n_clicks"), Input("pause_but", "n_clicks"),
              prevent_initial_call=True)
def change_slide_state(play, pause):
    caller = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'play' in caller:
        return False
    elif 'pause' in caller:
        return True


# @app.callback(Output({"type": "feasible_but", "index": MATCH}, "style"),
#               Input({"type": "feasible_but", "index": MATCH}, "n_clicks"))
# def set_feasible(value):
#     app_object.feasible_button_style["background"] = "green" if value else "red"
#     return app_object.feasible_button_style


@app.callback(Output("path_dropdown", "options"), Output("path_dropdown", "value"), Input("path_input", "value"),
              State("path_dropdown", "options"), prevent_initial_call=True)
def update_dropdown_options(value, options):
    if options is None:
        options = []
    options.append({"label": f"{value}", "value": value})
    return options, value


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


@app.callback(Output("path_input", "disabled"), Output("path_dropdown", "disabled"), Input("model_dropdown", "value"),
              prevent_initial_call=True)
def enable_path_selector(ckpts):
    if not isinstance(ckpts, list):
        return dash.no_update
    if len(ckpts) > 0:
        app_object.vis.update_models(ckpts)
        return False, False
    else:
        return True, True


if __name__ == '__main__':
    app_object = DashApp()
    app_object.run()
