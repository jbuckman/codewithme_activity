import torch
from bokeh.io import push_notebook
from bokeh.layouts import column, row
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Span, Slider, Toggle
from bokeh.models.callbacks import CustomJS
from math import pi as π

X_RANGE = (-1., 1.)
X_LINSPACE = torch.linspace(*X_RANGE, 200)
BASIS = [lambda x: 1,
         lambda x: x,
         lambda x: x**2,
         lambda x: x**3,
         lambda x: torch.sin(2**-1 * 2*π*x),
         lambda x: torch.sin(2**0 * 2*π*x),
         lambda x: torch.sin(2**1 * 2*π*x),
         lambda x: torch.sin(2**2 * 2*π*x)]

def function_from_parameters(p):
    def fn(x): return sum(p[i]*BASIS[i](x if isinstance(x, torch.Tensor) else torch.tensor(x)) for i in range(len(BASIS)))
    return fn

def get_random_target():
    p = torch.randn(8)
    target_fn = function_from_parameters(p)
    return target_fn

def get_examples(fn, n_datapoints):
    x = (torch.rand(n_datapoints) * (X_RANGE[1] - X_RANGE[0]) + X_RANGE[0])
    y = fn(x)
    return list(zip(x.tolist(), y.tolist()))

class task:
    def __init__(self, p, n):
        self.target_function = function_from_parameters(p)
        self.dataset = get_examples(self.target_function, n)

def interactive_plot(train_points=None, target_function=None):
    def plotter(doc):
        plot = figure(height=400, width=600, x_range=X_RANGE, y_range=[-6, 6])
        vline = Span(location=0, dimension='height', line_color='black', line_width=2)
        hline = Span(location=0, dimension='width', line_color='black', line_width=2)
        plot.renderers.extend([vline, hline])

        my_function_datasource = ColumnDataSource(data=dict(x=X_LINSPACE.tolist(), y=(0.*X_LINSPACE).tolist()))
        mfl = plot.line('x', 'y', source=my_function_datasource, color='blue', line_width=3)
        if train_points is not None:
            x, y = zip(*train_points)
            tpc = plot.circle(x, y, color='red', size=5)
        if target_function is not None:
            y = target_function(X_LINSPACE)
            tfl = plot.line(X_LINSPACE.tolist(), y.tolist(), color='red', line_dash="dashed", line_width=3, line_alpha=0. if train_points is not None else 1.)

        widgets = []

        sliders = [Slider(value=0.0, start=-3.0, end=3.0, step=0.25) for _ in range(8)]
        def update_data(attrname, old, new):
            parameters = [s.value for s in sliders]
            my_function = function_from_parameters(parameters)
            my_function_datasource.data = dict(x=X_LINSPACE.tolist(), y=my_function(X_LINSPACE).tolist())
        for s in sliders: s.on_change('value', update_data)
        widgets += sliders

        if target_function is not None and train_points is not None:
            target_toggle = Toggle(label="Reveal target function", active=False)
            def update_target_visible(new):
                if new == True: tfl.glyph.line_alpha = 1.; target_toggle.label = "Hide target function"
                else:           tfl.glyph.line_alpha = 0.; target_toggle.label = "Reveal target function"
            target_toggle.on_click(update_target_visible)
            widgets += [target_toggle]

        inputs = column(*widgets)
        doc.add_root(row(inputs, plot))
    return plotter

def live_plot(train_points, my_function, target_function):
    plot = figure(height=400, width=600, x_range=X_RANGE, y_range=[-6, 6])
    vline = Span(location=0, dimension='height', line_color='black', line_width=2)
    hline = Span(location=0, dimension='width', line_color='black', line_width=2)
    plot.renderers.extend([vline, hline])

    my_function_datasource = ColumnDataSource(data=dict(x=X_LINSPACE.tolist(), y=my_function(X_LINSPACE).tolist()))
    plot.line('x', 'y', source=my_function_datasource, color='blue', line_width=3)
    plot.circle(*zip(*train_points), color='red', size=5)
    tfl = plot.line(X_LINSPACE.tolist(), target_function(X_LINSPACE).tolist(), color='red', line_dash="dashed", line_width=3, line_alpha=0.)

    sliders = [Slider(value=0.0, start=-3.0, end=3.0, step=.01, disabled=True) for _ in range(8)]
    target_toggle = Toggle(label="Reveal target function", active=False)
    callback = CustomJS(args=dict(tflg=tfl.glyph, tt=target_toggle), code="""
        if (cb_obj.active == true) {tflg.line_alpha = 1.; tt.label = 'Hide target function'}
        else                       {tflg.line_alpha = 0.; tt.label = 'Reveal target function'};
    """)
    target_toggle.js_on_click(callback)

    widgets = sliders + [target_toggle]
    return row(column(*widgets), plot), my_function_datasource, sliders

def update_live_plot(parameters, my_function_datasource, sliders):
    for p, s in zip(parameters, sliders): s.value = p.item()
    my_function_datasource.data = dict(x=X_LINSPACE.tolist(), y=function_from_parameters(parameters)(X_LINSPACE).tolist())
    push_notebook()

easy_target_1 = function_from_parameters([0., -.5, 0., 0., 0., 0., 0., 0.])
easy_target_2 = function_from_parameters([2., 0., 0., 0., 0., 0., 0., 0.])
easy_target_3 = function_from_parameters([0., 0., 0., 0., 0., 0., 1.5, 0.])
easy_target_4 = function_from_parameters([1., -1., 0., 0., 0., 0., 0., 0.])
medium_target_1 = function_from_parameters([-1., 0.5, 0., 0., 0., 0., 1., 0.])
medium_target_2 = function_from_parameters([0., 0., 0., 0., 1., 0., 0.5, 0.5])
medium_target_3 = function_from_parameters([0., -1., 0.5, 0., 0., 0.1, 0., 0.])
medium_target_4 = function_from_parameters([-2., 0., 0., 0.5, -1., 1., 0., 0.])
hard_target_1 = function_from_parameters([-0.5000,  0.2500,  1.7500,  0.5000, -1.0000, -0.7500,  0.7500,  0.7500])
hard_target_2 = function_from_parameters([ 0.0000,  1.5000, -1.5000, -1.0000,  0.7500, -0.2500,  1.2500, -1.0000])
hard_target_3 = function_from_parameters([ 2.5000, -1.5000, -0.0000, -0.2500,  0.5000, -0.2500,  0.2500,  0.2500])
hard_target_4 = function_from_parameters([ 0.5000,  1.5000,  0.7500, -1.0000,  0.5000, -1.0000,  2.5000,  1.0000])


easy_target_small_dataset  = task([0., 0., 0., 0.5, 0., 0., 0., 0.], 5)
easy_target_medium_dataset = task([1., 0., -1., 0., 0., 0., 0., 0.], 25)
easy_target_large_dataset  = task([-1., 0., 0., 0., 0., 1., 0., 0.], 1000)
medium_target_small_dataset  = task([0.5, -1., 0., 0., 0.5, 0., 0., 0.], 5)
medium_target_medium_dataset = task([0., 0., 0., 0., -1., 0., 0., 1.], 25)
medium_target_large_dataset  = task([2., 0., -1.5, 0., 0., 1., 1., 0.], 1000)
hard_target_small_dataset  = task([-1.5000,  0.5000,  1.0000, -2.7500, -0.2500,  0.5000,  0.7500,  0.2500], 5)
hard_target_medium_dataset = task([ 0.2500, -0.7500,  0.7500,  0.5000, -1.5000, -0.7500,  0.2500, -0.5000], 25)
hard_target_large_dataset  = task([-0.2500,  0.0000, -0.5000, -0.5000,  0.0000, -0.2500, -0.2500, -0.5000], 1000)