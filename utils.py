# utility library to keep main notebook or python file clean
# for each class or function, write detailed docstring for readability

from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd

def plot_ts_open_hatch(dfi=None, fac_id=None, t_drone_open_hatch=None, plot_dir=None):
    """ 
    plotting function to plot time series of one single facility
    dfi: pd.DataFrame with two columns, one is timestamp, one is pressure_osi
    t_drone_open_hatch: datetime or string in a format that could be converted to datetime
    fac_id: facility id
    fac_name: facility name
    plot_dir: directory to save the figure
    """
    assert isinstance(dfi, pd.DataFrame)
    assert 'timestamp' in dfi
    assert 'pressure_osi' in dfi

    fig = px.scatter(dfi, x='timestamp', y='pressure_osi')

    if t_drone_open_hatch is not None:
        assert pd.to_datetime(t_drone_open_hatch), 'needs to in a format convertable to datetime'
        t_drone_open_hatch = pd.to_datetime(t_drone_open_hatch)
        fig.add_traces(px.line(x=[t_drone_open_hatch, t_drone_open_hatch], y=[dfi.pressure_osi.min(), dfi.pressure_osi.max()], color_discrete_sequence=['red']).data)

    fig.update_layout(height=900, width=1600, title=f'Facility ID: {fac_id}, hatch open: {t_drone_open_hatch}', xaxis_title='DateTime', yaxis_title='Pressure, OSI', font=dict(size=18))

    if plot_dir is not None: fig.write_image(f'{plot_dir}/{fac_id}.png', engine='orca')
    
    return fig


def plot_prediction_validation(df, df_pred, facility_id):

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True)

    # plot pressure with drone detected open hatch time
    fig_ts = plot_ts_open_hatch(dfi=df, fac_id=facility_id)

    fig.add_trace(fig_ts.data[0], row=1, col=1)

    # plot groud truth bar chart
    fig_truth = px.bar(df_pred, x='TimeStamp', y='Status_Truth')
    fig.add_trace(fig_truth.data[0], row=2, col=1)

    # plot model results bar chart
    fig_pred = px.bar(df_pred, x='TimeStamp', y='Status_Predicted')
    fig.add_trace(fig_pred.data[0], row=3, col=1)

    # plot model results bar chart
    fig_conf = px.line(df_pred, x='TimeStamp', y='Model_Confidence')
    fig.add_trace(fig_conf.data[0], row=4, col=1)

    fig.update_xaxes(title_text='TimeStamp', row=4, col=1)
    fig.update_yaxes(title_text = 'Pressure', row=1, col=1)
    fig.update_yaxes(title_text = 'Ground Truth', row=2, col=1)
    fig.update_yaxes(title_text = 'Prediction', row=3, col=1)
    fig.update_yaxes(title_text = 'Confidence', row=4, col=1)

    fig.update_layout(title_text= f'Model predicted hatch status vs groud truth {facility_id}', height=900, width=1600, font=dict(size=16))

    return fig