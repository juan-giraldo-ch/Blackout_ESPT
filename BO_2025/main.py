import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np



# Load and preprocess the CSV
df = pd.read_csv('28042025_Spain and Portugal_UTCtime.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# === Compute phase angles ===
def compute_phase_angle(df, freq_col, nominal_freq=50.0):
    timestamps = df['Timestamp']
    freqs = df[freq_col].interpolate().bfill().ffill().values


    time_sec = (timestamps - timestamps.iloc[0]).dt.total_seconds().values
    dt = np.diff(time_sec, prepend=time_sec[0])
    delta_f = freqs - nominal_freq
    delta_theta = 2 * np.pi * delta_f * dt
    phase_angle = np.cumsum(delta_theta)
    return phase_angle

# Get all frequency columns
frequency_columns = [col for col in df.columns if 'Frequency' in col]

# Compute absolute phase angles
phase_angle_cols = []
for col in frequency_columns:
    phase_col = col.replace('Frequency', 'Phase')
    df[phase_col] = compute_phase_angle(df, col)
    phase_angle_cols.append(phase_col)

# Normalize phase angles to a reference
reference_station = 'DE_Ostrhauderfehn:Phase'
for col in phase_angle_cols:
    df[col + '_rel'] = df[col] - df[reference_station]

# === Dash App ===
app = dash.Dash(__name__)
app.title = "Frequency & Phase Angle Visualizer"


app.layout = html.Div([
    html.H2("Frequency and Phase Angle Visualization"),

    html.Div([
        html.Label("Select substations:"),
        dcc.Dropdown(
            id='station-selector',
            options=[{'label': col.replace(':Frequency', ''), 'value': col} for col in frequency_columns],
            value=frequency_columns[:2],
            multi=True
        )
    ], style={'marginBottom': '20px'}),

    html.Div([
        html.Label("Select signal type:"),
        dcc.RadioItems(
            id='mode-selector',
            options=[
                {'label': 'Frequency (Hz)', 'value': 'Frequency'},
                {'label': 'Phase angle (rad)', 'value': 'Phase'}
            ],
            value='Frequency',
            labelStyle={'display': 'inline-block', 'marginRight': '10px'}
        )
    ], style={'marginBottom': '20px'}),

    dcc.Graph(id='freq-graph')
])

@app.callback(
    Output('freq-graph', 'figure'),
    [Input('station-selector', 'value'),
     Input('mode-selector', 'value')]
)
def update_graph(selected_stations, mode):
    traces = []
    for freq_col in selected_stations:
        if mode == 'Frequency':
            col_name = freq_col
        else:
            col_name = freq_col.replace('Frequency', 'Phase') + '_rel'

        label = freq_col.replace(':Frequency', '')
        traces.append(go.Scatter(
            x=df['Timestamp'],
            y=df[col_name],
            mode='lines',
            name=f"{label} ({mode})",
            connectgaps=True
        ))

    yaxis_title = "Frequency (Hz)" if mode == 'Frequency' else "Phase Angle (rad)"

    return {
        'data': traces,
        'layout': go.Layout(
            xaxis={'title': 'Timestamp'},
            yaxis={'title': yaxis_title},
            hovermode='closest',
            height=600
        )
    }

if __name__ == '__main__':
    app.run(debug=True)