"""
======================================================================
 Real-Time Frequency and RoCoF Visualizer for European Substations
======================================================================

This Dash web application allows users to visualize high-resolution
electrical frequency measurements from various substations across Europe.
The tool supports interactive analysis of:
 - System frequency trends
 - Estimated phase angles (relative)
 - Rate of Change of Frequency (RoCoF)
 - Threshold exceedances based on ENTSO-E operational limits

Users can select individual substations, compare metrics over time,
and download annotated plots for further analysis. Frequency safety
bands and RoCoF thresholds are included for enhanced situational awareness.

© 2025 Juan S. Giraldo | TNO Energy and Materials Transition
Email: juan.giraldo@tno.nl
"""


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

    # RoCoF
    rocof_col = col.replace('Frequency', 'RoCoF')
    freq_series = df[col].interpolate().bfill().ffill()
    df[rocof_col] = freq_series.diff() / df['Timestamp'].diff().dt.total_seconds()


# Normalize phase angles to a reference
reference_station = 'DE_Ostrhauderfehn:Phase'
for col in phase_angle_cols:
    df[col + '_rel'] = df[col] - df[reference_station]

COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

layout_watermark = dict(
    text='juan.giraldo@tno.nl',
    xref='paper', yref='paper',
    x=0.5, y=0.5,
    showarrow=False,
    font=dict(size=40, color='rgba(100,100,100,0.2)', family='Arial'),
    xanchor='center', yanchor='middle',
    opacity=0.2
)

# === Dash App ===
app = dash.Dash(__name__)
server = app.server  # 👈 this is required
app.title = "Frequency & Phase Angle Visualizer"


app.layout = html.Div([
    html.H2("European Grid Frequency, Phase Angle & RoCoF", style={'textAlign': 'center'}),

    html.Div([
        html.Div([
            html.Label("Select substations:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='station-selector',
                options=[{'label': col.replace(':Frequency', ''), 'value': col} for col in frequency_columns],
                value=frequency_columns[:2],
                multi=True
            )
        ], style={'width': '48%', 'display': 'inline-block', 'paddingRight': '1%'}),

        html.Div([
            html.Label("Select signal type:", style={'fontWeight': 'bold'}),
            dcc.RadioItems(
                id='mode-selector',
                options=[
                    {'label': 'Frequency (Hz)', 'value': 'Frequency'},
                    {'label': 'Phase angle (rad)', 'value': 'Phase'},
                    {'label': 'RoCoF (Hz/s)', 'value': 'RoCoF'}
                ],
                value='Frequency',
                labelStyle={'display': 'inline-block', 'marginRight': '15px'}
            )
        ], style={'width': '48%', 'display': 'inline-block'})
    ], style={'marginBottom': '20px'}),

    dcc.Graph(id='freq-graph', config={'displayModeBar': True})
], style={'padding': '30px', 'fontFamily': 'Arial, sans-serif'})

@app.callback(
    Output('freq-graph', 'figure'),
    [Input('station-selector', 'value'),
     Input('mode-selector', 'value')]
)
def update_graph(selected_stations, mode):
    traces = []
    shapes = []
    if mode == 'Frequency':
        shapes = [
            dict(
                type='line',
                xref='paper', yref='y',
                x0=0, x1=1,
                y0=49.8, y1=49.8,
                line=dict(color='red', width=1.5, dash='dash'),
                name='Lower Limit'
            ),
            dict(
                type='line',
                xref='paper', yref='y',
                x0=0, x1=1,
                y0=50.2, y1=50.2,
                line=dict(color='red', width=1.5, dash='dash'),
                name='Upper Limit'
            ),
             # Optional: dashed lines for the band edges
            dict(type='line', xref='paper', x0=0, x1=1, y0=49.95, y1=49.95,
                 line=dict(color='green', width=1, dash='dash')),
            dict(type='line', xref='paper', x0=0, x1=1, y0=50.05, y1=50.05,
                 line=dict(color='green', width=1, dash='dash')),

        ]

    for i, freq_col in enumerate(selected_stations):
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]

        if mode == 'Frequency':
            col_name = freq_col
            yaxis_title = "Frequency (Hz)"
        elif mode == 'Phase':
            col_name = freq_col.replace('Frequency', 'Phase') + '_rel'
            yaxis_title = "Relative Phase Angle (rad)"
        else:  # RoCoF
            col_name = freq_col.replace('Frequency', 'RoCoF')
            yaxis_title = "RoCoF (Hz/s)"

        label = freq_col.replace(':Frequency', '')
        traces.append(go.Scatter(
            x=df['Timestamp'],
            y=df[col_name],
            mode='lines',
            name=f"{label} ({mode})",
            line=dict(color=color),
            connectgaps=True
        ))

    return {
        'data': traces,
        'layout': go.Layout(
            title=f"{mode} over Time",
            annotations=[layout_watermark],
            xaxis=dict(
                title='Timestamp',
                showgrid=True,
                gridcolor='#eee',
                tickformat='%H:%M:%S.%L',
                tickangle=30,
            ),
            yaxis=dict(title=yaxis_title, showgrid=True, gridcolor='#eee'),
            hovermode='x unified',
            plot_bgcolor='#fafafa',
            paper_bgcolor='#ffffff',
            font=dict(size=14),
            height=600,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            shapes=shapes
        )
    }

if __name__ == '__main__':
    app.run(debug=True)