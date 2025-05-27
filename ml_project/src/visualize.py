import plotly.graph_objects as go
import matplotlib.pyplot as plt
import math
import numpy as np
from plotly.subplots import make_subplots
import seaborn as sns
def calculate_subplot_grid(n_items, cols):
    rows = math.ceil(n_items / cols)
    return (rows, cols)

def generate_bar_trace(x, y, name, text=None, color='rgb(156, 39, 176)', width=0.2):
    return go.Bar(x=x, y=y, name=name, text=text if text is not None else y, textposition='outside', width=width, marker=dict(color=color, line=dict(width=1.5, color='rgba(0,0,0,1)'), opacity=1))

def plot_feature_bars(df, features, target='depression', mode='numerical', subplot_cols=3):
    """
    mode: 'numerical' → plots average value by target
          'categorical' → plots count of feature categories by target
    """
    (rows, cols) = calculate_subplot_grid(len(features), subplot_cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"{('Avg' if mode == 'numerical' else 'Count')} {f} vs {target}" for f in features], horizontal_spacing=0.1, vertical_spacing=0.1)
    for (idx, feature) in enumerate(features):
        row = idx // cols + 1
        col = idx % cols + 1
        if mode == 'numerical':
            grouped = df.groupby(target)[feature].mean()
            trace = generate_bar_trace(x=grouped.index, y=grouped.values, name=f'{feature} avg', text=grouped.round(2))
            fig.add_trace(trace, row=row, col=col)
        elif mode == 'categorical':
            grouped = df.groupby([feature, target])[feature].count().unstack().fillna(0)
            for target_val in [0, 1]:
                y_vals = grouped[target_val] if target_val in grouped else [0] * len(grouped)
                trace = generate_bar_trace(x=grouped.index, y=y_vals, name=f'{feature} - Depressed {target_val}', text=y_vals.round(2))
                fig.add_trace(trace, row=row, col=col)
        else:
            raise ValueError("Invalid mode. Use 'numerical' or 'categorical'.")
    fig.update_layout(height=rows * 300, width=1200, plot_bgcolor='rgba(34, 34, 34, 1)', paper_bgcolor='rgba(34, 34, 34, 1)', font=dict(color='white'), barmode='group' if mode == 'categorical' else 'stack', showlegend=False, title=f"{('Numerical' if mode == 'numerical' else 'Categorical')} Features vs {target.capitalize()}")
    return fig

def plot_correlation_heatmap(df, cols=None, method='pearson', title='Correlation Heatmap', colorscale='Viridis', show_values=False, mask_triangle=None, width=800, height=700):
    """
    Plots a correlation heatmap from a DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        cols (list): Subset of columns to include (default: all numeric)
        method (str): Correlation method - 'pearson', 'spearman', or 'kendall'
        show_values (bool): Overlay correlation values on heatmap
        mask_triangle (str or None): 'upper', 'lower', or None to mask triangle
    """
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[cols].corr(method=method)
    if mask_triangle == 'upper':
        corr = corr.where(np.tril(np.ones(corr.shape)).astype(bool))
    elif mask_triangle == 'lower':
        corr = corr.where(np.triu(np.ones(corr.shape)).astype(bool))
    heatmap = go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale=colorscale, colorbar=dict(title='Correlation'), zmin=-1, zmax=1)
    fig = go.Figure(data=[heatmap])
    if show_values:
        annotations = []
        for i in range(len(corr)):
            for j in range(len(corr)):
                val = corr.values[i][j]
                if not np.isnan(val):
                    annotations.append(dict(x=corr.columns[j], y=corr.columns[i], text=f'{val:.2f}', showarrow=False, font=dict(color='white' if abs(val) > 0.5 else 'black')))
        fig.update_layout(annotations=annotations)
    fig.update_layout(title=title, xaxis_title='Features', yaxis_title='Features', width=width, height=height, template='plotly_dark', margin=dict(l=50, r=50, t=80, b=50))
    return fig

