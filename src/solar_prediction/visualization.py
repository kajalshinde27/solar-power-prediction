"""Plotly charts for historical solar output and forecasts."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_historical_solar(
    df: pd.DataFrame,
    date_column: str = "timestamp",
    value_column: str = "solar_output",
    title: str = "Historical Solar Output",
    height: int = 400,
) -> go.Figure:
    """Plot historical solar output as an interactive line chart with dark theme.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing historical data with timestamp and solar_output columns.
    date_column : str
        Name of the timestamp column.
    value_column : str
        Name of the solar output column.
    title : str
        Chart title.
    height : int
        Chart height in pixels.

    Returns
    -------
    go.Plotly Figure
        Interactive line chart of historical solar output with customized styling.
    """
    if date_column not in df.columns:
        raise ValueError(f"DataFrame missing date column: {date_column}")
    if value_column not in df.columns:
        raise ValueError(f"DataFrame missing value column: {value_column}")

    # Use plotly Express for a more modern look
    fig = px.line(
        df,
        x=date_column,
        y=value_column,
        title=title,
        template="plotly_dark",
        labels={value_column: "Solar Output (W/m²)", date_column: "Time"},
    )

    # Customize the trace for a nicer look
    fig.update_traces(
        line=dict(color="#ffd700", width=3),
        hovertemplate="<b>%{x|%H:%M}</b><br>Solar Output: %{y:.1f}<extra></extra>",
    )

    # Dark theme styling updates
    fig.update_layout(
        title=dict(x=0.5, font=dict(size=20, color="#ffd700")),
        xaxis=dict(
            gridcolor="#555555",
            gridwidth=1,
            tickfont=dict(color="#cccccc"),
        ),
        yaxis=dict(
            gridcolor="#555555",
            gridwidth=1,
            tickfont=dict(color="#cccccc"),
        ),
        legend=dict(font=dict(color="#ffffff")),
        margin=dict(l=60, r=60, t=60, b=60),
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e",
    )

    return fig


def plot_historical_vs_forecast(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    historical_date_column: str = "timestamp",
    forecast_date_column: str = "timestamp",
    historical_value_column: str = "solar_output",
    forecast_value_column: str = "predicted_solar_output",
    historical_name: str = "Historical",
    forecast_name: str = "Predicted",
    title: str = "Historical Solar Output vs. Forecast",
    height: int = 500,
) -> go.Figure:
    """Plot historical solar output and future predictions on shared time axis.

    Clearly distinguishes observed historical values from predicted future values
    with a dark theme style using gold and orange-red lines.

    Parameters
    ----------
    historical_df : pd.DataFrame
        Historical data with timestamp and solar output columns.
    forecast_df : pd.DataFrame
        Forecast data with timestamp and predicted solar output columns.
    historical_date_column : str
        Name of the timestamp column in historical DataFrame.
    forecast_date_column : str
        Name of the forecast date column in forecast DataFrame.
    historical_value_column : str
        Name of the solar output column in historical DataFrame.
    forecast_value_column : str
        Name of the predicted solar output column in forecast DataFrame.
    historical_name : str
        Legend label for historical data.
    forecast_name : str
        Legend label for predicted data.
    title : str
        Chart title.
    height : int
        Chart height in pixels.

    Returns
    -------
    go.Plotly Figure
        Interactive chart showing historical and forecast solar output with dark theme styling.
    """
    if historical_date_column not in historical_df.columns:
        raise ValueError(
            f"Historical DataFrame missing date column: {historical_date_column}"
        )
    if historical_value_column not in historical_df.columns:
        raise ValueError(
            f"Historical DataFrame missing value column: {historical_value_column}"
        )
    if forecast_date_column not in forecast_df.columns:
        raise ValueError(
            f"Forecast DataFrame missing date column: {forecast_date_column}"
        )
    if forecast_value_column not in forecast_df.columns:
        raise ValueError(
            f"Forecast DataFrame missing value column: {forecast_value_column}"
        )

    # Add historical trace with gold color
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=historical_df[historical_date_column],
            y=historical_df[historical_value_column],
            mode="lines",
            name=historical_name,
            line=dict(color="#ffd700", width=3),
            hovertemplate="<b>%{x|%H:%M}</b><br>%{name}: %{y:.1f}<extra></extra>",
        ),
    )

    # Add forecast trace with orange-red color
    fig.add_trace(
        go.Scatter(
            x=forecast_df[forecast_date_column],
            y=forecast_df[forecast_value_column],
            mode="lines+markers",
            name=forecast_name,
            line=dict(color="#ff6b6b", width=3),
            marker=dict(size=6, color="#ffd700"),
            hovertemplate="<b>%{x|%H:%M}</b><br>%{name}: %{y:.1f}<extra></extra>",
        ),
    )

    # Add vertical line separating historical from forecast
    if len(historical_df) > 0 and len(forecast_df) > 0:
        last_hist_time = historical_df[historical_date_column].iloc[-1]
        first_forecast_time = forecast_df[forecast_date_column].iloc[0]
        fig.add_vline(
            x=last_hist_time,
            line_dash="dash",
            line_color="#555555",
            annotation_text="Forecast Start",
            annotation_position="top left",
        )

    fig.update_layout(
        title=dict(x=0.5, font=dict(size=18, color="#ffd700")),
        xaxis=dict(
            title="Time",
            gridcolor="#555555",
            gridwidth=1,
            tickfont=dict(color="#cccccc"),
        ),
        yaxis=dict(
            title="Solar Output (W/m²)",
            gridcolor="#555555",
            gridwidth=1,
            tickfont=dict(color="#cccccc"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#ffffff"),
        ),
        height=height,
        template="plotly_dark",
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e",
        margin=dict(l=60, r=60, t=60, b=60),
    )

    return fig

