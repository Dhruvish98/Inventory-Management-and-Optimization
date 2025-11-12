"""
Visualization functions for inventory management dashboard
Uses Plotly for interactive charts
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict, Optional


def plot_inventory_depletion(
    months: List[int],
    inventory_levels: List[float],
    reorder_points: List[int],
    eoq: float
) -> go.Figure:
    """
    Create line plot showing inventory depletion over time
    
    Args:
        months: List of month numbers
        inventory_levels: List of inventory levels for each month
        reorder_points: List of months where reorder occurred
        eoq: Economic Order Quantity value
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Main inventory line
    fig.add_trace(go.Scatter(
        x=months,
        y=inventory_levels,
        mode='lines+markers',
        name='Inventory Level',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8, color='#1f77b4'),
        hovertemplate='Month: %{x}<br>Inventory: %{y:.0f} units<extra></extra>'
    ))
    
    # Add labels for each point
    for i, (month, inv) in enumerate(zip(months, inventory_levels)):
        fig.add_annotation(
            x=month,
            y=inv,
            text=f'{inv:.0f}',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor='#636363',
            ax=0,
            ay=-30,
            font=dict(size=9, color='#636363')
        )
    
    # Mark reorder points
    if reorder_points:
        reorder_inventory = [inventory_levels[m-1] for m in reorder_points if m <= len(inventory_levels)]
        fig.add_trace(go.Scatter(
            x=reorder_points[:len(reorder_inventory)],
            y=reorder_inventory,
            mode='markers',
            name='Reorder Point',
            marker=dict(
                size=12,
                color='red',
                symbol='diamond',
                line=dict(width=2, color='darkred')
            ),
            hovertemplate='Reorder at Month: %{x}<br>Inventory: %{y:.0f} units<extra></extra>'
        ))
    
    # Add EOQ reference line
    fig.add_hline(
        y=eoq,
        line_dash="dash",
        line_color="green",
        annotation_text=f"EOQ: {eoq:.0f}",
        annotation_position="right"
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Inventory Depletion Over Time',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Month",
        yaxis_title="Inventory Level (units)",
        hovermode='x unified',
        template="plotly_white",
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            dtick=1
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        )
    )
    
    return fig


def plot_eoq_comparison(
    items: List[str],
    eoq_values: List[float],
    total_costs: List[float]
) -> go.Figure:
    """
    Create comparison chart for EOQ values across items
    
    Args:
        items: List of item names
        eoq_values: List of EOQ values
        total_costs: List of total costs
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=items,
        y=eoq_values,
        name='EOQ',
        marker_color='#1f77b4',
        hovertemplate='Item: %{x}<br>EOQ: %{y:.0f} units<extra></extra>'
    ))
    
    fig.update_layout(
        title='EOQ Comparison Across Items',
        xaxis_title="Items",
        yaxis_title="EOQ (units)",
        template="plotly_white",
        height=400,
        xaxis=dict(tickangle=45)
    )
    
    return fig


def plot_abc_analysis(abc_results: pd.DataFrame) -> go.Figure:
    """
    Create visualization for ABC analysis
    
    Args:
        abc_results: DataFrame with ABC analysis results
    
    Returns:
        Plotly figure object
    """
    # Prepare data
    abc_results_sorted = abc_results.sort_values('cumulative_percentage')
    
    # Color mapping
    color_map = {'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#95E1D3'}
    colors = [color_map.get(cat, '#95E1D3') for cat in abc_results_sorted['category']]
    
    # Create bar chart
    fig = go.Figure()
    
    for category in ['A', 'B', 'C']:
        category_data = abc_results_sorted[abc_results_sorted['category'] == category]
        if len(category_data) > 0:
            fig.add_trace(go.Bar(
                x=category_data['item'],
                y=category_data['cumulative_percentage'],
                name=f'Category {category}',
                marker_color=color_map[category],
                hovertemplate='Item: %{x}<br>Cumulative %: %{y:.2f}%<extra></extra>'
            ))
    
    # Add cumulative percentage line
    fig.add_trace(go.Scatter(
        x=abc_results_sorted['item'],
        y=abc_results_sorted['cumulative_percentage'],
        mode='lines',
        name='Cumulative %',
        line=dict(color='black', width=2, dash='dash'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='ABC Analysis - Cumulative Value Distribution',
        xaxis_title="Items (sorted by value)",
        yaxis_title="Cumulative Percentage (%)",
        template="plotly_white",
        height=500,
        barmode='group',
        xaxis=dict(tickangle=45),
        yaxis2=dict(
            title="Cumulative %",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        hovermode='x unified'
    )
    
    return fig


def plot_forecast_comparison(
    historical: np.ndarray,
    ma_forecast: np.ndarray,
    es_forecast: np.ndarray,
    forecast_periods: int
) -> go.Figure:
    """
    Create comparison chart for different forecasting methods
    
    Args:
        historical: Historical demand data
        ma_forecast: Moving average forecast
        es_forecast: Exponential smoothing forecast
        forecast_periods: Number of forecast periods
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Historical data
    historical_periods = list(range(1, len(historical) + 1))
    fig.add_trace(go.Scatter(
        x=historical_periods,
        y=historical,
        mode='lines+markers',
        name='Historical Demand',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))
    
    # Forecast periods
    forecast_periods_list = list(range(len(historical) + 1, len(historical) + forecast_periods + 1))
    
    # Moving Average forecast
    fig.add_trace(go.Scatter(
        x=forecast_periods_list,
        y=ma_forecast,
        mode='lines+markers',
        name='Moving Average Forecast',
        line=dict(color='#FF6B6B', width=2, dash='dash'),
        marker=dict(size=6, symbol='square')
    ))
    
    # Exponential Smoothing forecast
    fig.add_trace(go.Scatter(
        x=forecast_periods_list,
        y=es_forecast,
        mode='lines+markers',
        name='Exponential Smoothing Forecast',
        line=dict(color='#4ECDC4', width=2, dash='dot'),
        marker=dict(size=6, symbol='diamond')
    ))
    
    # Add vertical line separating historical and forecast
    fig.add_vline(
        x=len(historical) + 0.5,
        line_dash="dash",
        line_color="gray",
        annotation_text="Forecast Start",
        annotation_position="top"
    )
    
    fig.update_layout(
        title='Demand Forecasting Comparison',
        xaxis_title="Period",
        yaxis_title="Demand (units)",
        template="plotly_white",
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def plot_metrics_comparison(
    ma_metrics: Dict[str, float],
    es_metrics: Dict[str, float]
) -> go.Figure:
    """
    Create bar chart comparing forecast accuracy metrics
    
    Args:
        ma_metrics: Metrics for Moving Average
        es_metrics: Metrics for Exponential Smoothing
    
    Returns:
        Plotly figure object
    """
    metrics = ['MAE', 'RMSE', 'MAPE (%)']
    ma_values = [
        ma_metrics.get('mae', 0),
        ma_metrics.get('rmse', 0),
        ma_metrics.get('mape', 0)
    ]
    es_values = [
        es_metrics.get('mae', 0),
        es_metrics.get('rmse', 0),
        es_metrics.get('mape', 0)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Moving Average',
        x=metrics,
        y=ma_values,
        marker_color='#FF6B6B'
    ))
    
    fig.add_trace(go.Bar(
        name='Exponential Smoothing',
        x=metrics,
        y=es_values,
        marker_color='#4ECDC4'
    ))
    
    fig.update_layout(
        title='Forecast Accuracy Metrics Comparison',
        xaxis_title="Metric",
        yaxis_title="Value",
        template="plotly_white",
        height=400,
        barmode='group',
        hovermode='x unified'
    )
    
    return fig


def plot_cost_breakdown(
    ordering_cost: float,
    holding_cost: float,
    total_cost: float
) -> go.Figure:
    """
    Create pie chart for cost breakdown
    
    Args:
        ordering_cost: Total ordering cost
        holding_cost: Total holding cost
        total_cost: Total cost
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure(data=[go.Pie(
        labels=['Ordering Cost', 'Holding Cost'],
        values=[ordering_cost, holding_cost],
        hole=0.4,
        marker_colors=['#1f77b4', '#FF6B6B']
    )])
    
    fig.update_layout(
        title=f'Cost Breakdown (Total: ${total_cost:,.2f})',
        template="plotly_white",
        height=400
    )
    
    return fig


def plot_ai_enhanced_comparison(
    months: List[int],
    normal_levels: List[float],
    normal_reorder_points: List[int],
    normal_eoq: float,
    ai_levels: List[float],
    ai_reorder_points: List[int],
    ai_eoq: Optional[float]
) -> go.Figure:
    """
    Create comparison plot showing normal vs AI-enhanced inventory management
    
    Args:
        months: List of month numbers
        normal_levels: Normal inventory levels
        normal_reorder_points: Months where normal reorder occurred
        normal_eoq: Normal EOQ value
        ai_levels: AI-enhanced inventory levels
        ai_reorder_points: Months where AI reorder occurred
        ai_eoq: AI-optimized EOQ value
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Normal inventory line
    fig.add_trace(go.Scatter(
        x=months,
        y=normal_levels,
        mode='lines+markers',
        name='Normal EOQ Management',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8, color='#1f77b4'),
        hovertemplate='Month: %{x}<br>Inventory: %{y:.0f} units<extra></extra>'
    ))
    
    # AI-enhanced inventory line
    fig.add_trace(go.Scatter(
        x=months,
        y=ai_levels,
        mode='lines+markers',
        name='AI-Enhanced Management',
        line=dict(color='#4ECDC4', width=2, dash='dash'),
        marker=dict(size=8, color='#4ECDC4', symbol='diamond'),
        hovertemplate='Month: %{x}<br>AI Inventory: %{y:.0f} units<extra></extra>'
    ))
    
    # Add labels for normal inventory points
    for i, (month, inv) in enumerate(zip(months, normal_levels)):
        if i % 2 == 0:  # Label every other point to avoid clutter
            fig.add_annotation(
                x=month,
                y=inv,
                text=f'{inv:.0f}',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor='#636363',
                ax=0,
                ay=-30,
                font=dict(size=8, color='#636363')
            )
    
    # Mark normal reorder points
    if normal_reorder_points:
        normal_reorder_inventory = [normal_levels[m-1] for m in normal_reorder_points if m <= len(normal_levels)]
        fig.add_trace(go.Scatter(
            x=normal_reorder_points[:len(normal_reorder_inventory)],
            y=normal_reorder_inventory,
            mode='markers',
            name='Normal Reorder',
            marker=dict(
                size=12,
                color='red',
                symbol='circle',
                line=dict(width=2, color='darkred')
            ),
            hovertemplate='Normal Reorder at Month: %{x}<br>Inventory: %{y:.0f} units<extra></extra>'
        ))
    
    # Mark AI reorder points
    if ai_reorder_points:
        ai_reorder_inventory = [ai_levels[m-1] for m in ai_reorder_points if m <= len(ai_levels)]
        fig.add_trace(go.Scatter(
            x=ai_reorder_points[:len(ai_reorder_inventory)],
            y=ai_reorder_inventory,
            mode='markers',
            name='AI Reorder',
            marker=dict(
                size=12,
                color='green',
                symbol='diamond',
                line=dict(width=2, color='darkgreen')
            ),
            hovertemplate='AI Reorder at Month: %{x}<br>Inventory: %{y:.0f} units<extra></extra>'
        ))
    
    # Add EOQ reference lines
    fig.add_hline(
        y=normal_eoq,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Normal EOQ: {normal_eoq:.0f}",
        annotation_position="right",
        annotation_font_size=10
    )
    
    if ai_eoq:
        fig.add_hline(
            y=ai_eoq,
            line_dash="dot",
            line_color="green",
            annotation_text=f"AI EOQ: {ai_eoq:.0f}",
            annotation_position="right",
            annotation_font_size=10
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Normal vs AI-Enhanced Inventory Management Comparison',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Month",
        yaxis_title="Inventory Level (units)",
        hovermode='x unified',
        template="plotly_white",
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            dtick=1
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        )
    )
    
    return fig

