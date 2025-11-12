"""
Inventory Management Models and Calculations
Includes EOQ, ABC Analysis, Safety Stock, Reorder Point, and Forecasting models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def calculate_eoq(annual_demand: float, ordering_cost: float, holding_cost: float) -> float:
    """
    Calculate Economic Order Quantity (EOQ)
    
    Formula: EOQ = sqrt((2 * D * S) / H)
    where:
    D = Annual demand
    S = Ordering cost per order
    H = Holding cost per unit per year
    
    Args:
        annual_demand: Annual demand in units
        ordering_cost: Cost per order
        holding_cost: Holding cost per unit per year
    
    Returns:
        EOQ value
    """
    if annual_demand <= 0 or ordering_cost <= 0 or holding_cost <= 0:
        return 0.0
    
    eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
    return max(1, round(eoq, 2))


def calculate_safety_stock(
    lead_time_demand: float,
    lead_time_variance: float,
    service_level: float = 0.95
) -> float:
    """
    Calculate Safety Stock
    
    Formula: Safety Stock = Z * sqrt(LT * Var(D) + D^2 * Var(LT))
    where Z is the z-score for the desired service level
    
    Args:
        lead_time_demand: Average demand during lead time
        lead_time_variance: Variance of demand during lead time
        service_level: Desired service level (default 0.95 = 95%)
    
    Returns:
        Safety stock value
    """
    try:
        from scipy import stats
        z_score = stats.norm.ppf(service_level)
    except ImportError:
        # Fallback: approximate z-score for common service levels
        z_scores = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
        z_score = z_scores.get(service_level, 1.65)  # Default to 95% if not found
    
    safety_stock = z_score * np.sqrt(lead_time_variance)
    return max(0, round(safety_stock, 2))


def calculate_reorder_point(
    average_demand: float,
    lead_time_days: float,
    safety_stock: float = 0,
    demand_variance: float = 0
) -> float:
    """
    Calculate Reorder Point
    
    Formula: ROP = (Average Daily Demand × Lead Time) + Safety Stock
    
    Args:
        average_demand: Average daily demand
        lead_time_days: Lead time in days
        safety_stock: Safety stock level
        demand_variance: Variance of demand
    
    Returns:
        Reorder point value
    """
    daily_demand = average_demand / 365 if average_demand > 365 else average_demand / 30
    reorder_point = (daily_demand * lead_time_days) + safety_stock
    return max(0, round(reorder_point, 2))


def abc_analysis(
    df: pd.DataFrame,
    value_column: str,
    item_column: str,
    threshold_a: float = 80,
    threshold_b: float = 95
) -> pd.DataFrame:
    """
    Perform ABC Analysis
    
    Categorizes items into A, B, and C categories based on their value contribution:
    - Category A: Top items contributing to threshold_a% of total value
    - Category B: Next items contributing to threshold_b% of total value
    - Category C: Remaining items
    
    Args:
        df: DataFrame with inventory data
        value_column: Column name containing item values
        item_column: Column name containing item identifiers
        threshold_a: Percentage threshold for Category A (default 80%)
        threshold_b: Percentage threshold for Category B (default 95%)
    
    Returns:
        DataFrame with ABC classification
    """
    # Create a copy to avoid modifying original
    result_df = df.copy()
    
    # Calculate total value per item
    if value_column not in result_df.columns:
        # Try to calculate from demand and unit cost
        demand_col = next((col for col in result_df.columns if 'demand' in col.lower()), None)
        cost_col = next((col for col in result_df.columns if 'cost' in col.lower() or 'price' in col.lower()), None)
        
        if demand_col and cost_col:
            result_df['calculated_value'] = result_df[demand_col] * result_df[cost_col]
            value_column = 'calculated_value'
        else:
            raise ValueError(f"Value column '{value_column}' not found and cannot be calculated")
    
    # Sort by value descending
    result_df = result_df.sort_values(by=value_column, ascending=False).reset_index(drop=True)
    
    # Calculate cumulative values and percentages
    result_df['cumulative_value'] = result_df[value_column].cumsum()
    total_value = result_df[value_column].sum()
    result_df['cumulative_percentage'] = (result_df['cumulative_value'] / total_value) * 100
    
    # Classify items
    result_df['category'] = 'C'  # Default to C
    result_df.loc[result_df['cumulative_percentage'] <= threshold_a, 'category'] = 'A'
    result_df.loc[
        (result_df['cumulative_percentage'] > threshold_a) & 
        (result_df['cumulative_percentage'] <= threshold_b),
        'category'
    ] = 'B'
    
    # Rename item column for consistency
    if item_column in result_df.columns:
        result_df = result_df.rename(columns={item_column: 'item'})
    else:
        result_df['item'] = result_df.index.astype(str)
    
    return result_df


def moving_average_forecast(
    historical_data: np.ndarray,
    forecast_periods: int,
    window: int = 3
) -> np.ndarray:
    """
    Generate forecast using Moving Average method
    
    Args:
        historical_data: Array of historical demand values
        forecast_periods: Number of periods to forecast
        window: Moving average window size
    
    Returns:
        Array of forecasted values
    """
    if len(historical_data) < window:
        # If not enough data, use average of available data
        avg = np.mean(historical_data) if len(historical_data) > 0 else 0
        return np.full(forecast_periods, avg)
    
    # Calculate moving average
    ma_value = np.mean(historical_data[-window:])
    
    # Forecast future periods with the same value (simple approach)
    # In practice, you might want to add trend or seasonality
    forecast = np.full(forecast_periods, ma_value)
    
    return forecast


def exponential_smoothing_forecast(
    historical_data: np.ndarray,
    forecast_periods: int,
    alpha: float = 0.3
) -> np.ndarray:
    """
    Generate forecast using Exponential Smoothing method
    
    Args:
        historical_data: Array of historical demand values
        forecast_periods: Number of periods to forecast
        alpha: Smoothing constant (0 < alpha <= 1)
    
    Returns:
        Array of forecasted values
    """
    if len(historical_data) == 0:
        return np.zeros(forecast_periods)
    
    alpha = max(0.01, min(1.0, alpha))  # Ensure alpha is in valid range
    
    # Calculate exponentially smoothed value
    smoothed_value = historical_data[0]
    for value in historical_data[1:]:
        smoothed_value = alpha * value + (1 - alpha) * smoothed_value
    
    # Forecast future periods
    forecast = np.full(forecast_periods, smoothed_value)
    
    return forecast


def calculate_metrics(actual: np.ndarray, forecast: np.ndarray) -> Dict[str, float]:
    """
    Calculate forecast accuracy metrics
    
    Metrics:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - MAPE: Mean Absolute Percentage Error
    
    Args:
        actual: Array of actual values
        forecast: Array of forecasted values
    
    Returns:
        Dictionary with metric names and values
    """
    if len(actual) != len(forecast) or len(actual) == 0:
        return {'mae': 0, 'rmse': 0, 'mape': 0}
    
    # Remove any zero or negative values for MAPE calculation
    actual_positive = np.where(actual > 0, actual, 1)
    
    errors = actual - forecast
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2
    percentage_errors = (abs_errors / actual_positive) * 100
    
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(squared_errors))
    mape = np.mean(percentage_errors)
    
    return {
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'mape': round(mape, 2)
    }


def calculate_total_cost(
    annual_demand: float,
    ordering_cost: float,
    holding_cost: float,
    order_quantity: float
) -> Dict[str, float]:
    """
    Calculate total inventory costs
    
    Args:
        annual_demand: Annual demand
        ordering_cost: Cost per order
        holding_cost: Holding cost per unit per year
        order_quantity: Order quantity
    
    Returns:
        Dictionary with cost breakdown
    """
    if order_quantity <= 0:
        return {
            'ordering_cost': 0,
            'holding_cost': 0,
            'total_cost': 0
        }
    
    ordering_cost_total = (ordering_cost * annual_demand) / order_quantity
    holding_cost_total = (holding_cost * order_quantity) / 2
    total_cost = ordering_cost_total + holding_cost_total
    
    return {
        'ordering_cost': round(ordering_cost_total, 2),
        'holding_cost': round(holding_cost_total, 2),
        'total_cost': round(total_cost, 2)
    }

