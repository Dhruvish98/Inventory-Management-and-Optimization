"""
Sample Inventory Data Generator
Creates realistic inventory data for demonstration purposes
"""

import pandas as pd
import numpy as np
from typing import Optional


def generate_sample_inventory_data(
    num_items: int = 50,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Generate sample inventory data with realistic distributions
    
    Args:
        num_items: Number of inventory items to generate
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with inventory data
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate item IDs
    item_ids = [f"ITEM_{i+1:03d}" for i in range(num_items)]
    
    # Generate product names (mix of categories)
    categories = ["Electronics", "Clothing", "Food", "Furniture", "Toys", "Books", "Sports"]
    product_names = [
        f"{np.random.choice(categories)} Product {i+1}" 
        for i in range(num_items)
    ]
    
    # Generate annual demand (skewed distribution - some items more popular)
    # Using log-normal distribution for realistic demand patterns
    annual_demand = np.random.lognormal(mean=6.5, sigma=1.2, size=num_items).astype(int)
    annual_demand = np.clip(annual_demand, 100, 50000)  # Reasonable bounds
    
    # Generate unit costs (correlated with demand - popular items might be cheaper or more expensive)
    # Mix of low, medium, and high value items
    unit_cost = np.random.lognormal(mean=2.5, sigma=1.0, size=num_items)
    unit_cost = np.clip(unit_cost, 1.0, 500.0)
    
    # Generate ordering costs (setup costs vary by item complexity)
    ordering_cost = np.random.lognormal(mean=3.5, sigma=0.8, size=num_items)
    ordering_cost = np.clip(ordering_cost, 10.0, 500.0)
    
    # Generate holding cost rates (as percentage of unit cost)
    holding_cost_rate = np.random.normal(loc=20, scale=5, size=num_items)
    holding_cost_rate = np.clip(holding_cost_rate, 5, 40)
    
    # Calculate holding cost per unit
    holding_cost = (holding_cost_rate / 100) * unit_cost
    
    # Generate lead times (in days)
    lead_time = np.random.gamma(shape=2, scale=7, size=num_items).astype(int)
    lead_time = np.clip(lead_time, 1, 30)
    
    # Generate demand variance (coefficient of variation)
    demand_variance = np.random.gamma(shape=1.5, scale=0.1, size=num_items)
    demand_variance = np.clip(demand_variance, 0.05, 0.5)
    
    # Generate current stock levels (random between 0.5x to 2x of monthly demand)
    monthly_demand = annual_demand / 12
    current_stock = np.random.uniform(0.5, 2.0, size=num_items) * monthly_demand
    current_stock = current_stock.astype(int)
    
    # Create DataFrame
    data = {
        'item_id': item_ids,
        'product_name': product_names,
        'annual_demand': annual_demand,
        'unit_cost': np.round(unit_cost, 2),
        'ordering_cost': np.round(ordering_cost, 2),
        'holding_cost_rate': np.round(holding_cost_rate, 2),
        'holding_cost': np.round(holding_cost, 2),
        'lead_time_days': lead_time,
        'demand_variance': np.round(demand_variance, 3),
        'current_stock': current_stock,
        'total_value': np.round(annual_demand * unit_cost, 2)
    }
    
    df = pd.DataFrame(data)
    
    return df


def generate_historical_demand_data(
    item_id: str,
    annual_demand: float,
    months: int = 12,
    variance: float = 0.2,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate historical monthly demand data for an item
    
    Args:
        item_id: Item identifier
        annual_demand: Annual demand value
        months: Number of months of historical data
        variance: Coefficient of variation for demand
        seed: Random seed
    
    Returns:
        DataFrame with monthly demand data
    """
    if seed is not None:
        np.random.seed(hash(item_id) % 1000 if seed == 42 else seed)
    
    monthly_avg = annual_demand / 12
    monthly_demand = np.random.normal(
        loc=monthly_avg,
        scale=monthly_avg * variance,
        size=months
    )
    monthly_demand = np.maximum(monthly_demand, 0)  # No negative demand
    
    # Add some seasonality (optional)
    seasonality = 1 + 0.1 * np.sin(2 * np.pi * np.arange(months) / 12)
    monthly_demand = monthly_demand * seasonality
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=months, freq='M')
    
    df = pd.DataFrame({
        'item_id': item_id,
        'date': dates,
        'demand': np.round(monthly_demand, 0).astype(int)
    })
    
    return df

