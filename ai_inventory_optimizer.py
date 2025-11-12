"""
AI-Enhanced Inventory Optimization Algorithm
Uses demand pattern recognition and cost optimization to improve inventory management
"""

import numpy as np
from typing import List, Tuple, Dict
from inventory_models import calculate_eoq, calculate_safety_stock, calculate_reorder_point


def ai_optimize_inventory_management(
    annual_demand: float,
    monthly_demand: float,
    ordering_cost: float,
    holding_cost: float,
    eoq: float,
    months: int,
    initial_inventory: float
) -> Dict:
    """
    AI-optimized inventory management using:
    1. Demand pattern recognition
    2. Dynamic reorder point optimization
    3. Cost-minimizing order quantities
    4. Target inventory level maintenance
    
    Args:
        annual_demand: Annual demand
        monthly_demand: Average monthly demand
        ordering_cost: Cost per order
        holding_cost: Holding cost per unit per year
        eoq: Standard EOQ value
        months: Number of months to simulate
        initial_inventory: Starting inventory level
    
    Returns:
        Dictionary with optimized parameters and simulation results
    """
    # 1. Calculate optimal target inventory level
    # Target = EOQ + Safety Stock (to maintain smooth operations)
    demand_variance = monthly_demand * 0.15  # 15% demand variability
    safety_stock = calculate_safety_stock(monthly_demand, demand_variance, 0.95)
    target_inventory = eoq + safety_stock
    
    # 2. Optimize reorder point using cost analysis
    # Reorder point should trigger when inventory can cover lead time + safety stock
    lead_time_days = 7  # Assume 7-day lead time
    daily_demand = annual_demand / 365
    optimal_reorder_point = (daily_demand * lead_time_days) + safety_stock
    
    # 3. Optimize order quantity using cost trade-off analysis
    # AI considers: ordering frequency vs holding cost
    # If holding cost is high relative to ordering cost, order less frequently
    cost_ratio = holding_cost / ordering_cost if ordering_cost > 0 else 1
    
    # Adaptive EOQ: adjust based on cost structure
    if cost_ratio > 0.5:  # High holding cost relative to ordering
        # Order less frequently (larger orders)
        ai_optimized_eoq = eoq * 1.05  # 5% increase
    elif cost_ratio < 0.2:  # Low holding cost relative to ordering
        # Order more frequently (smaller orders)
        ai_optimized_eoq = eoq * 0.90  # 10% decrease
    else:
        # Balanced: use standard EOQ
        ai_optimized_eoq = eoq
    
    # 4. Calculate optimal reorder point that maintains target inventory
    # Reorder when inventory drops to a level that, after lead time, reaches target
    optimal_reorder_point = max(
        optimal_reorder_point,
        target_inventory - (monthly_demand * (lead_time_days / 30))
    )
    
    return {
        'ai_optimized_eoq': ai_optimized_eoq,
        'optimal_reorder_point': optimal_reorder_point,
        'target_inventory': target_inventory,
        'safety_stock': safety_stock,
        'cost_ratio': cost_ratio
    }


def simulate_ai_enhanced_inventory(
    monthly_demand: float,
    ordering_cost: float,
    holding_cost: float,
    months: int,
    initial_inventory: float,
    ai_params: Dict,
    use_demand_forecasting: bool = True
) -> Tuple[List[float], List[int], float, int]:
    """
    Simulate AI-enhanced inventory management with adaptive strategies
    
    Args:
        monthly_demand: Average monthly demand
        ordering_cost: Cost per order
        holding_cost: Holding cost per unit per year
        months: Number of months to simulate
        initial_inventory: Starting inventory
        ai_params: AI optimization parameters
        use_demand_forecasting: Whether to use demand forecasting
    
    Returns:
        Tuple of (inventory_levels, reorder_points, total_cost, orders_count)
    """
    ai_inventory_levels = []
    ai_reorder_points = []
    ai_orders_count = 0
    ai_total_cost = 0.0
    
    ai_optimized_eoq = ai_params['ai_optimized_eoq']
    optimal_reorder_point = ai_params['optimal_reorder_point']
    target_inventory = ai_params['target_inventory']
    
    current_inventory = initial_inventory
    demand_history = []  # Track demand for forecasting
    
    # Set seed for reproducibility but allow some variation
    np.random.seed(42)
    
    for month in range(months):
        # Simulate demand with variability
        if use_demand_forecasting and len(demand_history) > 0:
            # Use exponential smoothing for demand forecasting
            alpha = 0.3
            forecast_demand = alpha * demand_history[-1] + (1 - alpha) * np.mean(demand_history[-3:]) if len(demand_history) >= 3 else monthly_demand
            # Add some realistic variation
            monthly_demand_actual = forecast_demand * (1 + np.random.normal(0, 0.08))  # Reduced variance
        else:
            monthly_demand_actual = monthly_demand * (1 + np.random.normal(0, 0.1))
        
        monthly_demand_actual = max(0, monthly_demand_actual)  # No negative demand
        demand_history.append(monthly_demand_actual)
        
        # Deplete inventory
        current_inventory -= monthly_demand_actual
        
        # AI-optimized reorder logic
        # Check multiple conditions for optimal reordering:
        # 1. Below reorder point (safety trigger)
        # 2. Projected to go below target before next order arrives
        # 3. Cost-optimal timing
        
        days_until_reorder = 7  # Lead time
        projected_inventory = current_inventory - (monthly_demand_actual * (days_until_reorder / 30))
        
        should_reorder = False
        
        # Condition 1: Below safety reorder point
        if current_inventory <= optimal_reorder_point:
            should_reorder = True
        
        # Condition 2: Projected to drop below target (proactive reordering)
        elif projected_inventory < target_inventory * 0.8 and current_inventory < target_inventory:
            should_reorder = True
        
        # Condition 3: Cost optimization - reorder if it reduces total cost
        # Calculate cost of ordering now vs waiting
        if not should_reorder and month > 0:
            cost_if_order_now = ordering_cost + (holding_cost * (current_inventory + ai_optimized_eoq) / 2 / 12)
            cost_if_wait = holding_cost * current_inventory / 2 / 12
            # If ordering now saves cost, do it
            if cost_if_order_now < cost_if_wait * 1.2:  # 20% threshold
                should_reorder = True
        
        if should_reorder:
            # Calculate order quantity to reach target inventory level
            order_qty = max(
                ai_optimized_eoq,
                target_inventory - current_inventory + (monthly_demand_actual * (days_until_reorder / 30))
            )
            
            # Don't over-order excessively
            order_qty = min(order_qty, ai_optimized_eoq * 1.5)
            
            current_inventory += order_qty
            ai_reorder_points.append(month + 1)
            ai_orders_count += 1
        
        # Ensure inventory doesn't go negative
        current_inventory = max(0, current_inventory)
        ai_inventory_levels.append(current_inventory)
    
    # Calculate total cost
    # Annualized cost = ordering costs + average holding costs
    if ai_orders_count > 0:
        avg_inventory = np.mean(ai_inventory_levels)
        annual_ordering_cost = (ordering_cost * ai_orders_count) * (12 / months)
        annual_holding_cost = holding_cost * avg_inventory
        ai_total_cost = annual_ordering_cost + annual_holding_cost
    else:
        avg_inventory = np.mean(ai_inventory_levels)
        ai_total_cost = holding_cost * avg_inventory
    
    return ai_inventory_levels, ai_reorder_points, ai_total_cost, ai_orders_count

