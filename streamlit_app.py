import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from inventory_models import (
    calculate_eoq, calculate_safety_stock, calculate_reorder_point,
    abc_analysis, moving_average_forecast, exponential_smoothing_forecast,
    calculate_metrics
)
from data_generator import generate_sample_inventory_data
from visualizations import (
    plot_inventory_depletion, plot_eoq_comparison, plot_abc_analysis,
    plot_forecast_comparison, plot_metrics_comparison
)
from ai_analysis import get_gemini_analysis, generate_pdf_report

# Page configuration
st.set_page_config(
    page_title="Inventory Management & Optimization Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'inventory_data' not in st.session_state:
    st.session_state.inventory_data = None
if 'generated_data' not in st.session_state:
    st.session_state.generated_data = False

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📦 Inventory Dashboard")
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigate to:",
        ["🏠 Home", "📊 EOQ Analysis", "🔤 ABC Analysis", "📈 Forecasting", "📉 Performance Metrics", "🤖 AI Analysis & Reports"],
        key="navigation"
    )
    
    st.markdown("---")
    
    # Data Generation Section
    st.subheader("📥 Data Management")
    
    if st.button("🔄 Generate Sample Data", use_container_width=True):
        with st.spinner("Generating sample inventory data..."):
            st.session_state.inventory_data = generate_sample_inventory_data()
            st.session_state.generated_data = True
            st.success("✅ Sample data generated successfully!")
    
    # File upload
    uploaded_file = st.file_uploader("Or upload CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            st.session_state.inventory_data = pd.read_csv(uploaded_file)
            st.session_state.generated_data = True
            st.success("✅ Data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    # Display data info
    if st.session_state.inventory_data is not None:
        st.markdown("---")
        st.subheader("📋 Data Info")
        st.write(f"**Items:** {len(st.session_state.inventory_data)}")
        st.write(f"**Columns:** {', '.join(st.session_state.inventory_data.columns.tolist())}")

# Main content based on selected page
if page == "🏠 Home":
    st.markdown('<div class="main-header">Inventory Management & Optimization Dashboard</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **📊 EOQ Analysis**
        - Economic Order Quantity calculation
        - Inventory depletion visualization
        - Cost optimization
        """)
    
    with col2:
        st.info("""
        **🔤 ABC Analysis**
        - Categorize items by value
        - Priority-based management
        - Resource allocation
        """)
    
    with col3:
        st.info("""
        **📈 Forecasting**
        - Demand forecasting models
        - Multiple techniques comparison
        - Accuracy metrics
        """)
    
    st.markdown("---")
    
    if st.session_state.inventory_data is not None:
        st.subheader("📋 Current Inventory Data Preview")
        st.dataframe(st.session_state.inventory_data.head(10), use_container_width=True)
    else:
        st.warning("⚠️ Please generate sample data or upload a CSV file to get started.")
        st.markdown("""
        ### Getting Started:
        1. Click **"Generate Sample Data"** in the sidebar to create sample inventory data
        2. Or upload your own CSV file with inventory data
        3. Navigate through different sections to explore various inventory optimization techniques
        
        ### Sample Data Format:
        Your CSV should include columns like:
        - `item_id` or `product_name`
        - `demand` or `annual_demand`
        - `ordering_cost` or `setup_cost`
        - `holding_cost` or `carrying_cost`
        - `unit_cost` or `price`
        - `lead_time` (optional)
        - `demand_variance` (optional)
        """)

elif page == "📊 EOQ Analysis":
    st.title("📊 Economic Order Quantity (EOQ) Analysis")
    
    if st.session_state.inventory_data is None:
        st.warning("⚠️ Please generate sample data or upload a CSV file first.")
    else:
        df = st.session_state.inventory_data.copy()
        
        # Parameters configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚙️ Model Parameters")
            
            # Select item if item_id exists
            if 'item_id' in df.columns or 'product_name' in df.columns:
                item_col = 'item_id' if 'item_id' in df.columns else 'product_name'
                selected_item = st.selectbox("Select Item", df[item_col].unique())
                item_data = df[df[item_col] == selected_item].iloc[0]
            else:
                item_data = df.iloc[0]
                st.info("Using first item from dataset")
            
            # Get or calculate parameters
            demand_col = next((col for col in df.columns if 'demand' in col.lower()), None)
            ordering_col = next((col for col in df.columns if 'order' in col.lower() or 'setup' in col.lower()), None)
            holding_col = next((col for col in df.columns if 'hold' in col.lower() or 'carry' in col.lower()), None)
            unit_cost_col = next((col for col in df.columns if 'cost' in col.lower() or 'price' in col.lower()), None)
            
            annual_demand = st.number_input(
                "Annual Demand (units)",
                min_value=1,
                value=int(item_data[demand_col]) if demand_col else 1000,
                step=100
            )
            
            ordering_cost = st.number_input(
                "Ordering Cost ($)",
                min_value=0.01,
                value=float(item_data[ordering_col]) if ordering_col else 50.0,
                step=10.0,
                format="%.2f"
            )
            
            holding_cost_rate = st.number_input(
                "Holding Cost Rate (%)",
                min_value=0.01,
                max_value=100.0,
                value=20.0,
                step=1.0,
                format="%.2f"
            )
            
            unit_cost = st.number_input(
                "Unit Cost ($)",
                min_value=0.01,
                value=float(item_data[unit_cost_col]) if unit_cost_col else 10.0,
                step=1.0,
                format="%.2f"
            )
            
            holding_cost = (holding_cost_rate / 100) * unit_cost
        
        with col2:
            st.subheader("📈 EOQ Results")
            
            # Calculate EOQ
            eoq = calculate_eoq(annual_demand, ordering_cost, holding_cost)
            total_cost = (ordering_cost * annual_demand / eoq) + (holding_cost * eoq / 2)
            ordering_cost_total = ordering_cost * annual_demand / eoq
            holding_cost_total = holding_cost * eoq / 2
            orders_per_year = annual_demand / eoq
            time_between_orders = 365 / orders_per_year if orders_per_year > 0 else 0
            
            st.metric("EOQ", f"{eoq:.0f} units")
            st.metric("Total Annual Cost", f"${total_cost:,.2f}")
            st.metric("Orders per Year", f"{orders_per_year:.2f}")
            st.metric("Time Between Orders", f"{time_between_orders:.1f} days")
            
            st.markdown("---")
            st.subheader("💰 Cost Breakdown")
            st.metric("Ordering Cost", f"${ordering_cost_total:,.2f}")
            st.metric("Holding Cost", f"${holding_cost_total:,.2f}")
        
        # Inventory depletion simulation
        st.markdown("---")
        st.subheader("📉 Inventory Depletion Over 12 Months")
        
        months = st.slider("Number of Months", 6, 24, 12)
        
        # Simulate inventory depletion
        monthly_demand = annual_demand / 12
        initial_inventory = eoq * 2  # Start with 2 EOQ cycles worth
        
        inventory_levels = []
        months_list = []
        reorder_points = []
        
        current_inventory = initial_inventory
        for month in range(months):
            months_list.append(month + 1)
            
            # Deplete inventory
            current_inventory -= monthly_demand
            
            # Reorder when reaching reorder point
            reorder_point = calculate_reorder_point(monthly_demand * 12, monthly_demand / 30, 0, 0)
            if current_inventory <= reorder_point:
                current_inventory += eoq
                reorder_points.append(month + 1)
            
            inventory_levels.append(max(0, current_inventory))
        
        # Create visualization
        fig = plot_inventory_depletion(months_list, inventory_levels, reorder_points, eoq)
        st.plotly_chart(fig, use_container_width=True)
        
        # EOQ Sensitivity Analysis
        st.markdown("---")
        st.subheader("🔍 EOQ Sensitivity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            demand_range = st.slider(
                "Demand Range (%)",
                min_value=-50,
                max_value=50,
                value=(-20, 20),
                step=5
            )
        
        with col2:
            cost_range = st.slider(
                "Cost Range (%)",
                min_value=-50,
                max_value=50,
                value=(-20, 20),
                step=5
            )
        
        # Calculate sensitivity
        base_eoq = eoq
        demand_values = [annual_demand * (1 + d/100) for d in range(demand_range[0], demand_range[1]+1, 5)]
        eoq_values = [calculate_eoq(d, ordering_cost, holding_cost) for d in demand_values]
        
        fig_sensitivity = go.Figure()
        fig_sensitivity.add_trace(go.Scatter(
            x=demand_values,
            y=eoq_values,
            mode='lines+markers',
            name='EOQ',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8)
        ))
        fig_sensitivity.add_hline(y=base_eoq, line_dash="dash", line_color="red", 
                                  annotation_text=f"Base EOQ: {base_eoq:.0f}")
        fig_sensitivity.update_layout(
            title="EOQ Sensitivity to Demand Changes",
            xaxis_title="Annual Demand (units)",
            yaxis_title="EOQ (units)",
            hovermode='x unified',
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_sensitivity, use_container_width=True)

elif page == "🔤 ABC Analysis":
    st.title("🔤 ABC Analysis")
    
    if st.session_state.inventory_data is None:
        st.warning("⚠️ Please generate sample data or upload a CSV file first.")
    else:
        df = st.session_state.inventory_data.copy()
        
        # ABC Analysis parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚙️ Analysis Parameters")
            
            # Find value column
            value_col = next((col for col in df.columns if 'value' in col.lower() or 'cost' in col.lower() or 'price' in col.lower()), None)
            
            if value_col:
                st.info(f"Using '{value_col}' column for value calculation")
            else:
                st.warning("No value column found. Using default values.")
            
            # Get quantity column
            quantity_col = next((col for col in df.columns if 'quantity' in col.lower() or 'demand' in col.lower() or 'stock' in col.lower()), None)
            
            if not value_col or not quantity_col:
                # Create synthetic value if needed
                if 'demand' in df.columns and 'unit_cost' not in df.columns:
                    df['total_value'] = df['demand'] * 10  # Default unit cost
                    value_col = 'total_value'
                elif 'demand' in df.columns and 'unit_cost' in df.columns:
                    df['total_value'] = df['demand'] * df['unit_cost']
                    value_col = 'total_value'
            
            # Get item identifier
            item_col = next((col for col in df.columns if 'item' in col.lower() or 'product' in col.lower() or 'name' in col.lower()), 'index')
            if item_col == 'index':
                df['item_id'] = df.index.astype(str)
                item_col = 'item_id'
        
        with col2:
            st.subheader("📊 ABC Thresholds")
            threshold_a = st.slider("Category A Threshold (%)", 0, 100, 80, step=5)
            threshold_b = st.slider("Category B Threshold (%)", 0, 100, 95, step=5)
        
        # Perform ABC Analysis
        abc_results = abc_analysis(df, value_col, item_col, threshold_a, threshold_b)
        
        # Display results
        st.markdown("---")
        st.subheader("📋 ABC Classification Results")
        
        col1, col2, col3 = st.columns(3)
        
        category_counts = abc_results['category'].value_counts()
        
        with col1:
            st.metric("Category A Items", category_counts.get('A', 0))
        with col2:
            st.metric("Category B Items", category_counts.get('B', 0))
        with col3:
            st.metric("Category C Items", category_counts.get('C', 0))
        
        # Display table
        st.dataframe(abc_results[['item', 'cumulative_value', 'cumulative_percentage', 'category']], 
                    use_container_width=True)
        
        # Visualization
        st.markdown("---")
        st.subheader("📈 ABC Analysis Visualization")
        
        fig = plot_abc_analysis(abc_results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Category breakdown
        st.markdown("---")
        st.subheader("📊 Category Breakdown")
        
        category_summary = abc_results.groupby('category').agg({
            'value': ['sum', 'mean', 'count']
        }).round(2)
        category_summary.columns = ['Total Value', 'Average Value', 'Item Count']
        st.dataframe(category_summary, use_container_width=True)

elif page == "📈 Forecasting":
    st.title("📈 Demand Forecasting")
    
    if st.session_state.inventory_data is None:
        st.warning("⚠️ Please generate sample data or upload a CSV file first.")
    else:
        df = st.session_state.inventory_data.copy()
        
        # Select item
        item_col = next((col for col in df.columns if 'item' in col.lower() or 'product' in col.lower()), None)
        if item_col:
            selected_item = st.selectbox("Select Item", df[item_col].unique())
            item_data = df[df[item_col] == selected_item]
        else:
            item_data = df.iloc[[0]]
        
        # Get historical demand
        demand_col = next((col for col in df.columns if 'demand' in col.lower()), None)
        
        if demand_col:
            # Generate historical data if we only have annual demand
            if len(item_data) == 1:
                annual_demand = item_data[demand_col].iloc[0]
                # Generate 12 months of historical data with some variation
                np.random.seed(42)
                historical_demand = np.random.normal(annual_demand/12, annual_demand/12*0.2, 12)
                historical_demand = np.maximum(historical_demand, 0)
            else:
                historical_demand = item_data[demand_col].values
        else:
            st.error("No demand column found in data")
            st.stop()
        
        # Forecasting parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚙️ Forecasting Parameters")
            forecast_periods = st.number_input("Forecast Periods", min_value=1, max_value=12, value=6)
            ma_window = st.number_input("Moving Average Window", min_value=2, max_value=12, value=3)
            alpha = st.slider("Exponential Smoothing Alpha", 0.0, 1.0, 0.3, step=0.1)
        
        with col2:
            st.subheader("📊 Historical Data")
            st.line_chart(pd.Series(historical_demand, name="Demand"))
            st.metric("Average Demand", f"{np.mean(historical_demand):.2f}")
            st.metric("Std Deviation", f"{np.std(historical_demand):.2f}")
        
        # Generate forecasts
        ma_forecast = moving_average_forecast(historical_demand, forecast_periods, ma_window)
        es_forecast = exponential_smoothing_forecast(historical_demand, forecast_periods, alpha)
        
        # Calculate metrics
        # For comparison, use last part of historical data as "test" set
        test_size = min(3, len(historical_demand) // 3)
        if test_size > 0:
            test_data = historical_demand[-test_size:]
            ma_test = moving_average_forecast(historical_demand[:-test_size], test_size, ma_window)
            es_test = exponential_smoothing_forecast(historical_demand[:-test_size], test_size, alpha)
            
            ma_metrics = calculate_metrics(test_data, ma_test)
            es_metrics = calculate_metrics(test_data, es_test)
        else:
            ma_metrics = {}
            es_metrics = {}
        
        # Visualization
        st.markdown("---")
        st.subheader("📈 Forecast Comparison")
        
        fig = plot_forecast_comparison(historical_demand, ma_forecast, es_forecast, forecast_periods)
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics comparison
        st.markdown("---")
        st.subheader("📊 Forecast Accuracy Metrics")
        
        if ma_metrics and es_metrics:
            metrics_df = pd.DataFrame({
                'Metric': ['MAE', 'RMSE', 'MAPE (%)'],
                'Moving Average': [
                    ma_metrics.get('mae', 0),
                    ma_metrics.get('rmse', 0),
                    ma_metrics.get('mape', 0)
                ],
                'Exponential Smoothing': [
                    es_metrics.get('mae', 0),
                    es_metrics.get('rmse', 0),
                    es_metrics.get('mape', 0)
                ]
            })
            
            st.dataframe(metrics_df, use_container_width=True)
            
            fig_metrics = plot_metrics_comparison(ma_metrics, es_metrics)
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Forecast values table
        st.markdown("---")
        st.subheader("📋 Forecast Values")
        
        forecast_df = pd.DataFrame({
            'Period': range(1, forecast_periods + 1),
            'Moving Average': ma_forecast,
            'Exponential Smoothing': es_forecast
        })
        st.dataframe(forecast_df, use_container_width=True)

elif page == "📉 Performance Metrics":
    st.title("📉 Performance Metrics Comparison")
    
    if st.session_state.inventory_data is None:
        st.warning("⚠️ Please generate sample data or upload a CSV file first.")
    else:
        df = st.session_state.inventory_data.copy()
        
        # Calculate metrics for all items
        results = []
        
        demand_col = next((col for col in df.columns if 'demand' in col.lower()), None)
        ordering_col = next((col for col in df.columns if 'order' in col.lower() or 'setup' in col.lower()), None)
        holding_col = next((col for col in df.columns if 'hold' in col.lower() or 'carry' in col.lower()), None)
        unit_cost_col = next((col for col in df.columns if 'cost' in col.lower() or 'price' in col.lower()), None)
        item_col = next((col for col in df.columns if 'item' in col.lower() or 'product' in col.lower()), 'index')
        
        if not all([demand_col, ordering_col]):
            st.error("Required columns not found. Please ensure data has demand and ordering cost columns.")
        else:
            for idx, row in df.iterrows():
                annual_demand = row[demand_col] if demand_col else 1000
                ordering_cost = row[ordering_col] if ordering_col else 50
                unit_cost = row[unit_cost_col] if unit_cost_col else 10
                holding_cost_rate = 20  # Default 20%
                holding_cost = (holding_cost_rate / 100) * unit_cost
                
                eoq = calculate_eoq(annual_demand, ordering_cost, holding_cost)
                total_cost = (ordering_cost * annual_demand / eoq) + (holding_cost * eoq / 2)
                turnover_ratio = annual_demand / eoq if eoq > 0 else 0
                
                item_name = row[item_col] if item_col != 'index' else f"Item {idx}"
                
                results.append({
                    'Item': item_name,
                    'Annual Demand': annual_demand,
                    'EOQ': eoq,
                    'Total Cost': total_cost,
                    'Turnover Ratio': turnover_ratio,
                    'Orders per Year': annual_demand / eoq if eoq > 0 else 0
                })
            
            results_df = pd.DataFrame(results)
            
            # Display metrics
            st.subheader("📊 Overall Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Items", len(results_df))
            with col2:
                st.metric("Avg EOQ", f"{results_df['EOQ'].mean():.0f}")
            with col3:
                st.metric("Total Annual Cost", f"${results_df['Total Cost'].sum():,.2f}")
            with col4:
                st.metric("Avg Turnover Ratio", f"{results_df['Turnover Ratio'].mean():.2f}")
            
            # Detailed table
            st.markdown("---")
            st.subheader("📋 Detailed Metrics Table")
            st.dataframe(results_df, use_container_width=True)
            
            # Visualizations
            st.markdown("---")
            st.subheader("📈 Performance Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.bar(
                    results_df.head(20),
                    x='Item',
                    y='Total Cost',
                    title="Total Cost by Item (Top 20)",
                    labels={'Total Cost': 'Total Cost ($)'}
                )
                fig1.update_xaxes(tickangle=45)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = px.scatter(
                    results_df,
                    x='Annual Demand',
                    y='EOQ',
                    size='Total Cost',
                    color='Turnover Ratio',
                    title="Demand vs EOQ",
                    hover_data=['Item']
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Cost distribution
            st.markdown("---")
            fig3 = px.histogram(
                results_df,
                x='Total Cost',
                nbins=30,
                title="Total Cost Distribution",
                labels={'Total Cost': 'Total Cost ($)', 'count': 'Number of Items'}
            )
            st.plotly_chart(fig3, use_container_width=True)

elif page == "🤖 AI Analysis & Reports":
    st.title("🤖 AI-Powered Analysis & Reports")
    
    # Check for Gemini API key
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
        if not api_key:
            st.warning("⚠️ Gemini API key not found in Streamlit secrets. Please add GEMINI_API_KEY to your secrets.")
            st.info("To add secrets: Go to your Streamlit app settings → Secrets → Add GEMINI_API_KEY")
    except:
        api_key = None
        st.warning("⚠️ Unable to access Streamlit secrets. Please ensure GEMINI_API_KEY is configured.")
    
    if st.session_state.inventory_data is None:
        st.warning("⚠️ Please generate sample data or upload a CSV file first.")
    else:
        df = st.session_state.inventory_data.copy()
        
        # Analysis options
        st.subheader("🔍 Analysis Options")
        
        analysis_type = st.selectbox(
            "Select Analysis Type",
            [
                "📊 Overall Inventory Summary",
                "📈 EOQ Analysis Interpretation",
                "🔤 ABC Analysis Insights",
                "📉 Forecasting Analysis",
                "💰 Cost Optimization Recommendations",
                "📋 Comprehensive Report"
            ]
        )
        
        # Prepare data summary for AI
        data_summary = {
            "total_items": len(df),
            "columns": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "summary_stats": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
        
        # Calculate key metrics if possible
        demand_col = next((col for col in df.columns if 'demand' in col.lower()), None)
        if demand_col:
            data_summary["total_demand"] = df[demand_col].sum()
            data_summary["avg_demand"] = df[demand_col].mean()
        
        if st.button("🚀 Generate AI Analysis", type="primary", use_container_width=True):
            if not api_key:
                st.error("❌ Gemini API key is required for AI analysis. Please configure it in Streamlit secrets.")
            else:
                with st.spinner("🤖 AI is analyzing your inventory data..."):
                    try:
                        analysis = get_gemini_analysis(
                            api_key,
                            analysis_type,
                            data_summary,
                            df.head(100).to_dict() if len(df) > 100 else df.to_dict()
                        )
                        
                        st.markdown("---")
                        st.subheader("✨ AI Analysis Results")
                        st.markdown(analysis)
                        
                        # Store analysis in session state
                        st.session_state.last_analysis = analysis
                        st.session_state.last_analysis_type = analysis_type
                        
                    except Exception as e:
                        st.error(f"❌ Error generating analysis: {str(e)}")
                        st.info("Please check your API key and try again.")
        
        # Report generation
        st.markdown("---")
        st.subheader("📄 Report Generation")
        
        if 'last_analysis' in st.session_state:
            col1, col2 = st.columns(2)
            
            with col1:
                report_title = st.text_input("Report Title", value=f"Inventory Analysis Report - {datetime.now().strftime('%Y-%m-%d')}")
            
            with col2:
                include_charts = st.checkbox("Include Charts", value=True)
            
            if st.button("📥 Generate PDF Report", use_container_width=True):
                if not api_key:
                    st.error("❌ Gemini API key is required for PDF generation.")
                else:
                    with st.spinner("📄 Generating PDF report..."):
                        try:
                            pdf_data = generate_pdf_report(
                                api_key,
                                report_title,
                                st.session_state.last_analysis,
                                st.session_state.last_analysis_type,
                                data_summary
                            )
                            
                            st.success("✅ PDF report generated successfully!")
                            st.download_button(
                                label="⬇️ Download PDF Report",
                                data=pdf_data,
                                file_name=f"inventory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"❌ Error generating PDF: {str(e)}")
        else:
            st.info("💡 Generate an AI analysis first to create a PDF report.")

