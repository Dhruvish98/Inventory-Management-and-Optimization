# Inventory Management & Optimization Dashboard

A comprehensive Streamlit dashboard for retail inventory optimization using various techniques including EOQ (Economic Order Quantity) model, ABC analysis, demand forecasting, and AI-powered insights.

## Features

### 📊 Core Functionalities

1. **EOQ Analysis**
   - Economic Order Quantity calculation using demand, ordering cost, and holding cost
   - Interactive inventory depletion visualization over 12+ months
   - Each point labeled with remaining units
   - EOQ sensitivity analysis
   - Cost breakdown and optimization

2. **ABC Analysis**
   - Automatic categorization of items into A, B, and C categories
   - Value-based prioritization
   - Interactive visualizations
   - Category-wise statistics and recommendations

3. **Demand Forecasting**
   - Multiple forecasting techniques:
     - Moving Average
     - Exponential Smoothing
   - Forecast accuracy metrics (MAE, RMSE, MAPE)
   - Model comparison and visualization
   - Interactive parameter tuning

4. **Performance Metrics**
   - Comprehensive metrics dashboard
   - Cost analysis across all items
   - Turnover ratio calculations
   - Visual comparisons and distributions

5. **AI-Powered Analysis & Reports**
   - Integration with Google Gemini AI (free tier)
   - Multiple analysis types:
     - Overall inventory summary
     - EOQ interpretation
     - ABC analysis insights
     - Forecasting analysis
     - Cost optimization recommendations
     - Comprehensive reports
   - PDF report generation with download option

### 🎨 Interactive Features

- **Sample Data Generation**: Generate realistic inventory data with one click
- **CSV Upload**: Upload your own inventory data
- **Real-time Calculations**: All metrics update dynamically
- **Interactive Visualizations**: Plotly charts with hover details and zoom
- **Parameter Tuning**: Adjust model parameters and see results instantly

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Dhruvish98/Inventory-Management-and-Optimization.git
cd Inventory-Management-and-Optimization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Configuration

### Gemini API Key Setup

To enable AI-powered analysis and report generation:

1. Get a free Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

2. For local development, create a `.streamlit/secrets.toml` file:
```toml
GEMINI_API_KEY = "your-api-key-here"
```

3. For Streamlit Cloud deployment:
   - Go to your app settings
   - Navigate to "Secrets"
   - Add: `GEMINI_API_KEY = "your-api-key-here"`

## Usage

### Getting Started

1. **Generate Sample Data**: Click "Generate Sample Data" in the sidebar to create sample inventory data
2. **Or Upload CSV**: Upload your own CSV file with inventory data
3. **Navigate**: Use the sidebar to navigate between different analysis sections

### Data Format

Your CSV file should include columns like:
- `item_id` or `product_name` - Item identifier
- `annual_demand` or `demand` - Annual demand in units
- `ordering_cost` or `setup_cost` - Cost per order
- `holding_cost` or `carrying_cost` - Holding cost per unit per year
- `unit_cost` or `price` - Unit cost
- `lead_time_days` or `lead_time` - Lead time in days (optional)
- `demand_variance` - Variance of demand (optional)

### Example Workflow

1. **Generate/Upload Data** → Start with sample data or your own
2. **EOQ Analysis** → Calculate optimal order quantities
3. **ABC Analysis** → Categorize items by value
4. **Forecasting** → Predict future demand
5. **Performance Metrics** → Compare across all items
6. **AI Analysis** → Get AI-powered insights and generate PDF reports

## Deployment on Streamlit Cloud

1. Push your code to GitHub:
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app"
4. Connect your GitHub repository
5. Select branch and main file (`streamlit_app.py`)
6. Add your `GEMINI_API_KEY` in the Secrets section
7. Deploy!

## Project Structure

```
Inventory-Management-and-Optimization/
│
├── streamlit_app.py          # Main Streamlit application
├── inventory_models.py       # EOQ, ABC, forecasting models
├── data_generator.py         # Sample data generation
├── visualizations.py         # Plotly visualization functions
├── ai_analysis.py           # Gemini AI integration
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Key Models & Techniques

### EOQ Model
- Formula: `EOQ = √(2DS/H)`
- Where D = Annual demand, S = Ordering cost, H = Holding cost
- Minimizes total inventory costs

### ABC Analysis
- Category A: Top 80% of value (high priority)
- Category B: Next 15% of value (medium priority)
- Category C: Remaining 5% of value (low priority)

### Forecasting Methods
- **Moving Average**: Simple average of recent periods
- **Exponential Smoothing**: Weighted average with emphasis on recent data

### Metrics
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error

## Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Plotly**: Interactive visualizations
- **SciPy**: Statistical functions
- **Google Gemini AI**: AI-powered analysis
- **ReportLab**: PDF generation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Support

For issues, questions, or contributions, please open an issue on GitHub.

## Acknowledgments

- Built with Streamlit
- AI insights powered by Google Gemini
- Visualization powered by Plotly

