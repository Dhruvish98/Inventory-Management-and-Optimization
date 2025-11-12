# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Gemini API (Optional but Recommended)
1. Get free API key from: https://makersuite.google.com/app/apikey
2. For local: Create `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
3. For Streamlit Cloud: Add in app settings → Secrets

### Step 3: Run the App
```bash
streamlit run streamlit_app.py
```

## 📋 First Steps in the Dashboard

1. **Generate Sample Data**: Click "Generate Sample Data" in sidebar
2. **Explore EOQ Analysis**: See inventory depletion visualization
3. **Try ABC Analysis**: Categorize items by value
4. **Test Forecasting**: Compare different forecasting methods
5. **Get AI Insights**: Generate AI-powered analysis (requires API key)

## 💡 Tips

- **Sample Data**: Use the built-in generator to explore features
- **CSV Upload**: Your CSV should have columns like `annual_demand`, `ordering_cost`, `unit_cost`
- **Interactive Charts**: Hover over charts for detailed information
- **PDF Reports**: Generate comprehensive reports with AI insights

## 🐛 Troubleshooting

**Issue**: "Module not found"
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: "Gemini API error"
- **Solution**: Check your API key in secrets. The app works without AI features too!

**Issue**: "No data showing"
- **Solution**: Click "Generate Sample Data" or upload a CSV file first

## 📚 Need More Help?

Check the full [README.md](README.md) for detailed documentation.

