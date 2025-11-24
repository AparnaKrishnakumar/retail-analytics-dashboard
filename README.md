# Retail_Analysis_Dashboard

# 🛒 Smart Retail Analytics Dashboard

> Transforming raw retail transaction data into actionable business insights through automated pipelines, ML forecasting, and interactive visualization.

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-in%20development-yellow)](https://github.com/yourusername/retail-analytics-dashboard)

---

## Project Overview

A comprehensive retail analytics solution that processes transaction data through an automated pipeline, generates ML-powered sales forecasts, and presents insights via an interactive dashboard.

**Built to showcase:** End-to-end data engineering, machine learning implementation, and production-ready analytics delivery.

---

## Key Features

### Analytics Pipeline
- Automated data ingestion and validation
- Data quality monitoring and cleaning
- Feature engineering for ML models
- Structured processing with clear data flow

### ML-Powered Insights
- **Sales Forecasting**: 30-day ahead predictions using time series models
- **Customer Segmentation**: RFM (Recency, Frequency, Monetary) analysis
- **Product Performance**: Category and item-level analytics
- **Trend Detection**: Seasonality and growth pattern identification

### Interactive Dashboard
- Real-time KPI tracking
- Multi-dimensional data exploration
- Customizable filters and date ranges
- Export functionality for reports

---

## Architecture
```
┌─────────────┐
│  Raw Data   │
│   (CSV)     │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Data Pipeline  │
│  - Validation   │
│  - Cleaning     │
│  - Transform    │
└──────┬──────────┘
       │
       ├──────────────┐
       │              │
       ▼              ▼
┌─────────────┐  ┌──────────┐
│  ML Model   │  │ Analytics│
│  Training   │  │  Engine  │
└──────┬──────┘  └────┬─────┘
       │              │
       └──────┬───────┘
              │
              ▼
       ┌─────────────┐
       │  Dashboard  │
       │ (Streamlit) │
       └─────────────┘
```

---

## Project Structure
```
retail-analytics-dashboard/
│
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned data
│   └── output/                 # Pipeline results
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_model_evaluation.ipynb
│   └── 03_insights_summary.ipynb
│
├── scripts/
│   ├── 01_data_ingestion.py    # Data loading and validation
│   ├── 02_train_model.py       # ML model training
│   └── 03_analytics_pipeline.py # Full pipeline orchestration
│
├── dashboard/
│   ├── app.py                  # Main Streamlit application
│   └── components/             # Reusable dashboard components
│
├── models/
│   └── sales_forecast.pkl      # Trained ML models
│
├── docs/
│   ├── pipeline_architecture.md
│   ├── data_dictionary.md
│   └── setup_guide.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Git

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/retail-analytics-dashboard.git
cd retail-analytics-dashboard

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset
# Place your retail dataset in data/raw/retail_data.csv
# Sample datasets: Kaggle Online Retail or Superstore
```

### Running the Pipeline
```bash
# Step 1: Data Ingestion
python scripts/01_data_ingestion.py

# Step 2: Train ML Model
python scripts/02_train_model.py

# Step 3: Run Full Analytics Pipeline
python scripts/03_analytics_pipeline.py
```

### Launch Dashboard
```bash
streamlit run dashboard/app.py
```

Dashboard will open at `http://localhost:8501`

---

## Tech Stack

### Data Processing
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

### Machine Learning
- **Prophet** - Time series forecasting
- **Scikit-learn** - Customer segmentation and classification
- **Statsmodels** - Statistical modeling

### Visualization
- **Plotly** - Interactive charts
- **Streamlit** - Dashboard framework
- **Matplotlib/Seaborn** - Statistical plots

### Development
- **Jupyter** - Exploratory analysis
- **Git** - Version control

---

## Data Requirements

### Expected Input Format

The pipeline expects retail transaction data with the following structure:

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| InvoiceNo | string | Transaction ID | Yes |
| StockCode | string | Product ID | Yes |
| Description | string | Product name | No |
| Quantity | integer | Units purchased | Yes |
| InvoiceDate | datetime | Transaction timestamp | Yes |
| UnitPrice | float | Price per unit | Yes |
| CustomerID | string | Customer identifier | Yes |
| Country | string | Customer location | No |

### Sample Data Sources
- [Kaggle Online Retail Dataset](https://www.kaggle.com/datasets/vijayuv/onlineretail)
- [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/online+retail)
- Your own retail transaction exports

---

## 📈 Key Metrics & KPIs

### Revenue Metrics
- Total Revenue
- Average Order Value (AOV)
- Revenue Growth Rate
- Revenue by Category/Product

### Customer Metrics
- Customer Lifetime Value (CLV)
- Customer Acquisition trends
- RFM Segmentation
- Churn Risk Score

### Product Metrics
- Best/Worst Performers
- Inventory Turnover
- Profit Margins
- Cross-sell Opportunities

### Predictive Metrics
- 30-Day Sales Forecast
- Demand Predictions
- Seasonality Patterns

---

## Roadmap

### Phase 1: Foundation (Weeks 1-2) ✅
- [x] Project setup and data ingestion
- [x] Exploratory data analysis
- [x] Basic pipeline structure
- [ ] ML model training
- [ ] Dashboard MVP

### Phase 2: Enhancement (Weeks 3-4)
- [ ] Advanced customer segmentation
- [ ] Anomaly detection
- [ ] Real-time data refresh
- [ ] Dashboard deployment (Streamlit Cloud)

### Phase 3: Production (Future)
- [ ] API development (FastAPI)
- [ ] Automated pipeline scheduling (Airflow)
- [ ] Cloud deployment (AWS/GCP)
- [ ] CI/CD implementation
- [ ] Advanced ML models (LSTM, XGBoost)
- [ ] A/B testing framework

---

## Model Performance

### Sales Forecasting
- **Model**: Prophet (Facebook)
- **Evaluation Metric**: MAPE (Mean Absolute Percentage Error)
- **Target**: <15% MAPE on test set
- **Training Data**: Last 12 months of transactions
- **Forecast Horizon**: 30 days

### Customer Segmentation
- **Method**: RFM + K-Means Clustering
- **Segments**: 4-5 customer groups
- **Validation**: Silhouette Score

*Detailed model evaluation available in `notebooks/02_model_evaluation.ipynb`*

---

## Documentation

- [Pipeline Architecture](docs/pipeline_architecture.md) - Detailed system design
- [Data Dictionary](docs/data_dictionary.md) - Column definitions and transformations
- [Setup Guide](docs/setup_guide.md) - Detailed installation instructions
- [API Reference](docs/api_reference.md) - Coming soon
- [Why Data Ingestion Matters](docs/why_data_ingestion.md) - **← ADD THIS**

---

## Contributing

This is a personal portfolio project, but suggestions and feedback are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 About

**Aparna Krishnakumar**  
Tech Lead | MSc in AI | Building AI-Powered Solutions

Transitioning from technical leadership back to hands-on ML engineering. This project demonstrates end-to-end capabilities in data engineering, machine learning, and product development.

🔗 [LinkedIn](your-linkedin-url)  
🌐 [Portfolio](your-portfolio-url)  
📧 [Email](your-email)

---

## Acknowledgments

- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- Inspiration: Real-world retail analytics challenges
- Built as part of a 6-month skill development journey

---

## Project Status

**Current Phase**: Week 1 - Foundation  
**Last Updated**: November 20, 2024  
**Status**: 🟡 In Active Development

### Recent Updates
- ✅ Initial repository setup
- ✅ Data ingestion pipeline
- 🔄 Exploratory data analysis (in progress)
- ⏳ ML model development (planned)

---

*Built with ❤️ and lots of ☕*

**⭐ Star this repo if you find it useful!**
