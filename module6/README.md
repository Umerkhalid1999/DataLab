# Advanced Feature Engineering Module (Module 6)

A comprehensive Flask-based web application for automated feature engineering and selection as part of a Final Year Project (FYP). This module provides intelligent feature analysis, creation, and optimization capabilities with an intuitive web interface.

## Features

### 🤖 Automated Feature Importance Ranking
- **Random Forest**: Tree-based feature importance using ensemble methods
- **XGBoost**: Gradient boosting feature importance scores
- **Mutual Information**: Statistical dependency measurement between features and target
- **Correlation Analysis**: Pearson correlation coefficients with target variable

### 🎯 Intelligent Feature Creation
- **Polynomial Features**: Automatic interaction terms between numeric features
- **Ratio Features**: Mathematical ratios between feature pairs (feature1/feature2)
- **Log Transformations**: Natural log transformations for positive-valued features
- **Binning Features**: Categorical binned versions of continuous features
- **Statistical Aggregations**: Mean, standard deviation, min, max across feature groups

### 📊 Dimensionality Reduction with Visual Explanations
- **PCA (Principal Component Analysis)**: Linear dimensionality reduction with variance explanation
- **t-SNE**: Non-linear dimensionality reduction for data visualization
- **Interactive Visualizations**: Scatter plots and variance explanation charts
- **Component Analysis**: Detailed breakdown of principal components

### ⚖️ Feature Set Comparison
- **Performance Impact Metrics**: Cross-validation scoring for different feature sets
- **A/B Testing Capabilities**: Compare multiple feature combinations
- **Statistical Significance**: Mean and standard deviation of performance metrics
- **Model Type Support**: Automatic detection or manual specification (classification/regression)

### 🏭 Domain-Specific Feature Templates
Pre-built feature engineering templates for common industries:

- **Financial**: Ratios, moving averages, volatility indicators, technical analysis
- **Healthcare**: Vital sign ratios, age interactions, risk scores, comorbidity counts
- **E-commerce**: Customer behavior metrics, product features, engagement rates
- **Marketing**: Campaign metrics, audience features, channel analysis, timing optimization
- **Manufacturing**: Quality metrics, process features, maintenance indicators, operational efficiency

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone or download the module**
   ```bash
   cd module6
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure OpenAI (Optional but Recommended)**
   - Get your OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
   - Edit the `.env` file and replace `your_openai_api_key_here` with your actual API key
   - **Note**: Without an API key, the module will still work but AI insights will be disabled

5. **Run the application**
   ```bash
   python start.py
   ```
   Or alternatively:
   ```bash
   python app.py
   ```

6. **Access the web interface**
   The browser will open automatically, or navigate to: `http://localhost:5000`

## Usage Guide

**🚀 Ultra-Simple 3-Step Process:**

### 1. Upload Your Dataset
- Simply drag and drop your CSV or Excel file OR click "Choose File"
- Target column is automatically detected (no manual input needed!)
- Supported formats: CSV, Excel (.xlsx)

### 2. Automatic Analysis
- Click anywhere or just wait - the analysis runs automatically!
- AI performs comprehensive feature engineering in the background
- Progress is shown with real-time updates

### 3. Review Results & Recommendations
- Get instant insights about your most important features
- See automatically created intelligent features
- Review performance comparisons and AI recommendations
- Export results in JSON or CSV format

**That's it! No configuration, no technical knowledge required.**

### What Happens Automatically:
- ✅ **Target Detection**: Automatically identifies the target column
- ✅ **Feature Importance**: Ranks features using 4 different AI methods
- ✅ **Feature Creation**: Generates 20+ intelligent new features
- ✅ **Dimensionality Analysis**: PCA and t-SNE analysis with visualizations
- ✅ **Performance Comparison**: Tests different feature combinations
- ✅ **🤖 LLM-Powered Analysis**: GPT-3.5-turbo provides expert insights
- ✅ **Smart Recommendations**: AI-generated actionable insights
- ✅ **Domain Intelligence**: Industry-specific feature suggestions
- ✅ **Performance Explanations**: Why certain features work better

## 🤖 LLM-Powered Features

The module integrates OpenAI's GPT-3.5-turbo to provide intelligent analysis:

### Dataset Analysis
- **Data Quality Assessment**: Identifies potential issues and biases
- **Feature Engineering Strategy**: Suggests domain-specific approaches
- **Missing Value Handling**: Intelligent recommendations for data cleaning
- **Data Leakage Detection**: Warns about potential overfitting risks

### Feature Intelligence
- **Importance Pattern Analysis**: Explains why certain features are important
- **Domain Recognition**: Automatically identifies the industry/domain
- **Feature Interaction Suggestions**: Recommends meaningful feature combinations
- **Performance Insights**: Explains model performance differences

### Strategic Recommendations
- **Priority Actions**: Most important next steps for your ML project
- **Model Selection**: Recommended algorithms based on data characteristics
- **Validation Strategy**: How to properly validate your models
- **Production Considerations**: Important factors for deployment
- **Risk Assessment**: Potential issues to watch for

### Configuration
The LLM features are configured via environment variables in `.env`:
```
OPENAI_API_KEY=your_api_key_here     # Required for LLM features
OPENAI_MODEL=gpt-3.5-turbo           # Model to use (optional)
OPENAI_TEMPERATURE=0.7               # Creativity level (optional)
```

## API Endpoints

The module provides RESTful API endpoints for programmatic access:

- `POST /upload` - Upload dataset
- `POST /feature_importance` - Calculate feature importance
- `POST /create_features` - Create intelligent features
- `POST /dimensionality_reduction` - Perform dimensionality reduction
- `POST /compare_features` - Compare feature sets
- `GET /domain_templates` - Get domain-specific templates
- `GET /visualize/<viz_type>` - Generate visualizations

## File Structure

```
module6/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/
│   └── index.html        # Web interface template
├── static/               # Static files (CSS, JS, images)
└── module_6.txt          # Original module specification
```

## Technical Implementation

### Backend Framework
- **Flask**: Lightweight Python web framework
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **XGBoost**: Gradient boosting framework
- **Matplotlib/Seaborn**: Data visualization

### Frontend Technologies
- **Bootstrap 5**: Responsive CSS framework
- **JavaScript (ES6+)**: Interactive web functionality
- **Font Awesome**: Icon library
- **Plotly.js**: Interactive charting library

### Machine Learning Algorithms
- Random Forest (Classification/Regression)
- XGBoost (Classification/Regression)
- Principal Component Analysis (PCA)
- t-Distributed Stochastic Neighbor Embedding (t-SNE)
- Mutual Information scoring
- Cross-validation for model evaluation

## Integration with Main FYP

This module is designed to be integrated into a larger FYP system. Key integration points:

1. **API Endpoints**: Can be called from other modules
2. **Data Format**: Accepts standard CSV/Excel formats
3. **Result Format**: Returns JSON responses for easy integration
4. **Modular Design**: Can be deployed separately and integrated via API calls

## Future Enhancements

Potential areas for expansion:
- Advanced feature selection algorithms (RFE, LASSO)
- Deep learning-based feature extraction
- Automated hyperparameter tuning for feature creation
- Real-time feature monitoring and drift detection
- Integration with popular ML platforms (MLflow, Kubeflow)

## Troubleshooting

### Common Issues

1. **File Upload Errors**: Ensure CSV/Excel files are properly formatted
2. **Memory Issues**: For large datasets, consider increasing system memory or using sampling
3. **Missing Dependencies**: Run `pip install -r requirements.txt` to install all dependencies
4. **Port Conflicts**: Change the port in `app.py` if 5000 is already in use

### Support

For issues related to this module, please check:
1. Error messages in the web interface
2. Console output when running the Flask app
3. Browser developer console for frontend errors

## License

This module is part of a Final Year Project and is intended for educational and research purposes.

---

**Note**: This module is designed as a standalone component that can be integrated into larger machine learning workflows and platforms.
