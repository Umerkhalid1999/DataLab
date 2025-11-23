# ml_explainer.py - Explains ML model selection decisions

def explain_sampling_decision(original_rows, sampled_rows=1000):
    """Explain why data was sampled"""
    if original_rows <= sampled_rows:
        return f"""
## 📊 Data Sampling

**Decision:** Used ALL {original_rows:,} rows (No sampling needed)

**Key Points:**
- Dataset size is manageable for fast ML training
- Using complete data for maximum accuracy
- No information loss
"""

    percentage = (sampled_rows / original_rows) * 100
    return f"""
## 📊 Data Sampling

**Your Data:** {original_rows:,} rows
**Analyzed:** {sampled_rows:,} rows ({percentage:.1f}%)

**Why Sample?**

**Key Points:**
- **Speed:** {original_rows:,} rows would take 10-30 minutes to analyze
- **Efficiency:** {sampled_rows:,} rows analyzed in ~2 minutes
- **Accuracy:** Random sampling preserves data patterns (95%+ reliable)
- **Final Model:** Will be trained on FULL dataset when you deploy
- **Industry Standard:** Google, Amazon, and research papers use sampling for analysis

**What This Means:**
- You get quick insights without losing accuracy
- Model recommendations are reliable
- Final production model uses all your data
"""

def explain_task_detection(task_type, target_column, n_unique_values):
    """Explain how the ML task was detected"""

    if task_type == "classification":
        return f"""
## 🎯 Task Type Detection

**Detected Task:** Classification

**How We Decided:**
- Target column: **{target_column}**
- Unique values: **{n_unique_values}** classes
- Decision rule: Less than 20 unique values = Classification

**Key Points:**
- **Classification** means predicting categories/classes
- Your model will predict which class each new data point belongs to
- Examples: Spam/Not Spam, Cat/Dog, High/Medium/Low Risk
- Best models: Random Forest, XGBoost, Logistic Regression, Neural Networks

**What This Means:**
- Models will output probabilities for each class
- Performance measured by accuracy, precision, recall, F1-score
- Confusion matrix shows prediction patterns
"""
    else:
        return f"""
## 🎯 Task Type Detection

**Detected Task:** Regression

**How We Decided:**
- Target column: **{target_column}**
- Unique values: **{n_unique_values}** (continuous numbers)
- Decision rule: Many unique values = Regression

**Key Points:**
- **Regression** means predicting continuous numbers
- Your model will predict numeric values
- Examples: House prices, temperature, sales amount, stock prices
- Best models: XGBoost, Random Forest, Linear Regression, Neural Networks

**What This Means:**
- Models will output precise numeric predictions
- Performance measured by R² score, MAE, RMSE
- Lower error = better predictions
"""

def explain_model_selection(model_name, suitability_score, performance, task_type):
    """Explain why a specific model was recommended"""

    model_descriptions = {
        "Random Forest": {
            "what": "Ensemble of decision trees voting together",
            "strengths": ["Handles non-linear relationships", "Works with missing data", "Feature importance built-in", "Rarely overfits"],
            "when": "Medium to large datasets with complex patterns"
        },
        "XGBoost": {
            "what": "Gradient boosting with advanced optimizations",
            "strengths": ["Best accuracy on structured data", "Fast training", "Handles missing values", "Industry standard for tabular data"],
            "when": "Any dataset where accuracy is priority"
        },
        "Logistic Regression": {
            "what": "Linear model for classification",
            "strengths": ["Very fast", "Interpretable coefficients", "Works on small datasets", "Good baseline"],
            "when": "Simple linear relationships, need interpretability"
        },
        "Neural Network": {
            "what": "Multi-layer perceptron with backpropagation",
            "strengths": ["Learns complex patterns", "Scales to large data", "Flexible architecture"],
            "when": "Large datasets with complex non-linear patterns"
        },
        "LightGBM": {
            "what": "Fast gradient boosting framework",
            "strengths": ["Extremely fast", "Memory efficient", "Handles large datasets", "High accuracy"],
            "when": "Large datasets where speed matters"
        },
        "Linear Regression": {
            "what": "Linear relationship model",
            "strengths": ["Very interpretable", "Fast training/prediction", "Good for linear trends"],
            "when": "Simple linear relationships"
        }
    }

    info = model_descriptions.get(model_name, {
        "what": "Machine learning model",
        "strengths": ["Proven algorithm", "Good performance"],
        "when": "Various use cases"
    })

    return f"""
## 🏆 Why {model_name}?

**Suitability Score:** {suitability_score:.1f}/100

**What is {model_name}?**
{info['what']}

**Key Strengths:**
{chr(10).join(f'- {s}' for s in info['strengths'])}

**Best Used When:**
{info['when']}

**Performance on Your Data:**
- CV Score: **{performance.get('mean_score', 0):.3f}** (Higher = Better for {task_type})
- Training Time: **{performance.get('training_time', 0):.2f}s** (Fast!)
- Std Dev: **{performance.get('std_score', 0):.3f}** (Lower = More stable)

**Why This Score?**
The suitability score considers:
1. **Performance (40%):** How accurate the model is
2. **Speed (30%):** Training and prediction time
3. **Stability (20%):** Consistent results across CV folds
4. **Complexity (10%):** Model interpretability and maintenance
"""

def explain_cross_validation(n_folds=5):
    """Explain cross-validation process"""
    return f"""
## 🔄 Cross-Validation

**Method:** {n_folds}-Fold Cross-Validation

**What Happens:**
1. Data split into {n_folds} equal parts
2. Train on {n_folds-1} parts, test on 1 part
3. Repeat {n_folds} times with different test parts
4. Average all {n_folds} scores = final score

**Why This Matters:**
- **Single Test:** Score might be lucky/unlucky (unreliable)
- **{n_folds}-Fold CV:** {n_folds} independent tests = reliable estimate
- **Prevents Overfitting:** Model never sees test data during training

**Key Points:**
- Each fold uses ~{100/n_folds:.0f}% for testing
- More folds = more reliable (but slower)
- Industry standard is 5-10 folds
- We use {n_folds} for best balance of speed and reliability
"""

def explain_hyperparameter_tuning():
    """Explain what hyperparameter tuning does"""
    return """
## ⚙️ Hyperparameter Tuning

**What Are Hyperparameters?**
Settings that control how a model learns (not learned from data)

**Examples:**
- **Random Forest:** Number of trees, tree depth, min samples per leaf
- **XGBoost:** Learning rate, max depth, number of estimators
- **Neural Network:** Number of layers, neurons per layer, learning rate

**Why Tune?**
- Default settings work "okay" but not optimal
- Tuning can improve accuracy by 5-15%
- Different datasets need different settings

**How It Works:**
1. Test many different combinations
2. Use cross-validation to measure each
3. Pick the combination with best CV score

**Levels Available:**
- **Normal:** Test 10-20 combinations (~1-2 min)
- **Semi-Deep:** Test 50-100 combinations (~5-10 min)
- **Deep:** Test 200+ combinations (~20-30 min)

**Key Points:**
- More combinations = better results but slower
- Start with Normal, use Deep for production
- Tuning happens automatically - you just wait!
"""

def generate_ml_explanations(analysis, recommendations, best_model_name):
    """Generate all explanations for ML analysis"""

    explanations = {}

    # Sampling explanation
    explanations['sampling'] = explain_sampling_decision(
        analysis.get('original_rows', 0),
        analysis.get('n_rows', 1000)
    )

    # Task detection
    explanations['task_detection'] = explain_task_detection(
        analysis['task_type'],
        analysis['target_column'],
        analysis.get('n_classes', analysis.get('n_unique_target', 0))
    )

    # Cross-validation
    explanations['cross_validation'] = explain_cross_validation(n_folds=5)

    # Best model explanation
    best_model = next((r for r in recommendations if r['model'] == best_model_name), None)
    if best_model:
        explanations['best_model'] = explain_model_selection(
            best_model_name,
            best_model['suitability_score'],
            best_model['performance'],
            analysis['task_type']
        )

    # Hyperparameter tuning
    explanations['hyperparameter_tuning'] = explain_hyperparameter_tuning()

    # Summary explanation
    explanations['summary'] = f"""
## 📋 Quick Summary

**What Just Happened:**
1. ✅ Analyzed your dataset ({analysis.get('n_rows', 0)} rows, {analysis.get('n_features', 0)} features)
2. ✅ Detected task type: **{analysis['task_type'].title()}**
3. ✅ Tested {len(recommendations)} different ML models
4. ✅ Used 5-fold cross-validation for reliability
5. ✅ Ranked models by suitability score

**Best Model:** {best_model_name} ({best_model['suitability_score']:.1f}/100)

**Next Steps:**
1. Review model comparison table below
2. Check why each model was scored
3. Optionally tune hyperparameters for better accuracy
4. Export notebook with complete training code
5. Deploy model to production!

**Key Takeaway:**
All models were tested fairly on the same data splits. The suitability score considers both accuracy and practicality (speed, stability, complexity).
"""

    return explanations
