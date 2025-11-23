# module6_enhancements.py - Enhanced Feature Engineering with Explainable AI
"""
This module contains enhancements for the Feature Engineering module including:
- Enhanced file upload validation
- Real-time progress tracking
- Comprehensive explainable AI
- Smart automated preprocessing
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Tuple
import chardet
from werkzeug.utils import secure_filename
import os

class EnhancedFileValidator:
    """Enhanced file validation with better error handling"""

    def __init__(self):
        self.MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
        self.ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}
        self.MIN_ROWS = 10
        self.MIN_COLS = 2

    def validate_file(self, file) -> Tuple[bool, str, Dict]:
        """
        Comprehensive file validation with detailed feedback
        Returns: (is_valid, message, metadata)
        """
        metadata = {}

        # 1. Check if file exists
        if not file or file.filename == '':
            return False, "No file selected. Please choose a file to upload.", metadata

        # 2. Validate filename
        filename = secure_filename(file.filename)
        if not filename:
            return False, "Invalid filename. Please use a valid file name.", metadata

        metadata['filename'] = filename

        # 3. Check file extension
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in self.ALLOWED_EXTENSIONS:
            return False, f"Unsupported file type '{file_ext}'. Please upload CSV or Excel files (.csv, .xlsx, .xls).", metadata

        metadata['extension'] = file_ext

        # 4. Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # Reset to beginning

        metadata['size_bytes'] = file_size
        metadata['size_mb'] = round(file_size / (1024 * 1024), 2)

        if file_size == 0:
            return False, "File is empty. Please upload a file with data.", metadata

        if file_size > self.MAX_FILE_SIZE:
            return False, f"File too large ({metadata['size_mb']}MB). Maximum size is {self.MAX_FILE_SIZE/(1024*1024)}MB.", metadata

        # 5. Try to read and validate content
        try:
            df = self._read_file(file, file_ext)

            # Check dimensions
            rows, cols = df.shape
            metadata['rows'] = rows
            metadata['columns'] = cols

            if rows < self.MIN_ROWS:
                return False, f"File has only {rows} rows. Minimum required is {self.MIN_ROWS} rows for meaningful analysis.", metadata

            if cols < self.MIN_COLS:
                return False, f"File has only {cols} columns. Minimum required is {self.MIN_COLS} columns.", metadata

            # Check for completely empty columns
            empty_cols = df.columns[df.isnull().all()].tolist()
            if empty_cols:
                metadata['empty_columns'] = empty_cols
                return False, f"File contains completely empty columns: {', '.join(empty_cols)}. Please clean your data first.", metadata

            # All validations passed
            metadata['column_names'] = df.columns.tolist()
            metadata['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}
            metadata['missing_percentage'] = round((df.isnull().sum().sum() / (rows * cols)) * 100, 2)

            return True, "File validated successfully!", metadata

        except pd.errors.EmptyDataError:
            return False, "File contains no data. Please upload a file with data.", metadata
        except pd.errors.ParserError as e:
            return False, f"File parsing error: {str(e)}. Please check if your CSV file is properly formatted.", metadata
        except Exception as e:
            return False, f"Error reading file: {str(e)}. Please ensure the file is not corrupted.", metadata

    def _read_file(self, file, file_ext):
        """Read file with encoding detection for CSVs"""
        if file_ext == '.csv':
            # Detect encoding
            raw_data = file.read(10000)  # Read first 10KB
            file.seek(0)  # Reset

            detected = chardet.detect(raw_data)
            encoding = detected['encoding'] if detected['confidence'] > 0.7 else 'utf-8'

            try:
                return pd.read_csv(file, encoding=encoding)
            except UnicodeDecodeError:
                # Fallback encodings
                file.seek(0)
                for fallback_encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        return pd.read_csv(file, encoding=fallback_encoding)
                    except:
                        file.seek(0)
                        continue
                raise ValueError("Could not decode CSV file with any common encoding")

        elif file_ext in ['.xlsx', '.xls']:
            return pd.read_excel(file)

        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")


class ExplainableAI:
    """Enhanced explainable AI with step-by-step explanations"""

    def __init__(self):
        self.explanations = {}
        self.progress_steps = []

    def explain_data_quality(self, quality_metrics: Dict) -> str:
        """Generate human-readable explanation of data quality"""
        score = quality_metrics.get('overall_quality_score', 0)
        grade = quality_metrics.get('quality_grade', 'Unknown')

        explanation = f"""
## 🎯 Data Quality Assessment

**Overall Quality Score: {score:.1f}/100 (Grade: {grade})**

### What does this mean?
"""

        if score >= 90:
            explanation += """
Your data is in excellent condition! 🎉

- **Very few missing values** - Your dataset is nearly complete
- **Good consistency** - Data types and values are appropriate
- **Minimal duplicates** - Rows are mostly unique
- **Strong feature quality** - Features show good correlation with the target

✅ **You can proceed with confidence!** This dataset is ready for advanced machine learning.
"""
        elif score >= 70:
            explanation += """
Your data is in good condition with minor issues. 👍

- **Some missing values** - A small percentage of data is missing
- **Generally consistent** - Most data follows expected patterns
- **Few duplicates** - Dataset is mostly unique
- **Decent feature quality** - Features have moderate predictive power

⚠️ **Recommendation:** Review and handle missing values for optimal results.
"""
        elif score >= 50:
            explanation += """
Your data has moderate quality issues that should be addressed. ⚠️

- **Noticeable missing values** - Significant data gaps present
- **Some consistency issues** - Data may have outliers or incorrect types
- **Moderate duplicates** - Some rows are repeated
- **Average feature quality** - Features may need engineering

📋 **Action Required:**
1. Handle missing values (imputation or removal)
2. Check for and remove outliers
3. Remove duplicate rows
4. Consider creating new features
"""
        else:
            explanation += """
Your data requires significant cleaning before machine learning. 🔧

- **Many missing values** - Large portions of data are incomplete
- **Consistency problems** - Data types or values are problematic
- **High duplication** - Many repeated rows
- **Weak features** - Current features have low predictive power

❌ **Critical Actions Needed:**
1. Investigate and fix missing data (over {quality_metrics.get('missing_values_percentage', 0):.1f}%)
2. Remove {quality_metrics.get('duplicate_rows', 0)} duplicate rows
3. Address {quality_metrics.get('consistency_issues', 0)} consistency issues
4. Perform extensive feature engineering
"""

        # Add specific metrics
        explanation += f"""

### 📊 Detailed Breakdown

| Metric | Score | What It Means |
|--------|-------|---------------|
| **Completeness** | {quality_metrics.get('completeness_percentage', 0):.1f}% | {self._explain_completeness(quality_metrics.get('completeness_percentage', 0))} |
| **Consistency** | {quality_metrics.get('consistency_percentage', 0):.1f}% | {self._explain_consistency(quality_metrics.get('consistency_percentage', 0))} |
| **Uniqueness** | {quality_metrics.get('uniqueness_percentage', 0):.1f}% | {self._explain_uniqueness(quality_metrics.get('uniqueness_percentage', 0))} |
| **Feature Quality** | {quality_metrics.get('feature_quality_percentage', 0):.1f}% | {self._explain_feature_quality(quality_metrics.get('feature_quality_percentage', 0))} |
"""

        return explanation

    def explain_feature_importance(self, importance_results: Dict, top_n: int = 5) -> str:
        """Explain feature importance in the simplest possible terms"""
        explanation = """
## 🎯 Which Columns Matter Most? (Feature Importance)

### What Did We Do?
We figured out which columns in your data are most helpful for making predictions. Like finding out which ingredients in a recipe make the biggest difference to taste.

### Simple Example:
Imagine predicting if someone will buy a car:
- **Income** → Very important (rich people buy more cars)
- **Age** → Somewhat important (middle-aged buy more)
- **Favorite color** → Not important at all

### How We Figured This Out:
We used 2 different methods to double-check our results:

**Method 1: Random Forest** - Like asking 100 experts and seeing which columns they pay attention to
**Method 2: Correlation** - Measuring how closely each column follows the target pattern

"""

        # Extract top features from each method
        top_features_by_method = {}
        for method, results in importance_results.items():
            if isinstance(results, list) and len(results) > 0:
                top_features = results[:top_n]
                top_features_by_method[method] = top_features

                method_name = "Random Forest Test" if method == 'random_forest' else "Correlation Test"
                explanation += f"""
### 📊 {method_name}

**Top {min(top_n, len(top_features))} Most Important Columns:**

"""
                for i, (feature, score) in enumerate(top_features, 1):
                    # Create visual bar
                    bar_length = int(score * 20) if score <= 1 else int((score / max(s for _, s in top_features)) * 20)
                    bar = '█' * bar_length

                    # Simple importance rating
                    if i == 1:
                        rating = "⭐⭐⭐ SUPER IMPORTANT"
                    elif i <= 3:
                        rating = "⭐⭐ VERY IMPORTANT"
                    else:
                        rating = "⭐ IMPORTANT"

                    explanation += f"{i}. **{feature}** {rating}\n"
                    explanation += f"   {bar} Score: {score:.4f}\n\n"

        # Find consensus features
        if len(top_features_by_method) > 1:
            all_top_features = {}
            for features in top_features_by_method.values():
                for feature, _ in features:
                    all_top_features[feature] = all_top_features.get(feature, 0) + 1

            consensus_features = [f for f, count in all_top_features.items() if count > 1]

            if consensus_features:
                explanation += f"""
### 🎯 **Super Reliable Columns** (Both Methods Agree!)

These columns showed up as important in BOTH tests. They are your **most trustworthy** columns:

"""
                for feature in consensus_features:
                    explanation += f"✨ **{feature}** - Keep this column! Very reliable!\n"

                explanation += f"""

### 💡 What This Means:
**Simple Rule:** Focus on the columns listed above. They matter most.

**What To Do:**
1. ✅ **Always keep** the columns listed above
2. ⚠️ **Consider removing** columns not in the top {top_n}
3. 🎯 **Build your model** using mainly these important columns

**Why Remove Unimportant Columns?**
- Makes your model faster
- Reduces confusion (focuses on what matters)
- Prevents the model from learning wrong patterns from useless data
"""
            else:
                explanation += """
### 💡 What This Means:
**Keep your top columns** (listed above) and consider testing with and without the others.
"""

        return explanation

    def explain_feature_creation(self, created_features: Dict) -> str:
        """Explain what features were created in the simplest possible terms"""
        explanation = """
## 🎨 Creating New Helpful Columns (Feature Engineering)

### What Did We Do?
We created brand new columns from your existing columns. Like a chef combining basic ingredients to create new flavors!

### Why Create New Columns?
Sometimes the original columns don't tell the full story. New combinations can reveal hidden patterns.

**Real Example:**
- Original: Height = 180cm, Weight = 80kg
- New Column: BMI = Weight ÷ (Height × Height) = 24.7
- BMI is better for predicting health than height and weight separately!

### 🔧 Types of New Columns We Created:

"""

        feature_types = {
            'poly_': ('🔀 Combination Columns', 'Multiplying two columns together (like Area = Length × Width)', 'Example: "age × income" might predict spending better than age or income alone'),
            'ratio_': ('➗ Division Columns', 'Dividing one column by another (like Speed = Distance ÷ Time)', 'Example: "income ÷ debt" shows financial health better than just income'),
            'log_': ('📊 Compressed Columns', 'Shrinking very large numbers to smaller ones (makes patterns clearer)', 'Example: Converting $1,000,000 to 6.0 and $100,000 to 5.0'),
            'binned_': ('📦 Category Columns', 'Grouping numbers into categories (like Age Groups: Teen, Adult, Senior)', 'Example: Converting ages 25,35,45 to categories "Young","Middle","Old"'),
            'mean_': ('📈 Average Columns', 'Average of multiple columns', 'Example: "average_score" from test1, test2, test3'),
            'std_': ('📉 Variability Columns', 'How much columns vary (spread out vs consistent)', 'Example: Someone with grades 90,91,89 is more consistent than 50,90,100'),
            'max_': ('🔝 Highest Value Columns', 'The biggest value across columns', 'Example: "highest_temperature" from morning, afternoon, evening temps'),
            'min_': ('🔻 Lowest Value Columns', 'The smallest value across columns', 'Example: "minimum_price" from various stores')
        }

        categorized = {}
        for feature_name in created_features.keys():
            for prefix, (category, short_desc, example) in feature_types.items():
                if feature_name.startswith(prefix):
                    if category not in categorized:
                        categorized[category] = []
                    categorized[category].append((feature_name, short_desc, example))
                    break
            else:
                if '📝 Other Columns' not in categorized:
                    categorized['📝 Other Columns'] = []
                categorized['📝 Other Columns'].append((feature_name, 'Custom calculated column', 'Special combination for your data'))

        for category, features in categorized.items():
            explanation += f"""
### {category}

**What it does:** {features[0][1]}

**Simple example:** {features[0][2]}

**Columns we created:**
"""
            for feature_name, _, _ in features:
                # Make feature names more readable
                readable_name = feature_name.replace('_', ' ').replace('poly ', '').replace('ratio ', '').replace('log ', '')
                explanation += f"- `{readable_name}`\n"
            explanation += "\n"

        explanation += f"""
### 💡 Why These New Columns Help:

**Think of a detective solving a case:**
- 🔍 **Original Columns** = Basic clues (height, weight, age)
- ✨ **New Columns** = Connections between clues (BMI, age-to-weight ratio)
- 🎯 **Result** = Easier to spot the pattern and solve the case!

### 📊 What Happens Next:

1. **Testing** - We test if these new columns actually help (not all will)
2. **Keeping Best Ones** - We only keep columns that improve predictions
3. **Throwing Away Bad Ones** - Useless new columns get removed automatically

### 📈 Total New Columns Created: {len(created_features)}

**Good News:** You don't have to understand each column in detail. The system automatically tests them and keeps only the helpful ones!

### 🎯 Bottom Line:
We created {len(created_features)} new columns from your original data. Some will help, some won't. The system will figure out which ones to keep!
"""

        return explanation

    def explain_dimensionality_reduction(self, method: str, results: Dict) -> str:
        """Explain dimensionality reduction in the simplest possible terms"""
        if method == 'pca':
            variance_ratio = results.get('explained_variance_ratio', [])
            cumulative = results.get('cumulative_variance', [])

            explanation = """
## 📉 Making Your Data Smaller (PCA)

### What Did We Do?
We took your data with many columns and squeezed it into fewer columns. Like compressing a large photo into a smaller file size - you lose a tiny bit of quality, but the main picture stays clear.

### Why Did We Use PCA Instead of t-SNE?
**Simple Answer:** PCA is faster, more reliable, and better for training models.

**Detailed Reasons:**
1. **Speed:** PCA works in seconds. t-SNE can take minutes or hours.
2. **Consistency:** PCA gives same results every time. t-SNE gives different results each run.
3. **New Data:** PCA can process new data later. t-SNE cannot.
4. **Model Training:** Your model can learn from PCA results. Models don't work well with t-SNE.

**When We Would Use t-SNE:**
- Only for creating pretty pictures to show in presentations
- Only when you just want to "see" your data visually
- Never for actual machine learning training

### 🎯 What Happened to Your Data:

"""
            for i, (var, cum) in enumerate(zip(variance_ratio[:5], cumulative[:5]), 1):
                bar = '█' * int(var * 20)
                explanation += f"""
**New Column {i}:** {bar} {var*100:.1f}%
- This new column captured {var*100:.1f}% of your original data's patterns
- Running total: {cum*100:.1f}% of information saved so far
"""

            if len(cumulative) >= 2:
                info_kept = cumulative[1]*100
                info_lost = 100 - info_kept
                original_cols = len(variance_ratio)

                explanation += f"""

### 💡 Bottom Line:
**Before:** {original_cols} columns
**After:** 2 columns
**Information Kept:** {info_kept:.1f}%
**Information Lost:** {info_lost:.1f}%

### Is This Good?
"""
                if info_kept >= 90:
                    explanation += f"✅ **EXCELLENT!** We kept {info_kept:.1f}% of information with just 2 columns. This is great compression!"
                elif info_kept >= 70:
                    explanation += f"✅ **GOOD!** We kept {info_kept:.1f}% of information. Some detail lost, but most patterns preserved."
                else:
                    explanation += f"⚠️ **OKAY.** We kept {info_kept:.1f}% of information. Your original {original_cols} columns had very different information - hard to compress."

                explanation += f"""

### Why This Matters:
1. **Faster Training:** Fewer columns = faster model training (maybe 10x-100x faster!)
2. **Less Overfitting:** Fewer columns = model less likely to memorize noise
3. **Easier to Understand:** 2 columns are easier to visualize than {original_cols} columns
4. **Same Performance:** You usually get similar accuracy with these {len(cumulative[:2])} columns as with all {original_cols} columns
"""

        elif method == 'tsne':
            explanation = """
## 🗺️ Making a Map of Your Data (t-SNE)

### What Did We Do?
We created a 2D map where similar data points sit close together. Like arranging similar items on a table - shoes near shoes, hats near hats.

### Why Did We Use t-SNE?
**We DIDN'T use t-SNE for your analysis!** We only used PCA.

### Why Not t-SNE?
1. **Too Slow:** Takes much longer to calculate
2. **Can't Train Models:** Machine learning models can't use t-SNE results
3. **Unpredictable:** Gives different results each time you run it
4. **Can't Process New Data:** Only works on the exact data you gave it

### When You WOULD Use t-SNE:
- **Only for visualization** - Making pretty pictures of your data clusters
- **Only for exploration** - Seeing if similar items group together
- **Never for training** - Your actual model should use PCA, not t-SNE

### 💡 Think of It This Way:
- **PCA** = GPS coordinates (precise, consistent, useful for navigation/models)
- **t-SNE** = Hand-drawn sketch map (pretty, good for seeing layout, not precise)

For machine learning, we need GPS coordinates (PCA), not sketch maps (t-SNE).
"""

        return explanation

    def explain_model_comparison(self, comparison_results: Dict) -> str:
        """Explain feature set comparison in the simplest possible terms"""
        explanation = """
## ⚖️ Which Columns Work Best? (Testing Different Feature Sets)

### What Did We Do?
We tested different groups of columns to see which group makes the best predictions. Like trying different ingredient combinations in a recipe to find the tastiest one.

### Why Did We Do This?
1. **Find the best columns** - Some columns help predictions, some don't
2. **Save time** - Fewer columns = faster training
3. **Avoid confusion** - Too many columns can confuse the model

### 📊 Test Results:

"""

        # Sort by performance
        valid_results = {k: v for k, v in comparison_results.items() if 'error' not in v}
        if valid_results:
            # Determine if classification or regression
            metric = list(valid_results.values())[0].get('metric', 'unknown')
            is_classification = metric == 'accuracy'

            # Sort (higher is better for accuracy, lower is better for MSE)
            sorted_results = sorted(valid_results.items(),
                                  key=lambda x: x[1].get('mean_score', 0),
                                  reverse=is_classification)

            best_set = sorted_results[0]
            best_score = best_set[1]['mean_score']
            best_metric = best_set[1]['metric']

            explanation += f"""
### 🏆 Winner: {best_set[0].replace('_', ' ').title()}

"""

            # Explain the metric in simple terms
            if is_classification:
                percentage = best_score * 100
                explanation += f"""
**Accuracy Score: {percentage:.1f}%**

### What Does This Mean?
- Out of 100 predictions, the model gets **{percentage:.0f} correct**
- It makes mistakes on **{100-percentage:.0f} out of 100**

### Is This Good?
"""
                if percentage >= 90:
                    explanation += f"✅ **EXCELLENT!** {percentage:.1f}% is very high accuracy. Your model is very reliable!"
                elif percentage >= 80:
                    explanation += f"✅ **GOOD!** {percentage:.1f}% is good accuracy. Model works well."
                elif percentage >= 70:
                    explanation += f"👍 **DECENT.** {percentage:.1f}% is okay. Room for improvement."
                else:
                    explanation += f"⚠️ **NEEDS WORK.** {percentage:.1f}% is low. Model needs improvement."

            else:  # MSE
                explanation += f"""
**MSE (Mean Squared Error): {best_score:.4f}**

### What is MSE?
MSE measures how wrong your predictions are, on average.

**Think of it like this:**
- You're guessing house prices
- Real price: $300,000
- Your guess: $310,000
- Error: $10,000
- Squared error: $10,000 × $10,000 = $100,000,000
- MSE is the average of all these squared errors

### Why Square the Errors?
1. **No negatives:** -$10,000 error and +$10,000 error both count as bad (squaring makes both positive)
2. **Punish big mistakes:** A $20,000 error is worse than two $10,000 errors
3. **Math reasons:** Makes calculations easier for the computer

### Is This MSE Good?
"""
                if best_score < 0.1:
                    explanation += f"✅ **EXCELLENT!** MSE of {best_score:.4f} is very low. Very accurate predictions!"
                elif best_score < 1.0:
                    explanation += f"✅ **GOOD!** MSE of {best_score:.4f} is reasonably low. Good predictions."
                elif best_score < 10.0:
                    explanation += f"👍 **OKAY.** MSE of {best_score:.4f} is moderate. Predictions have some error."
                else:
                    explanation += f"⚠️ **HIGH.** MSE of {best_score:.4f} is high. Predictions are often quite wrong."

                explanation += f"""

### How to Think About MSE:
- **Lower is better** (unlike accuracy where higher is better)
- Compare it to the range of your target values
- If you're predicting prices from $100K-$500K, and MSE is 0.5, that's VERY good
- If you're predicting test scores from 0-100, and MSE is 50, that's BAD
"""

            explanation += f"""

**Uses {best_set[1]['n_features']} columns**
**Consistency: ±{best_set[1]['std_score']:.4f}** (lower is more consistent)

"""

            # Show all results in simple terms
            explanation += "### All Groups We Tested:\n\n"
            for i, (set_name, results) in enumerate(sorted_results, 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
                score = results['mean_score']

                explanation += f"{emoji} **{set_name.replace('_', ' ').title()}**\n"

                if is_classification:
                    explanation += f"   - Accuracy: {score*100:.1f}% ({results['n_features']} columns)\n"
                else:
                    explanation += f"   - Error (MSE): {score:.4f} ({results['n_features']} columns)\n"
                explanation += "\n"

            # Add simple insights
            if len(sorted_results) > 1:
                explanation += f"""
### 💡 What We Learned:

"""
                if best_set[0] == 'top_10_features':
                    explanation += f"""
**Less is More!** Using just the top 10 most important columns works best.

**Why This is Great:**
- ⚡ **Faster:** Model trains in seconds instead of minutes
- 🎯 **Simpler:** Easier to understand and explain
- 💪 **Better:** Fewer columns means model focuses on what matters
- 🛡️ **Safer:** Less likely to memorize noise

**Bottom Line:** Keep only your top 10 columns. Throw away the rest!
"""
                elif best_set[0] == 'all_features':
                    explanation += f"""
**More is Better!** Using all columns together works best.

**Why:**
- Each column adds some useful information
- Your data has complex patterns that need all columns
- No columns are truly useless

**Bottom Line:** Keep all {best_set[1]['n_features']} columns for best results.
"""

        else:
            explanation += "⚠️ No test results available. Something went wrong during testing."

        return explanation

    def generate_processing_steps_explanation(self, steps: List[str]) -> str:
        """Generate explanation for preprocessing steps"""
        explanation = """
## 🔄 Data Processing Pipeline

### Steps Taken to Optimize Your Dataset:

"""

        step_details = {
            'Selected': 'Feature Selection - Kept only the most important features',
            'Added': 'Feature Engineering - Created new derived features',
            'Removed outliers': 'Outlier Removal - Eliminated extreme values that could skew results',
            'Applied': 'Feature Scaling - Normalized feature ranges for better model performance',
            'dimensionality reduction': 'Dimension Reduction - Reduced feature count while preserving information'
        }

        for i, step in enumerate(steps, 1):
            explanation += f"\n### Step {i}: "

            # Find matching detail
            for key, detail in step_details.items():
                if key.lower() in step.lower():
                    explanation += f"{detail}\n"
                    explanation += f"```\n{step}\n```\n"
                    break
            else:
                explanation += f"{step}\n"

        explanation += """

### 🎯 Why These Steps?

Each preprocessing step was automatically selected based on:
1. **Data Quality**: Addressing specific issues in your dataset
2. **ML Best Practices**: Applying transformations known to improve model performance
3. **Domain Relevance**: Considering the type of problem you're solving

### 📈 Expected Benefits:
- Improved model accuracy
- Faster training times
- More stable predictions
- Better generalization to new data
"""

        return explanation

    # Helper methods
    def _explain_completeness(self, score: float) -> str:
        if score >= 95:
            return "Excellent - Very few missing values"
        elif score >= 85:
            return "Good - Some missing values but manageable"
        elif score >= 70:
            return "Fair - Moderate missing values, imputation recommended"
        else:
            return "Poor - Significant missing data, major cleanup needed"

    def _explain_consistency(self, score: float) -> str:
        if score >= 90:
            return "Excellent - Data types and ranges are appropriate"
        elif score >= 75:
            return "Good - Minor consistency issues detected"
        elif score >= 60:
            return "Fair - Some outliers or type issues present"
        else:
            return "Poor - Major consistency problems found"

    def _explain_uniqueness(self, score: float) -> str:
        if score >= 95:
            return "Excellent - Almost all rows are unique"
        elif score >= 85:
            return "Good - Few duplicate rows"
        elif score >= 70:
            return "Fair - Moderate duplication detected"
        else:
            return "Poor - High duplication, consider deduplication"

    def _explain_feature_quality(self, score: float) -> str:
        if score >= 70:
            return "Excellent - Features show strong predictive signal"
        elif score >= 50:
            return "Good - Features have moderate correlation with target"
        elif score >= 30:
            return "Fair - Weak feature-target relationships"
        else:
            return "Poor - Features may need engineering"


class ProgressTracker:
    """Track and report analysis progress"""

    def __init__(self):
        self.steps = []
        self.current_step = 0
        self.total_steps = 0

    def initialize(self, step_names: List[str]):
        """Initialize with list of step names"""
        self.steps = [{'name': name, 'status': 'pending', 'message': ''} for name in step_names]
        self.total_steps = len(step_names)
        self.current_step = 0

    def update_step(self, step_index: int, status: str, message: str = ''):
        """Update a specific step"""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index]['status'] = status
            self.steps[step_index]['message'] = message
            if status == 'in_progress':
                self.current_step = step_index

    def get_progress(self) -> Dict:
        """Get current progress status"""
        completed = sum(1 for step in self.steps if step['status'] == 'completed')
        return {
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'completed_steps': completed,
            'progress_percentage': int((completed / self.total_steps) * 100) if self.total_steps > 0 else 0,
            'steps': self.steps
        }


# Global instances
file_validator = EnhancedFileValidator()
explainer = ExplainableAI()
progress_tracker = ProgressTracker()
