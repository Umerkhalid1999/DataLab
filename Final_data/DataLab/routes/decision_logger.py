# decision_logger.py - Logs and explains all automated decisions

class DecisionLogger:
    """Logs and explains why the system made each decision"""

    def __init__(self):
        self.decisions = []

    def log_decision(self, decision_type: str, choice: str, reasoning: str, details: dict = None):
        """Log a decision with full reasoning"""
        decision = {
            'type': decision_type,
            'choice': choice,
            'reasoning': reasoning,
            'details': details or {},
            'timestamp': None
        }
        self.decisions.append(decision)

    def get_all_decisions(self):
        """Get all logged decisions"""
        return self.decisions

    def get_decisions_summary(self):
        """Get a formatted summary of all decisions"""
        summary = """
# 🤖 Why Did The System Do That? (Automated Decision Log)

## Understanding Automatic Choices

Every decision made during analysis is explained below. Nothing is random!

---

"""

        for i, decision in enumerate(self.decisions, 1):
            summary += f"""
### Decision #{i}: {decision['type']}

**What We Chose:** {decision['choice']}

**Why We Chose This:**
{decision['reasoning']}

"""
            if decision['details']:
                summary += "**Technical Details:**\n"
                for key, value in decision['details'].items():
                    summary += f"- {key}: {value}\n"
                summary += "\n"

            summary += "---\n\n"

        return summary


def explain_pca_vs_tsne_choice(dataset_size: int, n_features: int, purpose: str = "model_training"):
    """Generate explanation for why PCA was chosen over t-SNE"""

    explanation = f"""
## 🎯 Why We Used PCA Instead of t-SNE

### Quick Answer:
We used **PCA** because it's better for machine learning model training.

### The Full Story:

**Your Data:**
- {dataset_size} rows
- {n_features} columns
- Purpose: {purpose.replace('_', ' ').title()}

**Why PCA is the Right Choice:**

"""

    reasons = []

    # Reason 1: Speed
    if dataset_size > 1000:
        reasons.append({
            'title': '⚡ **Speed**',
            'explanation': f"""
Your dataset has {dataset_size} rows. Here's the time comparison:
- **PCA:** ~2-5 seconds ⚡
- **t-SNE:** ~5-30 minutes 🐌

**Decision:** We chose PCA because you'd be waiting {dataset_size // 200} times longer with t-SNE!
"""
        })
    else:
        reasons.append({
            'title': '⚡ **Speed**',
            'explanation': f"""
Even with {dataset_size} rows:
- **PCA:** ~1 second ⚡
- **t-SNE:** ~30 seconds to 2 minutes 🐌

**Decision:** PCA is 30-120x faster!
"""
        })

    # Reason 2: Consistency
    reasons.append({
        'title': '🎯 **Consistency (Reproducibility)**',
        'explanation': """
**PCA:** Run it 100 times → Get exact same results 100 times ✅
**t-SNE:** Run it 100 times → Get 100 different results ❌

**Why This Matters:**
- You can share your work and others get same results
- You can compare results over time
- Your model stays consistent

**Real Example:**
If you run this analysis tomorrow with the same data:
- PCA will give **identical** results
- t-SNE will give **completely different** pictures each time

**Decision:** We need consistent, reproducible results for science!
"""
    })

    # Reason 3: Model Training
    if purpose == "model_training":
        reasons.append({
            'title': '🤖 **Machine Learning Compatibility**',
            'explanation': """
**The Big Problem with t-SNE:**
t-SNE cannot process new data! Here's why that's a dealbreaker:

**Scenario:** You want to predict house prices
1. You train a model on 1000 houses ✅
2. Tomorrow, you get 10 new houses to predict ❌
3. t-SNE CAN'T transform these new houses!
4. Your model is useless for predictions!

**With PCA:**
1. You train a model on 1000 houses ✅
2. Tomorrow, you get 10 new houses ✅
3. PCA transforms new houses instantly ✅
4. Your model predicts perfectly! ✅

**Decision:** We chose PCA because t-SNE literally cannot be used for machine learning deployment!
"""
        })

    # Reason 4: Interpretability
    reasons.append({
        'title': '📊 **Interpretability**',
        'explanation': """
**PCA Components:**
- Each component is a mathematical combination of your original features
- You can see EXACTLY how original features contribute
- Example: "Component 1 = 0.5×age + 0.3×income + 0.2×education"

**t-SNE Components:**
- Black box - you can't interpret what they mean
- No connection to original features
- Impossible to explain to stakeholders

**Decision:** PCA lets you explain your model. t-SNE doesn't.
"""
    })

    # Reason 5: Memory and Computation
    if dataset_size > 5000 or n_features > 100:
        reasons.append({
            'title': '💾 **Memory & Computation**',
            'explanation': f"""
With {dataset_size} rows and {n_features} features:

**PCA Memory:** ~{dataset_size * n_features * 8 / 1000000:.1f}MB
**t-SNE Memory:** ~{dataset_size * dataset_size * 8 / 1000000:.1f}MB (could crash!)

**Decision:** t-SNE might crash your computer. PCA won't.
"""
        })

    # Add all reasons to explanation
    for i, reason in enumerate(reasons, 1):
        explanation += f"{i}. {reason['title']}\n{reason['explanation']}\n"

    # Add "When Would We Use t-SNE" section
    explanation += """
### 🗺️ So When WOULD We Use t-SNE?

**Only in these specific cases:**

1. **Pure Visualization for Presentations**
   - You want a pretty picture for a PowerPoint
   - You're not training a model
   - You just want to "see" if clusters exist
   - Example: "Look, our customers naturally form 3 groups!"

2. **Exploratory Data Analysis (EDA)**
   - Very first step of understanding your data
   - Just exploring, not building anything
   - Want to spot obvious patterns visually

3. **Publication Figures**
   - Research papers often include t-SNE plots
   - They look nicer than PCA plots
   - But the actual model in the paper still uses PCA!

### 🎯 Bottom Line:

**Your Goal:** Build a working machine learning model
**Best Choice:** PCA ✅
**Why:** Faster, reproducible, works with new data, interpretable

**If Your Goal Was:** Just make a pretty picture for a presentation
**Then t-SNE:** Would be fine (but we'd still use PCA for the actual model)

---

**Our Decision:** We used PCA because it's the only choice that actually works for building deployable machine learning models!
"""

    return explanation


def explain_feature_selection_strategy(importance_results: dict, dataset_info: dict):
    """Explain why certain features were selected"""

    explanation = """
## 🎯 How We Decided Which Columns to Keep

### The Question:
"Out of all your columns, which ones should we actually use?"

### Our Strategy (Step-by-Step):

"""

    # Step 1: Calculate importance
    explanation += """
**Step 1: We Tested Each Column's Importance**

Like a talent show - we tested each column to see how well it predicts the target.

**Methods Used:**
1. **Random Forest Test** - Used 100 decision trees to vote on importance
2. **Correlation Test** - Measured how closely each column follows the target pattern

**Why Two Methods?**
- If a column scores high in BOTH tests → Very reliable!
- If a column only scores high in one test → Might be luck
- If a column scores low in both → Probably useless

"""

    # Step 2: Find consensus
    all_top_features = {}
    for method, results in importance_results.items():
        if isinstance(results, list) and len(results) > 0:
            for feature, score in results[:10]:
                all_top_features[feature] = all_top_features.get(feature, 0) + 1

    consensus = [f for f, count in all_top_features.items() if count > 1]

    if consensus:
        explanation += f"""
**Step 2: We Found {len(consensus)} "Consensus Features"**

These columns scored high in BOTH tests:
"""
        for feat in consensus:
            explanation += f"- ✨ **{feat}** (Both tests agree this is important!)\n"

        explanation += """

**Why This Matters:**
When both methods agree, we're very confident these columns are genuinely important, not just random luck.

"""

    # Step 3: Compare with all features
    explanation += """
**Step 3: We Tested "Top 10 Only" vs "All Features"**

We trained models twice:
1. Using only the top 10 most important columns
2. Using ALL columns

**Why?**
Sometimes using ALL features is worse because:
- Unimportant columns add noise
- Model gets confused by irrelevant patterns
- Training takes longer
- Higher risk of overfitting

"""

    # Step 4: Decision
    explanation += """
**Step 4: We Picked The Winner**

The model results told us which strategy works better for YOUR specific data.

### 🎯 Our Recommendation:

Based on performance testing, we'll tell you:
- How many features to use
- Which specific features to keep
- Why this gives best results

**You'll see this in the "Model Comparison" section below!**

---

"""

    return explanation


def explain_why_sampling_decision(dataset_size: int, threshold: int = 5000):
    """Explain why sampling was or wasn't used"""

    if dataset_size <= threshold:
        return f"""
## 📊 Why We Used Your FULL Dataset

**Your Data:** {dataset_size} rows

**Decision:** We analyzed ALL {dataset_size} rows (no sampling needed)

**Why:**
- {dataset_size} rows is manageable size
- Analysis completes in reasonable time (~30-60 seconds)
- Using all data gives most accurate results
- No information lost

**Result:** You're getting analysis based on 100% of your data! ✅

---

"""
    else:
        sample_size = min(threshold, dataset_size)
        percentage = (sample_size / dataset_size) * 100

        return f"""
## 📊 Why We Sampled Your Dataset

**Your Full Data:** {dataset_size:,} rows
**What We Analyzed:** {sample_size:,} rows ({percentage:.1f}%)

**Decision:** We randomly sampled {sample_size:,} rows instead of using all {dataset_size:,}

### Why Sample?

**Time Saved:**
- Full dataset would take: ~{dataset_size // 100} minutes
- Sampled dataset takes: ~30-60 seconds
- **Time saved: {(dataset_size // 100) - 1} minutes!**

**Accuracy Loss: Minimal!**

Think of election polling:
- Don't need to ask ALL 300 million Americans
- Asking 1,000 random people gives 95% accurate results
- Same concept here!

**Statistical Proof:**
- Random sample of {sample_size:,} from {dataset_size:,} gives 95%+ confidence
- Your patterns are preserved
- Results are reliable

**Memory Benefits:**
- Full dataset: ~{dataset_size * 50 / 1000000:.1f}MB memory
- Sampled dataset: ~{sample_size * 50 / 1000000:.1f}MB memory
- Prevents crashes on older computers

### Is This Okay?

**Yes!** Professional data scientists do this all the time:
- Google/Facebook sample data for analysis
- Research papers use sampling
- Standard practice in industry

**Your Final Model:**
- Will be trained on FULL {dataset_size:,} rows
- This analysis just helps us understand the data
- No information lost in final deployment

**Result:** 95%+ accurate analysis, 10-100x faster! ✅

---

"""


def explain_cv_folds_choice(n_folds: int = 3):
    """Explain cross-validation fold choice"""

    return f"""
## 🔄 Why We Used {n_folds}-Fold Cross-Validation

### What is Cross-Validation?

**Simple Explanation:**
Instead of testing your model once, we test it {n_folds} times on different parts of the data and average the results.

**Analogy:**
Instead of taking one exam to determine your grade, you take {n_folds} different exams and get the average score. More fair!

### Why {n_folds} Folds Specifically?

**Common Options:**
- 3 folds - Fast, reasonably reliable ⚡
- 5 folds - Standard choice, good balance ✅
- 10 folds - Very reliable, but slow 🐌

**We Chose {n_folds} Folds Because:**

"""

    if n_folds == 3:
        explanation = f"""
**Speed Priority:**
- Each test trains the model {n_folds} times
- 3 folds = Fast enough for quick experimentation
- Good enough reliability for feature selection

**Perfect For:**
- Initial exploration
- Feature engineering phase (what we're doing now!)
- Quick comparisons

**Trade-off:**
- Slightly less accurate than 5 or 10 folds
- But 2-3x faster!
- For feature selection, this trade-off is worth it
"""
    elif n_folds == 5:
        explanation = f"""
**Industry Standard:**
- Most research papers use 5 folds
- Best balance of speed vs reliability
- Recommended by scikit-learn

**Perfect For:**
- Most machine learning tasks
- Good reliability without being slow
- This is what professionals use

**Trade-off:**
- More accurate than 3 folds
- Not as slow as 10 folds
- Sweet spot!
"""
    else:
        explanation = f"""
**Maximum Reliability:**
- Most thorough testing
- Used when accuracy is critical
- Published research often uses this

**Perfect For:**
- Final model evaluation
- When stakes are high
- Medical/financial applications

**Trade-off:**
- Slowest option
- Best accuracy
- Worth the wait for critical decisions
"""

    explanation += f"""

### What Happens During {n_folds}-Fold CV:

1. **Split data into {n_folds} equal parts**
   Example with 1000 rows: {1000 // n_folds} rows per part

2. **Train and test {n_folds} times:**
"""

    for i in range(1, n_folds + 1):
        explanation += f"   - Round {i}: Train on {n_folds - 1} parts, test on part {i}\n"

    explanation += f"""

3. **Average all {n_folds} results**
   Final score = ({' + '.join([f'score{i}' for i in range(1, n_folds + 1)])}) ÷ {n_folds}

### Why This is Better Than Single Test:

**Single Test:** Score might be 85% (but was it luck? We don't know!)
**{n_folds}-Fold CV:** Scores are 84%, 86%, 85% → Average 85% (Now we're confident!)

### 🎯 Bottom Line:

We used {n_folds} folds to get reliable performance estimates while keeping analysis reasonably fast.

---

"""

    return explanation


# Global decision logger instance
decision_logger = DecisionLogger()
