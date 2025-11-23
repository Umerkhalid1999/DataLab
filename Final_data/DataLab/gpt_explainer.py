"""GPT-3.5-turbo based explainer for ML and Feature Engineering results"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def explain_with_gpt(context, question):
    """Get explanation from GPT-3.5-turbo"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return "OpenAI API key not configured. Please set OPENAI_API_KEY in .env file."
    
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a data science expert. Format your response with clear structure: use **bold** for important terms, numbered lists (1. 2. 3.), and clear section headings ending with colon (:). Keep explanations simple and well-organized."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting explanation: {str(e)}"

def explain_ml_results(model_name, performance, dataset_info):
    """Explain ML model results"""
    context = f"""
    Model: {model_name}
    Performance Score: {performance.get('mean_score', 'N/A')}
    Training Time: {performance.get('training_time', 'N/A')}s
    Dataset: {dataset_info.get('rows', 'N/A')} rows, {dataset_info.get('columns', 'N/A')} columns
    Task: {dataset_info.get('task_type', 'N/A')}
    """
    
    question = f"""Explain this model's performance in a well-structured format:
    
    What This Model Did:
    Why These Results:
    What This Means:
    Recommendation:
    
    Use **bold** for key metrics and keep it simple."""
    return explain_with_gpt(context, question)

def explain_feature_engineering(operation, results):
    """Explain feature engineering results"""
    # Extract specific metrics from results
    results_str = str(results)[:1000]
    
    context = f"""
    Feature Engineering Operation Performed: {operation}
    Specific Results from this operation: {results_str}
    """
    
    question = f"""Explain this feature engineering operation in a clear, structured format:
    
    Operation Performed:
    What Changed:
    Key Metrics:
    Impact on Model:
    
    Focus on the SPECIFIC results shown. Use **bold** for important numbers and terms."""
    
    return explain_with_gpt(context, question)
