"""
OpenAI GPT-3.5-turbo Helper for Data Preprocessing Explainability
"""
import os
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

class PreprocessingExplainer:
    def __init__(self):
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)
    
    def explain_preprocessing(self, transformations, data_stats):
        """Generate explanation for preprocessing steps using GPT-3.5-turbo"""
        prompt = self._build_prompt(transformations, data_stats)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data science expert explaining preprocessing steps in simple terms for FYP presentations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"Error generating explanation: {str(e)}"
    
    def _build_prompt(self, transformations, data_stats):
        """Build prompt for GPT-3.5"""
        prompt = f"""Explain these data preprocessing steps in 3-4 sentences for a university FYP presentation:

Dataset: {data_stats.get('rows', 0)} rows, {data_stats.get('columns', 0)} columns
Original Quality Score: {data_stats.get('original_quality', 0):.1f}%

Transformations Applied:
"""
        for i, t in enumerate(transformations, 1):
            prompt += f"{i}. {t['type'].title()}: {', '.join(t.get('columns', []))}\n"
        
        prompt += "\nExplain WHY each step was necessary and HOW it improves data quality. Be specific and technical."
        return prompt
