from openai import OpenAI
import json
import os
import logging
from functools import wraps

logger = logging.getLogger(__name__)

class OpenAIAPI:
    def __init__(self, api_key=None):
        """Initialize the OpenAI API client"""
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it in the environment variable OPENAI_API_KEY or pass it directly.")

        # Initialize OpenAI client with the new format
        self.client = OpenAI(api_key=self.api_key)
        
        # Default model - you can change this to gpt-4, gpt-3.5-turbo, etc.
        self.model = "gpt-3.5-turbo"

    def is_data_science_related(self, prompt):
        """
        Check if the prompt is related to data science, AI, or data analysis
        """
        data_science_keywords = [
            # Core data science terms
            'data', 'dataset', 'dataframe', 'analysis', 'analytics', 'statistics', 'statistical',
            'machine learning', 'ml', 'ai', 'artificial intelligence', 'deep learning', 'neural network',
            'algorithm', 'model', 'training', 'prediction', 'classification', 'regression', 'clustering',
            
            # Data processing terms
            'preprocessing', 'cleaning', 'transformation', 'feature', 'variable', 'column', 'row',
            'missing values', 'outliers', 'normalization', 'scaling', 'encoding', 'imputation',
            
            # Visualization terms
            'visualization', 'chart', 'graph', 'plot', 'histogram', 'scatter', 'correlation', 'heatmap',
            'dashboard', 'plotly', 'matplotlib', 'seaborn', 'tableau',
            
            # Tools and libraries
            'python', 'pandas', 'numpy', 'scikit-learn', 'sklearn', 'tensorflow', 'pytorch', 'keras',
            'jupyter', 'notebook', 'sql', 'database', 'csv', 'json', 'excel',
            
            # Statistical terms
            'mean', 'median', 'mode', 'variance', 'standard deviation', 'distribution', 'probability',
            'hypothesis', 'testing', 'p-value', 'confidence interval', 'anova', 'chi-square',
            
            # Business intelligence
            'kpi', 'metrics', 'insights', 'trends', 'patterns', 'reporting', 'etl', 'data warehouse',
            'business intelligence', 'bi', 'data mining', 'big data'
        ]
        
        prompt_lower = prompt.lower()
        
        # Check if any data science keywords are present
        for keyword in data_science_keywords:
            if keyword in prompt_lower:
                return True
        
        # Additional checks for question patterns
        data_patterns = [
            'how to analyze', 'how to process', 'how to clean', 'how to visualize',
            'what is the best way to', 'how do i handle', 'how can i improve',
            'what algorithm', 'which model', 'what visualization', 'how to interpret'
        ]
        
        for pattern in data_patterns:
            if pattern in prompt_lower:
                return True
        
        return False

    def get_redirect_message(self, prompt):
        """
        Get a specific redirect message based on the type of off-topic question
        """
        prompt_lower = prompt.lower()
        
        # Common off-topic categories and responses
        if any(word in prompt_lower for word in ['movie', 'film', 'cinema', 'netflix']):
            return "I'm a data science specialist and can't recommend movies. However, I can help you analyze movie datasets, build recommendation systems using collaborative filtering, or create visualizations of movie trends and ratings! What data science project would you like to work on?"
        
        elif any(word in prompt_lower for word in ['recipe', 'cook', 'food', 'restaurant']):
            return "I can't help with recipes, but I'd love to help you analyze food data! I can assist with restaurant rating analysis, nutritional data preprocessing, food sales forecasting, or creating dashboards for food service analytics. What food-related data project interests you?"
        
        elif any(word in prompt_lower for word in ['weather', 'temperature', 'climate']):
            return "I can't provide weather forecasts, but I can help you analyze weather and climate data! I can assist with time series analysis of temperature data, climate change visualization, weather pattern analysis, or building predictive models for weather data. What weather analytics project can I help with?"
        
        elif any(word in prompt_lower for word in ['sport', 'football', 'soccer', 'basketball', 'game']):
            return "I can't discuss sports in general, but I'm excellent at sports analytics! I can help you analyze player performance data, create sports dashboards, build predictive models for game outcomes, or visualize team statistics. What sports data analysis would you like to explore?"
        
        elif any(word in prompt_lower for word in ['joke', 'funny', 'humor']):
            return "I'm focused on data science rather than humor! But here's something fun: did you know that data scientists spend 80% of their time cleaning data? I can help make that remaining 20% more productive with machine learning, visualization, and statistical analysis. What data challenge can I help solve?"
        
        else:
            return "I'm a specialized AI assistant focused on data science, machine learning, and data analysis. I can help you with:\n\n📊 Data preprocessing and cleaning\n🤖 Machine learning algorithms and models\n📈 Data visualization and charts\n📋 Statistical analysis\n🔧 Python/SQL for data science\n📊 Business intelligence and analytics\n\nPlease ask me questions related to data science, and I'll be happy to help! What data-related challenge can I assist you with today?"

    def generate_response(self, prompt, max_tokens=1000, temperature=0.7):
        """
        Generate a response from OpenAI model, but only for data science related questions
        """
        try:
            # Check if the question is data science related
            if not self.is_data_science_related(prompt):
                redirect_message = self.get_redirect_message(prompt)
                return {
                    "choices": [
                        {
                            "message": {
                                "content": redirect_message
                            }
                        }
                    ]
                }
            
            logger.info(f"Calling OpenAI API with model: {self.model}")
            
            # Enhanced system prompt for data science focus
            system_prompt = """You are DataLab AI, a specialized assistant focused exclusively on data science, machine learning, data analysis, and related technical topics. 

Your expertise includes:
- Data preprocessing, cleaning, and transformation
- Statistical analysis and hypothesis testing
- Machine learning algorithms and model selection
- Data visualization and dashboard creation
- Python libraries (pandas, numpy, scikit-learn, matplotlib, seaborn, plotly)
- SQL and database operations
- Business intelligence and analytics
- Data quality assessment and improvement

IMPORTANT RESTRICTIONS:
- ONLY answer questions related to data science, analytics, AI/ML, statistics, or data-related topics
- If asked about non-data topics (movies, sports, general life advice, etc.), politely redirect to your specialization
- Provide practical, actionable advice with code examples when appropriate
- Focus on helping users solve real data problems

Always be helpful, accurate, and provide clear explanations with examples when possible."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            logger.info(f"OpenAI API response received successfully")
            
            # Return in the same format as HuggingFace for compatibility
            return {
                "choices": [
                    {
                        "message": {
                            "content": response.choices[0].message.content
                        }
                    }
                ]
            }

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            # Handle various OpenAI errors
            error_msg = str(e)
            if "rate limit" in error_msg.lower():
                return {"error": "Rate limit exceeded. Please try again later."}
            elif "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                return {"error": "Invalid API key. Please check your OpenAI API key."}
            elif "quota" in error_msg.lower():
                return {"error": "API quota exceeded. Please check your OpenAI account."}
            else:
                return {"error": f"API error: {str(e)}"}

    def analyze_dataset(self, dataset_info, analysis_type="summary"):
        """Analyze dataset with OpenAI"""
        prompt_templates = {
            "summary": """Analyze this dataset and provide a comprehensive summary including:
1. Dataset overview and structure
2. Key characteristics of the data
3. Potential insights and patterns
4. Recommendations for analysis

Dataset details: {dataset_details}""",
            
            "quality": """Analyze this dataset and identify potential quality issues including:
1. Missing values and their impact
2. Data consistency problems
3. Outliers and anomalies
4. Data type issues
5. Recommendations for data cleaning

Dataset details: {dataset_details}""",
            
            "preprocessing": """Suggest comprehensive preprocessing steps for this dataset including:
1. Data cleaning strategies
2. Feature engineering opportunities
3. Normalization/scaling recommendations
4. Handling of categorical variables
5. Best practices for this type of data

Dataset details: {dataset_details}"""
        }

        if analysis_type not in prompt_templates:
            analysis_type = "summary"

        dataset_details = json.dumps(dataset_info, indent=2)
        prompt = prompt_templates[analysis_type].format(dataset_details=dataset_details)

        return self.generate_response(prompt, max_tokens=1500)

    def suggest_visualizations(self, columns_info):
        """Suggest appropriate visualizations for the given columns"""
        prompt = f"""Based on these dataset columns, suggest the most appropriate data visualizations:

Column information: {json.dumps(columns_info, indent=2)}

Please provide:
1. Specific chart types for each column type
2. Relationships that should be explored
3. Best practices for visualization
4. Interactive visualization opportunities"""

        return self.generate_response(prompt, max_tokens=1200)

    def recommend_preprocessing(self, data_sample, column_types):
        """Recommend preprocessing steps based on data sample"""
        prompt = f"""Based on this data sample and column types, recommend specific preprocessing steps:

Data sample: {json.dumps(data_sample, indent=2)}
Column types: {json.dumps(column_types, indent=2)}

Please provide:
1. Specific preprocessing steps for each column
2. Order of operations
3. Potential challenges and solutions
4. Quality improvement strategies"""

        return self.generate_response(prompt, max_tokens=1500)

    def chat_response(self, message, context=None):
        """Generate a conversational response for the AI assistant"""
        # Apply the same filtering as generate_response
        if context:
            prompt = f"Context: {context}\n\nUser question: {message}"
        else:
            prompt = message

        return self.generate_response(prompt, max_tokens=800, temperature=0.7) 