"""Test GPT-3.5-turbo API connection"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def test_gpt():
    api_key = os.getenv('OPENAI_API_KEY')
    
    print("=" * 50)
    print("Testing GPT-3.5-turbo Connection")
    print("=" * 50)
    
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not found in .env file")
        return False
    
    print(f"[OK] API Key found: {api_key[:20]}...{api_key[-10:]}")
    
    try:
        client = OpenAI(api_key=api_key)
        print("\n[INFO] Sending test request to GPT-3.5-turbo...")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful data science assistant."},
                {"role": "user", "content": "Explain what Random Forest algorithm does in one sentence."}
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        result = response.choices[0].message.content
        
        print("\n[SUCCESS] GPT-3.5-turbo is working!")
        print("\n" + "=" * 50)
        print("GPT Response:")
        print("=" * 50)
        print(result)
        print("=" * 50)
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n[ERROR] {type(e).__name__}")
        print(f"Details: {error_msg}")
        
        if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            print("\nPlease check your OPENAI_API_KEY in .env file")
        elif "rate limit" in error_msg.lower():
            print("\nYour API key has exceeded the rate limit")
        
        return False

if __name__ == "__main__":
    success = test_gpt()
    
    if success:
        print("\n[SUCCESS] Your GPT system is ready for the presentation!")
    else:
        print("\n[ERROR] Please fix the errors above before your presentation")
