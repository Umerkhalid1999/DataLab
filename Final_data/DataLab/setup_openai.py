#!/usr/bin/env python3
"""
Setup script for OpenAI integration in DataLab
Run this script to configure your OpenAI API key and test the connection.
"""

import os
import sys

def setup_openai():
    print("🤖 DataLab OpenAI Integration Setup")
    print("=" * 40)
    
    # Check if OpenAI is installed
    try:
        import openai
        print("✅ OpenAI package is installed")
    except ImportError:
        print("❌ OpenAI package not found. Installing...")
        os.system("pip install openai>=1.3.0")
        print("✅ OpenAI package installed")
    
    # Get API key from user
    api_key = input("\n🔑 Please enter your OpenAI API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Exiting...")
        return False
    
    # Test the API key
    print("\n🔍 Testing API key...")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Test with a simple request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        
        print("✅ API key is valid!")
        print(f"📋 Test response: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ API key test failed: {str(e)}")
        return False
    
    # Set environment variable
    print("\n⚙️  Setting up environment...")
    
    # For Windows
    if os.name == 'nt':
        os.system(f'setx OPENAI_API_KEY "{api_key}"')
        print("✅ Environment variable set (Windows)")
        print("⚠️  Please restart your terminal/IDE for changes to take effect")
    
    # For Unix/Linux/Mac
    else:
        # Add to .bashrc or .zshrc
        shell_rc = os.path.expanduser("~/.bashrc")
        if os.path.exists(os.path.expanduser("~/.zshrc")):
            shell_rc = os.path.expanduser("~/.zshrc")
        
        with open(shell_rc, "a") as f:
            f.write(f'\nexport OPENAI_API_KEY="{api_key}"\n')
        
        print(f"✅ Added to {shell_rc}")
        print("⚠️  Please run 'source ~/.bashrc' or restart your terminal")
    
    # Create .env file for local development
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    with open(env_file, 'w') as f:
        f.write(f"OPENAI_API_KEY={api_key}\n")
    
    print(f"✅ Created .env file at {env_file}")
    
    print("\n🎉 Setup complete!")
    print("\n📝 Next steps:")
    print("1. Restart your application")
    print("2. The AI Assistant will now use OpenAI instead of Hugging Face")
    print("3. Try chatting with the AI Assistant in DataLab")
    
    return True

if __name__ == "__main__":
    setup_openai() 