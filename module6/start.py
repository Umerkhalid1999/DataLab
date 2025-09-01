#!/usr/bin/env python3
"""
Simple startup script for the Automated Feature Engineering Module
"""

import os
import sys
import webbrowser
import time
from threading import Timer

def open_browser():
    """Open the web browser after a short delay"""
    time.sleep(2)
    webbrowser.open('http://localhost:5000')

def main():
    print("🚀 Starting Automated Feature Engineering Module...")
    print("=" * 60)
    print()
    print("📋 Module Features:")
    print("   ✅ Automatic target column detection")
    print("   ✅ Multi-method feature importance analysis")
    print("   ✅ Intelligent feature creation (20+ new features)")
    print("   ✅ Dimensionality reduction with visualizations")
    print("   ✅ Performance comparison of feature sets")
    print("   ✅ Domain-specific feature templates")
    print("   ✅ AI-powered recommendations")
    print()
    print("🌐 Web Interface: http://localhost:5000")
    print("📁 Upload any CSV or Excel file to get started!")
    print()
    print("=" * 60)
    
    # Open browser automatically after 2 seconds
    Timer(2.0, open_browser).start()
    
    # Start Flask app
    try:
        from app import app
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down the application...")
        print("Thank you for using the Automated Feature Engineering Module!")
    except Exception as e:
        print(f"\n❌ Error starting the application: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("   2. Check if port 5000 is available")
        print("   3. Run 'python app.py' directly for more detailed error messages")

if __name__ == "__main__":
    main()
