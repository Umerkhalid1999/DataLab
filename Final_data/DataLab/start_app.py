#!/usr/bin/env python3
"""
Start the DataLab application with fixed authentication
"""

from main import app

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🔐 DataLab - AUTHENTICATION FIXED")
    print("="*60)
    print(f"🌐 Server: http://localhost:5000")
    print(f"🔑 Login: http://localhost:5000/login")
    print(f"🏠 Dashboard: http://localhost:5000/dashboard")
    print(f"🧪 Feature Engineering: http://localhost:5000/feature_engineering/fixed")
    print(f"🔧 Force login: http://localhost:5000/force-login")
    print("="*60)
    print("🔒 Authentication Flow:")
    print("  1. Visit any URL → Redirects to login")
    print("  2. Complete Firebase login → Redirects to dashboard")
    print("  3. Access dashboard and other features")
    print("  4. Use /force-login to clear auth if needed")
    print("🐛 Debug Tools:")
    print("  • Debug auth state: http://localhost:5000/debug-auth")
    print("  • Clear all auth: http://localhost:5000/force-login")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000, host='localhost')
