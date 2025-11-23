# Test unified workflow registration
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from routes.unified_workflow_routes import unified_workflow_bp

app = Flask(__name__)
app.config['SECRET_KEY'] = 'test'
app.register_blueprint(unified_workflow_bp)

print("\n=== UNIFIED WORKFLOW TEST ===")
print(f"Blueprint registered: {unified_workflow_bp.name}")
print(f"URL prefix: {unified_workflow_bp.url_prefix}")
print("\nRegistered routes:")
for rule in app.url_map.iter_rules():
    if 'workflow' in rule.rule:
        print(f"  {rule.rule} -> {rule.endpoint}")

print("\n✓ Unified workflow is properly configured!")
print("Access at: http://localhost:5000/workflow")
