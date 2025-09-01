from flask import Blueprint

# Create dashboard blueprint (placeholder since main app handles dashboard directly)
dashboard_bp = Blueprint('dashboard', __name__, url_prefix='/dash')

@dashboard_bp.route('/status')
def dashboard_status():
    return {"status": "Dashboard handled by main app"} 