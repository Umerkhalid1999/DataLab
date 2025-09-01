from flask import Blueprint

# Create AI blueprint (placeholder since main app handles AI directly)
ai_bp = Blueprint('ai', __name__, url_prefix='/ai_routes')

@ai_bp.route('/status')
def ai_status():
    return {"status": "AI routes handled by main app"} 