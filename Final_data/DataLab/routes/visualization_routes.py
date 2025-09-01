from flask import Blueprint

# Create visualization blueprint (placeholder since main app handles visualization directly)
visualization_bp = Blueprint('visualization', __name__, url_prefix='/viz_routes')

@visualization_bp.route('/status')
def visualization_status():
    return {"status": "Visualization routes handled by main app"} 