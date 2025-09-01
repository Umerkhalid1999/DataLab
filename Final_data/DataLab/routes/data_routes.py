from flask import Blueprint

# Create data blueprint (placeholder since main app handles data directly)
data_bp = Blueprint('data', __name__, url_prefix='/data_routes')

@data_bp.route('/status')
def data_status():
    return {"status": "Data routes handled by main app"} 