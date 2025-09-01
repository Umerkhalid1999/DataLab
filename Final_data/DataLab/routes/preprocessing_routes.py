from flask import Blueprint

# Create preprocessing blueprint (placeholder since main app handles preprocessing directly)
preprocessing_bp = Blueprint('preprocessing', __name__, url_prefix='/prep_routes')

@preprocessing_bp.route('/status')
def preprocessing_status():
    return {"status": "Preprocessing routes handled by main app"} 