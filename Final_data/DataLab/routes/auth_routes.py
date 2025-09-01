from flask import Blueprint

# Create auth blueprint (placeholder since main app handles auth directly)
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

@auth_bp.route('/status')
def auth_status():
    return {"status": "Auth handled by main app"} 