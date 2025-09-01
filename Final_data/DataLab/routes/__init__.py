# routes/__init__.py
from .auth_routes import auth_bp
from .dashboard_routes import dashboard_bp
from .data_routes import data_bp
from .visualization_routes import visualization_bp
from .preprocessing_routes import preprocessing_bp
from .ai_routes import ai_bp
from .ml_routes import ml_bp
from .workflow_routes import workflow_bp
from .module6_routes import module6_bp

def register_routes(app):
    """Register all blueprints with the Flask app"""
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(data_bp)
    app.register_blueprint(visualization_bp)
    app.register_blueprint(preprocessing_bp)
    app.register_blueprint(ai_bp)
    app.register_blueprint(ml_bp)
    app.register_blueprint(workflow_bp)
    app.register_blueprint(module6_bp) 