# routes/unified_workflow_routes.py - Unified Workflow using existing modules
from flask import Blueprint, render_template, session, redirect, url_for
from functools import wraps
import logging

logger = logging.getLogger(__name__)

unified_workflow_bp = Blueprint('unified_workflow', __name__, url_prefix='/workflow')

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            logger.warning('User not logged in, redirecting to login')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@unified_workflow_bp.route('/')
@login_required
def unified_workflow_page():
    """Render the unified workflow page that integrates all existing modules"""
    logger.info('Rendering unified workflow page')
    return render_template('unified_workflow.html')
