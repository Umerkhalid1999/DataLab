# routes/mode_selector_routes.py - Mode Selector
from flask import Blueprint, render_template, session
from functools import wraps

mode_selector_bp = Blueprint('mode_selector', __name__, url_prefix='/mode-selector')

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            from flask import redirect, url_for
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@mode_selector_bp.route('/')
@login_required
def mode_selector_page():
    return render_template('mode_selector.html')
