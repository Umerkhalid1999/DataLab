import logging
from functools import wraps
from flask import request, redirect, url_for, session, make_response
from firebase_admin import auth

logger = logging.getLogger(__name__)

def verify_firebase_token(id_token):
    """
    Verify Firebase ID token and return user information

    Args:
        id_token (str): Firebase ID token

    Returns:
        dict: User information or None if token is invalid
    """
    try:
        # Verify and decode the ID token
        decoded_token = auth.verify_id_token(id_token)

        # Get user information
        user = auth.get_user(decoded_token['uid'])

        return {
            'uid': user.uid,
            'email': user.email,
            'display_name': user.display_name
        }
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return None

def login_required(f):
    """Authentication middleware"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get the token from cookies
        token = request.cookies.get('token')
        logger.info(f"login_required - Cookie token present: {token is not None}")

        # If no token, redirect to login
        if not token:
            logger.warning("No token in cookies, redirecting to login")
            return redirect(url_for('login'))

        try:
            # Verify the token
            user_info = verify_firebase_token(token)

            if not user_info:
                logger.warning("Token verification failed")
                # Clear invalid token
                response = make_response(redirect(url_for('login')))
                response.set_cookie('token', '', expires=0)
                return response

            logger.info(f"User authenticated: {user_info['email']}")

            # Set user info in session if not already there
            if 'user_id' not in session or session['user_id'] != user_info['uid']:
                logger.info(f"Setting session for user: {user_info['uid']}")
                session['user_id'] = user_info['uid']
                session['email'] = user_info['email']
                session['username'] = user_info['display_name'] or user_info['email'].split('@')[0]

            return f(*args, **kwargs)

        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            # Clear session and redirect to login
            session.clear()
            return redirect(url_for('login'))

    return decorated_function 