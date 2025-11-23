# routes/community_routes.py - Real-time Community Platform with Firebase
from flask import Blueprint, render_template, request, jsonify, session
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import firebase_admin
from firebase_admin import firestore, storage
import os
import uuid

# Create Blueprint
community_bp = Blueprint('community', __name__, url_prefix='/community')

# Get Firestore client
db = firestore.client()

# File upload configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'uploads', 'profiles')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_user():
    """Get current user from session"""
    user_id = session.get('user_id', 'anonymous')
    email = session.get('email', 'user@example.com')
    username = session.get('username', email.split('@')[0])

    # Get user profile from Firestore
    try:
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()

        if user_doc.exists:
            user_data = user_doc.to_dict()
            return {
                'id': user_id,
                'username': user_data.get('username', username),
                'email': user_data.get('email', email),
                'profile_image': user_data.get('profile_image', '/static/img/default-avatar.png')
            }
        else:
            # Create new user profile
            user_data = {
                'id': user_id,
                'username': username,
                'email': email,
                'profile_image': '/static/img/default-avatar.png',
                'created_at': datetime.now().isoformat(),
                'bio': ''
            }
            user_ref.set(user_data)
            return {
                'id': user_id,
                'username': username,
                'email': email,
                'profile_image': '/static/img/default-avatar.png'
            }
    except Exception as e:
        print(f"Error getting user: {e}")
        return {
            'id': user_id,
            'username': username,
            'email': email,
            'profile_image': '/static/img/default-avatar.png'
        }

def update_user_presence(user_id, username, profile_image):
    """Update user's online presence in Firestore"""
    try:
        presence_ref = db.collection('user_presence').document(user_id)
        presence_ref.set({
            'username': username,
            'profile_image': profile_image,
            'last_active': firestore.SERVER_TIMESTAMP,
            'online': True
        }, merge=True)
    except Exception as e:
        print(f"Error updating presence: {e}")

# Main page
@community_bp.route('/')
def index():
    user = get_user()
    update_user_presence(user['id'], user['username'], user['profile_image'])
    return render_template('community.html', user=user)

# Get posts from Firestore
@community_bp.route('/api/posts', methods=['GET'])
def get_posts():
    try:
        user = get_user()
        update_user_presence(user['id'], user['username'], user['profile_image'])

        category = request.args.get('category', 'all')

        # Query Firestore
        posts_ref = db.collection('community_posts')

        if category != 'all':
            query = posts_ref.where('category', '==', category)
        else:
            query = posts_ref

        # Get all posts and sort by timestamp
        posts = []
        for doc in query.stream():
            post_data = doc.to_dict()
            post_data['id'] = doc.id
            posts.append(post_data)

        # Sort by timestamp (newest first)
        posts.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        return jsonify({'success': True, 'posts': posts, 'current_user_id': user['id']})
    except Exception as e:
        print(f"Error getting posts: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

# Create post
@community_bp.route('/api/posts', methods=['POST'])
def create_post():
    try:
        user = get_user()
        data = request.get_json()

        # Parse tags - handle both string and list formats
        tags_input = data.get('tags', [])
        if isinstance(tags_input, str):
            # Split by comma and clean up
            tags = [tag.strip() for tag in tags_input.split(',') if tag.strip()]
        elif isinstance(tags_input, list):
            tags = [str(tag).strip() for tag in tags_input if str(tag).strip()]
        else:
            tags = []

        new_post = {
            'user_id': user['id'],
            'username': user['username'],
            'profile_image': user['profile_image'],
            'category': data.get('type', 'discussion'),
            'title': data.get('title', ''),
            'content': data.get('content', ''),
            'tags': tags,
            'timestamp': datetime.now().isoformat(),
            'likes': 0,
            'liked_by': [],
            'comments': []
        }

        # Add to Firestore
        doc_ref = db.collection('community_posts').add(new_post)
        new_post['id'] = doc_ref[1].id

        return jsonify({'success': True, 'post': new_post})
    except Exception as e:
        print(f"Error creating post: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

# Like post
@community_bp.route('/api/posts/<post_id>/like', methods=['POST'])
def like_post(post_id):
    try:
        user = get_user()
        post_ref = db.collection('community_posts').document(post_id)
        post_doc = post_ref.get()

        if not post_doc.exists:
            return jsonify({'success': False, 'error': 'Post not found'}), 404

        post_data = post_doc.to_dict()
        liked_by = post_data.get('liked_by', [])

        if user['id'] in liked_by:
            # Unlike
            liked_by.remove(user['id'])
            likes = max(0, post_data.get('likes', 0) - 1)
        else:
            # Like
            liked_by.append(user['id'])
            likes = post_data.get('likes', 0) + 1

        # Update Firestore
        post_ref.update({
            'liked_by': liked_by,
            'likes': likes
        })

        return jsonify({'success': True, 'likes': likes, 'liked': user['id'] in liked_by})
    except Exception as e:
        print(f"Error liking post: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

# Add comment
@community_bp.route('/api/posts/<post_id>/comment', methods=['POST'])
def add_comment(post_id):
    try:
        user = get_user()
        data = request.get_json()
        post_ref = db.collection('community_posts').document(post_id)
        post_doc = post_ref.get()

        if not post_doc.exists:
            return jsonify({'success': False, 'error': 'Post not found'}), 404

        comment = {
            'id': f"comment_{int(datetime.now().timestamp() * 1000)}",
            'user_id': user['id'],
            'username': user['username'],
            'profile_image': user['profile_image'],
            'content': data.get('content', ''),
            'timestamp': datetime.now().isoformat()
        }

        # Update Firestore
        post_data = post_doc.to_dict()
        comments = post_data.get('comments', [])
        comments.append(comment)
        post_ref.update({'comments': comments})

        return jsonify({'success': True, 'comment': comment})
    except Exception as e:
        print(f"Error adding comment: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

# Get active/online users
@community_bp.route('/api/active-users', methods=['GET'])
def get_active_users():
    try:
        user = get_user()
        update_user_presence(user['id'], user['username'], user['profile_image'])

        # Get users active in last 5 minutes
        cutoff = datetime.now() - timedelta(minutes=5)

        active_users = []
        presence_ref = db.collection('user_presence')

        for doc in presence_ref.stream():
            presence_data = doc.to_dict()
            last_active = presence_data.get('last_active')

            if last_active:
                # Convert Firestore timestamp to datetime
                if hasattr(last_active, 'timestamp'):
                    last_active_dt = datetime.fromtimestamp(last_active.timestamp())
                else:
                    try:
                        last_active_dt = datetime.fromisoformat(str(last_active))
                    except:
                        continue

                if last_active_dt > cutoff:
                    active_users.append({
                        'id': doc.id,
                        'username': presence_data.get('username'),
                        'profile_image': presence_data.get('profile_image', '/static/img/default-avatar.png'),
                        'online': True
                    })

        return jsonify({'success': True, 'users': active_users, 'count': len(active_users)})
    except Exception as e:
        print(f"Error getting active users: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

# Get trending topics (based on tags)
@community_bp.route('/api/trending', methods=['GET'])
def get_trending():
    try:
        week_ago = datetime.now() - timedelta(days=7)
        tag_counts = {}

        # Get all posts from last 7 days
        posts_ref = db.collection('community_posts')

        for doc in posts_ref.stream():
            post_data = doc.to_dict()
            timestamp_str = post_data.get('timestamp', '')

            try:
                post_time = datetime.fromisoformat(timestamp_str)
                if post_time > week_ago:
                    tags = post_data.get('tags', [])
                    for tag in tags:
                        tag_clean = tag.strip().lower()
                        if tag_clean:
                            tag_counts[tag_clean] = tag_counts.get(tag_clean, 0) + 1
            except:
                continue

        # Get top 5 trending tags
        trending = [
            {'tag': tag, 'count': count}
            for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

        return jsonify({'success': True, 'trending': trending})
    except Exception as e:
        print(f"Error getting trending topics: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

# Get stats
@community_bp.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        user = get_user()
        update_user_presence(user['id'], user['username'], user['profile_image'])

        # Count posts
        posts_count = len(list(db.collection('community_posts').stream()))

        # Count comments
        total_comments = 0
        for doc in db.collection('community_posts').stream():
            post_data = doc.to_dict()
            total_comments += len(post_data.get('comments', []))

        # Count unique tags
        all_tags = set()
        for doc in db.collection('community_posts').stream():
            post_data = doc.to_dict()
            for tag in post_data.get('tags', []):
                all_tags.add(tag.strip().lower())

        # Count active members
        cutoff = datetime.now() - timedelta(minutes=5)
        active_count = 0

        for doc in db.collection('user_presence').stream():
            presence_data = doc.to_dict()
            last_active = presence_data.get('last_active')

            if last_active:
                if hasattr(last_active, 'timestamp'):
                    last_active_dt = datetime.fromtimestamp(last_active.timestamp())
                else:
                    try:
                        last_active_dt = datetime.fromisoformat(str(last_active))
                    except:
                        continue

                if last_active_dt > cutoff:
                    active_count += 1

        return jsonify({
            'success': True,
            'stats': {
                'active_members': active_count,
                'total_posts': posts_count,
                'total_comments': total_comments,
                'trending_topics': len(all_tags)
            }
        })
    except Exception as e:
        print(f"Error getting stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

# Upload profile image
@community_bp.route('/api/upload-profile-image', methods=['POST'])
def upload_profile_image():
    try:
        user = get_user()

        if 'profile_image' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400

        file = request.files['profile_image']

        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        if file and allowed_file(file.filename):
            # Generate unique filename
            ext = file.filename.rsplit('.', 1)[1].lower()
            filename = f"{user['id']}_{int(datetime.now().timestamp())}.{ext}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            # Save file
            file.save(filepath)

            # Update user profile in Firestore
            profile_image_url = f"/static/uploads/profiles/{filename}"
            user_ref = db.collection('users').document(user['id'])
            user_ref.update({'profile_image': profile_image_url})

            # Update presence with new image
            update_user_presence(user['id'], user['username'], profile_image_url)

            return jsonify({
                'success': True,
                'profile_image': profile_image_url,
                'message': 'Profile image uploaded successfully'
            })
        else:
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
    except Exception as e:
        print(f"Error uploading profile image: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

# Get user profile
@community_bp.route('/api/user/<user_id>', methods=['GET'])
def get_user_profile(user_id):
    try:
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()

        if not user_doc.exists:
            return jsonify({'success': False, 'error': 'User not found'}), 404

        user_data = user_doc.to_dict()
        return jsonify({'success': True, 'user': user_data})
    except Exception as e:
        print(f"Error getting user profile: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

# Update user profile
@community_bp.route('/api/update-profile', methods=['POST'])
def update_profile():
    try:
        user = get_user()
        data = request.get_json()

        user_ref = db.collection('users').document(user['id'])

        update_data = {}
        if 'username' in data:
            update_data['username'] = data['username']
        if 'bio' in data:
            update_data['bio'] = data['bio']

        if update_data:
            user_ref.update(update_data)
            return jsonify({'success': True, 'message': 'Profile updated successfully'})
        else:
            return jsonify({'success': False, 'error': 'No data to update'}), 400
    except Exception as e:
        print(f"Error updating profile: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

# ==================== PRIVATE MESSAGING ====================

# Send private message
@community_bp.route('/api/messages/send', methods=['POST'])
def send_message():
    try:
        sender = get_user()
        data = request.get_json()

        recipient_id = data.get('recipient_id')
        content = data.get('content', '').strip()

        if not recipient_id or not content:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400

        # Create conversation ID (alphabetically sorted to ensure consistency)
        conv_id = '_'.join(sorted([sender['id'], recipient_id]))

        message = {
            'sender_id': sender['id'],
            'sender_username': sender['username'],
            'sender_profile_image': sender['profile_image'],
            'recipient_id': recipient_id,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'seen': False,
            'seen_at': None
        }

        # Add message to conversation
        conv_ref = db.collection('conversations').document(conv_id)
        conv_doc = conv_ref.get()

        if conv_doc.exists:
            conv_data = conv_doc.to_dict()
            status = conv_data.get('status', 'pending')

            # Check if conversation is accepted
            if status == 'accepted':
                # Update existing accepted conversation
                messages = conv_data.get('messages', [])
                messages.append(message)
                conv_ref.update({
                    'messages': messages,
                    'last_message': content,
                    'last_message_time': datetime.now().isoformat(),
                    'updated_at': firestore.SERVER_TIMESTAMP
                })
            elif status == 'rejected':
                return jsonify({'success': False, 'error': 'Message request was rejected'}), 403
            else:
                # Pending - allow sender to add more messages to pending request
                if conv_data.get('initiator_id') == sender['id']:
                    messages = conv_data.get('messages', [])
                    messages.append(message)
                    conv_ref.update({
                        'messages': messages,
                        'last_message': content,
                        'last_message_time': datetime.now().isoformat(),
                        'updated_at': firestore.SERVER_TIMESTAMP
                    })
                else:
                    return jsonify({'success': False, 'error': 'Please wait for the recipient to accept your message request'}), 403
        else:
            # Create new conversation with pending status
            conv_ref.set({
                'participants': [sender['id'], recipient_id],
                'initiator_id': sender['id'],  # Who started the conversation
                'recipient_id': recipient_id,   # Who needs to accept/reject
                'status': 'pending',
                'messages': [message],
                'last_message': content,
                'last_message_time': datetime.now().isoformat(),
                'created_at': firestore.SERVER_TIMESTAMP,
                'updated_at': firestore.SERVER_TIMESTAMP
            })

        return jsonify({'success': True, 'message': message, 'status': 'pending'})
    except Exception as e:
        print(f"Error sending message: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

# Get conversations list
@community_bp.route('/api/messages/conversations', methods=['GET'])
def get_conversations():
    try:
        user = get_user()

        accepted_conversations = []
        pending_requests = []
        conv_ref = db.collection('conversations')

        for doc in conv_ref.stream():
            conv_data = doc.to_dict()
            participants = conv_data.get('participants', [])
            status = conv_data.get('status', 'accepted')  # Default to accepted for old conversations

            if user['id'] in participants and status != 'rejected':
                # Get other participant's info
                other_user_id = [p for p in participants if p != user['id']][0]
                other_user_ref = db.collection('users').document(other_user_id)
                other_user_doc = other_user_ref.get()

                if other_user_doc.exists:
                    other_user_data = other_user_doc.to_dict()

                    # Count unseen messages (only for accepted conversations)
                    unread_count = 0
                    if status == 'accepted':
                        for msg in conv_data.get('messages', []):
                            if msg.get('recipient_id') == user['id'] and not msg.get('seen', False):
                                unread_count += 1

                    conversation_obj = {
                        'conversation_id': doc.id,
                        'other_user': {
                            'id': other_user_id,
                            'username': other_user_data.get('username'),
                            'profile_image': other_user_data.get('profile_image', '/static/img/default-avatar.png')
                        },
                        'last_message': conv_data.get('last_message', ''),
                        'last_message_time': conv_data.get('last_message_time', ''),
                        'unread_count': unread_count,
                        'status': status
                    }

                    # Separate into accepted conversations and pending requests
                    if status == 'accepted':
                        accepted_conversations.append(conversation_obj)
                    elif status == 'pending':
                        # Only show as pending request if current user is the recipient
                        if conv_data.get('recipient_id') == user['id']:
                            conversation_obj['is_request'] = True
                            pending_requests.append(conversation_obj)
                        else:
                            # Show as sent request for the initiator
                            conversation_obj['is_sent_request'] = True
                            accepted_conversations.append(conversation_obj)

        # Sort by last message time
        accepted_conversations.sort(key=lambda x: x.get('last_message_time', ''), reverse=True)
        pending_requests.sort(key=lambda x: x.get('last_message_time', ''), reverse=True)

        return jsonify({
            'success': True,
            'conversations': accepted_conversations,
            'pending_requests': pending_requests,
            'pending_count': len(pending_requests)
        })
    except Exception as e:
        print(f"Error getting conversations: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

# Get messages in a conversation
@community_bp.route('/api/messages/<recipient_id>', methods=['GET'])
def get_messages(recipient_id):
    try:
        user = get_user()

        # Create conversation ID
        conv_id = '_'.join(sorted([user['id'], recipient_id]))

        conv_ref = db.collection('conversations').document(conv_id)
        conv_doc = conv_ref.get()

        if not conv_doc.exists:
            return jsonify({'success': True, 'messages': [], 'status': 'none'})

        conv_data = conv_doc.to_dict()
        status = conv_data.get('status', 'accepted')

        # Get other user's online status
        presence_ref = db.collection('user_presence').document(recipient_id)
        presence_doc = presence_ref.get()
        is_online = False
        if presence_doc.exists:
            presence_data = presence_doc.to_dict()
            last_active = presence_data.get('last_active')
            if last_active:
                cutoff = datetime.now() - timedelta(minutes=5)
                if hasattr(last_active, 'timestamp'):
                    last_active_dt = datetime.fromtimestamp(last_active.timestamp())
                else:
                    try:
                        last_active_dt = datetime.fromisoformat(str(last_active))
                    except:
                        last_active_dt = datetime.min
                is_online = last_active_dt > cutoff

        # Only show messages if conversation is accepted OR if user is viewing their own pending request
        if status == 'accepted' or (status == 'pending' and conv_data.get('recipient_id') == user['id']):
            messages = conv_data.get('messages', [])

            # Mark messages as seen (previously read) only if accepted
            if status == 'accepted':
                updated_messages = []
                has_updates = False
                for msg in messages:
                    if msg.get('recipient_id') == user['id'] and not msg.get('seen', False):
                        msg['seen'] = True
                        msg['seen_at'] = datetime.now().isoformat()
                        has_updates = True
                    updated_messages.append(msg)

                if has_updates:
                    conv_ref.update({'messages': updated_messages})
                return jsonify({'success': True, 'messages': updated_messages, 'status': status, 'recipient_online': is_online})
            else:
                # Pending request - show messages but don't mark as seen
                return jsonify({'success': True, 'messages': messages, 'status': status, 'recipient_online': is_online})
        elif status == 'rejected':
            return jsonify({'success': False, 'error': 'Message request was rejected', 'status': status}), 403
        else:
            return jsonify({'success': False, 'error': 'Please wait for approval', 'status': status}), 403

    except Exception as e:
        print(f"Error getting messages: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

# Accept message request
@community_bp.route('/api/messages/accept/<sender_id>', methods=['POST'])
def accept_message_request(sender_id):
    try:
        user = get_user()

        # Create conversation ID
        conv_id = '_'.join(sorted([user['id'], sender_id]))

        conv_ref = db.collection('conversations').document(conv_id)
        conv_doc = conv_ref.get()

        if not conv_doc.exists:
            return jsonify({'success': False, 'error': 'Conversation not found'}), 404

        conv_data = conv_doc.to_dict()

        # Verify user is the recipient
        if conv_data.get('recipient_id') != user['id']:
            return jsonify({'success': False, 'error': 'Unauthorized'}), 403

        # Update status to accepted
        conv_ref.update({
            'status': 'accepted',
            'accepted_at': firestore.SERVER_TIMESTAMP,
            'updated_at': firestore.SERVER_TIMESTAMP
        })

        return jsonify({
            'success': True,
            'message': 'Message request accepted',
            'conversation_id': conv_id
        })
    except Exception as e:
        print(f"Error accepting message request: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

# Reject message request
@community_bp.route('/api/messages/reject/<sender_id>', methods=['POST'])
def reject_message_request(sender_id):
    try:
        user = get_user()

        # Create conversation ID
        conv_id = '_'.join(sorted([user['id'], sender_id]))

        conv_ref = db.collection('conversations').document(conv_id)
        conv_doc = conv_ref.get()

        if not conv_doc.exists:
            return jsonify({'success': False, 'error': 'Conversation not found'}), 404

        conv_data = conv_doc.to_dict()

        # Verify user is the recipient
        if conv_data.get('recipient_id') != user['id']:
            return jsonify({'success': False, 'error': 'Unauthorized'}), 403

        # Update status to rejected
        conv_ref.update({
            'status': 'rejected',
            'rejected_at': firestore.SERVER_TIMESTAMP,
            'updated_at': firestore.SERVER_TIMESTAMP
        })

        return jsonify({
            'success': True,
            'message': 'Message request rejected',
            'conversation_id': conv_id
        })
    except Exception as e:
        print(f"Error rejecting message request: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

# Get all community members for messaging
@community_bp.route('/api/members', methods=['GET'])
def get_all_members():
    try:
        user = get_user()

        members = []
        users_ref = db.collection('users')

        for doc in users_ref.stream():
            if doc.id != user['id']:  # Exclude current user
                user_data = doc.to_dict()
                members.append({
                    'id': doc.id,
                    'username': user_data.get('username'),
                    'profile_image': user_data.get('profile_image', '/static/img/default-avatar.png'),
                    'bio': user_data.get('bio', '')
                })

        # Sort alphabetically by username
        members.sort(key=lambda x: x.get('username', '').lower())

        return jsonify({'success': True, 'members': members})
    except Exception as e:
        print(f"Error getting members: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400
