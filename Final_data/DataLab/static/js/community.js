// Community Page JavaScript - Real-time Features
let currentCategory = 'all';

document.addEventListener('DOMContentLoaded', function() {
    console.log('Community page loaded');

    // Initialize theme
    initializeTheme();

    // Initialize components
    initializeCategoryFilters();
    initializeNewPostModal();
    initializeMessaging();
    initializeProfileManagement();

    // Load initial data
    loadPosts();
    loadStats();
    loadActiveUsers();
    loadTrending();
    loadMessageRequests();

    // Auto-refresh every 60 seconds for smooth performance
    setInterval(() => {
        loadPosts();
        loadStats();
        loadActiveUsers();
        loadTrending();
        loadMessageRequests();
    }, 60000);
});

// Initialize Theme
function initializeTheme() {
    const themeToggle = document.getElementById('themeToggle');
    const themeIcon = document.getElementById('themeIcon');

    // Load saved theme or default to light
    const savedTheme = localStorage.getItem('community-theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);

    // Toggle theme
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('community-theme', newTheme);
            updateThemeIcon(newTheme);

            // Add ripple effect
            themeToggle.style.animation = 'pulse 0.5s ease';
            setTimeout(() => {
                themeToggle.style.animation = '';
            }, 500);
        });
    }
}

// Update theme icon
function updateThemeIcon(theme) {
    const themeIcon = document.getElementById('themeIcon');
    if (themeIcon) {
        if (theme === 'dark') {
            themeIcon.className = 'fas fa-sun';
        } else {
            themeIcon.className = 'fas fa-moon';
        }
    }
}

// Category Filters
function initializeCategoryFilters() {
    const filterButtons = document.querySelectorAll('.filter-btn');

    filterButtons.forEach(button => {
        button.addEventListener('click', function() {
            const category = this.getAttribute('data-category');
            currentCategory = category;

            // Update active filter
            filterButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');

            // Load posts for this category
            loadPosts(category);
        });
    });
}

// Load Posts
async function loadPosts(category = null) {
    try {
        const url = category && category !== 'all'
            ? `/community/api/posts?category=${category}`
            : '/community/api/posts';

        const response = await fetch(url);
        const data = await response.json();

        if (data.success) {
            renderPosts(data.posts);
        }
    } catch (error) {
        console.error('Error loading posts:', error);
    }
}

// Render Posts
function renderPosts(posts) {
    const container = document.getElementById('postsContainer');
    const emptyState = document.getElementById('emptyState');

    if (!container) return;

    if (posts.length === 0) {
        if (emptyState) emptyState.style.display = 'block';
        const existingPosts = container.querySelectorAll('.post-card');
        existingPosts.forEach(post => post.remove());
        return;
    }

    if (emptyState) emptyState.style.display = 'none';

    // Clear existing posts
    const existingPosts = container.querySelectorAll('.post-card');
    existingPosts.forEach(post => post.remove());

    // Render each post
    posts.forEach(post => {
        const postCard = createPostCard(post);
        container.insertBefore(postCard, emptyState);
    });
}

// Create Post Card
function createPostCard(post) {
    const card = document.createElement('div');
    card.className = 'post-card';
    card.setAttribute('data-post-id', post.id);

    const timestamp = formatTimestamp(post.timestamp);
    const tags = Array.isArray(post.tags) ? post.tags : [];
    const comments = Array.isArray(post.comments) ? post.comments : [];

    card.innerHTML = `
        <div class="post-header">
            <div class="post-author">
                <div class="author-avatar">
                    <img src="${post.profile_image || '/static/img/default-avatar.png'}"
                         alt="${escapeHtml(post.username)}"
                         data-user-id="${post.user_id}"
                         onerror="this.onerror=null; this.src='/static/img/default-avatar.png';">
                </div>
                <div class="author-info">
                    <h4>${escapeHtml(post.username)}</h4>
                    <span class="post-time">${timestamp}</span>
                </div>
            </div>
            <span class="post-category ${post.category}">${getCategoryLabel(post.category)}</span>
        </div>
        <div class="post-body">
            <h3>${escapeHtml(post.title)}</h3>
            <p>${escapeHtml(post.content)}</p>
        </div>
        ${tags.length > 0 ? `
        <div class="post-tags">
            ${tags.map(tag => `<span class="tag">#${escapeHtml(tag)}</span>`).join('')}
        </div>
        ` : ''}
        <div class="post-footer">
            <button class="post-action" onclick="likePost('${post.id}')">
                <i class="fas fa-heart"></i> <span>${post.likes || 0}</span>
            </button>
            <button class="post-action" onclick="toggleComments('${post.id}')">
                <i class="fas fa-comment"></i> <span>${comments.length}</span>
            </button>
            <button class="post-action">
                <i class="fas fa-share"></i> Share
            </button>
        </div>
        <div class="post-comments" id="comments-${post.id}" style="display: none;">
            <div class="comments-list">
                ${comments.map(comment => `
                    <div class="comment">
                        <div class="comment-header">
                            <img src="${comment.profile_image || '/static/img/default-avatar.png'}"
                                 alt="${escapeHtml(comment.username)}"
                                 class="comment-avatar"
                                 onerror="this.onerror=null; this.src='/static/img/default-avatar.png';">
                            <div class="comment-info">
                                <strong>${escapeHtml(comment.username)}</strong>
                                <span class="comment-time">${formatTimestamp(comment.timestamp)}</span>
                            </div>
                        </div>
                        <p class="comment-content">${escapeHtml(comment.content)}</p>
                    </div>
                `).join('')}
            </div>
            <div class="comment-form">
                <input type="text" id="comment-input-${post.id}" placeholder="Write a comment..." class="form-input">
                <button onclick="addComment('${post.id}')" class="btn-primary">
                    <i class="fas fa-paper-plane"></i>
                </button>
            </div>
        </div>
    `;

    return card;
}

// Get Category Label
function getCategoryLabel(category) {
    const labels = {
        'questions': 'Question',
        'tutorials': 'Tutorial',
        'showcases': 'Showcase',
        'discussion': 'Discussion'
    };
    return labels[category] || category;
}

// Format Timestamp
function formatTimestamp(timestamp) {
    try {
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;

        const minutes = Math.floor(diff / 60000);
        const hours = Math.floor(diff / 3600000);
        const days = Math.floor(diff / 86400000);

        if (minutes < 1) return 'Just now';
        if (minutes < 60) return `${minutes}m ago`;
        if (hours < 24) return `${hours}h ago`;
        if (days < 7) return `${days}d ago`;

        return date.toLocaleDateString();
    } catch {
        return 'Recently';
    }
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Load Stats
async function loadStats() {
    try {
        const response = await fetch('/community/api/stats');
        const data = await response.json();

        if (data.success) {
            updateStatsDisplay(data.stats);
        }
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Update Stats Display
function updateStatsDisplay(stats) {
    const elements = {
        'activeMembers': stats.active_members || 0,
        'totalPosts': stats.total_posts || 0,
        'totalComments': stats.total_comments || 0,
        'trendingCount': stats.trending_topics || 0
    };

    Object.keys(elements).forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            animateNumber(element, parseInt(element.textContent) || 0, elements[id]);
        }
    });
}

// Animate Number
function animateNumber(element, from, to) {
    if (from === to) return;

    const duration = 500;
    const steps = 20;
    const increment = (to - from) / steps;
    let current = from;
    let step = 0;

    const timer = setInterval(() => {
        step++;
        current += increment;

        if (step >= steps) {
            element.textContent = to;
            clearInterval(timer);
        } else {
            element.textContent = Math.round(current);
        }
    }, duration / steps);
}

// Load Active Users
async function loadActiveUsers() {
    try {
        const response = await fetch('/community/api/active-users');
        const data = await response.json();

        if (data.success) {
            renderActiveUsers(data.users);
        }
    } catch (error) {
        console.error('Error loading active users:', error);
    }
}

// Render Active Users
function renderActiveUsers(users) {
    const container = document.getElementById('activeMembersList');
    if (!container) return;

    if (users.length === 0) {
        container.innerHTML = '<p class="text-muted">No active members</p>';
        return;
    }

    container.innerHTML = users.slice(0, 5).map(user => `
        <div class="member-item">
            <div class="member-avatar">
                <img src="${user.profile_image || '/static/img/default-avatar.png'}"
                     alt="${escapeHtml(user.username)}"
                     data-user-id="${user.id}"
                     onerror="this.onerror=null; this.src='/static/img/default-avatar.png';">
                ${user.online ? '<span class="online-indicator"></span>' : ''}
            </div>
            <span class="member-name">${escapeHtml(user.username)}</span>
        </div>
    `).join('');
}

// Load Trending
async function loadTrending() {
    try {
        const response = await fetch('/community/api/trending');
        const data = await response.json();

        if (data.success) {
            renderTrending(data.trending);
        }
    } catch (error) {
        console.error('Error loading trending:', error);
    }
}

// Render Trending
function renderTrending(trending) {
    const container = document.getElementById('trendingList');
    if (!container) return;

    if (trending.length === 0) {
        container.innerHTML = '<p class="text-muted">No trending topics</p>';
        return;
    }

    container.innerHTML = trending.map(item => `
        <div class="trending-item">
            <span class="trending-tag">#${escapeHtml(item.tag)}</span>
            <span class="trending-count">${item.count} posts</span>
        </div>
    `).join('');
}

// New Post Modal
function initializeNewPostModal() {
    const newPostBtn = document.getElementById('newPostBtn');
    const modal = document.getElementById('postModal');
    const closeBtn = document.getElementById('closeModal');
    const form = document.getElementById('postForm');

    if (!newPostBtn || !modal || !form) return;

    // Open modal
    newPostBtn.addEventListener('click', function() {
        modal.style.display = 'flex';
    });

    // Close modal
    if (closeBtn) {
        closeBtn.addEventListener('click', function() {
            modal.style.display = 'none';
        });
    }

    // Close when clicking outside
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });

    // Handle form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        const postData = {
            type: document.getElementById('postType').value,
            title: document.getElementById('postTitle').value,
            content: document.getElementById('postContent').value,
            tags: document.getElementById('postTags').value
                .split(',')
                .map(t => t.trim())
                .filter(t => t)
        };

        try {
            const response = await fetch('/community/api/posts', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(postData)
            });

            const data = await response.json();

            if (data.success) {
                // Close modal
                modal.style.display = 'none';

                // Reset form
                form.reset();

                // Reload posts
                loadPosts(currentCategory);
                loadStats();

                // Show success message
                showNotification('Post created successfully!', 'success');
            } else {
                showNotification(data.error || 'Failed to create post', 'error');
            }
        } catch (error) {
            console.error('Error creating post:', error);
            showNotification('Error creating post', 'error');
        }
    });
}

// Like Post
async function likePost(postId) {
    try {
        const response = await fetch(`/community/api/posts/${postId}/like`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            // Update the like count in the UI
            const postCard = document.querySelector(`[data-post-id="${postId}"]`);
            if (postCard) {
                const likeButton = postCard.querySelector('.post-action');
                const likeCount = likeButton.querySelector('span');
                if (likeCount) {
                    likeCount.textContent = data.likes;
                }
            }
        }
    } catch (error) {
        console.error('Error liking post:', error);
    }
}

// Toggle Comments
function toggleComments(postId) {
    const commentsSection = document.getElementById(`comments-${postId}`);
    if (commentsSection) {
        commentsSection.style.display =
            commentsSection.style.display === 'none' ? 'block' : 'none';
    }
}

// Add Comment
async function addComment(postId) {
    const input = document.getElementById(`comment-input-${postId}`);
    if (!input || !input.value.trim()) return;

    try {
        const response = await fetch(`/community/api/posts/${postId}/comment`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                content: input.value.trim()
            })
        });

        const data = await response.json();

        if (data.success) {
            input.value = '';
            loadPosts(currentCategory);
            showNotification('Comment added!', 'success');
        } else {
            showNotification(data.error || 'Failed to add comment', 'error');
        }
    } catch (error) {
        console.error('Error adding comment:', error);
        showNotification('Error adding comment', 'error');
    }
}

// Show Notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 25px;
        background: ${type === 'success' ? '#4caf50' : type === 'error' ? '#f44336' : '#2196f3'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        z-index: 10000;
        animation: slideIn 0.3s ease;
    `;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    :root {
        /* Light Theme (Default) */
        --bg-primary: #f0f2f5;
        --bg-main: #f5f7fa;
        --bg-card: #ffffff;
        --bg-sidebar: #1a1f36;
        --bg-header: rgba(255, 255, 255, 0.95);
        --bg-overlay: rgba(0, 0, 0, 0.5);
        --text-primary: #1a1f36;
        --text-secondary: #4a5568;
        --text-muted: #718096;
        --border-color: #e2e8f0;
        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.1);
        --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.15);
        --shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.2);
        --accent-primary: #667eea;
        --accent-secondary: #764ba2;
        --success-color: #4caf50;
        --error-color: #f44336;
        --warning-color: #ff9800;
    }

    [data-theme="dark"] {
        /* Dark Theme */
        --bg-primary: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        --bg-main: #0f0f1e;
        --bg-card: #1a1a2e;
        --bg-sidebar: #16213e;
        --bg-header: rgba(26, 26, 46, 0.95);
        --bg-overlay: rgba(0, 0, 0, 0.7);
        --text-primary: #e4e4e7;
        --text-secondary: #a1a1aa;
        --text-muted: #71717a;
        --border-color: #27272a;
        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.3);
        --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.4);
        --shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.5);
        --accent-primary: #8b5cf6;
        --accent-secondary: #6366f1;
        --success-color: #10b981;
        --error-color: #ef4444;
        --warning-color: #f59e0b;
    }

    body {
        background: var(--bg-primary);
        color: var(--text-primary);
        transition: background 0.3s ease, color 0.3s ease;
    }

    .community-main {
        background: var(--bg-main);
        transition: background 0.3s ease;
    }

    .sidebar-card, .post-card, .modal-container {
        background: var(--bg-card);
        color: var(--text-primary);
        border-color: var(--border-color);
        box-shadow: var(--shadow-sm);
        transition: all 0.3s ease;
    }

    .community-header {
        background: var(--bg-header);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }

    .stat-box {
        background: var(--bg-card);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }

    .btn-theme-toggle {
        width: 45px;
        height: 45px;
        border-radius: 50%;
        background: var(--bg-card);
        border: 2px solid var(--border-color);
        color: var(--text-primary);
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
    }

    .btn-theme-toggle:hover {
        background: var(--accent-primary);
        color: white;
        transform: rotate(15deg) scale(1.1);
        box-shadow: var(--shadow-md);
    }

    .header-actions {
        display: flex;
        gap: 15px;
        align-items: center;
    }

    .form-input, .form-input:focus {
        background: var(--bg-card);
        color: var(--text-primary);
        border-color: var(--border-color);
        transition: all 0.3s ease;
    }

    .trending-item, .member-item {
        border-bottom: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }

    .text-muted {
        color: var(--text-muted);
    }

    @keyframes slideIn {
        from { transform: translateX(400px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(400px); opacity: 0; }
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.15); }
        100% { transform: scale(1); }
    }
    .post-comments {
        padding: 15px;
        background: var(--bg-main);
        border-radius: 8px;
        margin-top: 15px;
        border: 1px solid var(--border-color);
    }
    .comments-list {
        margin-bottom: 15px;
    }
    .comment {
        padding: 12px;
        background: var(--bg-card);
        border-radius: 8px;
        margin-bottom: 12px;
        border: 1px solid var(--border-color);
        transition: all 0.2s ease;
    }
    .comment:hover {
        box-shadow: var(--shadow-sm);
    }
    .comment-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
    }
    .comment-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid var(--border-color);
        flex-shrink: 0;
    }
    .comment-info {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
    .comment-info strong {
        color: var(--accent-primary);
        font-size: 14px;
        font-weight: 600;
    }
    .comment-time {
        font-size: 11px;
        color: var(--text-muted);
    }
    .comment-content {
        margin: 0 0 0 42px;
        color: var(--text-secondary);
        font-size: 14px;
        line-height: 1.5;
    }
    .comment-form {
        display: flex;
        gap: 10px;
    }
    .comment-form input {
        flex: 1;
    }
    .comment-form button {
        padding: 10px 15px;
    }
    .modal-large {
        max-width: 900px;
        width: 90%;
        height: 80vh;
    }
    .messages-body {
        padding: 0;
        height: calc(80vh - 60px);
    }
    .messages-layout {
        display: flex;
        height: 100%;
    }
    .conversations-sidebar {
        width: 300px;
        border-right: 1px solid var(--border-color);
        display: flex;
        flex-direction: column;
        background: var(--bg-card);
    }
    .search-box {
        padding: 15px;
        border-bottom: 1px solid var(--border-color);
    }
    .conversations-list {
        flex: 1;
        overflow-y: auto;
    }
    .conversation-item {
        padding: 15px;
        border-bottom: 1px solid var(--border-color);
        cursor: pointer;
        transition: background 0.2s;
        color: var(--text-primary);
    }
    .conversation-item:hover {
        background: var(--bg-main);
    }
    .conversation-item.active {
        background: var(--accent-primary);
        color: white;
    }
    .conversation-item.pending-request {
        background: rgba(255, 152, 0, 0.1);
        border-left: 3px solid #ff9800;
    }
    .chat-area {
        flex: 1;
        display: flex;
        flex-direction: column;
    }
    .chat-placeholder {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: var(--text-muted);
    }
    .chat-header {
        padding: 15px;
        border-bottom: 1px solid var(--border-color);
        background: var(--bg-main);
        color: var(--text-primary);
    }
    .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 15px;
        background: var(--bg-main);
    }
    .chat-input {
        padding: 15px;
        border-top: 1px solid var(--border-color);
        display: flex;
        gap: 10px;
        background: var(--bg-card);
    }
    .message-bubble {
        max-width: 70%;
        padding: 10px 15px;
        border-radius: 15px;
        margin-bottom: 10px;
        word-wrap: break-word;
    }
    .message-sent {
        background: var(--accent-primary);
        color: white;
        margin-left: auto;
        text-align: right;
    }
    .message-received {
        background: var(--bg-card);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
    }
    .request-actions {
        display: flex;
        gap: 10px;
        margin-top: 10px;
    }
    .btn-icon {
        background: none;
        border: none;
        cursor: pointer;
        padding: 5px 10px;
        color: var(--accent-primary);
        font-size: 18px;
        transition: all 0.3s ease;
    }
    .btn-icon:hover {
        color: var(--accent-secondary);
        transform: scale(1.1);
    }
    .badge-count {
        background: #f44336;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin-left: 8px;
    }
    .members-list {
        max-height: 400px;
        overflow-y: auto;
    }
    .member-select-item {
        padding: 12px;
        border-bottom: 1px solid var(--border-color);
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 12px;
        color: var(--text-primary);
        transition: background 0.2s ease;
    }
    .member-select-item:hover {
        background: var(--bg-main);
    }
    .header-profile {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .header-avatar {
        position: relative;
        width: 60px;
        height: 60px;
        flex-shrink: 0;
    }
    .header-profile-img {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .quick-upload-btn {
        position: absolute;
        bottom: -2px;
        right: -2px;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: #667eea;
        color: white;
        border: 2px solid white;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s;
        font-size: 11px;
    }
    .quick-upload-btn:hover {
        background: #5568d3;
        transform: scale(1.1);
    }
    .header-text {
        flex: 1;
    }
    .header-text h1, .header-text p, .post-author h4, .member-name, strong {
        text-transform: capitalize;
    }
    .image-upload-area {
        border: 2px dashed var(--border-color);
        border-radius: 8px;
        padding: 40px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
        background: var(--bg-main);
    }
    .image-upload-area:hover {
        border-color: var(--accent-primary);
        background: var(--bg-card);
    }
    .upload-placeholder i {
        font-size: 48px;
        color: var(--accent-primary);
        margin-bottom: 15px;
    }
    .upload-placeholder p {
        margin: 10px 0 5px;
        color: var(--text-primary);
        font-weight: 500;
    }
    .upload-placeholder small {
        color: var(--text-muted);
    }
    .post-author .author-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        overflow: hidden;
        background: var(--bg-main);
        border: 1px solid var(--border-color);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        color: var(--text-muted);
    }
    .post-author img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    }
    .member-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        overflow: hidden;
        background: var(--bg-main);
        border: 1px solid var(--border-color);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        color: var(--text-muted);
    }
    .member-avatar img, .member-item img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    }
    .btn-secondary {
        background: var(--bg-card);
        color: var(--text-primary);
        border: 2px solid var(--border-color);
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    .btn-secondary:hover {
        background: var(--bg-main);
        border-color: var(--accent-primary);
        color: var(--accent-primary);
        transform: translateY(-2px);
    }
    .modal-overlay {
        background: var(--bg-overlay);
    }
`;
document.head.appendChild(style);

// ==================== MESSAGING FUNCTIONALITY ====================

let currentConversationUserId = null;
let messageInterval = null;

// Initialize Profile Management
function initializeProfileManagement() {
    const quickUploadBtn = document.getElementById('quickUploadBtn');
    const uploadImageModal = document.getElementById('uploadImageModal');
    const closeUploadImageModal = document.getElementById('closeUploadImageModal');
    const imageUploadArea = document.getElementById('imageUploadArea');
    const profileImageInput = document.getElementById('profileImageInput');

    // Open upload image modal from header
    if (quickUploadBtn) {
        quickUploadBtn.addEventListener('click', () => {
            uploadImageModal.style.display = 'flex';
        });
    }

    // Close upload image modal
    if (closeUploadImageModal) {
        closeUploadImageModal.addEventListener('click', () => {
            uploadImageModal.style.display = 'none';
            resetImageUpload();
        });
    }

    // Image upload area click
    if (imageUploadArea) {
        imageUploadArea.addEventListener('click', () => {
            profileImageInput.click();
        });
    }

    // Handle image selection
    if (profileImageInput) {
        profileImageInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                handleImagePreview(file);
            }
        });
    }

    // Close modal when clicking outside
    uploadImageModal?.addEventListener('click', (e) => {
        if (e.target === uploadImageModal) {
            uploadImageModal.style.display = 'none';
            resetImageUpload();
        }
    });
}

// Handle Image Preview
function handleImagePreview(file) {
    // Validate file size (5MB max)
    if (file.size > 5 * 1024 * 1024) {
        showNotification('Image size must be less than 5MB', 'error');
        return;
    }

    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showNotification('Please upload a valid image file (PNG, JPG, JPEG, GIF, WEBP)', 'error');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        const previewImg = document.getElementById('previewImg');
        const imagePreview = document.getElementById('imagePreview');
        const imageUploadArea = document.getElementById('imageUploadArea');

        if (previewImg && imagePreview && imageUploadArea) {
            previewImg.src = e.target.result;
            imageUploadArea.style.display = 'none';
            imagePreview.style.display = 'block';
        }
    };
    reader.readAsDataURL(file);
}

// Cancel Image Upload
function cancelImageUpload() {
    resetImageUpload();
}

// Reset Image Upload
function resetImageUpload() {
    const profileImageInput = document.getElementById('profileImageInput');
    const imagePreview = document.getElementById('imagePreview');
    const imageUploadArea = document.getElementById('imageUploadArea');

    if (profileImageInput) profileImageInput.value = '';
    if (imagePreview) imagePreview.style.display = 'none';
    if (imageUploadArea) imageUploadArea.style.display = 'block';
}

// Upload Profile Image
async function uploadProfileImage() {
    const profileImageInput = document.getElementById('profileImageInput');
    const file = profileImageInput.files[0];

    if (!file) {
        showNotification('Please select an image', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('profile_image', file);

    try {
        showNotification('Uploading...', 'info');

        const response = await fetch('/community/api/upload-profile-image', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            showNotification('Profile image updated successfully!', 'success');

            // Update the header profile image immediately
            const headerProfileImage = document.getElementById('headerProfileImage');
            if (headerProfileImage) {
                headerProfileImage.src = data.profile_image + '?t=' + new Date().getTime();
                headerProfileImage.style.display = 'block';
            }

            // Close modal and reset
            const uploadImageModal = document.getElementById('uploadImageModal');
            if (uploadImageModal) {
                uploadImageModal.style.display = 'none';
            }
            resetImageUpload();

            // Reload posts and active users to show updated profile images
            loadPosts(currentCategory);
            loadActiveUsers();
        } else {
            showNotification(data.error || 'Failed to upload image', 'error');
        }
    } catch (error) {
        console.error('Error uploading image:', error);
        showNotification('Error uploading image', 'error');
    }
}

// Initialize Messaging
function initializeMessaging() {
    const openMessagesBtn = document.getElementById('openMessagesBtn');
    const messagesModal = document.getElementById('messagesModal');
    const closeMessagesModal = document.getElementById('closeMessagesModal');
    const newMessageBtn = document.getElementById('newMessageBtn');
    const newMessageModal = document.getElementById('newMessageModal');
    const closeNewMessageModal = document.getElementById('closeNewMessageModal');

    // Open messages modal
    if (openMessagesBtn) {
        openMessagesBtn.addEventListener('click', () => {
            messagesModal.style.display = 'flex';
            loadConversations();
        });
    }

    // Close messages modal
    if (closeMessagesModal) {
        closeMessagesModal.addEventListener('click', () => {
            messagesModal.style.display = 'none';
            if (messageInterval) {
                clearInterval(messageInterval);
                messageInterval = null;
            }
        });
    }

    // Open new message modal
    if (newMessageBtn) {
        newMessageBtn.addEventListener('click', () => {
            newMessageModal.style.display = 'flex';
            loadAllMembers();
        });
    }

    // Close new message modal
    if (closeNewMessageModal) {
        closeNewMessageModal.addEventListener('click', () => {
            newMessageModal.style.display = 'none';
        });
    }

    // Close modals when clicking outside
    messagesModal?.addEventListener('click', (e) => {
        if (e.target === messagesModal) {
            messagesModal.style.display = 'none';
        }
    });

    newMessageModal?.addEventListener('click', (e) => {
        if (e.target === newMessageModal) {
            newMessageModal.style.display = 'none';
        }
    });
}

// Load Message Requests
async function loadMessageRequests() {
    try {
        const response = await fetch('/community/api/messages/conversations');
        const data = await response.json();

        if (data.success) {
            const requestsCard = document.getElementById('messageRequestsCard');
            const requestsList = document.getElementById('messageRequestsList');
            const requestBadge = document.getElementById('requestBadge');

            if (data.pending_count > 0) {
                requestsCard.style.display = 'block';
                requestBadge.textContent = data.pending_count;

                requestsList.innerHTML = data.pending_requests.map(req => `
                    <div class="request-item" style="padding: 12px; border-bottom: 1px solid #e0e0e0;">
                        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                            <div class="conversation-avatar" style="width: 40px; height: 40px;">
                                <img src="${req.other_user.profile_image || '/static/img/default-avatar.png'}"
                                     alt="${escapeHtml(req.other_user.username)}"
                                     onerror="this.onerror=null; this.src='/static/img/default-avatar.png';">
                            </div>
                            <div style="flex: 1;">
                                <strong>${escapeHtml(req.other_user.username)}</strong>
                                <p style="margin: 5px 0; font-size: 13px; color: #666;">${escapeHtml(req.last_message)}</p>
                            </div>
                        </div>
                        <div class="request-actions">
                            <button class="btn-primary" style="flex: 1; padding: 8px;" onclick="acceptMessageRequest('${req.other_user.id}')">
                                <i class="fas fa-check"></i> Accept
                            </button>
                            <button class="btn-secondary" style="flex: 1; padding: 8px; background: #f44336;" onclick="rejectMessageRequest('${req.other_user.id}')">
                                <i class="fas fa-times"></i> Reject
                            </button>
                        </div>
                    </div>
                `).join('');
            } else {
                requestsCard.style.display = 'none';
            }
        }
    } catch (error) {
        console.error('Error loading message requests:', error);
    }
}

// Accept Message Request
async function acceptMessageRequest(senderId) {
    try {
        const response = await fetch(`/community/api/messages/accept/${senderId}`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            showNotification('Message request accepted!', 'success');
            loadMessageRequests();
            loadConversations();
        } else {
            showNotification(data.error || 'Failed to accept request', 'error');
        }
    } catch (error) {
        console.error('Error accepting request:', error);
        showNotification('Error accepting request', 'error');
    }
}

// Reject Message Request
async function rejectMessageRequest(senderId) {
    try {
        const response = await fetch(`/community/api/messages/reject/${senderId}`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            showNotification('Message request rejected', 'info');
            loadMessageRequests();
        } else {
            showNotification(data.error || 'Failed to reject request', 'error');
        }
    } catch (error) {
        console.error('Error rejecting request:', error);
        showNotification('Error rejecting request', 'error');
    }
}

// Load Conversations
async function loadConversations() {
    try {
        const response = await fetch('/community/api/messages/conversations');
        const data = await response.json();

        if (data.success) {
            renderConversations(data.conversations);
        }
    } catch (error) {
        console.error('Error loading conversations:', error);
    }
}

// Render Conversations
function renderConversations(conversations) {
    const container = document.getElementById('conversationsList');
    if (!container) return;

    if (conversations.length === 0) {
        container.innerHTML = '<p class="text-muted" style="padding: 15px;">No conversations yet</p>';
        return;
    }

    container.innerHTML = conversations.map(conv => {
        const isPending = conv.is_sent_request;
        return `
            <div class="conversation-item ${isPending ? 'pending-request' : ''}" onclick="openConversation('${conv.other_user.id}', '${escapeHtml(conv.other_user.username)}', '${conv.other_user.profile_image || '/static/img/default-avatar.png'}')">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div class="conversation-avatar">
                        <img src="${conv.other_user.profile_image || '/static/img/default-avatar.png'}"
                             alt="${escapeHtml(conv.other_user.username)}"
                             onerror="this.onerror=null; this.src='/static/img/default-avatar.png';">
                    </div>
                    <div style="flex: 1; min-width: 0;">
                        <strong>${escapeHtml(conv.other_user.username)}</strong>
                        ${isPending ? '<span style="color: #ff9800; font-size: 12px;"> (Pending)</span>' : ''}
                        <p style="margin: 5px 0; font-size: 13px; color: #666; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                            ${escapeHtml(conv.last_message)}
                        </p>
                    </div>
                    ${conv.unread_count > 0 ? `<span class="badge-count">${conv.unread_count}</span>` : ''}
                </div>
            </div>
        `;
    }).join('');
}

// Open Conversation
async function openConversation(userId, username, profileImage = '/static/img/default-avatar.png') {
    currentConversationUserId = userId;

    const chatArea = document.getElementById('chatArea');
    chatArea.innerHTML = `
        <div class="chat-header">
            <div class="chat-user-info">
                <div class="chat-user-avatar">
                    <img src="${profileImage}"
                         alt="${escapeHtml(username)}"
                         onerror="this.onerror=null; this.src='/static/img/default-avatar.png';">
                    <span class="online-indicator" id="chatOnlineIndicator" style="display: none;"></span>
                </div>
                <div class="chat-user-status">
                    <strong>${escapeHtml(username)}</strong>
                    <span class="offline-status" id="chatUserStatus">Offline</span>
                </div>
            </div>
        </div>
        <div class="chat-messages" id="chatMessages">
            <p class="text-muted">Loading messages...</p>
        </div>
        <div class="chat-input">
            <input type="text" id="messageInput" placeholder="Type a message..." class="form-input" onkeypress="if(event.key === 'Enter') sendMessage()">
            <button class="btn-primary" onclick="sendMessage()">
                <i class="fas fa-paper-plane"></i> Send
            </button>
        </div>
    `;

    // Highlight active conversation
    document.querySelectorAll('.conversation-item').forEach(item => {
        item.classList.remove('active');
    });
    event.target.closest('.conversation-item')?.classList.add('active');

    // Load messages
    loadMessages(userId);

    // Set up auto-refresh for messages (every 5 seconds for smoother performance)
    if (messageInterval) clearInterval(messageInterval);
    messageInterval = setInterval(() => loadMessages(userId), 5000);
}

// Load Messages
async function loadMessages(userId) {
    try {
        const response = await fetch(`/community/api/messages/${userId}`);
        const data = await response.json();

        if (data.success) {
            renderMessages(data.messages, data.status);

            // Update online status indicator
            const onlineIndicator = document.getElementById('chatOnlineIndicator');
            const userStatus = document.getElementById('chatUserStatus');
            if (onlineIndicator && userStatus) {
                if (data.recipient_online) {
                    onlineIndicator.style.display = 'block';
                    userStatus.textContent = 'Online';
                    userStatus.className = 'online-status';
                } else {
                    onlineIndicator.style.display = 'none';
                    userStatus.textContent = 'Offline';
                    userStatus.className = 'offline-status';
                }
            }
        } else if (data.status === 'pending') {
            const chatMessages = document.getElementById('chatMessages');
            if (chatMessages) {
                chatMessages.innerHTML = '<p class="text-muted" style="padding: 15px;">Waiting for the recipient to accept your message request...</p>';
            }
        }
    } catch (error) {
        console.error('Error loading messages:', error);
    }
}

// Render Messages
function renderMessages(messages, status) {
    const container = document.getElementById('chatMessages');
    if (!container) return;

    if (messages.length === 0) {
        container.innerHTML = '<p class="text-muted" style="padding: 15px;">No messages yet. Start the conversation!</p>';
        return;
    }

    const currentUserId = '{{ user.id }}'; // This will be replaced by Flask template

    container.innerHTML = messages.map(msg => {
        const isSent = msg.sender_id !== currentConversationUserId;

        // Determine message status for sent messages
        let statusIndicator = '';
        if (isSent) {
            if (msg.seen) {
                statusIndicator = '<span class="message-status seen" title="Seen"><i class="fas fa-check-double"></i></span>';
            } else {
                statusIndicator = '<span class="message-status delivered" title="Delivered"><i class="fas fa-check"></i></span>';
            }
        }

        return `
            <div class="message-bubble ${isSent ? 'message-sent' : 'message-received'}">
                <p style="margin: 0;">${escapeHtml(msg.content)}</p>
                <div style="display: flex; align-items: center; justify-content: ${isSent ? 'flex-end' : 'flex-start'}; gap: 5px; margin-top: 3px;">
                    <small style="opacity: 0.7; font-size: 11px;">${formatTimestamp(msg.timestamp)}</small>
                    ${statusIndicator}
                </div>
            </div>
        `;
    }).join('');

    // Scroll to bottom
    container.scrollTop = container.scrollHeight;
}

// Send Message
async function sendMessage() {
    const input = document.getElementById('messageInput');
    if (!input || !input.value.trim() || !currentConversationUserId) return;

    const message = input.value.trim();
    input.value = '';

    try {
        const response = await fetch('/community/api/messages/send', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                recipient_id: currentConversationUserId,
                content: message
            })
        });

        const data = await response.json();

        if (data.success) {
            loadMessages(currentConversationUserId);
            loadConversations();
        } else {
            showNotification(data.error || 'Failed to send message', 'error');
            input.value = message; // Restore message
        }
    } catch (error) {
        console.error('Error sending message:', error);
        showNotification('Error sending message', 'error');
        input.value = message; // Restore message
    }
}

// Load All Members
async function loadAllMembers() {
    try {
        const response = await fetch('/community/api/members');
        const data = await response.json();

        if (data.success) {
            renderMembersForNewMessage(data.members);
        }
    } catch (error) {
        console.error('Error loading members:', error);
    }
}

// Render Members for New Message
function renderMembersForNewMessage(members) {
    const container = document.getElementById('newMessageMembersList');
    if (!container) return;

    if (members.length === 0) {
        container.innerHTML = '<p class="text-muted">No members found</p>';
        return;
    }

    container.innerHTML = members.map(member => `
        <div class="member-select-item" onclick="startNewConversation('${member.id}', '${escapeHtml(member.username)}', '${member.profile_image || '/static/img/default-avatar.png'}')">
            <div class="conversation-avatar">
                <img src="${member.profile_image || '/static/img/default-avatar.png'}"
                     alt="${escapeHtml(member.username)}"
                     onerror="this.onerror=null; this.src='/static/img/default-avatar.png';">
            </div>
            <div>
                <strong>${escapeHtml(member.username)}</strong>
                ${member.bio ? `<p style="margin: 5px 0; font-size: 13px; color: #666;">${escapeHtml(member.bio)}</p>` : ''}
            </div>
        </div>
    `).join('');
}

// Start New Conversation
function startNewConversation(userId, username, profileImage = '/static/img/default-avatar.png') {
    const newMessageModal = document.getElementById('newMessageModal');
    const messagesModal = document.getElementById('messagesModal');

    newMessageModal.style.display = 'none';
    messagesModal.style.display = 'flex';

    loadConversations();
    openConversation(userId, username, profileImage);
}
