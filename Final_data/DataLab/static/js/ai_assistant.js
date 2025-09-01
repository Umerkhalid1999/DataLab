// ai_assistant.js - Client-side JavaScript for AI assistant functionality
// Place this file in your static/js directory

document.addEventListener('DOMContentLoaded', function() {
    initializeAIAssistant();
});

function initializeAIAssistant() {
    // Get DOM elements
    const assistantButton = document.getElementById('aiAssistantButton');
    const assistantPanel = document.getElementById('aiAssistantPanel');
    const closeButton = document.getElementById('closeAssistant');
    const messageInput = document.getElementById('assistantMessage');
    const sendButton = document.getElementById('sendAssistantMessage');
    const chatHistory = document.getElementById('assistantChatHistory');
    const loadingIndicator = document.getElementById('assistantLoading');

    // If elements don't exist, return early
    if (!assistantButton || !assistantPanel) return;

    // Chat history storage
    let messages = [];

    // Toggle assistant panel
    assistantButton.addEventListener('click', function() {
        assistantPanel.classList.toggle('active');
        if (assistantPanel.classList.contains('active')) {
            messageInput.focus();
        }
    });

    // Close assistant panel
    if (closeButton) {
        closeButton.addEventListener('click', function() {
            assistantPanel.classList.remove('active');
        });
    }

    // Handle sending message
    if (sendButton && messageInput) {
        sendButton.addEventListener('click', function() {
            sendMessage();
        });

        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    // Send message function
    function sendMessage() {
        const message = messageInput.value.trim();
        if (!message) return;

        // Add user message to chat
        addMessageToChat('user', message);

        // Clear input
        messageInput.value = '';

        // Show loading indicator
        if (loadingIndicator) {
            loadingIndicator.classList.remove('d-none');
        }

        // Get current dataset ID if any
        const currentDatasetId = getCurrentDatasetId();

        // Call API
        fetch('/api/ai_assistant/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                history: messages.slice(-5), // Send last 5 messages for context
                dataset_id: currentDatasetId
            }),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Hide loading indicator
            if (loadingIndicator) {
                loadingIndicator.classList.add('d-none');
            }

            if (data.success) {
                // Add AI response to chat
                addMessageToChat('assistant', data.message);
            } else {
                // Show error
                addMessageToChat('system', 'Error: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(error => {
            console.error('Error:', error);

            // Hide loading indicator
            if (loadingIndicator) {
                loadingIndicator.classList.add('d-none');
            }

            // Show error message
            addMessageToChat('system', 'Sorry, there was an error processing your request.');
        });
    }

    // Add message to chat history
    function addMessageToChat(sender, message) {
        // Create message element
        const messageEl = document.createElement('div');
        messageEl.className = `chat-message ${sender}-message`;

        // Create message content
        const contentEl = document.createElement('div');
        contentEl.className = 'message-content';

        // Format message with markdown if it's from the assistant
        if (sender === 'assistant') {
            // Simple markdown-like formatting
            let formattedMessage = message
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
                .replace(/\*(.*?)\*/g, '<em>$1</em>') // Italic
                .replace(/`(.*?)`/g, '<code>$1</code>') // Inline code
                .replace(/\n/g, '<br>'); // New lines

            // Handle code blocks
            const codeBlockRegex = /```(.*?)\n([\s\S]*?)```/g;
            formattedMessage = formattedMessage.replace(codeBlockRegex, function(match, language, code) {
                return `<pre class="code-block"><code class="language-${language}">${code}</code></pre>`;
            });

            contentEl.innerHTML = formattedMessage;
        } else {
            // For user messages, just replace new lines
            contentEl.textContent = message;
            contentEl.innerHTML = contentEl.innerHTML.replace(/\n/g, '<br>');
        }

        // Add sender avatar/icon
        const iconEl = document.createElement('div');
        iconEl.className = 'message-icon';

        if (sender === 'user') {
            iconEl.innerHTML = '<i class="fas fa-user"></i>';
        } else if (sender === 'assistant') {
            iconEl.innerHTML = '<i class="fas fa-robot"></i>';
        } else {
            iconEl.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
        }

        // Add timestamp
        const timeEl = document.createElement('div');
        timeEl.className = 'message-time';
        timeEl.textContent = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});

        // Build message
        messageEl.appendChild(iconEl);
        messageEl.appendChild(contentEl);
        messageEl.appendChild(timeEl);

        // Add to chat history
        if (chatHistory) {
            chatHistory.appendChild(messageEl);

            // Scroll to bottom
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }

        // Store message
        messages.push({
            sender: sender,
            message: message,
            timestamp: new Date().toISOString()
        });
    }

    // Initialize with a welcome message
    if (chatHistory && chatHistory.children.length === 0) {
        addMessageToChat('assistant', 'Hello! I\'m your AI assistant for data analysis and preprocessing. How can I help you today?');
    }

    // Function to get current dataset ID
    function getCurrentDatasetId() {
        // Try to get from dataset cards
        const activeDataset = document.querySelector('.dataset-card.active');
        if (activeDataset) {
            return activeDataset.getAttribute('data-id');
        }

        // Try to get from URL if on a dataset page
        const urlParams = new URLSearchParams(window.location.search);
        const datasetId = urlParams.get('dataset_id');
        if (datasetId) {
            return datasetId;
        }

        return null;
    }

    // Additional functionality for dataset-specific actions
    const analyzeButtons = document.querySelectorAll('.analyze-with-ai');
    if (analyzeButtons.length) {
        analyzeButtons.forEach(button => {
            button.addEventListener('click', function(e) {
                e.preventDefault();
                const datasetId = this.getAttribute('data-id');
                const datasetName = this.getAttribute('data-name');

                // Open assistant panel
                assistantPanel.classList.add('active');

                // Add a starter message
                messageInput.value = `Analyze the dataset "${datasetName}" and suggest preprocessing steps.`;

                // Focus input
                messageInput.focus();
            });
        });
    }
}

// Function to add a dataset question to the AI assistant
function askAIAboutDataset(datasetId, datasetName, question) {
    const assistantPanel = document.getElementById('aiAssistantPanel');
    const messageInput = document.getElementById('assistantMessage');

    if (!assistantPanel || !messageInput) return;

    // Open assistant panel
    assistantPanel.classList.add('active');

    // Set the question
    messageInput.value = question.replace('{dataset}', datasetName);

    // Focus input
    messageInput.focus();
}