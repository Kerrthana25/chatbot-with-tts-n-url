document.addEventListener('DOMContentLoaded', function() {
    // Existing variables
    const chatIcon = document.getElementById('chat-icon');
    const chatContainer = document.querySelector('.chat-container');
    const refreshBtn = document.getElementById('refresh-btn');
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const voiceButton = document.getElementById('voice-button');
    const fileInput = document.getElementById('file-input');
    const filePreviewContainer = document.getElementById('file-preview-container');
    const filePreviewImage = document.getElementById('file-preview-image');
    const cancelUploadButton = document.getElementById('cancel-upload');


    function speakText(text) {
    const synth = window.speechSynthesis;
    const utterance = new SpeechSynthesisUtterance(text);

    // Optional: Set a specific system voice like Zira if available
    const voices = synth.getVoices();
    utterance.voice = voices.find(v => v.name.includes("Zira") || v.name.includes("Hazel")) || null;

    synth.speak(utterance);
}



    // New variable for tracking uploaded file
    let currentUploadedFile = null;

    // Toggle chat visibility
    chatIcon.addEventListener('click', function() {
        chatContainer.classList.toggle('visible');
    });

    // Refresh chat
    refreshBtn.addEventListener('click', function() {
        this.classList.add('refreshing');
        setTimeout(() => {
            chatMessages.innerHTML = `
                <div class="message bot-message">
                    Hello! I'm your AI assistant. How can I help you today?
                </div>
            `;
            this.classList.remove('refreshing');
        }, 500);
    });

    // Send message on button click
    sendButton.addEventListener('click', sendMessage);

    // Send message on Enter key
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // File input handling
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            currentUploadedFile = e.target.files[0];
            displayFilePreview(currentUploadedFile);
        }
    });

    // Cancel upload button
    cancelUploadButton.addEventListener('click', cancelFileUpload);

    // Voice button functionality (existing)
    voiceButton.addEventListener('click', function() {
        // Your existing voice recognition code
        console.log("Voice button clicked");
    });

    // Function to display file preview
    function displayFilePreview(file) {
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = function(e) {
                filePreviewImage.src = e.target.result;
                filePreviewContainer.style.display = 'block';
            };
            reader.readAsDataURL(file);
        } else {
            // For non-image files, show a placeholder
            filePreviewImage.src = 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(
                `<svg xmlns="http://www.w3.org/2000/svg" width="200" height="150" viewBox="0 0 24 24" fill="#4285f4" stroke="#666" stroke-width="1" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                    <line x1="16" y1="13" x2="8" y2="13"></line>
                    <line x1="16" y1="17" x2="8" y2="17"></line>
                    <polyline points="10 9 9 9 8 9"></polyline>
                </svg>`
            );
            filePreviewContainer.style.display = 'block';
        }
    }

    // Function to cancel file upload
    function cancelFileUpload() {
        fileInput.value = '';
        filePreviewContainer.style.display = 'none';
        currentUploadedFile = null;
    }

    // Function to send message (updated to handle files)
    async function sendMessage() {
        const messageText = userInput.value.trim();

        if (!messageText && !currentUploadedFile) return;

        // Add user message to chat
        if (messageText) {
            addMessageToChat(messageText, 'user');
        }

        // Handle file upload if present
        if (currentUploadedFile) {
            displayFileInChat(currentUploadedFile);
            await processUploadedFile(currentUploadedFile, messageText);
            cancelFileUpload();
        } else if (messageText) {
            // Handle text-only message
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: messageText })
            });

            const data = await response.json();
            if (data.response) {
                addMessageToChat(data.response, 'bot');
            }
        }

        userInput.value = '';
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Function to display file in chat
    function displayFileInChat(file) {
    // Only display the file once - remove any existing preview
    const existingPreviews = document.querySelectorAll('.file-preview-container');
    existingPreviews.forEach(preview => preview.remove());

    const fileMessage = document.createElement('div');
    fileMessage.className = 'message user-message';

    if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const img = document.createElement('img');
            img.src = e.target.result;
            img.className = 'chat-image';
            fileMessage.appendChild(img);

            const fileInfo = document.createElement('div');
            fileInfo.className = 'file-message';
            fileInfo.textContent = `Uploaded: ${file.name}`;
            fileMessage.appendChild(fileInfo);

            chatMessages.appendChild(fileMessage);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        };
        reader.readAsDataURL(file);
    } else {
        fileMessage.textContent = `Uploaded file: ${file.name}`;
        chatMessages.appendChild(fileMessage);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

function scrollToBottom() {
  const chatBox = document.getElementById("chatBox");
  chatBox.scrollTop = chatBox.scrollHeight;
}


    // Function to process uploaded file
    // Modify your processUploadedFile function like this:
async function processUploadedFile(file, userPrompt) {
    const formData = new FormData();
    formData.append('file', file);
    if (userPrompt) {
        formData.append('message', userPrompt);
    }

    try {
        showLoadingIndicator();
        const response = await fetch('/api/upload-and-chat', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
        }

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Remove the previous bot message if it exists
        const existingMessages = document.querySelectorAll('.bot-message');
        if (existingMessages.length > 0) {
            existingMessages[existingMessages.length - 1].remove();
        }

        if (data.response) {
            addMessageToChat(data.response, 'bot');
        }
    } catch (error) {
        console.error('Error processing file:', error);
        addMessageToChat("Sorry, I couldn't process that file. Please try again.", 'bot');
    } finally {
        hideLoadingIndicator();
    }
}

    // Helper function to add messages to chat
    // Helper function to add messages to chat (with avatars)
function addMessageToChat(text, sender) {
    const messageRow = document.createElement('div');
    messageRow.className = `message-row ${sender === 'user' ? 'user-row' : 'bot-row'}`;

    const avatar = document.createElement('div');
    avatar.className = `avatar ${sender === 'user' ? 'user-avatar' : 'bot-avatar'}`;
    avatar.innerHTML = sender === 'user' ? '👤' : '🤖';

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    messageDiv.textContent = text;

    if (sender === 'user') {
    messageRow.appendChild(messageDiv);
    messageRow.appendChild(avatar);
} else {
    messageRow.appendChild(avatar);
    messageRow.appendChild(messageDiv);

    // 🗣️ Only speak if toggle is enabled
    const speakToggle = document.getElementById('speak-toggle');
    if (speakToggle && speakToggle.checked) {
        speakText(text);
    }
}


    chatMessages.appendChild(messageRow);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}


});
