import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from flask import Flask, render_template, request, jsonify, send_from_directory
from config import Config
import google.generativeai as genai
import os
import time
import logging
from werkzeug.utils import secure_filename
from document_processor import DocumentProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config.from_object(Config)

# Now configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'png', 'jpg', 'jpeg', 'docx', 'txt', 'pptx'}

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Gemini
try:
    genai.configure(api_key=app.config['GEMINI_API_KEY'])
    model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Successfully initialized Gemini 1.5 Flash model")
except Exception as e:
    logger.error(f"Gemini initialization failed: {str(e)}")
    try:
        model = genai.GenerativeModel('gemini-pro')
        logger.info("Fell back to Gemini Pro model")
    except Exception as fallback_error:
        logger.error(f"Gemini fallback failed: {str(fallback_error)}")
        model = None

# Rate limiting
last_request_time = 0
REQUEST_INTERVAL = 1.5  # seconds

knowledge_base = {}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def is_valid_url(url):
    """Check if a URL is valid"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


def fetch_and_parse_url(url):
    """Fetch, parse, and clean webpage content using BeautifulSoup"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove noise: nav, footer, header, script, style, aside, iframe
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
            element.decompose()

        # Get clean text
        text = soup.get_text(separator='\n', strip=True)

        # Remove extra blank lines and duplicate lines
        cleaned_lines = []
        seen = set()
        for line in text.split('\n'):
            line = line.strip()
            if line and line not in seen:
                cleaned_lines.append(line)
                seen.add(line)

        cleaned_text = "\n".join(cleaned_lines)
        return cleaned_text

    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return None


def initialize_knowledge_base():
    """Initialize knowledge base with enhanced web scraping"""
    global knowledge_base
    knowledge_base.clear()
    kb_dir = app.config.get('KNOWLEDGE_BASE', 'knowledge_base')

    try:
        if not os.path.exists(kb_dir):
            os.makedirs(kb_dir)
            logger.info(f"Created knowledge base directory at {kb_dir}")

        # Process regular files
        for filename in os.listdir(kb_dir):
            file_path = os.path.join(kb_dir, filename)
            if os.path.isfile(file_path) and allowed_file(filename) and filename != 'urls.txt':
                try:
                    content = DocumentProcessor.process_file(file_path)
                    knowledge_base[filename] = {
                        'path': file_path,
                        'content': content[:50000] # Limit size
                    }
                    logger.info(f"Loaded {filename} into knowledge base")
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")

        # Process URLs from urls.txt
        # Process URLs from urls.txt
        urls_file = os.path.join(kb_dir, 'urls.txt')
        if os.path.exists(urls_file):
            with open(urls_file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]

            for url in urls:
                if not is_valid_url(url):
                    logger.error(f"Invalid URL format: {url}")
                    continue

                content = DocumentProcessor.crawl_website(url, depth=2)  # depth=2 or 3 as desired
                if content:
                    knowledge_base[url] = {
                        'path': url,
                        'content': content[:50000],  # Limit content size if needed
                        'type': 'url'
                    }
                    logger.info(f"Crawled and loaded content from {url}")


    except Exception as e:
        logger.error(f"Knowledge base initialization failed: {str(e)}")


initialize_knowledge_base()


def find_most_relevant_files(query, knowledge_base, top_n=10, chunk_size=1000):
    """Search using semantic similarity with chunking for better relevance."""
    if not knowledge_base:
        return []

    try:
        documents = []
        doc_chunks = []
        doc_refs = []

        # Split each document content into chunks
        for name, data in knowledge_base.items():
            content = data['content']
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i+chunk_size]
                doc_chunks.append(chunk)
                doc_refs.append(name)

        vectorizer = TfidfVectorizer(stop_words='english')
        doc_vectors = vectorizer.fit_transform(doc_chunks)
        query_vector = vectorizer.transform([query])

        similarities = cosine_similarity(query_vector, doc_vectors).flatten()
        most_similar_indices = similarities.argsort()[-top_n:][::-1]

        # Return the most relevant chunk and associated filename
        results = [(doc_refs[i], doc_chunks[i]) for i in most_similar_indices]
        return results

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return []


def generate_response(user_input):
    """Generate responses with web-scraped content integration
    Generate direct answers using knowledge base or general knowledge"""
    if not model:
        return "System error: AI model not available. Please contact support."

    try:
        user_input = user_input.strip()
        if not user_input:
            return "Please enter a valid question."

        # Special responses
        greetings = ["hello", "hi", "hey"]
        if user_input.lower() in greetings:
            return "Hello! How can I assist you today?"

        # Get relevant context
        relevant_files = find_most_relevant_files(user_input, knowledge_base)

        # Build optimized prompt with strict directives
        prompt = f"""
        Answer the following question concisely and directly as if you inherently know the information.
        Never mention:
        - Your knowledge sources
        - Whether information came from files/URLs
        - Phrases like "based on", "according to", or "the document states"

        Follow these rules:
        1. If the context below clearly answers the question, use ONLY that information
        2. If not, use your general knowledge
        3. Always deliver the answer as if it's your own knowledge

        Question: {user_input}
        """

        if relevant_files:
            context = "\n".join([content for _, content in relevant_files])
            prompt = f"""
            {prompt}
            Available Context:
            {context}
            """
        else:
            prompt = f"""
            {prompt}
            (Use only your general knowledge)
            """

        logger.info(f"Sending prompt to Gemini: {prompt[:200]}...")

        # Get response from Gemini
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 500
            }
        )

        if not response.text:
            raise ValueError("Empty response from Gemini")

        # Clean response (additional safeguard)
        text = response.text.strip()
        forbidden_phrases = [
            "according to", "based on", "the document states",
            "in the context", "from the file", "as mentioned in"
        ]
        for phrase in forbidden_phrases:
            text = text.replace(phrase, "")

        return text if text else "Please rephrase your question."

    except Exception as e:
        logger.error(f"Response generation failed: {str(e)}", exc_info=True)
        return "I'm having trouble answering right now. Please try again later."


@app.route('/')
def home():
    """Serve the main chat interface"""
    return render_template('index.html')


@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory(app.static_folder, filename)


@app.route('/api/chat', methods=['POST'])
def chat():
    global last_request_time

    try:
        # Rate limiting
        elapsed = time.time() - last_request_time
        if elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)
        last_request_time = time.time()

        user_input = request.json.get('message', '').strip()
        if not user_input:
            return jsonify({'response': "Please enter a valid question"})

        response = generate_response(user_input)
        return jsonify({'response': response})

    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}", exc_info=True)
        return jsonify({'response': "System error. Please try again later."}), 500


@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400

        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Ensure upload directory exists
            os.makedirs(os.path.dirname(upload_path), exist_ok=True)
            file.save(upload_path)

            try:
                content = DocumentProcessor.process_file(upload_path)
                knowledge_base[filename] = {
                    'path': upload_path,
                    'content': content[:10000]
                }
                return jsonify({'success': True, 'filename': filename})
            except Exception as e:
                logger.error(f"File processing error: {str(e)}")
                return jsonify({'error': 'Failed to process file'}), 500

        return jsonify({'error': 'File type not allowed'}), 400

    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        return jsonify({'error': 'System error during upload'}), 500


@app.route('/api/upload-and-chat', methods=['POST'])
def upload_and_chat():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400

        # Secure filename and create temp path
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            # Save file temporarily
            file.save(temp_path)

            # Process file content
            extracted_text = DocumentProcessor.process_file(temp_path)

            # Check if text extraction failed
            # Even if text is minimal or empty, return it when explicitly asked
            user_input = request.form.get('message', '').strip().lower()
            if user_input in ['give the text from the image', 'extract text']:
                return jsonify({'response': f"Extracted text:\n{extracted_text}"})

            # Handle failed OCR
            if not extracted_text or "Could not" in extracted_text or "Error" in extracted_text:
                return jsonify({'response': "I couldn't extract readable text from this image. Try a clearer image."})

            # Generate response
            user_input = request.form.get('message', '').strip()
            if user_input.lower() in ['give the text from the image', 'extract text']:
                # Return raw extracted text if user specifically asks for it
                return jsonify({'response': f"Extracted text:\n{extracted_text}"})
            else:
                # Otherwise generate a summarized response
                prompt = f"Extracted text from file:\n{extracted_text[:5000]}\n\nUser query: {user_input if user_input else 'Please summarize the key information from this file'}"
                response = generate_response(prompt)
                return jsonify({'response': response})

        except Exception as e:
            logger.error(f"File processing error: {str(e)}")
            return jsonify({'error': 'Failed to process file'}), 500

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.error(f"Failed to remove temp file: {str(e)}")

    except Exception as e:
        logger.error(f"Upload and chat error: {str(e)}", exc_info=True)
        return jsonify({'error': 'System error during processing'}), 500





@app.route('/api/crawl', methods=['POST'])
def crawl_url():
    url = request.json.get('url', '').strip()
    if not is_valid_url(url):
        return jsonify({'error': 'Invalid URL format'}), 400

    try:
        content = DocumentProcessor.crawl_website(url, depth=2)  # depth can be adjusted
        if content:
            knowledge_base[url] = {
                'path': url,
                'content': content[:10000],  # limit size
                'type': 'url'
            }
            return jsonify({'success': True, 'message': f'Crawled and added {url} to knowledge base'})
        else:
            return jsonify({'error': 'Failed to extract content'}), 500
    except Exception as e:
        logger.error(f"Crawling error: {e}")
        return jsonify({'error': 'Internal server error'}), 500



@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404


def create_default_files():
    """Create required directories and default files"""
    # Create required directories
    required_dirs = [
        app.config.get('UPLOAD_FOLDER', 'uploads'),
        app.config.get('KNOWLEDGE_BASE', 'knowledge_base'),
        'static',
        'templates'
    ]

    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)

    # Create default index.html if missing
    index_path = os.path.join('templates', 'index.html')
    if not os.path.exists(index_path):
        with open(index_path, 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Knowledge Base Chatbot</title>
    <link rel="stylesheet" href="/static/css/styles.css">
</head>
<body>
    <div id="chat-container">
        <h1>Knowledge Base Chatbot</h1>
        <div id="chat-messages"></div>
        <div id="user-input">
            <input type="text" id="message" placeholder="Type your message...">
            <button id="send">Send</button>
        </div>
    </div>
    <script src="/static/js/script.js"></script>
</body>
</html>""")


if __name__ == '__main__':
    create_default_files()
    app.run(debug=True, host='0.0.0.0', port=5000)
