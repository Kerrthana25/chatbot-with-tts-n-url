

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'xlsx', 'pptx', 'txt', 'jpg', 'jpeg', 'png', 'gif', 'mp4', 'csv','url'}
    UPLOAD_FOLDER = 'static/uploads'
    KNOWLEDGE_BASE = 'static/knowledge_base'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size