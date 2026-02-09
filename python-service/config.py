import os
from dotenv import load_dotenv

load_dotenv()

# Test User ID for development
TEST_USER_ID = "3194ab4e-3a4c-4e82-bde0-2f6c4152790d"

# Supabase Configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# AI Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")
DEFAULT_CHAT_MODEL = os.environ.get("DEFAULT_CHAT_MODEL", "llama-3.3-70b-versatile")

# Server Configuration
PORT = int(os.environ.get("PORT", 3001))
