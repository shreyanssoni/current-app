import os
import logging
from typing import Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from sentence_transformers import SentenceTransformer

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLMService")

logger.info("Loading local embedding model: all-MiniLM-L6-v2")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class LLMService:
    def __init__(self):
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.ollama_url = os.getenv("OLLAMA_BASE_URL")
        self.default_model = os.getenv("DEFAULT_CHAT_MODEL")
        logger.info("LLMService initialized")

    async def generate(self, prompt: str, provider: str = "groq", model: Optional[str] = None, prompt_type: Optional[str] = None):
        """
        Generic generate function that routes to different providers and applies specific prompts
        """
        logger.info(f"Incoming request - Provider: {provider}, Type: {prompt_type or 'DEFAULT'}")

        # Define System Prompts
        system_prompts = {
            "negotiate_resistance": (
                "You are an expert productivity coach using the 'Tiny Habits' method. "
                "Input: A user's task and a high resistance level. "
                "Output: Return ONLY a JSON object with 'deal_title' and a 'steps' array containing exactly 3 steps. "
                "Constraint: Each step must be executable in under 2 minutes. "
                "Constraint: Tone must be empathetic but extremely concise (max 10 words per step). "
                "Constraint: Do not use Markdown formatting."
            ),
            "auto_tag_capture": (
                "You are an ontology expert. Analyze the incoming thought.\n"
                "Rules:\n"
                "1. RETURN ONLY JSON. No markdown.\n"
                "2. 'type' MUST be exactly one of: ['TASK', 'PRINCIPLE', 'JOURNAL', 'IDEA', 'FEELING'].\n"
                "   - TASK: Actionable items for the near future.\n"
                "   - PRINCIPLE: Wisdom, rules for life, or facts to remember.\n"
                "   - JOURNAL: Personal reflection, diary entry, or rant.\n"
                "   - IDEA: A creative spark or project concept.\n"
                "   - FEELING: Pure emotion or mood statement.\n"
                "3. 'tags': Extract 1-3 crisp context tags (e.g., ['Work', 'Anxiety']).\n"
                "4. 'sentiment': 'POSITIVE', 'NEUTRAL', or 'NEGATIVE'.\n"
                "Structure: {{ 'tags': [], 'type': '', 'sentiment': '', 'confidence': 0.9 }}"
            ),
            "resurface_insight": (
                "You are a contextual reminder engine. Rewrite the following insight into a single, punchy 1-sentence reminder. "
                "Output: JSON object with field 'reminder'."
            ),
            "decompose_goal": (
                "You are an expert at first-principles thinking and atomic habits. "
                "Task: Break the input goal into 3-5 extremely small, immediate, actionable steps. "
                "Rules:\n"
                "1. Each step MUST start with a strong verb.\n"
                "2. Each step MUST be discrete and achievable today.\n"
                "3. RETURN ONLY a JSON object with a 'tasks' field containing a list of strings.\n"
                "Structure: {{ \"tasks\": [\"Step 1\", \"Step 2\"] }}"
            ),
            "select_mission": (
                "You are a master strategist. Analysis the user's list of tasks (Dock).\n"
                "Task: Select 3-5 tasks that together form a cohesive, high-impact day.\n"
                "Rules:\n"
                "1. Focus on urgency, importance, and logical flow.\n"
                "2. RETURN ONLY a JSON object with a 'selected_ids' field containing the list of task IDs.\n"
                "3. IMPORTANT: Use the EXACT IDs provided in the input. Do not hallucinate or shorten them.\n"
                "Input format will be: [{{ \"id\": \"...\", \"content\": \"...\" }}, ...]\n"
                "Structure: {{ \"selected_ids\": [\"uuid1\", \"uuid2\"] }}"
            ),
            "generate_schedule": (
                "You are an expert executive assistant. Task: Create a sequential, time-blocked schedule.\n"
                "Input: A list of tasks with estimated durations (minutes), and optionally a list of 'existing_tasks' that are already scheduled.\n"
                "Rules:\n"
                "1. Start at the provided 'user_start_time'.\n"
                "2. IMPORTANT: Do not schedule new tasks over 'existing_tasks'. These are fixed slots.\n"
                "3. Schedule new tasks into the available gaps between 'existing_tasks' or after them.\n"
                "4. Estimate realistic durations (30m to 90m) if not provided.\n"
                "5. Add 10-minute buffer breaks between tasks if possible.\n"
                "6. Format: RETURN ONLY a JSON object with a 'schedule' field containing a list of {{id, start_time, end_time}}.\n"
                "7. Times MUST be in HH:mm format (24h).\n"
                "Structure: {{ \"schedule\": [{{ \"id\": \"...\", \"start_time\": \"09:00\", \"end_time\": \"09:45\" }}] }}"
            ),
            "spark_idea": (
                "You are a productivity expert specializing in actionable task breakdown. "
                "Task: Analyze the following raw thought or idea and break it down into exactly 3 concrete, actionable steps.\n"
                "Rules:\n"
                "1. Each step MUST be a specific action (start with a verb).\n"
                "2. Each step MUST be 6 words or fewer.\n"
                "3. Steps should be immediately executable (no vague concepts).\n"
                "4. RETURN ONLY a JSON object with a 'tasks' field containing an array of strings.\n"
                "Structure: {{ \"tasks\": [\"Action 1\", \"Action 2\", \"Action 3\"] }}"
            )
        }

        system_msg = system_prompts.get(prompt_type, "You are a helpful AI assistant. Output should be concise.")
        
        if provider == "mock":
            if prompt_type == "negotiate_resistance":
                return {
                    "deal_title": "The 5-Minute Compromise",
                    "steps": ["Open the file.", "Write one word.", "Close it and celebrate."]
                }
            return {"response": f"MOCK RESPONSE for {prompt_type or 'default'}"}

        llm = None
        if provider == "groq":
            if not self.groq_key: raise ValueError("GROQ_API_KEY not set")
            llm = ChatGroq(temperature=0, model_name=model or self.default_model, groq_api_key=self.groq_key)
        elif provider == "openai":
            if not self.openai_key: raise ValueError("OPENAI_API_KEY not set")
            llm = ChatOpenAI(temperature=0, api_key=self.openai_key, model=model or "gpt-4-turbo")
        elif provider == "ollama":
            llm = ChatOllama(base_url=self.ollama_url, model=model or "llama3", temperature=0)

        if not llm:
            raise ValueError(f"Provider {provider} not supported")

        from langchain_core.prompts import ChatPromptTemplate
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", "{input}")
        ])

        try:
            logger.info("Invoking LLM...")
            # Use raw ainvoke and pass variables directly to avoid KeyError on input text containing { }
            raw_response = await llm.ainvoke(chat_prompt.invoke({"input": prompt}))
            content = raw_response.content

            # Cleaning Step: Strip Markdown code blocks
            import re
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)
            content = content.strip()

            import json
            try:
                parsed = json.loads(content)
                logger.info("Generation and parsing successful")
                return parsed
            except json.JSONDecodeError:
                logger.warning("Strict parsing failed, searching for JSON block...")
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    return json.loads(match.group())
                raise ValueError("Could not find valid JSON in response")

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            # Fallback for enrichment failures
            if prompt_type == "auto_tag_capture":
                logger.info("Returning fallback JSON for auto_tag_capture")
                return { 
                    "tags": ["Review"], 
                    "type": "JOURNAL", 
                    "sentiment": "NEUTRAL", 
                    "confidence": 0.5 
                }
            raise e

    async def generate_embedding(self, text: str):
        """
        Generate vector embedding using local sentence-transformers model
        """
        try:
            # Local generation is synchronous, so we run it in a thread if needed,
            # but for MiniLM it's usually fast enough to call directly.
            # Using .tolist() as requested by user.
            vector = embedding_model.encode(text).tolist()
            return vector
        except Exception as e:
            logger.error(f"Local embedding generation failed: {str(e)}")
            return None
