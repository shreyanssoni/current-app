import os
import logging
from typing import Optional
from dotenv import load_dotenv
import requests
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(token=os.environ.get("HUGGINGFACE_API_KEY"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLMService")

class LLMService:
    def __init__(self):
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.ollama_url = os.getenv("OLLAMA_BASE_URL")
        self.default_model = os.getenv("DEFAULT_CHAT_MODEL")
        
        # HuggingFace Config
        self.hf_token = os.environ.get("HUGGINGFACE_API_KEY")
        self.hf_api_url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"
        
        # Initialize OpenAI Client (for other tasks)
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
            ),
            "mindspace_vision": (
                "Describe this image in short. Identify the mood, objects, aesthetic, and any visible text. "
                "Be concise but descriptive. Provide a poetic but accurate description for a subconscious thought journal. "
                "Max 2 sentences. Return ONLY the description text. Be concise and answer in short."
            ),
            "mindspace_clustering": (
                "You are a strict Librarian. Your job is to categorize concepts based on a provided list of existing sections. "
                "Rules:\n"
                "1. heavily PREFER assigning to an 'existing_sections' from the list provided if the concept fits.\n"
                "2. Only suggest a NEW section if the input is completely unrelated to everything in the list.\n"
                "3. Use broad categories (e.g., 'Nature' instead of 'Ocean', 'Self-Development' instead of 'Morning Routine').\n"
                "4. Return ONLY a JSON object with a 'clusters' array of strings.\n"
                "Structure: {{ \"clusters\": [\"Theme1\", \"Theme2\"] }}"
            ),
            "mindspace_sentiment": (
                "Analyze the emotional tone of the input. Return a single-word emotional tone. "
                "Example: 'Nostalgic', 'Anxious', 'Determined', 'Calm'. "
                "Return ONLY a JSON object with 'tone'. "
                "Structure: {{ \"tone\": \"...\" }}"
            ),
            "mindspace_patterns": (
                "You are an expert psychological profiler and life coach. "
                "Analyze the provided thoughts and image descriptions to identify deep patterns.\n\n"
                "Return in response: Output a clear, structured insight report with these specific sections:\n"
                "1. **THOUGHT PATTERNS**: Recurring mental themes or cognitive habits.\n"
                "2. **INTERESTS**: What is currently exciting or capturing your attention.\n"
                "3. **DISLIKES**: Sources of friction, stress, or negativity identified.\n"
                "4. **AI SUGGESTED ACTIONS**: 2 practical, high-impact things you can do next based on this mindset.\n\n"
                "Format: Professional, insightful, and concise. Respond in one line only for each section. Return in beautiful markdown.. "
            ),
            "mindspace_chat": (
                "You are a Second Brain. You help users understand their own thoughts and experiences. "
                "Use the provided context (notes and image descriptions) to answer the user's query. "
                "Be helpful, insightful, and always reference dates if they are in the context. "
                "If the answer isn't in the context, say you don't recall that specific detail from their Mindspace. "
                "Maintain a clean, encouraging tone."
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
        if not llm:
            raise ValueError(f"Provider {provider} not supported")

        from langchain_core.prompts import ChatPromptTemplate
        
        # Support for Vision
        if prompt_type == "mindspace_vision" and provider == "groq":
            vision_model = "meta-llama/llama-4-scout-17b-16e-instruct"
            chat = ChatGroq(temperature=0, model_name=vision_model, groq_api_key=self.groq_key)
            
            # For vision, prompt is actually the image URL or base64
            # Assuming 'prompt' is the image URL here
            message = HumanMessage(
                content=[
                    {"type": "text", "text": system_msg},
                    {"type": "image_url", "image_url": {"url": prompt}},
                ]
            )
            try:
                res = await chat.ainvoke([message])
                return res.content
            except Exception as e:
                logger.error(f"Vision failed: {str(e)}")
                return "An image that words couldn't capture."

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
            # Prompt types that MUST return JSON
            JSON_REQUIRED_TYPES = {
                "negotiate_resistance", "auto_tag_capture", "resurface_insight", 
                "decompose_goal", "select_mission", "generate_schedule", 
                "spark_idea", "mindspace_clustering", "mindspace_sentiment"
            }

            try:
                parsed = json.loads(content)
                logger.info("Generation and parsing successful (Strict JSON)")
                return parsed
            except json.JSONDecodeError:
                # Try to find a JSON block with regex
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group())
                        logger.info("Generation and parsing successful (Regex JSON)")
                        return parsed
                    except json.JSONDecodeError:
                        pass
                
                # If JSON is not strictly required, return the raw text
                if prompt_type not in JSON_REQUIRED_TYPES:
                    logger.info(f"JSON parsing failed but not required for {prompt_type}. Returning raw content.")
                    return content
                
                raise ValueError(f"Could not find valid JSON in response for required type: {prompt_type}")

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
        Generate vector embedding using HuggingFace API (all-MiniLM-L6-v2)
        """
        if not self.hf_token:
            logger.error("HUGGINGFACE_API_KEY not set")
            return None

        headers = {"Authorization": f"Bearer {self.hf_token}"}
        
        try:
            # Running synchronous requests in async flow (can be improved with aiohttp but sticking to requests as per user snippet)
            # For production, consider run_in_executor
            import asyncio
            from functools import partial
            
            loop = asyncio.get_event_loop()
            # response = await loop.run_in_executor(
            #     None, 
            #     partial(
            #         requests.post, 
            #         self.hf_api_url, 
            #         headers=headers, 
            #         json={"inputs": {"source_sentence": text, "sentences" : [text]}, "options":{"wait_for_model":True}}
            #     )
            # )

            response = client.feature_extraction(
                text,
                model="sentence-transformers/all-MiniLM-L6-v2",
            )   

            import numpy as np
            vector = np.array(response).flatten().tolist()
            return vector

            if response.status_code != 200:
                logger.error(f"HF Embedding Error: {response.text}")
                return None
                
            return response.tolist()

        except Exception as e:
            logger.error(f"HF embedding generation failed: {str(e)}")
            return None
