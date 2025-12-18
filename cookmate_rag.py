"""
CookMate: Complete Voice-Driven Recipe Assistant with RAG
A context-aware cooking assistant using ASR, RAG, LLM, and TTS
All components are FREE and open-source
"""

import os
import sys
from typing import List, Dict, Optional, Tuple
import json
from dataclasses import dataclass, field
from datetime import datetime
import time
import re

# Silence warnings FIRST
import warnings
warnings.filterwarnings('ignore')

# Suppress pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

# Suppress ChromaDB telemetry warnings
import logging
logging.getLogger('chromadb').setLevel(logging.ERROR)
logging.getLogger('chromadb.telemetry').setLevel(logging.CRITICAL)

# Core libraries
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch

# Whisper for ASR
import whisper

# TTS
from gtts import gTTS
import pygame

# For audio recording
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

# Ollama for local LLM (best free option)
import requests
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight & fast
def embed_text(text: str):
    """Return vector embedding for a given text"""
    return embedding_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

FILLER_WORDS = {
    "ok", "okay", "okk", "k", "kk",
    "yeah", "yea", "ya", "yup", "yep",
    "hmm", "hmmm", "hm",
    "uh", "uhh", "uhhh",
    "huh", "huhu",
    "right", "alright", "aight",
    "sure", "fine",
    "cool", "nice", "great", "good",
    "hehe", "haha",
    "lol", "lmao",
    "hmm ok", "ok then",
    "got it", "gotcha",
    "makes sense", "i see",
    "hmm right",
    "hmm okay"
}
def is_filler_message(message: str) -> bool:
        message_clean = message.strip().lower()
        if len(message_clean) <= 2:
            return True
        if message_clean in FILLER_WORDS:
            return True
        for f in FILLER_WORDS:
            if message_clean.startswith(f) or message_clean.endswith(f):
                return True
            return False
@dataclass
class RecipeStep:
    step_number: int
    instruction: str
    duration: Optional[str] = None
    ingredients: Optional[List[str]] = None
    tips: Optional[str] = None


@dataclass
class ConversationState:
    current_recipe: Optional[str] = None
    current_recipe_name: Optional[str] = None
    current_step: int = 0
    total_steps: int = 0
    timer_start: Optional[float] = None
    conversation_history: List[Dict] = field(default_factory=list)
    last_response: str = ""

import requests

class GroqLLM:
    """Groq LLM for super-fast inference (completely free)"""
    
    def __init__(self, model: str = "llama-3.1-8b-instant", api_key: str = None):
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1"
        self.api_key = api_key or self._get_api_key()
        self.available = self._check_connection()
    
    def _get_api_key(self):
        
        """Get Groq API key from environment or user input"""
        import os
        api_key = os.getenv("")
        if not api_key:
            print("🔑 Please set your Groq API key:")
            print("   1. Get free API key from: https://console.groq.com/keys")
            print("   2. Set environment variable: GROQ_API_KEY=your_key_here")
            print("   3. Or enter it when prompted")
            api_key = input("Enter your Groq API key: ").strip()
        return api_key
    
    def _check_connection(self):
        """Check if Groq API is accessible"""
        if not self.api_key:
            print("❌ No Groq API key provided")
            return False
        
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10
            )
            if response.status_code == 200:
                print(f"✓ Connected to Groq ({self.model})")
                return True
            else:
                print(f"⚠ Groq API error: {response.status_code}")
                return False
        except Exception as e:
            print(f"⚠ Groq not available: {e}")
            print("   Using fallback mode")
            return False
    
    def generate(self, prompt: str, context: str = "") -> str:
        """Generate response using Groq API"""
        if not self.available:
            return self._fallback_response(prompt, context)
        
        try:
            # Prepare the message
            system_message = "You are CookMate, a helpful cooking assistant. Provide concise, practical cooking advice (2-3 sentences max)."
            
            if context:
                system_message += f"\n\nRecipe Context: {context}"
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 150,
                    "top_p": 0.9
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"].strip()
                if result:
                    return result
            else:
                print(f"⚠ Groq API error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            print("⚠ Groq response timed out, using fallback")
        except Exception as e:
            print(f"⚠ Groq error: {e}")
        
        return self._fallback_response(prompt, context)
    
    def _fallback_response(self, prompt: str, context: str) -> str:
        """Enhanced fallback when Groq unavailable"""
        prompt_lower = prompt.lower()
        
        # Handle common cooking questions
        if "substitute" in prompt_lower or "replace" in prompt_lower:
            if context:
                return f"Based on the recipe: {context[:200]}... Common substitutions: butter→oil, milk→cream, eggs→flax eggs."
            return "Common substitutions: butter→oil/margarine, eggs→flax eggs, milk→almond milk. What specific ingredient?"
        
        if "how long" in prompt_lower or "time" in prompt_lower:
            if context and any(word in context for word in ["minute", "hour", "second"]):
                return context[:250]
            return "Check the recipe timing. Most steps take 5-15 minutes. Which step are you asking about?"
        
        if any(word in prompt_lower for word in ["weather", "news", "sports", "politics"]):
            return "I'm CookMate, your cooking assistant! I focus on recipes and cooking techniques. What would you like to cook? 🍳"
        
        # Use context if available
        if context:
            sentences = context.split('.')[:2]
            return '. '.join(sentences) + '.'
        
        return "I'm here to help with cooking! Try asking about recipes, ingredients, or cooking techniques."
class OllamaLLM:
    """Local LLM using Ollama (completely free)"""
    
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.available = self._check_connection()
    
    def _check_connection(self):
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✓ Connected to Ollama")
                return True
        except Exception as e:
            print("⚠ Ollama not detected. Running in fallback mode.")
            print("   To enable LLM: Install Ollama from https://ollama.ai")
            print("   Then run: ollama pull llama3.2 && ollama serve")
            return False
        return False
    
    def generate(self, prompt: str, context: str = "") -> str:
        """Generate response from LLM"""
        if not self.available:
            return self._fallback_response(prompt, context)
        
        full_prompt = f"{context}\n\nUser query: {prompt}\n\nProvide a helpful, concise response (2-3 sentences max):"
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 150,  # Limit response length
                    }
                },
                timeout=60  # Increased timeout
            )
            
            if response.status_code == 200:
                result = response.json()["response"].strip()
                if result:
                    return result
        except requests.exceptions.Timeout:
            print("⚠ LLM response timed out, using fallback")
        except Exception as e:
            print(f"⚠ LLM error: {e}")
        
        return self._fallback_response(prompt, context)
    
    def _fallback_response(self, prompt: str, context: str) -> str:
        """Enhanced fallback when LLM unavailable"""
        prompt_lower = prompt.lower()
        
        # Handle common cooking questions without LLM
        if "substitute" in prompt_lower or "replace" in prompt_lower:
            if context:
                return f"Based on the recipe, here are some options: {context[:200]}... For specific substitutions, common alternatives usually work (e.g., butter→oil, milk→water)."
            return "Common substitutions: butter→oil/margarine, eggs→flax eggs, milk→almond milk. What specific ingredient are you asking about?"
        
        if "how long" in prompt_lower or "time" in prompt_lower:
            if context and any(word in context for word in ["minute", "hour", "second"]):
                return context[:300]
            return "Timing information depends on the specific step. Could you be more specific about which part of the recipe?"
        
        if any(word in prompt_lower for word in ["weather", "news", "sports", "politics"]):
            return "I'm CookMate, your cooking assistant! I focus on helping with recipes and cooking. What would you like to cook today?"
        
        # Use context if available
        if context:
            sentences = context.split('.')[:2]
            return '. '.join(sentences) + '.'
        
        return "I'm here to help with cooking! Try asking about recipes, ingredients, or cooking techniques."


class CookMateRAG:
    """Complete CookMate system with all features"""
    def __init__(
    self, 
    recipe_data_path: str = "recipes.json",
    use_whisper: bool = True,
    whisper_model: str = "base"
):
        print("🍳 Initializing CookMate...")
        self.state = ConversationState()

        print("📚 Loading recipes...")
        self.recipes = self._load_recipes(recipe_data_path)
    
        print("🧠 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        #step 1
        print("🔹 Preparing recipe embeddings...")
        self.recipe_chunks = []
        self._prepare_recipe_chunks()
        #step 2 
        print("💾 Setting up vector database...")
        self.chroma_client = chromadb.Client(Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=False  # Use in-memory for speed
        ))
        self.collection = self._setup_vector_db()
        print("🤖 Connecting to Groq LLM...")
        self.llm = GroqLLM(model="llama-3.1-8b-instant")  # ← Use the new model name
    
        self.whisper_model = None
        if use_whisper:
            print(f"🎤 Loading Whisper ({whisper_model})...")
            self.whisper_model = whisper.load_model(whisper_model)
        pygame.mixer.init()
        print("✅ CookMate ready!\n")

    def _prepare_recipe_chunks(self):
        for recipe_id, recipe in self.recipes.items():
            ingredients_text = f"Recipe: {recipe['name']}\nIngredients:\n" + "\n".join(recipe['ingredients'])
            self.recipe_chunks.append({
            "recipe_id": recipe_id,
            "text": ingredients_text,
            "vector": self.embedding_model.encode(ingredients_text, convert_to_numpy=True, normalize_embeddings=True)
        })

        # Steps as separate chunks
        for step in recipe['steps']:
            step_text = f"Recipe: {recipe['name']}\nStep {step['step']}: {step['instruction']}"
            if step.get("tips"):
                step_text += f" Tips: {step['tips']}"
            self.recipe_chunks.append({
                "recipe_id": recipe_id,
                "text": step_text,
                "vector": self.embedding_model.encode(step_text, convert_to_numpy=True, normalize_embeddings=True)
            })
    def _load_recipes(self, path: str) -> Dict:
        """Load recipes from JSON"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        
        # Create comprehensive sample recipes
        sample_recipes = {
            "pasta_carbonara": {
                "name": "Classic Pasta Carbonara",
                "prep_time": "10 minutes",
                "cook_time": "20 minutes",
                "servings": 4,
                "difficulty": "Medium",
                "ingredients": [
                    "400g spaghetti",
                    "200g pancetta or guanciale, diced",
                    "4 large eggs",
                    "100g Pecorino Romano cheese, grated",
                    "50g Parmesan cheese, grated",
                    "Freshly ground black pepper",
                    "Salt for pasta water"
                ],
                "steps": [
                    {
                        "step": 1,
                        "instruction": "Bring a large pot of salted water to a rolling boil. Add spaghetti and cook until al dente, about 8-10 minutes.",
                        "duration": "10 minutes",
                        "tips": "Save 1 cup of pasta water before draining!"
                    },
                    {
                        "step": 2,
                        "instruction": "While pasta cooks, heat a large pan over medium heat. Add diced pancetta and cook until golden and crispy.",
                        "duration": "5 minutes",
                        "tips": "No oil needed - pancetta has enough fat"
                    },
                    {
                        "step": 3,
                        "instruction": "In a bowl, whisk together eggs, Pecorino, Parmesan, and lots of black pepper until smooth and well combined.",
                        "duration": "2 minutes",
                        "tips": "Use room temperature eggs for best results"
                    },
                    {
                        "step": 4,
                        "instruction": "Drain pasta and immediately add to the pan with pancetta. Remove from heat.",
                        "duration": "1 minute",
                        "tips": "Work quickly while pasta is hot"
                    },
                    {
                        "step": 5,
                        "instruction": "Pour egg mixture over pasta and toss vigorously. Add pasta water gradually until creamy and silky.",
                        "duration": "2 minutes",
                        "tips": "Keep tossing to prevent scrambling. The residual heat cooks the eggs"
                    }
                ]
            },
            "chocolate_chip_cookies": {
                "name": "Perfect Chocolate Chip Cookies",
                "prep_time": "15 minutes",
                "cook_time": "12 minutes",
                "servings": 24,
                "difficulty": "Easy",
                "ingredients": [
                    "2 cups all-purpose flour",
                    "1 tsp baking soda",
                    "1 tsp salt",
                    "1 cup butter, softened",
                    "3/4 cup granulated sugar",
                    "3/4 cup brown sugar",
                    "2 large eggs",
                    "2 tsp vanilla extract",
                    "2 cups chocolate chips"
                ],
                "steps": [
                    {
                        "step": 1,
                        "instruction": "Preheat oven to 375°F (190°C). Line baking sheets with parchment paper.",
                        "duration": "2 minutes"
                    },
                    {
                        "step": 2,
                        "instruction": "Mix flour, baking soda, and salt in a bowl. Set aside.",
                        "duration": "2 minutes"
                    },
                    {
                        "step": 3,
                        "instruction": "Beat softened butter with both sugars until fluffy, about 3 minutes. Add eggs and vanilla, beat well.",
                        "duration": "5 minutes",
                        "tips": "Butter should be soft but not melted"
                    },
                    {
                        "step": 4,
                        "instruction": "Gradually add flour mixture to butter mixture. Stir in chocolate chips.",
                        "duration": "3 minutes"
                    },
                    {
                        "step": 5,
                        "instruction": "Drop rounded tablespoons of dough onto prepared sheets, 2 inches apart. Bake 9-11 minutes until golden.",
                        "duration": "12 minutes",
                        "tips": "Don't overbake - they'll firm up as they cool"
                    }
                ]
            }
        }
        
        # Save for future use
        with open(path, 'w') as f:
            json.dump(sample_recipes, f, indent=2)
        
        return sample_recipes
    
    def _setup_vector_db(self) -> chromadb.Collection:
        """Setup ChromaDB with recipe embeddings"""
        try:
            self.chroma_client.delete_collection("cookmate_recipes")
        except:
            pass
        
        collection = self.chroma_client.create_collection(
            name="cookmate_recipes",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Prepare documents
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        doc_id = 0
        
        for recipe_id, recipe in self.recipes.items():
            # Full recipe overview
            overview = f"""Recipe: {recipe['name']}
Prep: {recipe.get('prep_time', 'N/A')} | Cook: {recipe.get('cook_time', 'N/A')} | Serves: {recipe.get('servings', 'N/A')}
Difficulty: {recipe.get('difficulty', 'Medium')}

Ingredients:
{chr(10).join(['- ' + ing for ing in recipe['ingredients']])}"""
            
            documents.append(overview)
            metadatas.append({
                "recipe_id": recipe_id,
                "recipe_name": recipe['name'],
                "type": "overview"
            })
            ids.append(f"doc_{doc_id}")
            doc_id += 1
            
            # Individual steps
            for step_data in recipe['steps']:
                step_text = f"{recipe['name']} - Step {step_data['step']}: {step_data['instruction']}"
                if step_data.get('duration'):
                    step_text += f" (Takes {step_data['duration']})"
                if step_data.get('tips'):
                    step_text += f" Tip: {step_data['tips']}"
                
                documents.append(step_text)
                metadatas.append({
                    "recipe_id": recipe_id,
                    "recipe_name": recipe['name'],
                    "step_number": step_data['step'],
                    "type": "step"
                })
                ids.append(f"doc_{doc_id}")
                doc_id += 1
        
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Add to collection
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✓ Indexed {len(documents)} documents")
        return collection
    
    def record_audio(self, duration: int = 5, sample_rate: int = 16000) -> str:
        """Record audio from microphone"""
        print(f"🎤 Recording for {duration} seconds...")
        audio = sd.rec(int(duration * sample_rate), 
                      samplerate=sample_rate, 
                      channels=1, 
                      dtype=np.int16)
        sd.wait()
        print("✓ Recording complete")
        
        temp_path = "temp_audio.wav"
        write(temp_path, sample_rate, audio)
        return temp_path
    
    def speech_to_text(self, audio_path: str) -> str:
        """Convert speech to text"""
        if not self.whisper_model:
            return ""
        
        result = self.whisper_model.transcribe(audio_path)
        return result["text"].strip()
    
    def text_to_speech(self, text: str) -> str:
        """Convert text to speech"""
        tts = gTTS(text=text, lang='en', slow=False)
        output_path = f"tts_{int(time.time())}.mp3"
        tts.save(output_path)
        return output_path
    
    def play_audio(self, audio_path: str):
        """Play audio file"""
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        # Cleanup
        try:
            os.remove(audio_path)
        except:
            pass
    def retrieve_context(self, query: str, k: int = 3):
        query_vec = embed_text(query)
        if self.state.current_recipe:
            relevant_chunks = [c for c in self.recipe_chunks if c['recipe_id'] == self.state.current_recipe]
        else:
            relevant_chunks = self.recipe_chunks
        similarities = []
        for chunk in relevant_chunks:
            score = cosine_similarity(query_vec, chunk['vector'])
            similarities.append((score, chunk))
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [chunk for score, chunk in similarities[:k]]
    
    def _classify_query(self, query: str) -> str:
        """Classify query type to decide how to handle it"""
        query_lower = query.lower().strip()
        
        # Off-topic (highest priority)
        off_topic = ["weather", "news", "sports", "politics", "stock", "movie", "calculate"]
        if any(word in query_lower for word in off_topic):
            return "off_topic"
        
        # Navigation commands - EXPANDED and PRIORITIZED
        navigation_keywords = [
            "next", "repeat", "previous", "back", "continue", "start", "begin", 
            "make", "again", "go to step", "step", "jump to", "how many steps",
            "number of steps", "total steps", "last step", "what's next", "whats next",
            "next step", "continue", "proceed", "move on", "go ahead", "ok next", "next please"
        ]
        
        # If we have an active recipe, be ULTRA-AGGRESSIVE about classifying as navigation
        if self.state.current_recipe:
            # Short commands are ALWAYS navigation when we have an active recipe
            if len(query.split()) <= 4:  # Increased from 3 to 4 to catch "next step"
                if any(word in query_lower for word in ["next", "back", "previous", "again", "repeat", "continue", "go", "ok", "step"]):
                    return "navigation"
            
            # Specific handling for "next step" variations - THESE SHOULD NEVER FAIL
            if any(phrase in query_lower for phrase in ["next step", "next please", "whats next", "what's next", "ok next"]):
                return "navigation"
            
            # Common navigation patterns
            if any(keyword in query_lower for keyword in navigation_keywords):
                return "navigation"
        
        # Standard navigation detection for all cases
        if any(keyword in query_lower for keyword in navigation_keywords):
            return "navigation"
        
        # Time estimation queries
        time_keywords = ["how much time", "time left", "how long", "time remaining", "much more time", "time it will take", "how much longer"]
        if any(keyword in query_lower for keyword in time_keywords):
            return "cooking"
        
        # Cooking-related (if has active recipe, assume it's about that)
        if self.state.current_recipe:
            return "cooking"
        
        # Check for cooking keywords
        cooking_keywords = [
            "recipe", "cook", "bake", "ingredient", "substitute", "replace",
            "how long", "temperature", "oven", "stir", "mix", "chop",
            "boil", "fry", "pasta", "chicken", "food", "meal"
        ]
        if any(keyword in query_lower for keyword in cooking_keywords):
            return "cooking"
        
        return "general"
    
    def process_query(self, query: str) -> Tuple[str, float]:
        """Process user query with context awareness and smart routing"""
        start_time = time.time()
        query_lower = query.lower().strip()
        
        # Handle standalone numbers FIRST (before classification)
        if query.strip().isdigit():
            step_num = int(query.strip())
            if self.state.current_recipe:
                return self._go_to_specific_step(step_num), 0.1
            else:
                return "Please start a recipe first. Try 'start pasta carbonara' or 'make chocolate chip cookies'.", 0.1
        
        # ULTRA-AGGRESSIVE navigation detection for active recipes
        if self.state.current_recipe:
            # These patterns should ALWAYS be treated as next step navigation
            next_step_patterns = [
                "next", "next step", "next please", "whats next", "what's next",
                "continue", "proceed", "go ahead", "move on", "ok next"
            ]
            
            if any(pattern in query_lower for pattern in next_step_patterns):
                return self._handle_next_step(), 0.1
        
        # Classify the query
        query_type = self._classify_query(query)
        
        # Handle off-topic first
        if query_type == "off_topic":
            response = "I'm CookMate, your cooking assistant! 🍳 I help with recipes, ingredients, and cooking techniques. What would you like to cook today?"
        
        elif query_type == "navigation":
            # NEW: Handle "last step" specifically
            if "last step" in query_lower:
                if self.state.current_recipe:
                    response = self._go_to_specific_step(self.state.total_steps)
                else:
                    response = "Please start a recipe first."
            
            # NEW: Handle step count queries
            elif any(phrase in query_lower for phrase in ["how many steps", "number of steps", "total steps"]):
                if self.state.current_recipe:
                    response = f"This recipe has {self.state.total_steps} steps."
                else:
                    response = "Please start a recipe first to see the step count."
            
            # Handle "go to step X" commands - only when numbers are present
            elif any(phrase in query_lower for phrase in ["go to step", "jump to step"]) or ("step" in query_lower and re.search(r'\d+', query)):
                numbers = re.findall(r'\d+', query)
                if numbers:
                    step_num = int(numbers[0])
                    response = self._go_to_specific_step(step_num)
                else:
                    # Check for "last" keyword
                    if "last" in query_lower and self.state.current_recipe:
                        response = self._go_to_specific_step(self.state.total_steps)
                    else:
                        # If no number and not "last", assume they want next step
                        response = self._handle_next_step()
            
            # Context-aware navigation - FIXED: handle all "next" variations
            elif any(phrase in query_lower for phrase in ["what's next", "next step", "continue", "next", "whats next", "proceed", "move on", "go ahead", "ok next", "next please"]):
                response = self._handle_next_step()
            
            elif any(phrase in query_lower for phrase in ["repeat", "say that again", "what was that", "again"]):
                response = self._handle_repeat()
            
            elif any(phrase in query_lower for phrase in ["previous", "go back", "back"]):
                response = self._handle_previous_step()
            
            elif any(phrase in query_lower for phrase in ["start", "begin", "make"]):
                # If we already have a recipe, check if they want to restart
                if self.state.current_recipe:
                    current_recipe_name = self.state.current_recipe_name.lower()
                    # Check if they're referring to current recipe
                    if any(word in query_lower for word in current_recipe_name.split()):
                        # They're referring to current recipe, restart it
                        response = self._start_recipe(self.state.current_recipe, self.recipes[self.state.current_recipe])
                    else:
                        # They want a different recipe
                        response = self._handle_start_recipe(query)
                else:
                    response = self._handle_start_recipe(query)
            
            else:
                # Fallback: if we have active recipe and it's a short query, assume next step
                if self.state.current_recipe and len(query.split()) <= 3:
                    response = self._handle_next_step()
                else:
                    response = self._handle_general_query(query)
        
        elif query_type == "cooking":
            # NEW: Handle time estimation queries
            if any(phrase in query_lower for phrase in ["how much time", "time left", "how long", "time remaining", "much more time", "time it will take", "how much longer"]):
                response = self._handle_time_estimation()
            
            # Cooking-specific queries (use RAG)
            elif "substitute" in query_lower or "replace" in query_lower:
                response = self._handle_substitution(query)
            
            elif "ingredient" in query_lower:
                response = self._handle_ingredient_query(query)
            
            else:
                response = self._handle_general_query(query)
        
        elif query_type == "general":
            # General questions (LLM without RAG restriction)
            response = self._handle_general_question(query)
        
        else:
            response = self._handle_general_query(query)
        
        latency = time.time() - start_time
        self.state.last_response = response
        self.state.conversation_history.append({
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "latency": latency
        })
        
        return response, latency
    
    def _handle_time_estimation(self) -> str:
        """Calculate remaining time for current recipe"""
        if not self.state.current_recipe:
            return "Please start a recipe first to estimate cooking time."
        
        recipe = self.recipes.get(self.state.current_recipe)
        
        if self.state.current_step == 0:
            total_time = self._parse_time(recipe.get('prep_time', '0')) + self._parse_time(recipe.get('cook_time', '0'))
            return f"Total estimated time: {total_time} minutes. You haven't started cooking yet."
        
        # Calculate remaining time from current step to end
        remaining_time = 0
        for i in range(self.state.current_step - 1, len(recipe['steps'])):
            step = recipe['steps'][i]
            if step.get('duration'):
                remaining_time += self._parse_time(step['duration'])
        
        current_step = recipe['steps'][self.state.current_step - 1]
        current_step_name = f"Step {self.state.current_step}: {current_step['instruction'][:100]}..."
        
        steps_remaining = self.state.total_steps - self.state.current_step + 1
        
        return f"You're on step {self.state.current_step} of {self.state.total_steps}. " \
               f"Estimated time remaining: {remaining_time} minutes. " \
               f"Steps remaining: {steps_remaining}. " \
               f"Current: {current_step_name}"
    
    def _parse_time(self, time_str: str) -> int:
        """Parse time strings like '10 minutes' into minutes"""
        if not time_str:
            return 0
        
        numbers = re.findall(r'\d+', time_str)
        if numbers:
            minutes = int(numbers[0])
            # Handle hours
            if 'hour' in time_str.lower():
                minutes *= 60
            return minutes
        return 0
    
    def _go_to_specific_step(self, step_num: int) -> str:
        """Jump to a specific step number"""
        if not self.state.current_recipe:
            return "Please start a recipe first. Try saying 'Start pasta carbonara' or 'Make chocolate chip cookies'."
        
        recipe = self.recipes.get(self.state.current_recipe)
        
        if step_num < 1 or step_num > len(recipe['steps']):
            return f"That recipe only has {len(recipe['steps'])} steps. Please choose a step between 1 and {len(recipe['steps'])}."
        
        self.state.current_step = step_num
        step = recipe['steps'][step_num - 1]
        
        response = f"Jumping to Step {step['step']} of {self.state.total_steps}: {step['instruction']}"
        
        if step.get('duration'):
            response += f"\n⏱ This takes about {step['duration']}."
        
        if step.get('tips'):
            response += f"\n💡 Tip: {step['tips']}"
        
        return response
    
    def _handle_next_step(self) -> str:
        """Handle next step navigation"""
        if not self.state.current_recipe:
            return "Please start a recipe first. Try saying 'Start pasta carbonara' or 'Make chocolate chip cookies'."
        
        recipe = self.recipes.get(self.state.current_recipe)
        
        if self.state.current_step >= self.state.total_steps:
            return f"🎉 You've completed all {self.state.total_steps} steps! Your {recipe['name']} is ready. Enjoy!"
        
        self.state.current_step += 1
        step = recipe['steps'][self.state.current_step - 1]
        
        response = f"Step {step['step']} of {self.state.total_steps}: {step['instruction']}"
        
        if step.get('duration'):
            response += f"\n⏱ This takes about {step['duration']}."
        
        if step.get('tips'):
            response += f"\n💡 Tip: {step['tips']}"
        
        return response
    
    def _repeat_specific_step(self, step_num: int) -> str:
        """Repeat a specific step by number"""
        if not self.state.current_recipe:
            return "Please start a recipe first."
        
        recipe = self.recipes.get(self.state.current_recipe)
        if step_num < 1 or step_num > len(recipe['steps']):
            return f"That recipe only has {len(recipe['steps'])} steps."
        
        step = recipe['steps'][step_num - 1]
        response = f"Step {step['step']}: {step['instruction']}"
        
        if step.get('duration'):
            response += f"\n⏱ This takes about {step['duration']}."
        
        if step.get('tips'):
            response += f"\n💡 Tip: {step['tips']}"
        
        return response
    
    def _handle_repeat(self) -> str:
        """Repeat current step"""
        if not self.state.current_recipe or self.state.current_step == 0:
            if self.state.last_response:
                return self.state.last_response
            return "There's nothing to repeat yet."
        
        recipe = self.recipes.get(self.state.current_recipe)
        step = recipe['steps'][self.state.current_step - 1]
        
        response = f"Repeating Step {step['step']}: {step['instruction']}"
        if step.get('duration'):
            response += f"\n⏱ This takes about {step['duration']}."
        if step.get('tips'):
            response += f"\n💡 {step['tips']}"
        
        return response
    
    def _handle_previous_step(self) -> str:
        """Go back to previous step"""
        if not self.state.current_recipe:
            return "Please start a recipe first. Try saying 'Start pasta carbonara' or 'Make chocolate chip cookies'."
        
        if self.state.current_step <= 1:
            return "You're at the first step. Say 'next' to continue."
        
        self.state.current_step -= 1
        recipe = self.recipes.get(self.state.current_recipe)
        step = recipe['steps'][self.state.current_step - 1]
        
        response = f"Going back to Step {step['step']} of {self.state.total_steps}: {step['instruction']}"
        
        if step.get('duration'):
            response += f"\n⏱ This takes about {step['duration']}."
        
        if step.get('tips'):
            response += f"\n💡 Tip: {step['tips']}"
        
        return response
    
    def _handle_start_recipe(self, query: str) -> str:
        """Start a recipe with improved fuzzy matching"""
        query_lower = query.lower()
        
        # Remove common words that don't help matching
        stop_words = ['make', 'start', 'begin', 'cook', 'bake', 'prepare', 'lets', "let's", 'a', 'the']
        query_words = [w for w in query_lower.split() if w not in stop_words]
        
        best_match = None
        best_score = 0
        
        for recipe_id, recipe in self.recipes.items():
            recipe_name_lower = recipe['name'].lower()
            
            # Exact match (highest priority)
            if recipe_name_lower in query_lower or query_lower in recipe_name_lower:
                return self._start_recipe(recipe_id, recipe)
            
            # Word-by-word matching
            recipe_words = set(recipe_name_lower.split())
            query_word_set = set(query_words)
            
            # Calculate match score
            if query_word_set:
                matching_words = recipe_words & query_word_set
                score = len(matching_words) / len(query_word_set)
                
                # If any query word matches recipe, consider it
                if score > best_score:
                    best_score = score
                    best_match = (recipe_id, recipe)
        
        # Accept if at least one word matches (more lenient)
        if best_match and best_score > 0:
            return self._start_recipe(best_match[0], best_match[1])
        
        # No match found
        available = ', '.join([f'"{r["name"]}"' for r in self.recipes.values()])
        return f"I don't have that recipe. Available recipes: {available}. Try saying 'start [recipe name]'."
    
    def _start_recipe(self, recipe_id: str, recipe: Dict) -> str:
        """Helper to start a recipe"""
        self.state.current_recipe = recipe_id
        self.state.current_recipe_name = recipe['name']
        self.state.current_step = 0
        self.state.total_steps = len(recipe['steps'])
        
        response = f"🍳 Starting: {recipe['name']}\n"
        response += f"⏱ Prep: {recipe.get('prep_time', 'N/A')} | Cook: {recipe.get('cook_time', 'N/A')}\n"
        response += f"👥 Serves: {recipe.get('servings', 'N/A')} | Difficulty: {recipe.get('difficulty', 'Medium')}\n\n"
        response += "📝 Ingredients you'll need:\n"
        response += '\n'.join([f"  • {ing}" for ing in recipe['ingredients']])
        response += "\n\nSay 'next' when you're ready for step 1!"
        
        return response
    
    def _handle_substitution(self, query: str) -> str:
        """Handle ingredient substitution queries"""
        contexts = self.retrieve_context(query, k=2)
        
        if not contexts:
            return "I don't have specific substitution info for that. Generally, you can substitute similar ingredients (butter↔oil, milk↔cream, etc.)"
        
        context_str = "\n".join([c["text"] for c in contexts])
        
        # Try LLM only if available, otherwise give direct answer
        if self.llm.available:
            prompt = f"User is asking about ingredient substitutions: '{query}'"
            response = self.llm.generate(prompt, context_str)
            if response and len(response) > 20:
                return response
        
        # Fallback: use the retrieved context
        return f"Based on similar recipes:\n\n{context_str}\n\nCommon substitutions work well here (e.g., butter→oil, eggs→flax eggs)."
    
    def _handle_timing_query(self, query: str) -> str:
        """Handle timing-related questions"""
        if self.state.current_recipe and self.state.current_step > 0:
            recipe = self.recipes.get(self.state.current_recipe)
            step = recipe['steps'][self.state.current_step - 1]
            
            if step.get('duration'):
                return f"Step {self.state.current_step} takes {step['duration']}."
            else:
                return f"Step {self.state.current_step} doesn't have a specific time. Follow the instructions: {step['instruction']}"
        
        # No active step, search recipes
        contexts = self.retrieve_context(query, k=2)
        if contexts:
            return contexts[0]["text"]
        
        return "Please start a recipe first, then I can tell you timing for each step."
    
    def _handle_ingredient_query(self, query: str) -> str:
        """Handle ingredient-related questions"""
        if self.state.current_recipe:
            recipe = self.recipes.get(self.state.current_recipe)
            ingredients = '\n'.join([f"  • {ing}" for ing in recipe['ingredients']])
            return f"For {recipe['name']}, you need:\n{ingredients}"
        
        contexts = self.retrieve_context(query, k=2)
        return contexts[0]["text"] if contexts else "Please start a recipe first."
    
    def _handle_general_question(self, query: str) -> str:
        """Handle general non-cooking questions with better filtering"""
        query_lower = query.lower()
        
        # Detect negative statements (don't want X)
        negative_indicators = ["don't want", "do not want", "dont want", "not interested", "no thanks"]
        if any(neg in query_lower for neg in negative_indicators):
            return "No problem! What would you like to cook instead? I have recipes for pasta, cookies, and more. Just say 'start [recipe name]' or ask me what's available! 🍳"
        
        # Check if it's completely off-topic
        off_topic_keywords = [
            "weather", "temperature today", "forecast", "rain", "sunny",
            "news", "politics", "election", "president",
            "sports", "game", "score", "team",
            "stock", "market", "investment",
            "movie", "film", "tv show",
            "math", "calculate", "what is 2+2",
        ]
        
        if any(keyword in query_lower for keyword in off_topic_keywords):
            return "I'm CookMate, your cooking assistant! I specialize in recipes, ingredients, and cooking techniques. What would you like to cook today? 🍳"
        
        # Cooking mishaps/problems
        cooking_problems = {
            "overcooked": "If it's overcooked, it might be mushy. For pasta, you can still use it in a baked dish. For next time, test it a minute before the recommended time!",
            "overboiled": "Overboiled pasta gets mushy. You can still make it work by draining immediately and tossing with sauce. Next time, test for doneness 1-2 minutes early!",
            "undercooked": "Keep cooking! Just add more time. For pasta, it should be 'al dente' - firm but not hard.",
            "burned": "If it's burned, unfortunately you'll need to start fresh. Lower the heat next time and watch it closely!",
            "too salty": "Add water, cream, or unsalted ingredients to dilute. You can also add potato chunks to absorb salt.",
            "too spicy": "Add dairy (milk, cream, yogurt) or something sweet (sugar, honey) to balance the heat.",
            "not cooking": "Check your heat! Make sure burner is on and set to the right temperature. Give it more time.",
        }
        
        for problem, solution in cooking_problems.items():
            if problem in query_lower:
                return f"💡 {solution}"
        
        # If we have an active recipe, assume it's about that recipe
        if self.state.current_recipe:
            recipe = self.recipes.get(self.state.current_recipe)
            if recipe:
                # Try to answer based on current recipe context
                contexts = self.retrieve_context(f"{recipe['name']} {query}", k=2)
                context_str = "\n".join([c["content"] for c in contexts])
                
                # Use fallback with recipe context (no LLM - too unreliable)
                return f"For {recipe['name']}: {context_str[:300]}..."
        
        # For cooking-related general questions
        cooking_keywords = ["cook", "recipe", "ingredient", "food", "meal", "dish"]
        if any(keyword in query_lower for keyword in cooking_keywords):
            contexts = self.retrieve_context(query, k=2)
            if contexts:
                return contexts[0]["text"]
        
        # Last resort
        return "I'm here to help with cooking! Try asking about recipes, ingredients, cooking times, or substitutions. What would you like to know? 🍳"
    
    def _handle_general_query(self, query: str) -> str:
        """Handle general cooking questions with RAG"""
        query_lower = query.lower()
        
        # If user says "again", repeat last response
        if query_lower in ["again", "say again", "repeat again"]:
            return self._handle_repeat()
        
        # Retrieve relevant context
        contexts = self.retrieve_context(query, k=2)
        
        if not contexts:
            return "I don't have information about that in my recipes. Could you be more specific or ask about a different recipe?"
        
        # If we have active recipe, prioritize that
        if self.state.current_recipe:
            recipe = self.recipes.get(self.state.current_recipe)
            
            # Build context from current recipe
            recipe_context = f"Current recipe: {recipe['name']}\n"
            recipe_context += f"You are on step {self.state.current_step} of {self.state.total_steps}\n\n"
            
            # Add retrieved contexts
            for ctx in contexts:
                recipe_context += ctx["text"] + "\n\n"
            
            # Try LLM with timeout handling
            if self.llm.available:
                try:
                    response = self.llm.generate(query, recipe_context)
                    if response and len(response) > 20:
                        return response
                except:
                    pass
            
            # Fallback: return the most relevant context directly
            return contexts[0]["text"]
        
        # No active recipe - use RAG results directly
        context_str = contexts[0]["text"]
        
        # Try LLM
        if self.llm.available:
            response = self.llm.generate(query, context_str)
            if response and len(response) > 20:
                return response
        
        # Fallback: return context directly
        return f"Here's what I found:\n\n{context_str}"
    
    def chat(self, use_voice: bool = False):
        """Interactive chat interface"""
        print("\n" + "="*60)
        print("🍳 CookMate Voice Assistant")
        print("="*60)
        print("\n💬 Try saying:")
        print("  • 'Start pasta carbonara'")
        print("  • 'What's next?'")
        print("  • 'How do I substitute pancetta?'")
        print("  • 'How long does this take?'")
        print("  • 'Go to step 3'")
        print("  • 'Repeat step 1'")
        print("  • 'How many steps?'")
        print("  • 'Go to last step'")
        print("  • 'Next' (after any question)")
        print("\n⌨️  Type 'quit' to exit")
        print("="*60 + "\n")
        
        while True:
            try:
                if use_voice and self.whisper_model:
                    input("\n🎤 Press Enter to speak (or type 'text' for text mode)... ")
                    audio_path = self.record_audio(duration=5)
                    user_input = self.speech_to_text(audio_path)
                    os.remove(audio_path)
                    print(f"\n👤 You said: {user_input}")
                else:
                    user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'stop']:
                    print("\n👨‍🍳 Happy cooking! See you next time!")
                    break
                
                # Process query
                print("\n🤔 Processing...")
                response, latency = self.process_query(user_input)
                print(f"\n🤖 CookMate: {response}")
                print(f"\n⚡ Response time: {latency:.2f}s")
                
                # TTS if voice mode
                if use_voice:
                    print("🔊 Speaking...")
                    audio_file = self.text_to_speech(response)
                    self.play_audio(audio_file)
            
            except KeyboardInterrupt:
                print("\n\n👨‍🍳 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
    
    def get_statistics(self) -> Dict:
        """Get usage statistics"""
        total_queries = len(self.state.conversation_history)
        avg_latency = sum(q['latency'] for q in self.state.conversation_history) / total_queries if total_queries > 0 else 0
        
        return {
            "total_queries": total_queries,
            "average_latency": f"{avg_latency:.2f}s",
            "current_recipe": self.state.current_recipe_name,
            "current_step": f"{self.state.current_step}/{self.state.total_steps}"
        }
# Main execution
if __name__ == "__main__":
    import sys
    
    print("\n🚀 Starting CookMate...\n")
    
    # Initialize system
    cookmate = CookMateRAG(
        use_whisper=True,  # Set to False if no microphone
        whisper_model="base"  # Options: tiny, base, small
    )
    
    # Choose mode
    voice_mode = True
    if len(sys.argv) > 1 and sys.argv[1] == "--voice":
        voice_mode = True
    
    # Start chat
    cookmate.chat(use_voice=voice_mode)
    
    # Show stats
    print("\n📊 Session Statistics:")
    stats = cookmate.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
