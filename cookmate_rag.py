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

import requests

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠ python-dotenv not installed. Using system environment variables.")


@dataclass
class RecipeStep:
    step_number: int
    instruction: str
    duration: Optional[str] = None
    ingredients: Optional[List[str]] = None
    tips: Optional[str] = None


@dataclass
class ConversationMessage:
    role: str
    content: str
    timestamp: str
    sources: Optional[List[str]] = None


@dataclass
class ConversationState:
    current_recipe: Optional[str] = None
    current_recipe_name: Optional[str] = None
    current_step: int = 0
    total_steps: int = 0
    timer_start: Optional[float] = None
    messages: List[ConversationMessage] = field(default_factory=list)
    max_history: int = 10
    last_response: str = ""


class ImprovedGroqLLM:
    """Fixed Groq LLM with proper context and hallucination prevention"""
    
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1"
        self.api_key = self._get_api_key()
        self.available = self._check_connection()
    
    def _get_api_key(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("\n🔑 GROQ_API_KEY not found!")
            print("   Option 1: Create .env file with: GROQ_API_KEY=your_key")
            print("   Option 2: Set environment variable")
            print("   Option 3: Enter key now")
            api_key = input("\nEnter Groq API key (or press Enter to skip): ").strip()
            if not api_key:
                return None
        return api_key
    
    def _check_connection(self):
        if not self.api_key:
            print("⚠ Running without LLM (fallback mode)")
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
            return False
        except Exception as e:
            print(f"⚠ Groq unavailable: {e}")
            return False
    
    def generate_with_history(
        self, 
        prompt: str, 
        context: str, 
        conversation_history: List[ConversationMessage],
        recipe_name: str = None
    ) -> str:
        """Generate response with conversation history and strict grounding"""
        
        if not self.available:
            return self._safe_fallback(prompt, context)
        
        system_message = self._build_strict_system_prompt(context, recipe_name)
        messages = [{"role": "system", "content": system_message}]
        
        # Add recent conversation history (last 10 messages)
        recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
        
        for msg in recent_history:
            messages.append({"role": msg.role, "content": msg.content})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 200,
                    "top_p": 0.8,
                    "frequency_penalty": 0.3,
                    "presence_penalty": 0.3
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"].strip()
                
                if self._is_valid_response(result, context):
                    return result
                
                print("⚠ Response failed validation, using fallback")
                return self._safe_fallback(prompt, context)
            
            return self._safe_fallback(prompt, context)
            
        except Exception as e:
            print(f"⚠ LLM error: {e}")
            return self._safe_fallback(prompt, context)
    
    def _build_strict_system_prompt(self, context: str, recipe_name: str = None) -> str:
        recipe_info = f"for {recipe_name}" if recipe_name else ""
        
        return f"""You are CookMate, a precise cooking assistant {recipe_info}.

CRITICAL RULES:
1. ONLY use information from the Recipe Context below
2. If answer is NOT in context, say: "I don't have that information in the current recipe"
3. NEVER make up ingredients, steps, times, or temperatures
4. Keep responses under 3 sentences
5. DO NOT answer non-cooking questions

Recipe Context:
---
{context}
---

If question cannot be answered using ONLY the information above, say:
"I don't see that in the current recipe. Ask about something else?"
"""
    
    def _is_valid_response(self, response: str, context: str) -> bool:
        generic_phrases = ["generally", "typically", "usually", "most chefs", "traditionally"]
        response_lower = response.lower()
        
        generic_count = sum(1 for phrase in generic_phrases if phrase in response_lower)
        if generic_count >= 2:
            return False
        
        context_words = set(context.lower().split())
        response_words = set(response_lower.split())
        overlap = len(context_words & response_words)
        
        if overlap < 3 and len(response.split()) > 10:
            return False
        
        return True
    
    def _safe_fallback(self, prompt: str, context: str) -> str:
        prompt_lower = prompt.lower()
        
        off_topic = ["weather", "news", "sports", "politics", "stock", "time", "date"]
        if any(word in prompt_lower for word in off_topic):
            return "I'm CookMate - I only help with cooking! What would you like to cook? 🍳"
        
        if context:
            sentences = [s.strip() for s in context.split('.') if s.strip()]
            
            best_sentence = ""
            best_score = 0
            query_words = set(prompt_lower.split())
            
            for sentence in sentences[:5]:
                sentence_words = set(sentence.lower().split())
                score = len(query_words & sentence_words)
                if score > best_score:
                    best_score = score
                    best_sentence = sentence
            
            if best_sentence and best_score > 0:
                return best_sentence + "."
        
        return "I don't have that information in the current recipe. Ask about ingredients, steps, or timing?"


class CookMateRAG:
    """Complete CookMate system with all features and improved RAG"""
    
    def __init__(self, 
                 recipe_data_path: str = "recipes.json",
                 use_whisper: bool = True,
                 whisper_model: str = "base"):
        """
        Initialize complete CookMate system
        
        Args:
            recipe_data_path: Path to recipe JSON
            use_whisper: Enable voice input
            whisper_model: Whisper model size (tiny/base/small)
        """
        print("🍳 Initializing CookMate...")
        self.state = ConversationState()
        
        # Load recipes
        print("📚 Loading recipes...")
        self.recipes = self._load_recipes(recipe_data_path)
        
        # Initialize embedding model (free from HuggingFace)
        print("🧠 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        print("💾 Setting up vector database...")
        self.chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=False  # Use in-memory for speed
        ))
        self.collection = self._setup_vector_db()
        
        # Initialize LLM with improved Groq implementation
        print("🤖 Connecting to Groq LLM...")
        self.llm = ImprovedGroqLLM(model="llama-3.1-8b-instant")
        
        # Initialize ASR
        self.whisper_model = None
        if use_whisper:
            print(f"🎤 Loading Whisper ({whisper_model})...")
            self.whisper_model = whisper.load_model(whisper_model)
        
        # Initialize TTS
        pygame.mixer.init()
        
        print("✅ CookMate ready!\n")
    
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
        
        # Generate embeddings
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
    
    def retrieve_context(self, query: str, k: int = 3) -> Tuple[str, List[Dict]]:
        """Retrieve relevant context using RAG with filtering"""
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        where_filter = None
        if self.state.current_recipe:
            where_filter = {"recipe_id": self.state.current_recipe}
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k * 2,
            where=where_filter
        )
        
        contexts = []
        for i in range(len(results['documents'][0])):
            distance = results['distances'][0][i]
            
            if distance < 0.7:  # Quality threshold
                contexts.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": distance,
                    "relevance": 1 - distance
                })
        
        contexts.sort(key=lambda x: x['relevance'], reverse=True)
        contexts = contexts[:k]
        
        if not contexts:
            return "", []
        
        context_parts = [ctx['content'] for ctx in contexts]
        context_string = "\n\n".join(context_parts)
        
        return context_string, contexts
    
    def _is_off_topic(self, query: str) -> bool:
        """Check if query is off-topic"""
        off_topic = ["weather", "news", "sports", "politics", "stock", "movie", "time", "date"]
        return any(word in query.lower() for word in off_topic)
    
    def _handle_mixed_intent(self, query: str) -> Optional[str]:
        """Handle queries that have multiple intents"""
        query_lower = query.lower()
        
        mixed_indicators = [" and ", " also ", " plus ", " then ", " after that", " next "]
        if not any(indicator in query_lower for indicator in mixed_indicators):
            return None
        
        print("🎯 Detected mixed intent query")
        
        # Split and process each part
        responses = []
        
        # Common splitting patterns
        if " and " in query_lower:
            parts = query.split(" and ")
        elif " then " in query_lower:
            parts = query.split(" then ")
        elif " also " in query_lower:
            parts = query.split(" also ")
        else:
            parts = [query]  # Fallback
        
        for part in parts[:2]:  # Process max 2 parts to avoid complexity
            part = part.strip()
            if part:
                # Use existing process_query but don't update history yet
                response, _ = self.process_query(part)
                responses.append(f"{part}: {response}")
        
        return " | ".join(responses) if responses else None
    
    def _handle_recipe_switch(self, query: str) -> Optional[str]:
        """Handle requests to switch recipes"""
        query_lower = query.lower()
        
        switch_patterns = [
            "switch to", "change to", "try", "different recipe", 
            "new recipe", "another recipe", "let's try", "how about",
            "i want to make instead", "start over with"
        ]
        
        if any(pattern in query_lower for pattern in switch_patterns):
            print("🔄 Detected recipe switch request")
            
            # If we have a current recipe, confirm switch
            if self.state.current_recipe:
                current_recipe = self.state.current_recipe_name
                # Extract new recipe name from query
                for recipe_id, recipe in self.recipes.items():
                    if recipe['name'].lower() in query_lower:
                        self.state.current_recipe = None  # Reset current recipe
                        self.state.current_step = 0
                        return f"Switching from {current_recipe} to {recipe['name']}.\n\n{self._start_recipe(recipe_id, recipe)}"
            
            # If no specific recipe mentioned, show available options
            available = ', '.join([f'"{r["name"]}"' for r in self.recipes.values()])
            return f"Available recipes: {available}. Which one would you like to switch to?"
        
        return None
    def _handle_multi_step_request(self, query: str) -> Optional[str]:
        """Handle requests for multiple steps"""
        query_lower = query.lower()
        
        if not self.state.current_recipe:
            return None
        
        # Patterns for multiple steps
        multi_step_patterns = [
            "next two steps", "next 2 steps", "next few steps",
            "steps", "through", "to step", "until step"
        ]
        
        if not any(pattern in query_lower for pattern in multi_step_patterns):
            return None
        
        print("📚 Detected multi-step request")
        
        recipe = self.recipes.get(self.state.current_recipe)
        current_step = self.state.current_step
        
        # Extract step range from query
        numbers = re.findall(r'\d+', query)
        
        if "next two" in query_lower or "next 2" in query_lower:
            # Show next 2 steps
            end_step = min(current_step + 2, self.state.total_steps)
            steps_range = range(current_step + 1, end_step + 1)
        elif "next few" in query_lower:
            # Show next 3 steps
            end_step = min(current_step + 3, self.state.total_steps)
            steps_range = range(current_step + 1, end_step + 1)
        elif numbers and "to step" in query_lower:
            # Specific range like "steps 2 to 4"
            if len(numbers) >= 2:
                start_step = int(numbers[0])
                end_step = min(int(numbers[1]), self.state.total_steps)
                steps_range = range(start_step, end_step + 1)
            else:
                steps_range = range(current_step + 1, self.state.total_steps + 1)
        else:
            # Default: show remaining steps
            steps_range = range(current_step + 1, self.state.total_steps + 1)
        
        responses = []
        for step_num in steps_range:
            if step_num <= self.state.total_steps:
                step = recipe['steps'][step_num - 1]
                step_response = f"Step {step['step']}: {step['instruction']}"
                if step.get('duration'):
                    step_response += f" ⏱ {step['duration']}"
                if step.get('tips'):
                    step_response += f" 💡 {step['tips']}"
                responses.append(step_response)
        
        if responses:
            return "Here are the steps:\n\n" + "\n\n".join(responses)
        
        return None
    def _update_conversation_history(self, query: str, response: str):
            """Update conversation history with new messages"""
            self.state.messages.append(ConversationMessage(
                role="user",
                content=query,
                timestamp=datetime.now().isoformat()
            ))
            
            self.state.messages.append(ConversationMessage(
                role="assistant",
                content=response,
                timestamp=datetime.now().isoformat()
            ))
            
            # Trim history if too long
            if len(self.state.messages) > self.state.max_history * 2:
                self.state.messages = self.state.messages[-self.state.max_history * 2:]
            
            self.state.last_response = response

    def _handle_quantity_adjustment(self, query: str) -> Optional[str]:
        """Handle recipe scaling requests"""
        query_lower = query.lower()
        
        if not self.state.current_recipe:
            return None
        
        adjustment_patterns = [
            "half", "double", "triple", "quarter",
            "for 2 people", "for 4 people", "for 6 people",
            "make less", "make more", "scale", "adjust"
        ]
        
        if not any(pattern in query_lower for pattern in adjustment_patterns):
            return None
        
        print("⚖️ Detected quantity adjustment request")
        
        recipe = self.recipes.get(self.state.current_recipe)
        current_servings = recipe.get('servings', 4)
        
        # Extract target quantity
        numbers = re.findall(r'\d+', query)
        target_servings = None
        
        if "half" in query_lower:
            factor = 0.5
            target_servings = current_servings * 0.5
        elif "double" in query_lower or "2 times" in query_lower:
            factor = 2.0
            target_servings = current_servings * 2
        elif "triple" in query_lower or "3 times" in query_lower:
            factor = 3.0
            target_servings = current_servings * 3
        elif numbers:
            target_servings = int(numbers[0])
            factor = target_servings / current_servings
        else:
            return "Please specify how much you want to adjust (half, double, or specific number of servings)"
        
        if target_servings:
            response = f"🔄 Adjusting from {current_servings} to {target_servings:.0f} servings:\n\n"
            
            # Show adjusted ingredients (simplified)
            for ingredient in recipe['ingredients']:
                # Simple ingredient adjustment (this is basic - could be enhanced)
                adjusted_ingredient = f"• {ingredient} × {factor:.1f}"
                response += adjusted_ingredient + "\n"
            
            response += f"\n⏱ Cooking time remains similar. Adjust seasonings to taste."
            return response
        
        return None
    def _handle_equipment_substitution(self, query: str) -> Optional[str]:
        """Handle equipment substitution questions"""
        query_lower = query.lower()
        
        equipment_patterns = [
            "without", "don't have", "no ", "alternative to", 
            "substitute for", "can i use", "instead of"
        ]
        
        equipment_keywords = [
            "mixer", "oven", "baking sheet", "pan", "pot", 
            "whisk", "blender", "processor", "rolling pin"
        ]
        
        if not any(pattern in query_lower for pattern in equipment_patterns):
            return None
        
        if not any(equipment in query_lower for equipment in equipment_keywords):
            return None
        
        print("🔧 Detected equipment substitution request")
        
        # Basic equipment substitutions
        equipment_subs = {
            "mixer": "You can use a whisk and elbow grease, or a fork for mixing.",
            "oven": "For stovetop cooking, use a covered pan on low heat. For baking, consider a toaster oven or air fryer.",
            "baking sheet": "Use any flat oven-safe pan, pizza stone, or even foil shaped into a tray.",
            "whisk": "A fork works well for most mixing. For egg whites, use a clean glass jar and shake vigorously.",
            "blender": "Use a food processor, immersion blender, or manually chop/mash ingredients.",
            "rolling pin": "Use a wine bottle, sturdy glass, or even a clean cylindrical can.",
            "baking pan": "Use any oven-safe dish of similar size. Adjust cooking time if depth differs."
        }
        
        for equipment, substitution in equipment_subs.items():
            if equipment in query_lower:
                return f"🔧 {substitution}"
        
        return "For equipment substitutions: use similar-shaped items, adjust cooking times, and be creative with what you have!"
        
    def _handle_substitution_query(self, query: str) -> Optional[str]:
        """Handle ingredient substitution questions - ALWAYS USE LLM"""
        # Apply spell correction first
        corrected_query = self._correct_spelling(query)
        query_lower = corrected_query.lower().strip()
        
        print(f"🔧 Handling substitution query: '{query}' -> '{corrected_query}'")
        
        # Check if this is a substitution question with fuzzy matching
        substitution_patterns = [
            "substitute", "replace", "instead of", "alternative", 
            "instead", "can i use", "can i replace", "what can i use",
            "don't have", "no ", "without", "swap"
        ]
        
        is_substitution = any(self._fuzzy_match(pattern, query_lower, threshold=0.6) for pattern in substitution_patterns)
        if not is_substitution:
            return None
        
        print("🎯 Detected substitution question - USING LLM")
        
        # ALWAYS use LLM for substitution questions with enhanced context
        enhanced_context = self._get_enhanced_substitution_context(query)
        
        # Use LLM with enhanced context
        response = self.llm.generate_with_history(
            prompt=query,
            context=enhanced_context,
            conversation_history=self.state.messages,
            recipe_name=self.state.current_recipe_name
        )
        
        return response

    def _get_enhanced_substitution_context(self, query: str) -> str:
        """Get enhanced context for substitution questions"""
        # Get regular RAG context
        context, sources = self.retrieve_context(query, k=3)
        
        # Add general substitution knowledge
        substitution_knowledge = """
    General Ingredient Substitutions:
    - Eggs: 1 tbsp chia seeds + 3 tbsp water = 1 egg, or mashed banana, applesauce
    - Butter: Coconut oil, vegetable oil, margarine (1:1 ratio)
    - Milk: Any plant-based milk (almond, soy, oat, coconut)
    - Buttermilk: 1 cup milk + 1 tbsp lemon juice/vinegar
    - Flour: Almond flour, coconut flour, gluten-free blends (adjust ratios)
    - Sugar: Honey, maple syrup, coconut sugar (adjust liquids)
    - Baking powder: 1 tsp = 1/4 tsp baking soda + 1/2 tsp cream of tartar
    - Yogurt: Sour cream, buttermilk, coconut cream
    - Heavy cream: Milk + butter, coconut cream, evaporated milk
    - Wine: Broth with a splash of vinegar
    - Tomato paste: Tomato sauce reduced, ketchup (in small amounts)
    - Fresh herbs: Dried herbs (use 1/3 amount)
    - Pancetta/Bacon: Smoked tofu, mushrooms, tempeh (vegetarian)
    - Pasta: Zucchini noodles, spaghetti squash, rice noodles
        """
        
        # Add current recipe context if available
        recipe_context = ""
        if self.state.current_recipe:
            recipe = self.recipes.get(self.state.current_recipe)
            recipe_context = f"\nCurrent Recipe: {recipe['name']}\nIngredients: {', '.join(recipe['ingredients'])}"
        
        return f"{context}\n\n{substitution_knowledge}\n{recipe_context}"  

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
    

    def _correct_spelling(self, query: str) -> str:
        """Basic spell correction for common navigation words"""
        # Common misspellings and corrections
        corrections = {
            # Next variations
            "nex": "next", "nexy": "next", "nextt": "next", "nect": "next",
            "continew": "continue", "contnue": "continue", "contine": "continue",
            "proced": "proceed", "proceeed": "proceed",
            
            # Previous variations  
            "prev": "previous", "previuos": "previous", "prevous": "previous",
            "previouse": "previous", "bak": "back", "bck": "back",
            
            # Repeat variations
            "repat": "repeat", "repeet": "repeat", "repeate": "repeat",
            "agian": "again", "agane": "again", "agen": "again",
            
            # Step variations
            "stp": "step", "stepp": "step", "sep": "step",
            
            # General
            "whar": "what", "wat": "what", "wht": "what",
            "sho": "show", "howw": "how", "steo": "step",
        }
        
        words = query.lower().split()
        corrected_words = []
        
        for word in words:
            # Check if word is misspelled
            if word in corrections:
                corrected_words.append(corrections[word])
                print(f"🔤 Corrected '{word}' to '{corrections[word]}'")
            else:
                corrected_words.append(word)
        
        return " ".join(corrected_words)


    def _is_clear_navigation(self, query: str) -> bool:
        """Check for CLEAR navigation commands with fuzzy matching"""
        # Apply spell correction first
        corrected_query = self._correct_spelling(query)
        query_lower = corrected_query.lower().strip()
        
        # Clear navigation commands with fuzzy matching
        clear_nav_patterns = [
            "next", "next step", "continue", "proceed", 
            "previous", "back", "go back", "previous step",
            "repeat", "again", "say again", "repeat step",
            "step 1", "step 2", "step 3", "step 4", "step 5", 
            "go to step", "jump to step", "show step"
        ]
        
        # Use fuzzy matching for navigation detection
        for pattern in clear_nav_patterns:
            if self._fuzzy_match(pattern, query_lower, threshold=0.7):
                return True
        
        # Also check for very short queries that are likely navigation
        if len(query_lower.split()) <= 2:
            short_nav_words = ["next", "back", "prev", "again", "go", "yes", "ok"]
            return any(self._fuzzy_match(word, query_lower, 0.6) for word in short_nav_words)
        
        return False



    

    def _is_navigation(self, query: str) -> bool:
        """Check if query is navigation command - WITH SPELL CORRECTION"""
        if not self.state.current_recipe:
            return False
            
        # First, correct spelling
        corrected_query = self._correct_spelling(query)
        query_lower = corrected_query.lower().strip()
        
        print(f"🔤 After correction: '{query_lower}'")
        
        # Expanded navigation patterns
        navigation_patterns = {
            "next": ["next", "continue", "proceed", "go ahead", "move on", "forward"],
            "previous": ["previous", "back", "go back", "return", "backward"],
            "repeat": ["repeat", "again", "say again", "once more"],
            "step_jump": ["step", "go to step", "jump to step"],
            "info": ["how many", "total steps", "number of steps", "current step"]
        }
        
        # Flatten all patterns
        all_nav_words = []
        for category in navigation_patterns.values():
            all_nav_words.extend(category)
        
        # Check with fuzzy matching
        for nav_word in all_nav_words:
            if self._fuzzy_match(nav_word, query_lower):
                return True
        
        # Check short commands with fuzzy matching
        if len(query_lower.split()) <= 3:
            short_commands = ["next", "back", "prev", "again", "repeat", "step"]
            return any(self._fuzzy_match(cmd, query_lower) for cmd in short_commands)
        
        return False

    def _fuzzy_match(self, pattern: str, text: str, threshold: float = 0.7) -> bool:
        """Fuzzy string matching using simple similarity"""
        pattern = pattern.lower()
        text = text.lower()
        
        # Exact match
        if pattern in text:
            return True
        
        # Check if pattern is contained in text (for multi-word patterns)
        if " " in pattern and any(word in text for word in pattern.split()):
            return True
        
        # Simple similarity calculation for single words
        if " " not in pattern:
            words = text.split()
            for word in words:
                similarity = self._calculate_similarity(pattern, word)
                if similarity >= threshold:
                    print(f"🔤 Fuzzy match: '{pattern}' ~ '{word}' (score: {similarity:.2f})")
                    return True
        
        return False

    def _calculate_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two words (0.0 to 1.0)"""
        # Simple implementation - you can use more sophisticated algorithms
        if word1 == word2:
            return 1.0
        
        # Length-based similarity
        len_similarity = 1 - abs(len(word1) - len(word2)) / max(len(word1), len(word2))
        
        # Character overlap
        set1, set2 = set(word1), set(word2)
        char_similarity = len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0
        
        # Combined score
        return (len_similarity + char_similarity) / 2


    def _handle_navigation(self, query: str) -> str:
        """Handle navigation commands with SPELL CORRECTION & FUZZY MATCHING"""
        # Correct spelling first
        corrected_query = self._correct_spelling(query)
        query_lower = corrected_query.lower().strip()
        
        print(f"🎯 Handling navigation (corrected): '{query_lower}'")
        
        # Define patterns with fuzzy matching
        patterns = {
            "next": {
                "keywords": ["next", "continue", "proceed", "go ahead", "move on"],
                "weight": 1.0
            },
            "previous": {
                "keywords": ["previous", "back", "go back", "return"],
                "weight": 1.0
            },
            "repeat": {
                "keywords": ["repeat", "again", "say again", "once more"],
                "weight": 1.0
            },
            "step_specific": {
                "keywords": ["step", "go to step", "jump to step"],
                "weight": 0.8
            }
        }
        
        # Calculate scores with fuzzy matching
        scores = {}
        for category, data in patterns.items():
            score = 0
            for keyword in data["keywords"]:
                if self._fuzzy_match(keyword, query_lower, threshold=0.6):  # Lower threshold for fuzzy
                    score += data["weight"]
                    # Bonus for better matches
                    if keyword in query_lower:  # Exact match
                        score += 0.3
            scores[category] = score
        
        # Find best category
        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]
        
        print(f"🎯 Best category: {best_category} (fuzzy score: {best_score})")
        
        # Proceed with reasonable confidence (lower threshold for fuzzy)
        if best_score >= 0.4:  # 40% threshold for fuzzy matching
            if best_category == "next":
                return self._handle_next_step()
            elif best_category == "previous":
                return self._handle_previous_step()
            elif best_category == "repeat":
                return self._handle_repeat()
            elif best_category == "step_specific":
                # Extract numbers even with spelling errors
                numbers = re.findall(r'\d+', query)
                if numbers:
                    step_num = int(numbers[0])
                    return self._go_to_specific_step(step_num)
        
        # Ultra-forgiving fallback for very short queries
        if len(query_lower) <= 6:  # Very short queries
            if any(self._fuzzy_match(word, query_lower, 0.5) for word in ["next", "yes", "ok", "go"]):
                return self._handle_next_step()
            elif any(self._fuzzy_match(word, query_lower, 0.5) for word in ["back", "prev"]):
                return self._handle_previous_step()
            elif any(self._fuzzy_match(word, query_lower, 0.5) for word in ["repeat", "again"]):
                return self._handle_repeat()
        
        return "I'm not sure what you want to do. Say 'next', 'previous', or 'repeat'."
    

    def _smells_like_navigation(self, query: str) -> bool:
            """Ultra-flexible detection for navigation-like queries"""
            if not self.state.current_recipe:
                return False
                
            query_lower = query.lower().strip()
            
            # Very short queries are probably navigation
            if len(query_lower) <= 10 and len(query_lower.split()) <= 3:
                return True
            
            # Common affirmative patterns
            affirmatives = ["yes", "ok", "sure", "go", "ready", "yep", "yeah", "alright", "fine"]
            if any(affirmative in query_lower for affirmative in affirmatives):
                return True
            
            # Question patterns about progress
            progress_questions = ["what now", "what do i do", "what should i do", "and now", "then what"]
            if any(question in query_lower for question in progress_questions):
                return True
            
            return False
    

    def start_specific_recipe(self, recipe_name: str) -> str:
        """Direct method to start a specific recipe by name - for button clicks"""
        print(f"🎯 Direct recipe start requested: {recipe_name}")
        
        # Find exact match
        for recipe_id, recipe in self.recipes.items():
            if recipe['name'].lower() == recipe_name.lower():
                return self._start_recipe(recipe_id, recipe)
        
        # If not found, try contains match
        for recipe_id, recipe in self.recipes.items():
            if recipe_name.lower() in recipe['name'].lower():
                return self._start_recipe(recipe_id, recipe)
        
        return f"Recipe '{recipe_name}' not found. Available recipes: {', '.join([r['name'] for r in self.recipes.values()])}"
    
   
    
    def get_statistics(self) -> Dict:
        """Get usage statistics"""
        total_queries = len([m for m in self.state.messages if m.role == "user"])
        avg_latency = 0.1  # Placeholder
        
        return {
            "total_queries": total_queries,
            "average_latency": f"{avg_latency:.2f}s",
            "current_recipe": self.state.current_recipe_name,
            "current_step": f"{self.state.current_step}/{self.state.total_steps}",
            "conversation_history": len(self.state.messages)
        }
    
    def process_query(self, query: str) -> Tuple[str, float]:
            """Process user query with COMPLETE enhanced handlers"""
            start_time = time.time()
            
            # Apply spell correction first to all queries
            corrected_query = self._correct_spelling(query)
            query_lower = corrected_query.lower().strip()
            
            print(f"🔍 Processing: '{query}' -> '{corrected_query}'")

            # HIGHEST PRIORITY: Handle substitution questions (ALWAYS USE LLM)
            substitution_response = self._handle_substitution_query(query)
            if substitution_response:
                latency = time.time() - start_time
                self._update_conversation_history(query, substitution_response)
                return substitution_response, latency

            # HIGH PRIORITY: Handle mixed intents
            mixed_response = self._handle_mixed_intent(query)
            if mixed_response:
                latency = time.time() - start_time
                self._update_conversation_history(query, mixed_response)
                return mixed_response, latency

            # HIGH PRIORITY: Handle recipe switching
            switch_response = self._handle_recipe_switch(query)
            if switch_response:
                latency = time.time() - start_time
                self._update_conversation_history(query, switch_response)
                return switch_response, latency

            # MEDIUM PRIORITY: Handle multi-step requests
            multi_step_response = self._handle_multi_step_request(query)
            if multi_step_response:
                latency = time.time() - start_time
                self._update_conversation_history(query, multi_step_response)
                return multi_step_response, latency

            # MEDIUM PRIORITY: Handle quantity adjustments
            quantity_response = self._handle_quantity_adjustment(query)
            if quantity_response:
                latency = time.time() - start_time
                self._update_conversation_history(query, quantity_response)
                return quantity_response, latency

            # MEDIUM PRIORITY: Handle equipment substitutions
            equipment_response = self._handle_equipment_substitution(query)
            if equipment_response:
                latency = time.time() - start_time
                self._update_conversation_history(query, equipment_response)
                return equipment_response, latency

            # NOW: Handle navigation commands with spell correction and fuzzy matching
            if self.state.current_recipe and self._is_clear_navigation(query):
                print("🎯 Detected clear navigation command")
                response = self._handle_navigation(query)
                latency = time.time() - start_time
                self._update_conversation_history(query, response)
                return response, latency

            # Handle recipe start commands with fuzzy matching
            start_patterns = ["start", "make", "cook", "begin", "recipe", "prepare", "bake", "i want to make"]
            if any(self._fuzzy_match(pattern, query_lower, threshold=0.6) for pattern in start_patterns) and not self.state.current_recipe:
                print("🎯 Detected start command")
                response = self._handle_start_recipe(query)
                latency = time.time() - start_time
                self._update_conversation_history(query, response)
                return response, latency

            # If we have active recipe, use RAG with recipe context
            if self.state.current_recipe:
                print("🎯 Using RAG with recipe context")
                context, sources = self.retrieve_context(query, k=3)
                
                if context:
                    response = self.llm.generate_with_history(
                        prompt=query,
                        context=context,
                        conversation_history=self.state.messages,
                        recipe_name=self.state.current_recipe_name
                    )
                else:
                    # Check if this is a general cooking question
                    cooking_questions = ["how", "what", "why", "when", "can i", "should i", "do i need"]
                    if any(self._fuzzy_match(q_word, query_lower, 0.5) for q_word in cooking_questions):
                        response = "I'm not sure about that specific question for this recipe. Try asking about ingredients, steps, timing, or cooking techniques."
                    else:
                        # Only show current step for very short queries that might be navigation
                        if len(query_lower.split()) <= 2:
                            recipe = self.recipes.get(self.state.current_recipe)
                            if self.state.current_step > 0:
                                current_step = recipe['steps'][self.state.current_step - 1]
                                response = f"Step {self.state.current_step}: {current_step['instruction']}"
                            else:
                                response = "Say 'next' to start the first step!"
                        else:
                            recipe = self.recipes.get(self.state.current_recipe)
                            response = f"I'm helping you with {recipe['name']}. Ask me about the recipe, ingredients, or say 'next' to continue!"
                
                latency = time.time() - start_time
                self._update_conversation_history(query, response)
                return response, latency

            # No active recipe - general cooking help
            print("🎯 General cooking query")
            context, sources = self.retrieve_context(query, k=3)
            
            if context:
                response = self.llm.generate_with_history(
                    prompt=query,
                    context=context,
                    conversation_history=self.state.messages,
                    recipe_name=None
                )
            else:
                # Suggest starting a recipe
                available = ', '.join([f'"{r["name"]}"' for r in self.recipes.values()])
                response = f"I'd love to help you cook! Available recipes: {available}. Try 'start pasta carbonara' or 'make chocolate chip cookies'."
            
            latency = time.time() - start_time
            self._update_conversation_history(query, response)
            return response, latency
    
    def chat(self, use_voice: bool = False):
        """Interactive chat interface"""
        print("\n" + "="*60)
        print("🍳 CookMate Voice Assistant (Improved RAG + History)")
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


# Main execution
if __name__ == "__main__":
    import sys
    
    print("\n🚀 Starting CookMate (Improved RAG Version)...\n")
    
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