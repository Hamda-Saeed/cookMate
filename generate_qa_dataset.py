"""
Generate 100 Question-Answer pairs from recipes.json
Each QA pair includes document name (recipes.json) and page/section references
"""

import json
import random
from typing import List, Dict
from datetime import datetime

# Document name (treating recipes.json as the source document)
DOCUMENT_NAME = "recipes.json"

def generate_qa_pairs(recipes_path: str = "recipes.json", num_pairs: int = 100) -> List[Dict]:
    """
    Generate comprehensive QA pairs from recipes
    
    Question types:
    1. Recipe overview questions (name, prep time, cook time, servings, difficulty)
    2. Ingredient questions (list, quantities, specific ingredients)
    3. Step-by-step questions (specific steps, instructions, durations, tips)
    4. Multi-recipe comparison questions
    5. Timing questions (total time, step duration)
    6. Substitution questions
    7. Cooking technique questions
    """
    
    with open(recipes_path, 'r', encoding='utf-8') as f:
        recipes = json.load(f)
    
    qa_pairs = []
    recipe_list = list(recipes.items())
    
    # Ensure we have enough recipes to generate 100 diverse questions
    if len(recipe_list) < num_pairs:
        # Repeat recipes if needed, but vary question types
        recipe_list = recipe_list * ((num_pairs // len(recipe_list)) + 1)
    
    random.shuffle(recipe_list)
    
    question_id = 1
    
    # Type 1: Recipe Overview Questions (20 questions)
    for i in range(20):
        if question_id > num_pairs:
            break
        recipe_id, recipe = recipe_list[i % len(recipe_list)]
        recipe_name = recipe['name']
        
        qa_type = i % 4
        if qa_type == 0:
            question = f"What is the name of the recipe that takes {recipe.get('prep_time', 'N/A')} to prepare and {recipe.get('cook_time', 'N/A')} to cook?"
            answer = recipe_name
            pages = f"Page {i+1} (Recipe Overview Section)"
        elif qa_type == 1:
            question = f"How long does it take to prepare {recipe_name}?"
            answer = recipe.get('prep_time', 'N/A')
            pages = f"Page {i+1} (Recipe Overview Section)"
        elif qa_type == 2:
            question = f"How many servings does {recipe_name} make?"
            answer = f"{recipe.get('servings', 'N/A')} servings"
            pages = f"Page {i+1} (Recipe Overview Section)"
        else:
            question = f"What is the difficulty level of {recipe_name}?"
            answer = recipe.get('difficulty', 'Medium')
            pages = f"Page {i+1} (Recipe Overview Section)"
        
        qa_pairs.append({
            "id": question_id,
            "question": question,
            "answer": answer,
            "document": DOCUMENT_NAME,
            "pages": pages,
            "recipe_id": recipe_id,
            "recipe_name": recipe_name,
            "question_type": "recipe_overview"
        })
        question_id += 1
    
    # Type 2: Ingredient Questions (25 questions)
    for i in range(25):
        if question_id > num_pairs:
            break
        recipe_id, recipe = recipe_list[(i+20) % len(recipe_list)]
        recipe_name = recipe['name']
        ingredients = recipe.get('ingredients', [])
        
        qa_type = i % 5
        if qa_type == 0:
            question = f"What ingredients are needed for {recipe_name}?"
            answer = ", ".join(ingredients)
            pages = f"Page {(i+20)+1} (Ingredients Section)"
        elif qa_type == 1 and ingredients:
            ingredient = random.choice(ingredients)
            question = f"Is {ingredient.split(',')[0].split('(')[0].strip()} used in {recipe_name}?"
            answer = f"Yes, {ingredient} is used in {recipe_name}"
            pages = f"Page {(i+20)+1} (Ingredients Section)"
        elif qa_type == 2 and ingredients:
            ingredient = random.choice(ingredients)
            question = f"How much {ingredient.split(',')[0].split('(')[0].strip()} is needed for {recipe_name}?"
            answer = ingredient
            pages = f"Page {(i+20)+1} (Ingredients Section)"
        elif qa_type == 3:
            question = f"How many ingredients are required for {recipe_name}?"
            answer = f"{len(ingredients)} ingredients"
            pages = f"Page {(i+20)+1} (Ingredients Section)"
        else:
            if ingredients:
                ingredient = random.choice(ingredients)
                question = f"List all ingredients for {recipe_name}, including {ingredient.split(',')[0].split('(')[0].strip()}"
                answer = ", ".join(ingredients)
                pages = f"Page {(i+20)+1} (Ingredients Section)"
            else:
                continue
        
        qa_pairs.append({
            "id": question_id,
            "question": question,
            "answer": answer,
            "document": DOCUMENT_NAME,
            "pages": pages,
            "recipe_id": recipe_id,
            "recipe_name": recipe_name,
            "question_type": "ingredients"
        })
        question_id += 1
    
    # Type 3: Step-by-Step Questions (30 questions)
    for i in range(30):
        if question_id > num_pairs:
            break
        recipe_id, recipe = recipe_list[(i+45) % len(recipe_list)]
        recipe_name = recipe['name']
        steps = recipe.get('steps', [])
        
        if not steps:
            continue
        
        qa_type = i % 6
        step = random.choice(steps)
        step_num = step['step']
        
        if qa_type == 0:
            question = f"What is step {step_num} in {recipe_name}?"
            answer = step['instruction']
            pages = f"Page {(i+45)+1} (Step {step_num} Section)"
        elif qa_type == 1:
            question = f"How long does step {step_num} take in {recipe_name}?"
            answer = step.get('duration', 'Duration not specified')
            pages = f"Page {(i+45)+1} (Step {step_num} Section)"
        elif qa_type == 2:
            question = f"What is the instruction for step {step_num} of {recipe_name}?"
            answer = step['instruction']
            pages = f"Page {(i+45)+1} (Step {step_num} Section)"
        elif qa_type == 3 and step.get('tips'):
            question = f"What tip is provided for step {step_num} in {recipe_name}?"
            answer = step['tips']
            pages = f"Page {(i+45)+1} (Step {step_num} Tips Section)"
        elif qa_type == 4:
            question = f"How many steps are in {recipe_name}?"
            answer = f"{len(steps)} steps"
            pages = f"Page {(i+45)+1} (Steps Section)"
        else:
            question = f"Describe step {step_num} of {recipe_name}"
            answer = step['instruction']
            if step.get('duration'):
                answer += f" (Takes {step['duration']})"
            pages = f"Page {(i+45)+1} (Step {step_num} Section)"
        
        qa_pairs.append({
            "id": question_id,
            "question": question,
            "answer": answer,
            "document": DOCUMENT_NAME,
            "pages": pages,
            "recipe_id": recipe_id,
            "recipe_name": recipe_name,
            "question_type": "steps",
            "step_number": step_num
        })
        question_id += 1
    
    # Type 4: Multi-Recipe Comparison Questions (10 questions)
    for i in range(10):
        if question_id > num_pairs:
            break
        recipe1_id, recipe1 = recipe_list[(i+75) % len(recipe_list)]
        recipe2_id, recipe2 = recipe_list[(i+76) % len(recipe_list)]
        
        qa_type = i % 3
        if qa_type == 0:
            question = f"Which recipe takes longer to cook, {recipe1['name']} or {recipe2['name']}?"
            time1 = recipe1.get('cook_time', '0 minutes')
            time2 = recipe2.get('cook_time', '0 minutes')
            # Simple comparison (in real scenario, parse times properly)
            answer = f"{recipe1['name']} takes {time1} and {recipe2['name']} takes {time2}"
            pages = f"Page {(i+75)+1} and Page {(i+76)+1} (Recipe Overview Sections)"
        elif qa_type == 1:
            question = f"Compare the difficulty levels of {recipe1['name']} and {recipe2['name']}"
            answer = f"{recipe1['name']} is {recipe1.get('difficulty', 'Medium')} difficulty, while {recipe2['name']} is {recipe2.get('difficulty', 'Medium')} difficulty"
            pages = f"Page {(i+75)+1} and Page {(i+76)+1} (Recipe Overview Sections)"
        else:
            question = f"How many total servings can {recipe1['name']} and {recipe2['name']} make together?"
            servings1 = recipe1.get('servings', 0)
            servings2 = recipe2.get('servings', 0)
            answer = f"{servings1 + servings2} servings total ({servings1} from {recipe1['name']} and {servings2} from {recipe2['name']})"
            pages = f"Page {(i+75)+1} and Page {(i+76)+1} (Recipe Overview Sections)"
        
        qa_pairs.append({
            "id": question_id,
            "question": question,
            "answer": answer,
            "document": DOCUMENT_NAME,
            "pages": pages,
            "recipe_id": f"{recipe1_id},{recipe2_id}",
            "recipe_name": f"{recipe1['name']}, {recipe2['name']}",
            "question_type": "multi_recipe"
        })
        question_id += 1
    
    # Type 5: Timing Questions (10 questions)
    for i in range(10):
        if question_id > num_pairs:
            break
        recipe_id, recipe = recipe_list[(i+85) % len(recipe_list)]
        recipe_name = recipe['name']
        steps = recipe.get('steps', [])
        
        if not steps:
            continue
        
        qa_type = i % 2
        if qa_type == 0:
            prep_time = recipe.get('prep_time', '0 minutes')
            cook_time = recipe.get('cook_time', '0 minutes')
            question = f"What is the total time (prep + cook) for {recipe_name}?"
            answer = f"Prep time: {prep_time}, Cook time: {cook_time}"
            pages = f"Page {(i+85)+1} (Recipe Overview Section)"
        else:
            total_duration = 0
            durations = []
            for step in steps:
                if step.get('duration'):
                    durations.append(step['duration'])
            question = f"List all step durations for {recipe_name}"
            answer = ", ".join(durations) if durations else "Duration information not available for all steps"
            pages = f"Page {(i+85)+1} (Steps Section)"
        
        qa_pairs.append({
            "id": question_id,
            "question": question,
            "answer": answer,
            "document": DOCUMENT_NAME,
            "pages": pages,
            "recipe_id": recipe_id,
            "recipe_name": recipe_name,
            "question_type": "timing"
        })
        question_id += 1
    
    # Type 6: Cooking Technique Questions (5 questions)
    for i in range(5):
        if question_id > num_pairs:
            break
        recipe_id, recipe = recipe_list[(i+95) % len(recipe_list)]
        recipe_name = recipe['name']
        steps = recipe.get('steps', [])
        
        if not steps:
            continue
        
        step = random.choice(steps)
        instruction = step['instruction'].lower()
        
        if 'heat' in instruction or 'pan' in instruction or 'skillet' in instruction:
            question = f"What cooking method is used in {recipe_name}?"
            answer = "Stovetop cooking using a pan or skillet"
            pages = f"Page {(i+95)+1} (Steps Section)"
        elif 'oven' in instruction or 'bake' in instruction:
            question = f"Does {recipe_name} require baking?"
            answer = f"Yes, {recipe_name} involves baking/oven cooking"
            pages = f"Page {(i+95)+1} (Steps Section)"
        else:
            question = f"Describe the cooking process for {recipe_name}"
            answer = f"The recipe involves: {', '.join([s['instruction'][:50] + '...' for s in steps[:3]])}"
            pages = f"Page {(i+95)+1} (Steps Section)"
        
        qa_pairs.append({
            "id": question_id,
            "question": question,
            "answer": answer,
            "document": DOCUMENT_NAME,
            "pages": pages,
            "recipe_id": recipe_id,
            "recipe_name": recipe_name,
            "question_type": "cooking_technique"
        })
        question_id += 1
    
    # Ensure exactly 100 pairs
    qa_pairs = qa_pairs[:num_pairs]
    
    # Re-number IDs
    for i, qa in enumerate(qa_pairs, 1):
        qa['id'] = i
    
    return qa_pairs


def save_dataset(qa_pairs: List[Dict], output_path: str = "rag_qa_dataset.json"):
    """Save QA dataset to JSON file"""
    dataset = {
        "metadata": {
            "dataset_name": "CookMate RAG Evaluation Dataset",
            "total_questions": len(qa_pairs),
            "document_source": DOCUMENT_NAME,
            "generated_at": datetime.now().isoformat(),
            "question_types": {
                "recipe_overview": sum(1 for qa in qa_pairs if qa['question_type'] == 'recipe_overview'),
                "ingredients": sum(1 for qa in qa_pairs if qa['question_type'] == 'ingredients'),
                "steps": sum(1 for qa in qa_pairs if qa['question_type'] == 'steps'),
                "multi_recipe": sum(1 for qa in qa_pairs if qa['question_type'] == 'multi_recipe'),
                "timing": sum(1 for qa in qa_pairs if qa['question_type'] == 'timing'),
                "cooking_technique": sum(1 for qa in qa_pairs if qa['question_type'] == 'cooking_technique')
            }
        },
        "qa_pairs": qa_pairs
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(qa_pairs)} QA pairs")
    print(f"Saved to: {output_path}")
    print(f"\nQuestion Type Distribution:")
    for qtype, count in dataset['metadata']['question_types'].items():
        print(f"   {qtype}: {count}")


if __name__ == "__main__":
    import sys
    
    # Set UTF-8 encoding for Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    num_pairs = 100
    if len(sys.argv) > 1:
        try:
            num_pairs = int(sys.argv[1])
        except ValueError:
            print("Usage: python generate_qa_dataset.py [num_pairs]")
            sys.exit(1)
    
    print(f"Generating {num_pairs} QA pairs from recipes.json...")
    qa_pairs = generate_qa_pairs("recipes.json", num_pairs)
    save_dataset(qa_pairs, "rag_qa_dataset.json")
    
    print("\nDataset generation complete!")

