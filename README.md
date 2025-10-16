# 🍳 CookMate — AI Cooking Assistant

CookMate is an intelligent recipe and cooking assistant powered by a Retrieval-Augmented Generation (RAG) model.  
It helps users generate, understand, and follow recipes in a guided, conversational way.

---

## 🧰 Requirements

Before you start, make sure you have the correct Python version installed.

### ✅ Step 1: Install Python 3.11.9
CookMate is compatible with **Python 3.11.9**.  
You can download it directly from the official website:

👉 [Download Python 3.11.9](https://www.python.org/downloads/release/python-3119/)

Make sure to check the box **"Add Python to PATH"** during installation.

---

## ⚙️ Setup Instructions

### Step 2: Create and Activate a Virtual Environment

Open **PowerShell** or **Command Prompt** inside the project folder by pressing ctrl+` basically the terminal window (e.g., `C:\Users\User\Desktop\cook`).

Run the following commands:

```bash
# Create a virtual environment
python -m venv cook_env

# (For PowerShell) Allow running scripts
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# Activate the virtual environment
cook_env\Scripts\activate

Once the virtual environment is active, install all dependencies from requirements.txt:

pip install -r requirements.txt

Step 4: Add Your Groq API Key

Open the file cookmate_rag.py 
You’ll find a line like this near the top:

 api_key = os.getenv(key)


Replace "key" with your actual Groq API key, for example:

 api_key = os.getenv('actual key here')
Step 5: Run the Backend

Run the following command inside your activated environment:

python cookmate_rag.py

Step 6: Run the Streamlit App

Now start the Streamlit interface by running:

streamlit run streamlit_app.py


This will launch the CookMate web interface in your browser.

🧩 Troubleshooting

streamlit : command not found
→ Use python -m streamlit run streamlit_app.py

chromadb or numpy errors
→ Try upgrading your dependencies:

pip install --upgrade chromadb numpy


Permission error on activation
→ Run PowerShell as Administrator and reapply:

Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
