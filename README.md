# Sentinelamai-agent(Autonomous Multi-Tool AI Agent)
Unlock autonomous workflow automation with this Python AI agent powered by Groq LLMs. Features dynamic tool calling, sandboxed code execution, web scraping, and safety controls with human-in-the-loop approvals. Built with multi-turn context memory and model fallbacks to execute complex, multi-step technical task

## 🚀 Features

* **Dynamic Tool Calling:** Evaluates arithmetic, fetches current time, manages files, and performs HTTP requests dynamically.
* **Sandboxed Execution:** Executes generated Python code inside an isolated `/workspace` directory.
* **Human-in-the-Loop Safeguards:** Features a safety approval prompt (`AUTO_APPROVE`) before performing risky write operations or running scripts.
* **Groq LLM Integration:** Automatically detects and falls back to available high-performance open models on Groq.
* **Interactive CLI Session:** Keeps conversation memory across multi-step user instructions.

---

## 📁 Project Structure

'''text
my-agent/
├── my-agent.py         # Main agent script & tool definitions
├── workspace/           # Isolated directory where files/scripts are created and run
├── .gitignore           # Excludes sensitive files & environment variables
├── LICENSE              # MIT License
└── README.md            # Project documentation
🛠️ Setup & Installation
1. Prerequisites
Ensure you have Python 3.10+ installed on your system.

2. Install Dependencies
Bash
pip install openai
3. API Key Configuration
Get a free API key from console.groq.com. Set it in your system environment or paste it into my-agent.py:

# Set as an environment variable (Recommended)
export GROQ_API_KEY="your_groq_api_key_here"
💻 Usage
Run Interactively
Launch the interactive CLI prompt:

Bash
python my-agent.py
Run Single Command
Pass your instruction directly in the terminal:

Bash
python my-agent.py "Write hello.py that prints today's date, then run it"
🛡️ License
Distributed under the MIT License.
