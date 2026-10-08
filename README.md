# 🧠 valueadded - ReAct AI Agent (Reason + Action)

A pure-Python implementation of an Agentic AI system operating under the foundational **ReAct (Reasoning + Acting)** pattern (Yao et al., 2022). Built directly with standard Python libraries and the Google GenAI SDK—with **zero third-party agent frameworks** (no LangChain, no CrewAI, no AutoGen).

---

## 🔬 What is the ReAct Pattern?

**ReAct** combines **Reasoning** (chain-of-thought) with **Acting** (tool execution). Instead of blindly invoking tools or attempting to do everything in its head, the agent interleaves reasoning and execution:

```text
User Request
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. 🧠 THOUGHT (Reasoning)                                   │
│    "I need to find today's date first before adding days."  │
│                                                             │
│ 2. ⚡ ACTION (Tool) + 📥 ACTION INPUT (JSON)                │
│    Calls 'date_time' with action='get_current'              │
│                                                             │
│ 3. 👁️ OBSERVATION (Environment Feedback)                    │
│    Receives {"current_date": "2026-10-08", "day": "Thursday"}│
│                                                             │
│ 4. 🧠 THOUGHT (Reasoning)                                   │
│    "Now I need to add 10 days to 2026-10-08."               │
│                                                             │
│ 5. ⚡ ACTION (Tool) + 📥 ACTION INPUT (JSON)                │
│    Calls 'date_time' with action='add_days', days=10        │
│                                                             │
│ 6. 👁️ OBSERVATION (Environment Feedback)                    │
│    Receives {"resulting_date": "2026-10-18", "day": "Sunday"}│
│                                                             │
│ 7. 🏁 FINAL ANSWER                                          │
│    "In 10 days from today, it will be Sunday, Oct 18, 2026."│
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ The 4 Tools Available

All tools are declared in standard JSON Schema format and require an explicit `thought` reasoning property:

1. **`calculator`**: Arithmetic operations (`add`, `subtract`, `multiply`, `divide`, `power`, `modulo`).
2. **`web_search`**: Live web searches for real-time information, documentation, and external facts.
3. **`date_time`**: Current date, current time, day of the week, adding/subtracting days, and date differences.
4. **`document_search`**: PDF & document retrieval (such as `Dummy_Student_Database_for_RAG.pdf`) for student records, attendance, marks, placement eligibility, and text lookups.

---

## 🚀 How to Run

Launch the interactive terminal CLI:

```powershell
python calculator_agent.py
```

### Interactive Commands:
- `schema` : Displays the full JSON Schemas for all 4 tools.
- `clear`  : Clears conversational memory.
- `exit` / `quit` : Exits the CLI.

---

## 💡 Example Prompts to Try

1. **Document Search & RAG Retrieval:**
   ```text
   You > What is Priya Sharma's attendance and overall mark in the document?
   You > Which student has the highest overall mark?
   You > How many students are eligible for placement in the student database?
   ```

2. **Multi-Step Date Reasoning:**
   ```text
   You > What day of the week will it be 10 days from today?
   ```
   *Watch the agent reason, get the current date, calculate +10 days, and formulate the answer.*

3. **Math with Reasoning:**
   ```text
   You > What is 35 multiplied by 18?
   ```

4. **Web Search with Reasoning:**
   ```text
   You > Who created the Python programming language?
   ```

5. **Chained Multi-Tool Reasoning (Search + Math):**
   ```text
   You > What is the speed of light in km/s, and how far does it travel in 5 seconds?
   ```
