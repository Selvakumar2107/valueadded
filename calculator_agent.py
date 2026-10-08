"""
Calculator Agent - Agentic AI Practice
======================================
A lightweight, pure Python AI Agent that uses JSON tool definitions
and function-calling to perform arithmetic calculations.

No third-party agent frameworks (like LangChain or CrewAI) are used.
Built directly with the official Google GenAI SDK and Python standard library.
"""

import os
import sys
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus, unquote
import requests
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# 1. Configuration & API Key
# ---------------------------------------------------------------------------
DEFAULT_API_KEY = "AQ.Ab8RN6JxqoHwD6TmOIfCxE4bt6CL8HggxsfNcbylcHyGG2xiEw"
API_KEY = os.environ.get("GEMINI_API_KEY", DEFAULT_API_KEY)
# Multiple models provided so free-tier daily rate limits automatically fallback
AVAILABLE_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash"
]
CURRENT_MODEL = AVAILABLE_MODELS[0]

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    try:
        if sys.stdout.encoding != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr.encoding != "utf-8":
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Initialize Google GenAI Client
client = genai.Client(api_key=API_KEY)

# ---------------------------------------------------------------------------
# 2. Tool Definitions in JSON Schema Format (ReAct Pattern)
# ---------------------------------------------------------------------------
# In the ReAct (Reasoning + Acting) pattern, the LLM articulates its reasoning
# ('thought') before taking an action with structured inputs ('action input').

# Tool 1: Calculator
CALCULATOR_TOOL_JSON_SCHEMA = {
    "name": "calculator",
    "description": (
        "Performs arithmetic calculations between two numbers. "
        "Supports operations: 'add', 'subtract', 'multiply', 'divide', 'power', 'modulo'."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "thought": {
                "type": "STRING",
                "description": "ReAct Reasoning: Step-by-step internal thought explaining why this calculation is chosen and what values are being computed."
            },
            "operation": {
                "type": "STRING",
                "description": "The mathematical operation to execute.",
                "enum": ["add", "subtract", "multiply", "divide", "power", "modulo"]
            },
            "a": {
                "type": "NUMBER",
                "description": "The first number (left operand)."
            },
            "b": {
                "type": "NUMBER",
                "description": "The second number (right operand)."
            }
        },
        "required": ["thought", "operation", "a", "b"]
    }
}

# Tool 2: Web Search
WEB_SEARCH_TOOL_JSON_SCHEMA = {
    "name": "web_search",
    "description": (
        "Searches the web for up-to-date information, facts, real-world data, documentation, "
        "or questions that require external knowledge."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "thought": {
                "type": "STRING",
                "description": "ReAct Reasoning: Step-by-step internal thought explaining why search is needed and what specific information you are looking for."
            },
            "query": {
                "type": "STRING",
                "description": "The search query string to look up on the web."
            }
        },
        "required": ["thought", "query"]
    }
}

# Tool 3: Date & Time
DATE_TIME_TOOL_JSON_SCHEMA = {
    "name": "date_time",
    "description": (
        "Provides current date and time information, or performs date calculations "
        "such as adding/subtracting days from a date, calculating days between two dates, "
        "or determining the day of the week."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "thought": {
                "type": "STRING",
                "description": "ReAct Reasoning: Step-by-step internal thought explaining why date/time information or calculation is needed."
            },
            "action": {
                "type": "STRING",
                "description": (
                    "Operation to perform: 'get_current' (get current date, time, weekday), "
                    "'add_days' (add or subtract N days from a base date), "
                    "'date_difference' (calculate the number of days between two dates), "
                    "'day_of_week' (find the day of the week for any given date)."
                ),
                "enum": ["get_current", "add_days", "date_difference", "day_of_week"]
            },
            "base_date": {
                "type": "STRING",
                "description": "Base date in 'YYYY-MM-DD' format. If omitted or 'today', current date is used."
            },
            "days": {
                "type": "NUMBER",
                "description": "Number of days to add (positive) or subtract (negative) for 'add_days'."
            },
            "target_date": {
                "type": "STRING",
                "description": "Target date in 'YYYY-MM-DD' format for 'date_difference'."
            }
        },
        "required": ["thought", "action"]
    }
}

# Tool 4: Document Search (RAG / Document Retrieval)
DOCUMENT_SEARCH_TOOL_JSON_SCHEMA = {
    "name": "document_search",
    "description": (
        "Searches local PDF documents, databases, and files (such as 'Dummy_Student_Database_for_RAG.pdf') "
        "for student records, attendance, marks (Python, DSA, AI, Overall), placement status, contact details, "
        "or general document content and questions."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "thought": {
                "type": "STRING",
                "description": "ReAct Reasoning: Step-by-step internal thought explaining why document search is needed and what specific data to retrieve."
            },
            "query": {
                "type": "STRING",
                "description": "Search query or criteria (e.g. student name, ID like 'STU001', department like 'CSE', 'highest overall', 'attendance below 75%', 'placement eligible', 'AI above 90')."
            },
            "file_name": {
                "type": "STRING",
                "description": "Optional name or path of the document file to search. Defaults to 'Dummy_Student_Database_for_RAG.pdf'."
            }
        },
        "required": ["thought", "query"]
    }
}

# Convert JSON schemas to GenAI FunctionDeclarations
calculator_func_decl = types.FunctionDeclaration(
    name=CALCULATOR_TOOL_JSON_SCHEMA["name"],
    description=CALCULATOR_TOOL_JSON_SCHEMA["description"],
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "thought": types.Schema(
                type=types.Type.STRING,
                description="ReAct Reasoning: Step-by-step internal thought explaining why this calculation is chosen."
            ),
            "operation": types.Schema(
                type=types.Type.STRING,
                description="The mathematical operation to execute.",
                enum=["add", "subtract", "multiply", "divide", "power", "modulo"]
            ),
            "a": types.Schema(
                type=types.Type.NUMBER,
                description="The first number (left operand)."
            ),
            "b": types.Schema(
                type=types.Type.NUMBER,
                description="The second operand (right operand)."
            )
        },
        required=["thought", "operation", "a", "b"]
    )
)

web_search_func_decl = types.FunctionDeclaration(
    name=WEB_SEARCH_TOOL_JSON_SCHEMA["name"],
    description=WEB_SEARCH_TOOL_JSON_SCHEMA["description"],
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "thought": types.Schema(
                type=types.Type.STRING,
                description="ReAct Reasoning: Step-by-step internal thought explaining why search is needed."
            ),
            "query": types.Schema(
                type=types.Type.STRING,
                description="The search query string to look up on the web."
            )
        },
        required=["thought", "query"]
    )
)

date_time_func_decl = types.FunctionDeclaration(
    name=DATE_TIME_TOOL_JSON_SCHEMA["name"],
    description=DATE_TIME_TOOL_JSON_SCHEMA["description"],
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "thought": types.Schema(
                type=types.Type.STRING,
                description="ReAct Reasoning: Step-by-step internal thought explaining why date/time operation is needed."
            ),
            "action": types.Schema(
                type=types.Type.STRING,
                description="Operation: 'get_current', 'add_days', 'date_difference', 'day_of_week'",
                enum=["get_current", "add_days", "date_difference", "day_of_week"]
            ),
            "base_date": types.Schema(
                type=types.Type.STRING,
                description="Base date in 'YYYY-MM-DD' format."
            ),
            "days": types.Schema(
                type=types.Type.NUMBER,
                description="Number of days to add/subtract."
            ),
            "target_date": types.Schema(
                type=types.Type.STRING,
                description="Target date in 'YYYY-MM-DD' format."
            )
        },
        required=["thought", "action"]
    )
)

document_search_func_decl = types.FunctionDeclaration(
    name=DOCUMENT_SEARCH_TOOL_JSON_SCHEMA["name"],
    description=DOCUMENT_SEARCH_TOOL_JSON_SCHEMA["description"],
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "thought": types.Schema(
                type=types.Type.STRING,
                description="ReAct Reasoning: Step-by-step internal thought explaining why document search is needed."
            ),
            "query": types.Schema(
                type=types.Type.STRING,
                description="Search query, question, student name, ID, or criteria."
            ),
            "file_name": types.Schema(
                type=types.Type.STRING,
                description="Optional document file path (defaults to 'Dummy_Student_Database_for_RAG.pdf')."
            )
        },
        required=["thought", "query"]
    )
)

agent_tool = types.Tool(
    function_declarations=[
        calculator_func_decl,
        web_search_func_decl,
        date_time_func_decl,
        document_search_func_decl
    ]
)

agent_config = types.GenerateContentConfig(
    system_instruction=(
        "You are an AI Agent operating strictly under the ReAct (Reasoning + Acting) paradigm.\n\n"
        "### The ReAct Protocol:\n"
        "1. THOUGHT (Reasoning): Before executing ANY tool, you MUST formulate clear internal reasoning explaining why you need that tool and what you plan to accomplish. Provide this in the 'thought' field of your tool call.\n"
        "2. ACTION (Acting): Call the appropriate tool ('calculator', 'web_search', 'date_time', or 'document_search') with structured arguments.\n"
        "3. OBSERVATION: You will receive the tool result as JSON observation.\n"
        "4. REPEAT OR CONCLUDE: If more steps are needed, reason again and call another tool. Once you have all the facts, provide your clear Final Answer grounded in the observations."
    ),
    tools=[agent_tool],
    temperature=0.0
)

# ---------------------------------------------------------------------------
# 3. Tool Implementation (Python Logic)
# ---------------------------------------------------------------------------
def execute_calculator_tool(tool_input: dict) -> dict:
    """
    Executes the calculation based on the JSON input dictionary from the LLM.
    
    Expected input format:
    {
        "operation": "add" | "subtract" | "multiply" | "divide" | "power" | "modulo",
        "a": <number>,
        "b": <number>
    }
    """
    try:
        op = str(tool_input.get("operation", "")).strip().lower()
        a = float(tool_input["a"])
        b = float(tool_input["b"])
    except (KeyError, ValueError, TypeError) as err:
        return {
            "status": "error",
            "message": f"Invalid arguments received: {err}",
            "received_input": tool_input
        }

    if op == "add":
        result = a + b
    elif op == "subtract":
        result = a - b
    elif op == "multiply":
        result = a * b
    elif op == "divide":
        if b == 0:
            return {
                "status": "error",
                "message": "Division by zero is undefined.",
                "operation": op,
                "a": a,
                "b": b
            }
        result = a / b
    elif op == "power":
        try:
            result = math.pow(a, b)
        except OverflowError:
            return {
                "status": "error",
                "message": "Result caused an arithmetic overflow.",
                "operation": op,
                "a": a,
                "b": b
            }
    elif op == "modulo":
        if b == 0:
            return {
                "status": "error",
                "message": "Modulo by zero is undefined.",
                "operation": op,
                "a": a,
                "b": b
            }
        result = a % b
    else:
        return {
            "status": "error",
            "message": f"Unsupported operation '{op}'. Supported: add, subtract, multiply, divide, power, modulo."
        }

    # Format result: convert 5.0 to 5 for cleaner representation
    formatted_result = int(result) if result.is_integer() else result

    return {
        "status": "success",
        "operation": op,
        "a": int(a) if a.is_integer() else a,
        "b": int(b) if b.is_integer() else b,
        "result": formatted_result
    }


def execute_web_search_tool(tool_input: dict) -> dict:
    """
    Executes a web search based on the JSON input dictionary from the LLM.
    
    Expected input format:
    {
        "query": "<search query string>"
    }
    """
    query = str(tool_input.get("query", "")).strip()
    if not query:
        return {
            "status": "error",
            "message": "Search query cannot be empty.",
            "received_input": tool_input
        }

    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        resp = requests.post(url, headers=headers, data={"q": query}, timeout=10)
        html = resp.text

        titles_links = re.findall(
            r'<a class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            html,
            re.DOTALL
        )
        snippets = re.findall(
            r'<a class="result__snippet"[^>]*>(.*?)</a>',
            html,
            re.DOTALL
        )

        results = []
        max_results = 4
        count = min(len(titles_links), len(snippets), max_results)

        for i in range(count):
            link = titles_links[i][0]
            if "uddg=" in link:
                m = re.search(r'uddg=([^&]+)', link)
                if m:
                    link = unquote(m.group(1))

            title = re.sub(r'<[^>]+>', '', titles_links[i][1]).strip()
            snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip()
            results.append({
                "title": title,
                "snippet": snippet,
                "url": link
            })

        if not results and snippets:
            for i, snip in enumerate(snippets[:max_results]):
                snippet_text = re.sub(r'<[^>]+>', '', snip).strip()
                results.append({
                    "title": f"Result {i+1}",
                    "snippet": snippet_text,
                    "url": ""
                })

        if not results:
            return {
                "status": "success",
                "query": query,
                "results": [],
                "message": "No direct web search results found."
            }

        return {
            "status": "success",
            "query": query,
            "results": results
        }

    except Exception as err:
        return {
            "status": "error",
            "query": query,
            "message": f"Web search request failed: {err}"
        }


def execute_date_time_tool(tool_input: dict) -> dict:
    """
    Executes date/time operations based on the JSON input dictionary from the LLM.
    Pure Python datetime standard library - zero external frameworks.
    
    Expected input format:
    {
        "action": "get_current" | "add_days" | "date_difference" | "day_of_week",
        "base_date": "YYYY-MM-DD" (optional),
        "days": <number> (optional),
        "target_date": "YYYY-MM-DD" (optional)
    }
    """
    action = str(tool_input.get("action", "get_current")).strip().lower()
    now = datetime.now()

    def parse_date(date_str) -> datetime:
        if not date_str or str(date_str).strip().lower() in ("today", "now", "current"):
            return now
        date_str_clean = str(date_str).strip()
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(date_str_clean, fmt)
            except ValueError:
                continue
        raise ValueError(f"Could not parse date '{date_str}'. Expected format: YYYY-MM-DD")

    try:
        if action == "get_current":
            return {
                "status": "success",
                "action": "get_current",
                "current_date": now.strftime("%Y-%m-%d"),
                "current_time": now.strftime("%H:%M:%S"),
                "day_of_week": now.strftime("%A"),
                "formatted": now.strftime("%A, %B %d, %Y %H:%M:%S")
            }

        elif action == "add_days":
            base = parse_date(tool_input.get("base_date"))
            days_count = float(tool_input.get("days", 0))
            new_date = base + timedelta(days=days_count)
            return {
                "status": "success",
                "action": "add_days",
                "base_date": base.strftime("%Y-%m-%d"),
                "days_added": days_count,
                "resulting_date": new_date.strftime("%Y-%m-%d"),
                "day_of_week": new_date.strftime("%A"),
                "formatted": new_date.strftime("%A, %B %d, %Y")
            }

        elif action == "date_difference":
            base = parse_date(tool_input.get("base_date"))
            target = parse_date(tool_input.get("target_date"))
            diff_days = (target - base).days
            return {
                "status": "success",
                "action": "date_difference",
                "date1": base.strftime("%Y-%m-%d"),
                "date2": target.strftime("%Y-%m-%d"),
                "difference_days": diff_days,
                "absolute_days": abs(diff_days)
            }

        elif action == "day_of_week":
            base = parse_date(tool_input.get("base_date"))
            return {
                "status": "success",
                "action": "day_of_week",
                "date": base.strftime("%Y-%m-%d"),
                "day_of_week": base.strftime("%A")
            }

        else:
            return {
                "status": "error",
                "message": f"Unsupported action '{action}'. Supported: get_current, add_days, date_difference, day_of_week."
            }

    except Exception as err:
        return {
            "status": "error",
            "action": action,
            "message": f"Date/time operation failed: {err}"
        }


def execute_document_search_tool(tool_input: dict) -> dict:
    """
    Searches local documents (PDFs, text files, markdown) and extracts relevant information.
    Specifically parses tabular datasets like 'Dummy_Student_Database_for_RAG.pdf'
    for student records, filtering, and aggregate queries.
    
    Expected input format:
    {
        "query": <string>,
        "file_name": <string> (optional)
    }
    """
    query = str(tool_input.get("query", "")).strip()
    file_name = str(tool_input.get("file_name", "Dummy_Student_Database_for_RAG.pdf")).strip()
    if not query:
        return {"status": "error", "message": "Search query cannot be empty."}

    # Resolve document path
    doc_path = None
    candidates = [
        Path(file_name),
        Path(__file__).parent / file_name,
        Path(__file__).parent / "Dummy_Student_Database_for_RAG.pdf",
    ]
    for c in candidates:
        if c.is_file():
            doc_path = c
            break

    if not doc_path:
        return {
            "status": "error",
            "message": f"Document file '{file_name}' not found."
        }

    try:
        # Extract content based on file extension
        extracted_text = ""
        page_texts = []
        if doc_path.suffix.lower() == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(str(doc_path))
                for i, page in enumerate(reader.pages):
                    pt = page.extract_text() or ""
                    page_texts.append({"page": i + 1, "text": pt})
                    extracted_text += f"\n--- Page {i+1} ---\n" + pt
            except Exception as e:
                return {"status": "error", "message": f"Failed reading PDF: {e}"}
        else:
            with open(doc_path, "r", encoding="utf-8", errors="ignore") as f:
                extracted_text = f.read()
            page_texts.append({"page": 1, "text": extracted_text})

        # Try parsing structured student records if present
        records = []
        lines = [line.strip() for line in extracted_text.splitlines() if line.strip()]
        i = 0
        while i < len(lines):
            if re.match(r"^STU\d{3}$", lines[i]):
                try:
                    rec = {
                        "student_id": lines[i],
                        "name": lines[i+1],
                        "department": lines[i+2],
                        "year": lines[i+3],
                        "email": lines[i+4],
                        "phone": lines[i+5],
                        "attendance": lines[i+6],
                        "python": int(lines[i+7]),
                        "dsa": int(lines[i+8]),
                        "ai": int(lines[i+9]),
                        "overall": int(lines[i+10]),
                        "placement": lines[i+11]
                    }
                    records.append(rec)
                    i += 12
                    continue
                except (IndexError, ValueError):
                    pass
            i += 1

        q_lower = query.lower()

        # Handle student database structured queries
        if records:
            matched_records = []

            # 1. Check for specific student name / ID in query
            for r in records:
                if r["student_id"].lower() in q_lower:
                    matched_records.append(r)
                elif any(name_token.lower() in q_lower for name_token in r["name"].split() if len(name_token) > 2):
                    matched_records.append(r)

            # 2. Check for department filtering (e.g. CSE, AIDS, IT, ECE, EEE)
            dept_matches = re.findall(r"\b(cse|aids|it|ece|eee)\b", q_lower)
            if dept_matches and not matched_records:
                target_depts = [d.upper() for d in dept_matches]
                matched_records = [r for r in records if r["department"].upper() in target_depts]

            # 3. Handle criteria keywords
            if "highest overall" in q_lower or "top overall" in q_lower:
                top_student = max(records, key=lambda x: x["overall"])
                return {
                    "status": "success",
                    "document": doc_path.name,
                    "query": query,
                    "criterion": "Highest Overall Mark",
                    "top_student": top_student,
                    "all_records_count": len(records)
                }

            if "highest attendance" in q_lower or "top attendance" in q_lower:
                top_att = max(records, key=lambda x: int(x["attendance"].replace("%", "")))
                return {
                    "status": "success",
                    "document": doc_path.name,
                    "query": query,
                    "criterion": "Highest Attendance",
                    "top_student": top_att,
                    "all_records_count": len(records)
                }

            if "below 75" in q_lower or "< 75" in q_lower:
                low_att = [r for r in records if int(r["attendance"].replace("%", "")) < 75]
                return {
                    "status": "success",
                    "document": doc_path.name,
                    "query": query,
                    "criterion": "Attendance below 75%",
                    "count": len(low_att),
                    "matching_students": low_att
                }

            if "above 90 in ai" in q_lower or "ai > 90" in q_lower or "score above 90 in ai" in q_lower:
                high_ai = [r for r in records if r["ai"] > 90]
                return {
                    "status": "success",
                    "document": doc_path.name,
                    "query": query,
                    "criterion": "AI Score > 90",
                    "count": len(high_ai),
                    "matching_students": high_ai
                }

            if "placement eligible" in q_lower or "eligible for placement" in q_lower or "placement eligibility" in q_lower:
                eligible = [r for r in records if r["placement"].lower() == "eligible"]
                not_eligible = [r for r in records if r["placement"].lower() == "not eligible"]
                return {
                    "status": "success",
                    "document": doc_path.name,
                    "query": query,
                    "criterion": "Placement Eligibility",
                    "eligible_count": len(eligible),
                    "not_eligible_count": len(not_eligible),
                    "total_students": len(records),
                    "eligible_students": [f"{r['student_id']}: {r['name']} ({r['department']})" for r in eligible]
                }

            if matched_records:
                return {
                    "status": "success",
                    "document": doc_path.name,
                    "query": query,
                    "total_matches": len(matched_records),
                    "records": matched_records
                }

        # General text matching across pages/lines
        matching_snippets = []
        words = [w for w in re.split(r"\W+", q_lower) if len(w) > 2]
        for p in page_texts:
            for line in p["text"].splitlines():
                line_lower = line.lower()
                if any(w in line_lower for w in words):
                    matching_snippets.append({"page": p["page"], "text": line.strip()})

        return {
            "status": "success",
            "document": doc_path.name,
            "query": query,
            "total_matches": len(matching_snippets),
            "snippets": matching_snippets[:10],
            "records_available": len(records)
        }

    except Exception as err:
        return {
            "status": "error",
            "query": query,
            "document": file_name,
            "message": f"Document search failed: {err}"
        }

# ---------------------------------------------------------------------------
# 4. Agentic Interaction Loop
# ---------------------------------------------------------------------------
class CalculatorAgent:
    """
    The Agent core that manages conversation history, LLM interaction,
    tool call detection, JSON parameter passing, and tool execution.
    """

    def __init__(self):
        self.history = []

    def reset(self):
        """Reset conversation memory."""
        self.history = []

    def run(self, user_prompt: str) -> str:
        """
        Processes a single user turn through the complete Agentic AI lifecycle:
        1. User Input -> LLM
        2. LLM evaluates and decides whether to emit a Tool Call (in JSON format)
        3. Agent intercepts Tool Call, logs JSON input, and invokes tool
        4. Tool produces JSON result
        5. Agent sends JSON result back to LLM
        6. LLM formulates final answer
        """
        # Append user message
        self.history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_prompt)]
            )
        )

        max_tool_steps = 5  # Guard against infinite agent tool loops
        step = 0

        while step < max_tool_steps:
            step += 1

            # Call the LLM with automatic fallback across available models if quota is hit
            global CURRENT_MODEL
            response = None
            last_err = None
            for model_name in [CURRENT_MODEL] + [m for m in AVAILABLE_MODELS if m != CURRENT_MODEL]:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=self.history,
                        config=agent_config
                    )
                    CURRENT_MODEL = model_name
                    break
                except Exception as e:
                    last_err = e
                    err_str = str(e)
                    transient_errors = [
                        "RESOURCE_EXHAUSTED", "429", "503", "504", "UNAVAILABLE", 
                        "high demand", "NOT_FOUND", "DEADLINE_EXCEEDED", "timeout", "unreachable"
                    ]
                    if any(x in err_str for x in transient_errors):
                        continue
                    raise

            if response is None:
                raise last_err

            candidate = response.candidates[0]
            candidate_content = candidate.content
            self.history.append(candidate_content)

            # Check if LLM requested one or more tool calls
            tool_calls = [
                part.function_call
                for part in candidate_content.parts
                if part.function_call is not None
            ]

            if not tool_calls:
                # LLM finished reasoning and produced final textual answer
                return response.text or ""

            # Process each requested tool call
            tool_response_parts = []
            for fc in tool_calls:
                tool_name = fc.name
                tool_args = dict(fc.args)

                # Extract the ReAct reasoning (Thought) from the arguments
                react_thought = tool_args.pop("thought", "Evaluating action required for user request...")

                print("\n" + "=" * 68)
                print("🧠 [ReAct: THOUGHT / REASON]:")
                print(f"   {react_thought}")
                print(f"⚡ [ReAct: ACTION]: {tool_name}")
                print("📥 [ReAct: ACTION INPUT JSON]:")
                print(json.dumps(tool_args, indent=2))
                print("-" * 68)

                # Execute requested tool
                if tool_name == "calculator":
                    output = execute_calculator_tool(tool_args)
                elif tool_name == "web_search":
                    output = execute_web_search_tool(tool_args)
                elif tool_name == "date_time":
                    output = execute_date_time_tool(tool_args)
                elif tool_name == "document_search":
                    output = execute_document_search_tool(tool_args)
                else:
                    output = {"status": "error", "message": f"Unknown tool: {tool_name}"}

                print("👁️ [ReAct: OBSERVATION JSON]:")
                print(json.dumps(output, indent=2))
                print("=" * 68 + "\n")

                # Prepare function response part for the LLM
                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": output}
                    )
                )

            # Send tool response back to the LLM
            self.history.append(
                types.Content(
                    role="user",
                    parts=tool_response_parts
                )
            )

        return "Agent reached maximum tool step limit."


# ---------------------------------------------------------------------------
# 5. Interactive CLI
# ---------------------------------------------------------------------------
def main():
    agent = CalculatorAgent()

    print("=" * 68)
    print("   🧠 ReAct AGENT (Reason + Action) - ZERO FRAMEWORK")
    print("=" * 68)
    print(f"• Pattern        : ReAct (Thought -> Action -> Observation -> Final Answer)")
    print(f"• LLM Model      : {CURRENT_MODEL}")
    print("• Framework      : Pure Python + Google GenAI SDK (Zero Agent Frameworks)")
    print("• Tools Available: 1. 'calculator'   2. 'web_search'   3. 'date_time'   4. 'document_search'")
    print("• Commands       : 'exit'/'quit' to leave | 'clear' to reset | 'schema' to view schemas")
    print("=" * 68 + "\n")

    while True:
        try:
            user_input = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            agent.reset()
            print("🧹 Conversation memory cleared.\n")
            continue

        if user_input.lower() == "schema":
            print("\n📋 1. Calculator Tool JSON Schema:")
            print(json.dumps(CALCULATOR_TOOL_JSON_SCHEMA, indent=2))
            print("\n📋 2. Web Search Tool JSON Schema:")
            print(json.dumps(WEB_SEARCH_TOOL_JSON_SCHEMA, indent=2))
            print("\n📋 3. Date & Time Tool JSON Schema:")
            print(json.dumps(DATE_TIME_TOOL_JSON_SCHEMA, indent=2))
            print("\n📋 4. Document Search Tool JSON Schema:")
            print(json.dumps(DOCUMENT_SEARCH_TOOL_JSON_SCHEMA, indent=2))
            print()
            continue

        print("\nThinking...")
        try:
            answer = agent.run(user_input)
            print(f"Agent > {answer}\n")
        except Exception as e:
            print(f"❌ Error occurred: {e}\n")


if __name__ == "__main__":
    main()
