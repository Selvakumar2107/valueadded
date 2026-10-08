"""
LangChain Multi-Task Parallel Agent
Runs MULTIPLE tasks SIMULTANEOUSLY using asyncio.gather().
Tools: calculator, web_search, date_time
Framework: LangChain + LangGraph ReAct
"""

import os, sys, math, asyncio, re
from datetime import datetime, timedelta
from urllib.parse import quote_plus, unquote

import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

API_KEY = os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6JxqoHwD6TmOIfCxE4bt6CL8HggxsfNcbylcHyGG2xiEw")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
]

def make_llm(model):
    return ChatGoogleGenerativeAI(model=model, google_api_key=API_KEY, temperature=0.0)

@tool
def calculator(operation: str, a: float, b: float) -> dict:
    """Perform arithmetic: add, subtract, multiply, divide, power, modulo."""
    op = operation.strip().lower()
    try:
        if op == "add":        result = a + b
        elif op == "subtract": result = a - b
        elif op == "multiply": result = a * b
        elif op == "divide":
            if b == 0: return {"status": "error", "message": "Division by zero"}
            result = a / b
        elif op == "power":    result = math.pow(a, b)
        elif op == "modulo":
            if b == 0: return {"status": "error", "message": "Modulo by zero"}
            result = a % b
        else:
            return {"status": "error", "message": f"Unknown op: {op}"}
    except OverflowError:
        return {"status": "error", "message": "Arithmetic overflow"}
    res = int(result) if isinstance(result, float) and result.is_integer() else result
    return {"status": "success", "operation": op, "a": a, "b": b, "result": res}

@tool
def web_search(query: str) -> dict:
    """Search the web using DuckDuckGo for facts, news, or documentation."""
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    hdrs = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"}
    try:
        resp  = requests.post(url, headers=hdrs, data={"q": query}, timeout=10)
        html  = resp.text
        links = re.findall(r'<a class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', html, re.DOTALL)
        snips = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
        out, n = [], min(len(links), len(snips), 4)
        for i in range(n):
            lnk = links[i][0]
            m = re.search(r"uddg=([^&]+)", lnk)
            if m: lnk = unquote(m.group(1))
            out.append({
                "title":   re.sub(r"<[^>]+>", "", links[i][1]).strip(),
                "snippet": re.sub(r"<[^>]+>", "", snips[i]).strip(),
                "url":     lnk,
            })
        return {"status": "success", "query": query, "results": out}
    except Exception as err:
        return {"status": "error", "query": query, "message": str(err)}

@tool
def date_time(action: str, base_date: str = "today", days: float = 0, target_date: str = "") -> dict:
    """Date/time ops: get_current, add_days, date_difference, day_of_week."""
    now = datetime.now()
    def parse(s):
        if not s or s.strip().lower() in ("today", "now", "", "current"): return now
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"):
            try: return datetime.strptime(s.strip(), fmt)
            except ValueError: pass
        raise ValueError(f"Cannot parse: {s}")
    try:
        a = action.strip().lower()
        if a == "get_current":
            return {"status": "success", "date": now.strftime("%Y-%m-%d"),
                    "time": now.strftime("%H:%M:%S"), "day": now.strftime("%A"),
                    "formatted": now.strftime("%A, %B %d, %Y %H:%M:%S")}
        elif a == "add_days":
            base = parse(base_date); nd = base + timedelta(days=days)
            return {"status": "success", "base": base.strftime("%Y-%m-%d"),
                    "days": days, "result": nd.strftime("%Y-%m-%d"), "day": nd.strftime("%A")}
        elif a == "date_difference":
            b, t = parse(base_date), parse(target_date); diff = (t - b).days
            return {"status": "success", "from": b.strftime("%Y-%m-%d"),
                    "to": t.strftime("%Y-%m-%d"), "diff_days": diff, "abs_days": abs(diff)}
        elif a == "day_of_week":
            base = parse(base_date)
            return {"status": "success", "date": base.strftime("%Y-%m-%d"), "day": base.strftime("%A")}
        else:
            return {"status": "error", "message": f"Unknown action: {a}"}
    except Exception as err:
        return {"status": "error", "message": str(err)}

@tool
def document_search(query: str, file_name: str = "Dummy_Student_Database_for_RAG.pdf") -> dict:
    """Search local PDF document or database (e.g. Dummy_Student_Database_for_RAG.pdf) for student marks, attendance, rank, contact details, or document text."""
    try:
        from pathlib import Path
        import pypdf
        p = Path(file_name)
        if not p.is_file():
            p = Path(__file__).parent / file_name
        if not p.is_file():
            p = Path(__file__).parent / "Dummy_Student_Database_for_RAG.pdf"
        if not p.is_file():
            return {"status": "error", "message": f"File {file_name} not found"}

        reader = pypdf.PdfReader(str(p))
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        records = []
        i = 0
        while i < len(lines):
            if re.match(r"^STU\d{3}$", lines[i]):
                try:
                    records.append({
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
                    })
                    i += 12
                    continue
                except (IndexError, ValueError):
                    pass
            i += 1

        q_lower = query.lower()
        if "highest overall" in q_lower or "top overall" in q_lower:
            return {"status": "success", "top_student": max(records, key=lambda x: x["overall"])}
        if "highest attendance" in q_lower or "top attendance" in q_lower:
            return {"status": "success", "top_student": max(records, key=lambda x: int(x["attendance"].replace("%", "")))}
        if "below 75" in q_lower or "< 75" in q_lower:
            return {"status": "success", "results": [r for r in records if int(r["attendance"].replace("%", "")) < 75]}
        if "above 90 in ai" in q_lower or "ai > 90" in q_lower:
            return {"status": "success", "results": [r for r in records if r["ai"] > 90]}
        if "placement eligible" in q_lower:
            eligible = [r for r in records if r["placement"].lower() == "eligible"]
            return {"status": "success", "eligible_count": len(eligible), "eligible_students": eligible}

        matches = []
        for r in records:
            if r["student_id"].lower() in q_lower:
                matches.append(r)
            elif any(part.lower() in q_lower for part in r["name"].split() if len(part) > 2):
                matches.append(r)
        
        dept_match = re.findall(r"\b(cse|aids|it|ece|eee)\b", q_lower)
        if dept_match and not matches:
            target = [d.upper() for d in dept_match]
            matches = [r for r in records if r["department"].upper() in target]

        if matches:
            return {"status": "success", "count": len(matches), "results": matches}

        # Substring search
        snippets = [l for l in lines if any(w in l.lower() for w in q_lower.split() if len(w) > 3)]
        return {"status": "success", "snippets": snippets[:8]}
    except Exception as e:
        return {"status": "error", "message": str(e)}

TOOLS = [calculator, web_search, date_time, document_search]

def build_agent():
    for m in MODELS:
        try:
            agent = create_react_agent(make_llm(m), TOOLS)
            print(f"  [Agent] Model: {m}")
            return agent
        except Exception as e:
            if any(x in str(e) for x in ["429","503","504","NOT_FOUND","UNAVAILABLE","RESOURCE_EXHAUSTED"]):
                continue
            raise
    raise RuntimeError("All models exhausted.")

async def run_task(agent, tid: int, prompt: str) -> dict:
    t0 = datetime.now()
    print(f"  [Task-{tid}] START: {prompt[:60]}...")
    try:
        out    = await agent.ainvoke({"messages": [("user", prompt)]})
        answer = out["messages"][-1].content
        ela    = (datetime.now() - t0).total_seconds()
        print(f"  [Task-{tid}] DONE  ({ela:.1f}s)")
        return {"task_id": tid, "prompt": prompt, "answer": answer, "elapsed": ela, "status": "success"}
    except Exception as e:
        ela = (datetime.now() - t0).total_seconds()
        print(f"  [Task-{tid}] ERROR ({ela:.1f}s): {e}")
        return {"task_id": tid, "prompt": prompt, "answer": None, "error": str(e), "elapsed": ela, "status": "error"}

async def run_parallel(tasks: list) -> list:
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  PARALLEL EXECUTION  ({len(tasks)} tasks simultaneously)")
    print(sep)
    agent = build_agent()
    t0    = datetime.now()
    results = await asyncio.gather(*[run_task(agent, i+1, t) for i, t in enumerate(tasks)])
    total = (datetime.now() - t0).total_seconds()
    print(f"\n{sep}")
    print(f"  ALL {len(tasks)} TASKS DONE in {total:.1f}s (parallel)")
    print(sep)
    return list(results)

def show_results(results):
    sep = "=" * 68
    print(f"\n{sep}")
    print("  RESULTS SUMMARY")
    print(sep)
    for r in results:
        print(f"\n[Task-{r['task_id']}] [{r['status'].upper()}] ({r['elapsed']:.1f}s)")
        print(f"  Prompt : {r['prompt']}")
        if r["status"] == "success":
            print(f"  Answer : {r['answer']}")
        else:
            print(f"  Error  : {r.get('error','unknown')}")
        print("-" * 68)

DEMO = [
    "What is 245 multiplied by 37?",
    "What day of the week will it be 30 days from today?",
    "Who invented the World Wide Web and when?",
    "What is 2 to the power of 16?",
    "How many days between 2026-01-01 and 2026-12-31?",
]

def main():
    sep = "=" * 68
    print(sep)
    print("   LangChain Multi-Task Parallel Agent")
    print(sep)
    print("  Framework : LangChain + LangGraph  (ReAct pattern)")
    print("  Execution : asyncio.gather()  -- TRUE parallel tasks")
    print("  Tools     : calculator  |  web_search  |  date_time  |  document_search")
    print("  Commands  : demo | add | list | clear | run | exit")
    print(sep)
    pending = []
    while True:
        try:
            cmd = input("\nCommand > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!"); break
        if not cmd: continue
        if cmd.lower() in ("exit", "quit"):
            print("Goodbye!"); break
        elif cmd.lower() == "demo":
            pending = list(DEMO)
            print(f"\nLoaded {len(pending)} demo tasks:")
            for i, t in enumerate(pending, 1): print(f"  [{i}] {t}")
            print("\nType 'run' to execute all in parallel.")
        elif cmd.lower() == "add":
            t = input("  Task prompt: ").strip()
            if t: pending.append(t); print(f"  Added. ({len(pending)} total)")
            else: print("  Empty -- skipped.")
        elif cmd.lower() == "list":
            if not pending: print("  No tasks pending.")
            else:
                print(f"  Pending ({len(pending)}):")
                for i, t in enumerate(pending, 1): print(f"    [{i}] {t}")
        elif cmd.lower() == "clear":
            pending.clear(); print("  Cleared.")
        elif cmd.lower() == "run":
            if not pending: print("  Nothing to run. Use 'add' or 'demo' first."); continue
            show_results(asyncio.run(run_parallel(pending)))
            pending.clear()
        else:
            show_results(asyncio.run(run_parallel([cmd])))

if __name__ == "__main__":
    main()
