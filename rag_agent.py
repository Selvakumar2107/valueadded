"""
LangChain RAG (Retrieval-Augmented Generation) System
Document: Dummy_Student_Database_for_RAG.pdf
Embeddings: models/gemini-embedding-001
LLM: Gemini 3.1 Flash Lite / Gemini 3 Flash / Gemini 3.7 Flash
VectorStore: InMemoryVectorStore
"""

import os
import sys
import re
from pathlib import Path
import pypdf

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

API_KEY = os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6JxqoHwD6TmOIfCxE4bt6CL8HggxsfNcbylcHyGG2xiEw")
PDF_PATH = Path(__file__).parent / "Dummy_Student_Database_for_RAG.pdf"

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

def make_llm():
    for model_name in MODELS:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=API_KEY,
                temperature=0.0
            )
            return llm, model_name
        except Exception:
            continue
    raise RuntimeError("Could not initialize any Gemini model.")

def make_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=API_KEY
    )

def extract_documents_from_pdf(pdf_path: Path) -> list[Document]:
    """
    Extracts text from the PDF and creates rich Document objects.
    Includes full page text, parsed student records, and summary tables
    for optimal retrieval on both granular and aggregate questions.
    """
    reader = pypdf.PdfReader(str(pdf_path))
    docs = []
    
    # 1. Whole page chunks
    full_text_pages = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        full_text_pages.append(page_text)
        docs.append(Document(
            page_content=f"Document Page {i+1} Content:\n{page_text}",
            metadata={"source": pdf_path.name, "page": i + 1, "type": "page_text"}
        ))

    # 2. Extract structured table rows from the extracted tokens
    # Columns: Student ID, Name, Dept., Year, Email, Phone, Attendance, Python, DSA, AI, Overall, Placement
    combined_text = "\n".join(full_text_pages)
    lines = [line.strip() for line in combined_text.splitlines() if line.strip()]

    # Find student ID tokens (STU001 to STU025)
    records = []
    i = 0
    while i < len(lines):
        line = lines[i]
        match = re.match(r"^STU\d{3}$", line)
        if match:
            # We have a student record starting here. Let's gather the 12 fields:
            # ID, Name, Dept, Year, Email, Phone, Attendance, Python, DSA, AI, Overall, Placement
            try:
                stu_id = lines[i]
                name = lines[i+1]
                dept = lines[i+2]
                year = lines[i+3]
                email = lines[i+4]
                phone = lines[i+5]
                attendance = lines[i+6]
                python_score = lines[i+7]
                dsa_score = lines[i+8]
                ai_score = lines[i+9]
                overall_score = lines[i+10]
                placement = lines[i+11]
                
                content = (
                    f"Student ID: {stu_id}\n"
                    f"Name: {name}\n"
                    f"Department: {dept}\n"
                    f"Year: {year}\n"
                    f"Email: {email}\n"
                    f"Phone Number: {phone}\n"
                    f"Attendance: {attendance}\n"
                    f"Python Score: {python_score}\n"
                    f"DSA Score: {dsa_score}\n"
                    f"AI Score: {ai_score}\n"
                    f"Overall Mark: {overall_score}\n"
                    f"Placement Status: {placement}"
                )
                
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "source": pdf_path.name,
                        "student_id": stu_id,
                        "name": name,
                        "department": dept,
                        "type": "student_record"
                    }
                ))
                records.append({
                    "id": stu_id, "name": name, "dept": dept, "year": year,
                    "email": email, "phone": phone, "att": attendance,
                    "py": python_score, "dsa": dsa_score, "ai": ai_score,
                    "overall": overall_score, "placement": placement
                })
                i += 12
                continue
            except IndexError:
                pass
        i += 1

    # 3. Add complete consolidated dataset summary chunk for aggregations/rankings
    summary_lines = [
        "Consolidated Student Roster and Academic Performance Table:",
        "ID | Name | Department | Attendance | Python | DSA | AI | Overall | Placement | Phone | Email"
    ]
    for r in records:
        summary_lines.append(
            f"{r['id']} | {r['name']} | {r['dept']} | {r['att']} | {r['py']} | {r['dsa']} | {r['ai']} | {r['overall']} | {r['placement']} | {r['phone']} | {r['email']}"
        )
    
    docs.append(Document(
        page_content="\n".join(summary_lines),
        metadata={"source": pdf_path.name, "type": "consolidated_roster"}
    ))

    return docs

def build_rag_chain():
    print(" [1/3] Loading and parsing PDF document...")
    docs = extract_documents_from_pdf(PDF_PATH)
    print(f"       Extracted {len(docs)} documents/chunks (records + page text + consolidated roster).")

    print(" [2/3] Embedding chunks into Vector Store (models/gemini-embedding-001)...")
    embeddings = make_embeddings()
    vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    print(" [3/3] Initializing Gemini LLM and RAG prompt pipeline...")
    llm, model_name = make_llm()
    print(f"       Active LLM: {model_name}")

    prompt = ChatPromptTemplate.from_template(
        """You are a precise document retrieval assistant answering queries based ONLY on the provided context from the student database document.
Context:
{context}

Question:
{question}

Instructions:
1. Provide a direct, concise, and 100% accurate answer supported by the context.
2. If relevant, include the Student ID, department, and exact values (marks, percentages, phone numbers).
3. Do not assume or invent information not present in the document.

Answer:"""
    )

    def retrieve_and_format(query: str):
        # We retrieve top-k chunks, and if the question looks like an aggregation/comparison, ensure consolidated roster is included
        retrieved_docs = retriever.invoke(query)
        is_aggregate = any(w in query.lower() for w in ["highest", "lowest", "list", "how many", "all", "which students", "department"])
        if is_aggregate:
            for d in docs:
                if d.metadata.get("type") == "consolidated_roster" and d not in retrieved_docs:
                    retrieved_docs.append(d)
                    break
        context_str = "\n\n---\n\n".join([d.page_content for d in retrieved_docs])
        return context_str, retrieved_docs

    rag_chain = prompt | llm | StrOutputParser()

    return retrieve_and_format, rag_chain

SUGGESTED_QUESTIONS = [
    "What is Priya Sharma's attendance and overall mark?",
    "Who has the highest overall mark?",
    "Which students belong to the CSE department?",
    "List students whose attendance is below 75%.",
    "Who scored above 90 in AI?",
    "How many students are placement eligible?",
    "What is Arjun Kumar's DSA mark?",
    "Which student has the highest attendance?",
    "Find all AIDS students with an overall mark above 80.",
    "What is the phone number of Fathima N?"
]

def run_suggested(retrieve_fn, chain):
    sep = "=" * 76
    print(f"\n{sep}")
    print("  EXECUTING RAG RETRIEVAL FOR ALL 10 SUGGESTED QUESTIONS")
    print(sep)

    for i, q in enumerate(SUGGESTED_QUESTIONS, 1):
        print(f"\n[Q{i}] Question: {q}")
        context, matched_docs = retrieve_fn(q)
        ans = chain.invoke({"context": context, "question": q})
        print(f"     Answer  : {ans.strip()}")
        sources = [d.metadata.get("student_id") or d.metadata.get("type") or "chunk" for d in matched_docs[:3]]
        print(f"     Retrieved Chunks: {', '.join(sources)}")
        print("-" * 76)

def main():
    retrieve_fn, chain = build_rag_chain()
    
    # Run the suggested questions first
    run_suggested(retrieve_fn, chain)

    print("\n" + "=" * 76)
    print("  INTERACTIVE RAG MODE (Type 'exit' to quit)")
    print("=" * 76)
    while True:
        try:
            q = input("\nAsk a question > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("Exiting.")
            break
        context, _ = retrieve_fn(q)
        ans = chain.invoke({"context": context, "question": q})
        print(f"\nAnswer:\n{ans.strip()}\n")

if __name__ == "__main__":
    main()
