"""Minimal local agent: asks questions about a SQLite DB using a small local model.

Setup:
    ollama pull qwen2.5:7b
    ollama pull nomic-embed-text
    pip install ollama sqlite-vec
    python agent.py


Using PGVECTOR:
    docker run -d --name pgvec -e POSTGRES_PASSWORD=pw -p 5432:5432 pgvector/pgvector:pg17
    CREATE EXTENSION vector;  -- in psql
    pip install "psycopg[binary]"
    export PG_DSN="postgresql://postgres:pw@localhost:5432/postgres"
    python agent.py

    

"""
import json, os, sqlite3
import ollama
from vector_store import VectorStore, EMBED_MODEL

MODEL = "qwen2.5:7b"   # any local model that supports tool calling
DB = "demo.db"
MEM_DB = "memory.db"   # long-term memory lives here
MAX_STEPS = 8          # guardrail: the loop can never run forever

# ---------- 0. Demo data (so you have something to query) ----------
if not os.path.exists(DB):
    con = sqlite3.connect(DB)
    con.executescript("""
    CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT, country TEXT);
    CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL, created_at TEXT);
    INSERT INTO customers VALUES (1,'Ana','US'),(2,'Ben','UK'),(3,'Chen','US');
    INSERT INTO orders VALUES (1,1,120.5,'2026-01-03'),(2,1,80,'2026-02-10'),
                              (3,2,200,'2026-02-11'),(4,3,50,'2026-03-01');
    """)
    con.commit(); con.close()

DEMO_NOTES = [  # (table, column or '' for the table itself, description)
    ("customers", "", "People who buy from us."),
    ("customers", "country", "Two-letter country code of the customer."),
    ("orders", "", "One row per purchase."),
    ("orders", "customer_id", "References customers.id."),
    ("orders", "amount", "Purchase amount in USD, excluding tax."),
    ("orders", "created_at", "Order date as YYYY-MM-DD text."),
]

DEMO_EXAMPLES = [  # (question, SQL) pairs that show YOUR preferred patterns
    ("Total purchase per customer",
     "SELECT c.name, SUM(o.amount) AS total FROM customers c "
     "JOIN orders o ON o.customer_id = c.id GROUP BY c.name ORDER BY total DESC"),
    ("Number of orders per country",
     "SELECT c.country, COUNT(*) AS orders FROM orders o "
     "JOIN customers c ON c.id = o.customer_id GROUP BY c.country"),
    ("Total revenue in a given month",
     "SELECT SUM(amount) FROM orders WHERE created_at >= '2026-02-01' AND created_at < '2026-03-01'"),
]

def connect_readonly():
    # Guardrail: the connection itself cannot write, whatever the model asks for
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

# ---------- 1. Tools: plain Python functions ----------
def table_names() -> list:
    rows = connect_readonly().execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
    return [r[0] for r in rows]

def list_tables() -> str:
    """Step 1 of schema discovery: every table plus a one-line description."""
    notes = dict(mem_con().execute(
        "SELECT table_name, note FROM schema_notes WHERE column_name=''").fetchall())
    return "\n".join(f"{t}: {notes.get(t, '(no description)')}" for t in table_names())

def describe_table(table: str) -> str:
    """Step 2: columns, types, human descriptions and sample values for ONE table."""
    if table not in table_names():              # validate: never trust model-supplied names
        return f"Error: unknown table '{table}'. Available: {', '.join(table_names())}"
    notes = dict(mem_con().execute(
        "SELECT column_name, note FROM schema_notes WHERE table_name=?", (table,)).fetchall())
    db = connect_readonly()
    lines = [f"Table {table}: {notes.get('', '(no description)')}"]
    for _, col, ctype, *_ in db.execute(f'PRAGMA table_info("{table}")').fetchall():
        samples = [str(r[0])[:30] for r in db.execute(
            f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL LIMIT 3')]
        lines.append(f"- {col} ({ctype}): {notes.get(col, 'no description')} "
                     f"| e.g. {', '.join(samples)}")
    return "\n".join(lines)

def run_sql(query: str) -> str:
    if not query.strip().lower().startswith("select"):
        return "Error: only SELECT queries are allowed."
    try:
        return json.dumps(connect_readonly().execute(query).fetchmany(50))
    except Exception as e:
        return f"Error: {e}"  # fed back to the model so it can fix its own SQL

# ---------- Long-term memory: a separate, writable SQLite file ----------
# Kept apart from DB so run_sql can never read or modify it.
def vstore():
    """Pick the vector backend: Postgres/pgvector if PG_DSN is set, else sqlite-vec."""
    if os.getenv("PG_DSN"):
        from vector_store_pg import VectorStore_pg   # imported lazily: psycopg only needed here
        return VectorStore_pg(os.environ["PG_DSN"])
    return VectorStore(MEM_DB)

def mem_con():
    con = sqlite3.connect(MEM_DB)
    con.execute("CREATE TABLE IF NOT EXISTS memories("
                "id INTEGER PRIMARY KEY, fact TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
    # Human-written data dictionary. column_name='' means a note about the whole table.
    # Edit this table with plain SQL: it's the highest-value thing you can maintain.
    con.execute("CREATE TABLE IF NOT EXISTS schema_notes("
                "table_name TEXT, column_name TEXT DEFAULT '', note TEXT, "
                "PRIMARY KEY(table_name, column_name))")
    if con.execute("SELECT COUNT(*) FROM schema_notes").fetchone()[0] == 0:
        con.executemany("INSERT INTO schema_notes VALUES (?,?,?)", DEMO_NOTES)
        con.commit()
    con.execute("CREATE TABLE IF NOT EXISTS examples(question TEXT PRIMARY KEY, sql TEXT)")
    if con.execute("SELECT COUNT(*) FROM examples").fetchone()[0] == 0:
        con.executemany("INSERT INTO examples VALUES (?,?)", DEMO_EXAMPLES)
        con.commit()
    return con

def find_examples(question: str, k: int = 3) -> list:
    """Pick the saved examples most similar in MEANING to the question."""
    rows = mem_con().execute("SELECT question, sql FROM examples").fetchall()
    try:
        vs = vstore()
        vs.sync("example", [(q, q) for q, _ in rows])    # index anything new
        sql_by_q = dict(rows)
        return [(q, sql_by_q[q]) for q, _, _ in vs.search("example", question, k)
                if q in sql_by_q]
    except Exception:                                    # embeddings unavailable -> word overlap
        words = set(question.lower().split())
        scored = sorted(((len(words & set(q.lower().split())), q, sql) for q, sql in rows),
                        reverse=True)
        return [(q, sql) for n, q, sql in scored[:k] if n > 0]

def remember(fact: str) -> str:
    fact = fact[:500]
    with mem_con() as con:                      # 'with' commits the write
        cur = con.execute("INSERT INTO memories(fact) VALUES (?)", (fact,))
    try:
        vstore().add("memory", str(cur.lastrowid), fact)
    except Exception:
        pass                                    # embeddings are an optimisation, not required
    return "Saved."

def recall(keyword: str = "") -> str:
    """With a query: the most relevant memories by meaning. Without: the latest ones."""
    if keyword:
        try:
            vs = vstore()
            vs.sync("memory", [(str(i), f) for i, f in mem_con().execute("SELECT id, fact FROM memories")])
            hits = vs.search("memory", keyword, k=5)
            return "\n".join(t for _, t, _ in hits) or "No memories found."
        except Exception:
            keyword = ""                        # embeddings down -> fall back to latest 20
    rows = mem_con().execute(
        "SELECT fact FROM memories WHERE fact LIKE ? ORDER BY id DESC LIMIT 20",
        (f"%{keyword}%",)).fetchall()
    return "\n".join(r[0] for r in rows) or "No memories found."

def search_schema(question: str) -> str:
    """Find the tables/columns most relevant to a question (for databases with many tables)."""
    try:
        vs = vstore()
        items = [((f"{t}.{c}" if c else t), (f"{t}.{c}: {n}" if c else f"{t}: {n}"))
                 for t, c, n in mem_con().execute(
                     "SELECT table_name, column_name, note FROM schema_notes")]
        vs.sync("schema", items)
        hits = vs.search("schema", question, k=8)
        if hits:
            tables = list(dict.fromkeys(key.split(".")[0] for key, _, _ in hits))
            return "Most relevant tables: " + ", ".join(tables) + "\n" + "\n".join(t for _, t, _ in hits)
    except Exception:
        pass
    return list_tables()                        # nothing found or embeddings down

REGISTRY = {"search_schema": search_schema, "list_tables": list_tables, "describe_table": describe_table,
            "run_sql": run_sql, "remember": remember, "recall": recall}  # allowlist

# ---------- 2. Tool descriptions: this is all the model ever sees ----------
TOOLS = [
    {"type": "function", "function": {
        "name": "search_schema",
        "description": "Find the tables and columns most relevant to the user's question. Call this first.",
        "parameters": {"type": "object",
                       "properties": {"question": {"type": "string", "description": "The user's question"}},
                       "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "list_tables",
        "description": "List ALL tables with a one-line description. Use if search_schema finds nothing.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "describe_table",
        "description": "Show columns, meanings and sample values of one table. "
                       "Call it for each table you plan to query.",
        "parameters": {"type": "object",
                       "properties": {"table": {"type": "string", "description": "Exact table name"}},
                       "required": ["table"]}}},
    {"type": "function", "function": {
        "name": "run_sql",
        "description": "Run a read-only SQLite SELECT query and return rows as JSON.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "A SELECT statement"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "remember",
        "description": "Save a short, useful fact for future conversations "
                       "(e.g. user preferences, what a column means, a query that worked).",
        "parameters": {"type": "object",
                       "properties": {"fact": {"type": "string"}},
                       "required": ["fact"]}}},
    {"type": "function", "function": {
        "name": "recall",
        "description": "Search saved memories by keyword. Empty keyword returns the latest ones.",
        "parameters": {"type": "object",
                       "properties": {"keyword": {"type": "string"}}}}},
]

SYSTEM = ("You are a data analyst. Use the tools to answer the question about the database. "
          "First call search_schema with the question, then describe_table for each table you need, before writing SQL. "
          "Trust the column descriptions. If a query errors, fix it and retry. "
          "Use remember() only for durable facts worth keeping, not for every answer. "
          "When you have the answer, reply in plain English.")

# ---------- Model check: pull the model if it isn't installed yet ----------
_model_ready = False
def ensure_model():
    global _model_ready
    if _model_ready:
        return
    try:
        installed = [m.model for m in ollama.list().models]
    except Exception:
        raise SystemExit("Cannot reach Ollama. Is the server running? (try: ollama serve)")
    for name in (MODEL, EMBED_MODEL):
        if name not in installed and f"{name}:latest" not in installed:
            print(f"Model {name} not found locally, downloading (one-time)...")
            ollama.pull(name)
    _model_ready = True

# ---------- 3. The agent loop ----------
def run_agent(question: str, show_sql: bool = True) -> str:
    # Small models rarely think to call recall(), so we also inject recent memories up front
    known = recall(question)                     # only memories relevant to THIS question
    system = SYSTEM + (f"\n\nThings you remember from before:\n{known}"
                       if known != "No memories found." else "")
    examples = find_examples(question)
    if examples:                                 # few-shot: show the model how YOU write SQL
        system += "\n\nExample questions with good SQL:\n" + "\n".join(
            f"Q: {q}\nSQL: {sql}" for q, sql in examples)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": question}]  # model is stateless: we keep the history

    ensure_model()
    queries = []  # successful SQL, recorded by our code (not left to the model's memory)

    for step in range(MAX_STEPS):
        resp = ollama.chat(model=MODEL, messages=messages, tools=TOOLS,
                           options={"temperature": 0})
        msg = resp.message
        messages.append(msg)                      # remember what the model said

        if not msg.tool_calls:                    # no tool requested -> final answer
            if show_sql and queries:
                # Last successful query = the one the answer is most likely based on
                return f"{msg.content}\n\nQuery used:\n{queries[-1]}"
            return msg.content

        for call in msg.tool_calls:               # model asked us to run tools
            name, args = call.function.name, call.function.arguments
            print(f"[step {step}] {name}({args})")
            fn = REGISTRY.get(name)
            result = fn(**args) if fn else f"Error: unknown tool {name}"
            print(f"         -> {str(result)[:200]}")
            if name == "run_sql" and not str(result).startswith("Error"):
                queries.append(args.get("query", ""))
            messages.append({"role": "tool", "tool_name": name, "content": str(result)})

    return "Stopped: reached max steps without a final answer."

if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or 'quit'): ")
        if q.strip().lower() == "quit":
            break
        print("\nAnswer:", run_agent(q))
