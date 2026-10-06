"""Vector store on PostgreSQL using the pgvector extension.

Drop-in replacement for VectorStore (same add / sync / search methods), so agent.py
only needs a different constructor. The difference: vectors live in a real `vector(N)`
column, are searched with the `<=>` (cosine distance) operator, and can use an HNSW index.

Setup:
    docker run -d --name pgvec -e POSTGRES_PASSWORD=pw -p 5432:5432 pgvector/pgvector:pg17
    pip install "psycopg[binary]" ollama
    export PG_DSN="postgresql://postgres:pw@localhost:5432/postgres"
"""
import re
import psycopg
from vector_store import embed   # same embedding function as the SQLite version


def to_pg(vec: list) -> str:
    """pgvector accepts vectors as text like '[0.1,0.2,0.3]'; we cast with ::vector in SQL."""
    return "[" + ",".join(str(x) for x in vec) + "]"


class VectorStore_pg:
    def __init__(self, dsn: str):
        self.dsn = dsn
        with psycopg.connect(self.dsn) as con:       # 'with' commits and closes
            con.execute("CREATE EXTENSION IF NOT EXISTS vector")   # needs permission once per database

    @staticmethod
    def _table(kind: str) -> str:
        assert re.fullmatch(r"\w+", kind), "kind must be a simple word"   # it becomes a table name
        return f"vec_{kind}"

    def add(self, kind: str, key: str, text: str):
        vec = embed(text)
        table = self._table(kind)
        with psycopg.connect(self.dsn) as con:
            # The vector size is fixed when the table is created (768 for nomic-embed-text).
            # One table per kind keeps HNSW searches exact about "only this kind".
            con.execute(f"CREATE TABLE IF NOT EXISTS {table}("
                        f"key TEXT PRIMARY KEY, text TEXT, embedding vector({len(vec)}))")
            con.execute(f"CREATE INDEX IF NOT EXISTS {table}_hnsw "
                        f"ON {table} USING hnsw (embedding vector_cosine_ops)")
            con.execute(f"INSERT INTO {table}(key, text, embedding) VALUES (%s, %s, %s::vector) "
                        f"ON CONFLICT (key) DO UPDATE SET text = EXCLUDED.text, "
                        f"embedding = EXCLUDED.embedding", (key, text, to_pg(vec)))

    def sync(self, kind: str, items: list):
        """Embed any (key, text) pairs that aren't indexed yet."""
        table = self._table(kind)
        with psycopg.connect(self.dsn) as con:
            if con.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0] is None:
                have = set()
            else:
                have = {r[0] for r in con.execute(f"SELECT key FROM {table}")}
        for key, text in items:
            if key not in have:
                self.add(kind, key, text)

    def search(self, kind: str, query: str, k: int = 3, min_score: float = 0.4) -> list:
        """Return [(key, text, score)] best first. score = cosine similarity (1 = identical)."""
        table = self._table(kind)
        with psycopg.connect(self.dsn) as con:
            if con.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0] is None:
                return []                              # nothing indexed for this kind yet
            q = to_pg(embed(query))
            rows = con.execute(
                f"SELECT key, text, 1 - (embedding <=> %s::vector) AS score FROM {table} "
                f"ORDER BY embedding <=> %s::vector LIMIT %s", (q, q, k)).fetchall()
        return [(key, text, float(score)) for key, text, score in rows if score >= min_score]
