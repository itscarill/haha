"""Vector store on SQLite using the sqlite-vec extension (vectors live in your .db file).

How it works:
  1. embed(text) turns text into a vector with a local embedding model.
  2. Plain table `vec_docs` holds the text; one `vec0` virtual table per kind holds the vectors.
     The two are linked by rowid.
  3. search(query) runs a KNN query (embedding MATCH ? AND k = N) and returns closest rows.

Setup:
    ollama pull nomic-embed-text
    pip install sqlite-vec ollama
"""
import re
import sqlite3
import ollama
try:
    import sqlite_vec
except ImportError:   # only needed for the SQLite backend
    sqlite_vec = None

EMBED_MODEL = "nomic-embed-text"


def embed(text: str) -> list:
    return ollama.embed(model=EMBED_MODEL, input=text).embeddings[0]


class VectorStore:
    def __init__(self, path: str):
        self.path = path
        with self._con() as con:
            con.execute("CREATE TABLE IF NOT EXISTS vec_docs("
                        "id INTEGER PRIMARY KEY, kind TEXT, key TEXT, text TEXT, UNIQUE(kind, key))")

    def _con(self):
        con = sqlite3.connect(self.path)
        con.enable_load_extension(True)       # some Pythons (e.g. macOS system python) block this
        sqlite_vec.load(con)
        con.enable_load_extension(False)
        return con

    @staticmethod
    def _table(kind: str) -> str:
        assert re.fullmatch(r"\w+", kind), "kind must be a simple word"   # it becomes a table name
        return f"vec_{kind}"

    def add(self, kind: str, key: str, text: str):
        vec = embed(text)
        table = self._table(kind)
        with self._con() as con:
            # The vector column's size is fixed when the table is created (768 for nomic-embed-text)
            con.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0("
                        f"embedding float[{len(vec)}] distance_metric=cosine)")
            row = con.execute("SELECT id FROM vec_docs WHERE kind=? AND key=?", (kind, key)).fetchone()
            if row:                            # update: replace text and vector
                doc_id = row[0]
                con.execute("UPDATE vec_docs SET text=? WHERE id=?", (text, doc_id))
                con.execute(f"DELETE FROM {table} WHERE rowid=?", (doc_id,))
            else:
                doc_id = con.execute("INSERT INTO vec_docs(kind, key, text) VALUES (?,?,?)",
                                     (kind, key, text)).lastrowid
            con.execute(f"INSERT INTO {table}(rowid, embedding) VALUES (?, ?)",
                        (doc_id, sqlite_vec.serialize_float32(vec)))

    def sync(self, kind: str, items: list):
        """Embed any (key, text) pairs that aren't indexed yet.
        Lets you INSERT rows by hand with plain SQL and have them picked up automatically."""
        have = {r[0] for r in self._con().execute("SELECT key FROM vec_docs WHERE kind=?", (kind,))}
        for key, text in items:
            if key not in have:
                self.add(kind, key, text)

    def search(self, kind: str, query: str, k: int = 3, min_score: float = 0.4) -> list:
        """Return [(key, text, score)] best first. score = cosine similarity (1 = identical)."""
        table = self._table(kind)
        con = self._con()
        if not con.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone():
            return []                          # nothing indexed for this kind yet
        rows = con.execute(
            f"SELECT d.key, d.text, v.distance FROM {table} v JOIN vec_docs d ON d.id = v.rowid "
            f"WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance",
            (sqlite_vec.serialize_float32(embed(query)), k)).fetchall()
        return [(key, text, 1 - dist) for key, text, dist in rows if 1 - dist >= min_score]
