
import os
import re
import sys
import json
import argparse
import textwrap
import pandas as pd

DB_DIR        = "./pico8_rag_db"
COLLECTION    = "pico8_games"
EMBED_MODEL   = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 4


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = df.fillna("")
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"[build] Loaded {len(df)} rows from {csv_path}")
    print(f"[build] Columns: {list(df.columns)}")
    return df


def make_document(row: pd.Series) -> str:
    parts = []
    if row.get("name"):
        parts.append(f"Game: {row['name']}")
    if row.get("author"):
        parts.append(f"Author: {row['author']}")
    if row.get("description"):
        parts.append(f"Description: {str(row['description'])[:800]}")
    if row.get("license"):
        parts.append(f"License: {row['license']}")
    if row.get("like_count"):
        parts.append(f"Likes: {row['like_count']}")
    if row.get("top5_comments"):
        comments = str(row["top5_comments"])[:400]
        parts.append(f"Community comments: {comments}")
    if row.get("game_code"):
        code = str(row["game_code"])
        code_snippet = code[:1500]
        parts.append(f"PICO-8 source code:\n{code_snippet}")
    return "\n".join(parts)


def make_metadata(row: pd.Series, idx: int) -> dict:
    return {
        "id":          str(idx),
        "name":        str(row.get("name", ""))[:200],
        "author":      str(row.get("author", ""))[:100],
        "like_count":  str(row.get("like_count", "0")),
        "artwork_url": str(row.get("artwork_url", ""))[:300],
        "license":     str(row.get("license", ""))[:100],
        "has_code":    "yes" if str(row.get("game_code", "")).strip() else "no",
    }


def build_database(csv_path: str):
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        print("ERROR: Run:  pip install chromadb sentence-transformers")
        sys.exit(1)

    df = load_csv(csv_path)

    print(f"[build] Loading embedding model '{EMBED_MODEL}'…")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    client     = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION)
        print("[build] Deleted existing collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    documents = []
    metadatas = []
    ids       = []

    for idx, row in df.iterrows():
        doc  = make_document(row)
        meta = make_metadata(row, idx)
        documents.append(doc)
        metadatas.append(meta)
        ids.append(f"game_{idx}")

    BATCH = 50
    for i in range(0, len(documents), BATCH):
        collection.add(
            documents=documents[i:i+BATCH],
            metadatas=metadatas[i:i+BATCH],
            ids=ids[i:i+BATCH],
        )
        print(f"[build] Indexed {min(i+BATCH, len(documents))}/{len(documents)} games…")

    print(f"\n[build] Database built at '{DB_DIR}' with {collection.count()} entries.")
    print("[build] Run queries with:  python rag_system.py --query \"your question\"")


def load_collection():
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        print("ERROR: Run:  pip install chromadb sentence-transformers")
        sys.exit(1)

    ef         = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client     = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(name=COLLECTION, embedding_function=ef)
    return collection


def retrieve(collection, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "document": doc,
            "metadata": meta,
            "score":    round(1 - dist, 4),
        })
    return hits


def build_prompt(query: str, hits: list[dict]) -> str:
    context_parts = []
    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        context_parts.append(
            f"--- Reference {i} (game: {meta['name']}, author: {meta['author']}, "
            f"similarity: {hit['score']}) ---\n{hit['document']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""You are an expert PICO-8 game developer. PICO-8 is a fantasy console that uses Lua.
PICO-8 constraints: 128x128 display, 16 colours, 8 music channels, 64 sound effects.
Key PICO-8 API: pset/pget, spr/sspr, map/mget/mset, sfx/music, btn/btnp, print, cls, camera, circ/rect/line.

You have been given {len(hits)} relevant PICO-8 games from a community database as reference.
Use their code patterns, techniques, and approaches to write the best possible PICO-8 code.

CONTEXT FROM RAG DATABASE:
{context}

USER REQUEST:
{query}

Write complete, working PICO-8 Lua code. Structure it with the standard PICO-8 sections:
- _init() for initialisation
- _update() or _update60() for game logic  
- _draw() for rendering

Add brief comments explaining key parts. Make the code fun and playable."""
    return prompt


def generate_with_claude(prompt: str, api_key: str) -> str:
    try:
        import anthropic
    except ImportError:
        print("WARNING: anthropic not installed. Run: pip install anthropic")
        return None

    client   = anthropic.Anthropic(api_key=api_key)
    message  = client.messages.create(
        model      = "claude-opus-4-5",
        max_tokens = 4096,
        messages   = [{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def format_context_only(query: str, hits: list[dict]) -> str:
    lines = [
        f"\n{'='*70}",
        f"QUERY: {query}",
        f"{'='*70}",
        f"Top {len(hits)} matching games from RAG database:\n",
    ]
    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        lines.append(f"  {i}. {meta['name']}  (author: {meta['author']}, "
                     f"likes: {meta['like_count']}, similarity: {hit['score']})")
        lines.append(f"     has_code: {meta['has_code']}  |  license: {meta['license'] or 'none'}")
        lines.append(f"     artwork: {meta['artwork_url'][:60] or 'n/a'}")

        doc_lines = hit["document"].split("\n")
        code_start = next((i for i, l in enumerate(doc_lines) if "source code" in l.lower()), None)
        if code_start is not None:
            snippet = "\n".join(doc_lines[code_start+1:code_start+15])
            lines.append(f"\n     Code snippet:\n{textwrap.indent(snippet, '     ')}")
        lines.append("")

    lines += [
        f"{'='*70}",
        "TIP: Set ANTHROPIC_API_KEY env var to generate code with Claude AI.",
        "     pip install anthropic",
        f"{'='*70}\n",
    ]
    return "\n".join(lines)


def query_rag(query: str, top_k: int = DEFAULT_TOP_K, api_key: str = None):
    print(f"\n[query] '{query}'  (top_k={top_k})")
    collection = load_collection()
    print(f"[query] Database has {collection.count()} games. Retrieving…")

    hits = retrieve(collection, query, top_k)
    print(f"[query] Retrieved {len(hits)} results.")

    if api_key:
        print("[query] Generating PICO-8 code with Claude…\n")
        prompt   = build_prompt(query, hits)
        response = generate_with_claude(prompt, api_key)
        if response:
            print("=" * 70)
            print(f"GENERATED PICO-8 CODE FOR: {query}")
            print("=" * 70)
            print(response)
            print("=" * 70)

            refs = [f"  - {h['metadata']['name']} by {h['metadata']['author']} (sim={h['score']})"
                    for h in hits]
            print("RETRIEVED REFERENCES:")
            print("\n".join(refs))
            print("=" * 70)

            output_file = f"generated_{re.sub(r'[^a-z0-9]', '_', query.lower())[:40]}.lua"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"-- Generated by PICO-8 RAG System\n-- Query: {query}\n")
                f.write(f"-- References: {', '.join(h['metadata']['name'] for h in hits)}\n\n")
                f.write(response)
            print(f"\n[saved] Code written to: {output_file}")
        else:
            print(format_context_only(query, hits))
    else:
        print(format_context_only(query, hits))
        print("To generate PICO-8 code, provide your Anthropic API key:")
        print("  python rag_system.py --query \"...\" --api-key sk-ant-...")
        print("  # or set env var:  ANTHROPIC_API_KEY=sk-ant-...")


def interactive_mode(api_key: str = None):
    print("\n" + "=" * 70)
    print("  PICO-8 RAG System – Interactive Mode")
    print("  Type your game idea and get matching code from the database.")
    print("  Commands: 'quit' or 'exit' to stop, 'top N' to change result count")
    print("=" * 70)

    collection = load_collection()
    print(f"  Database loaded: {collection.count()} PICO-8 games indexed.\n")

    top_k = DEFAULT_TOP_K
    while True:
        try:
            user_input = input("Query > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if user_input.lower().startswith("top "):
            try:
                top_k = int(user_input.split()[1])
                print(f"  top_k set to {top_k}")
            except ValueError:
                print("  Usage: top N  (e.g. top 5)")
            continue

        hits = retrieve(collection, user_input, top_k)
        if api_key:
            prompt   = build_prompt(user_input, hits)
            response = generate_with_claude(prompt, api_key)
            if response:
                print("\n" + "=" * 70)
                print(response)
                print("=" * 70 + "\n")
        else:
            print(format_context_only(user_input, hits))


def main():
    parser = argparse.ArgumentParser(description="PICO-8 RAG Database System")
    parser.add_argument("--build",       action="store_true",             help="Build the RAG database from CSV")
    parser.add_argument("--csv",         default="lexaloffle_games.csv",  help="Path to CSV (for --build)")
    parser.add_argument("--query",       type=str,                        help="Query the RAG database")
    parser.add_argument("--top-k",       type=int, default=DEFAULT_TOP_K, help="Number of results to retrieve")
    parser.add_argument("--interactive", action="store_true",             help="Interactive query mode")
    parser.add_argument("--api-key",     type=str, default=None,          help="Anthropic API key for code generation")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")

    if args.build:
        build_database(args.csv)
    elif args.query:
        query_rag(args.query, top_k=args.top_k, api_key=api_key)
    elif args.interactive:
        interactive_mode(api_key=api_key)
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  1. python rag_system.py --build --csv lexaloffle_games.csv")
        print("  2. python rag_system.py --query \"make a snake game\"")
        print("  3. python rag_system.py --interactive")


if __name__ == "__main__":
    main()
