#!/usr/bin/env python3
"""
vrom-cli — Command-line tool for managing vROM packages.

Fetch, list, inspect, and search pre-computed HNSW vector indexes
from the vROM Registry on Hugging Face Hub.

Usage:
    python vrom_cli.py list                          # List available vROMs
    python vrom_cli.py pull <vrom-id>                # Download a vROM
    python vrom_cli.py pull --all                    # Download all vROMs
    python vrom_cli.py info <vrom-id>                # Show vROM details
    python vrom_cli.py search <vrom-id> "query"      # Search a local vROM
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: 'requests' required. Install with: pip install requests")
    sys.exit(1)

# ─── Configuration ────────────────────────────────────────────────────────────

REGISTRY_URL = "https://huggingface.co/datasets/philipp-zettl/vrom-registry/resolve/main/registry.json"
VROM_HOME = Path(os.environ.get("VROM_HOME", Path.home() / ".vrom"))


def get_registry(force_refresh=False):
    """Fetch the vROM registry from CDN."""
    cache_path = VROM_HOME / "registry.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Use cache if fresh (< 1 hour)
    if not force_refresh and cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < 3600:
            with open(cache_path) as f:
                return json.load(f)

    print(f"Fetching registry from {REGISTRY_URL}...")
    resp = requests.get(REGISTRY_URL, timeout=30)
    resp.raise_for_status()
    registry = resp.json()

    with open(cache_path, "w") as f:
        json.dump(registry, f, indent=2)

    return registry


def find_vrom(registry, vrom_id):
    """Find a vROM entry by ID."""
    for v in registry["vroms"]:
        if v["id"] == vrom_id:
            return v
    return None


def download_file(url, dest_path, label=""):
    """Download a file with progress."""
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                bar = "█" * int(pct // 2) + "░" * (50 - int(pct // 2))
                print(f"\r  {label} [{bar}] {pct:.0f}% ({downloaded / 1e6:.1f}/{total / 1e6:.1f} MB)", end="", flush=True)
    if total > 0:
        print()  # newline after progress bar


# ─── Commands ─────────────────────────────────────────────────────────────────

def cmd_list(args):
    """List available vROMs."""
    registry = get_registry(args.refresh)

    print(f"\n{'ID':<30} {'Vectors':>8} {'Size':>8} {'Model':<25} {'Tags'}")
    print("─" * 100)

    for v in registry["vroms"]:
        badge = "★ " if v.get("official") else "  "
        tags = ", ".join(v.get("tags", [])[:4])
        print(f"{badge}{v['id']:<28} {v['vectors']:>8} {v['size_mb']:>6.1f}MB {v['model']:<25} {tags}")

    # Show local status
    print(f"\nLocal cache: {VROM_HOME}")
    for v in registry["vroms"]:
        local = VROM_HOME / "vroms" / v["id"]
        if (local / "index.json").exists():
            size = (local / "index.json").stat().st_size / 1e6
            print(f"  ✓ {v['id']} ({size:.1f} MB)")


def cmd_pull(args):
    """Download a vROM."""
    registry = get_registry()

    if args.all:
        vroms_to_pull = registry["vroms"]
    else:
        if not args.vrom_id:
            print("Error: specify a vrom-id or use --all")
            sys.exit(1)
        vrom = find_vrom(registry, args.vrom_id)
        if not vrom:
            print(f"Error: vROM '{args.vrom_id}' not found. Use 'vrom list' to see available.")
            sys.exit(1)
        vroms_to_pull = [vrom]

    for vrom in vroms_to_pull:
        dest = VROM_HOME / "vroms" / vrom["id"]

        # Check if already cached
        if not args.force and (dest / "index.json").exists():
            print(f"✓ {vrom['id']} already cached ({(dest / 'index.json').stat().st_size / 1e6:.1f} MB)")
            continue

        print(f"\n📥 Pulling {vrom['id']} ({vrom['size_mb']} MB)...")

        for name in ["manifest.json", "index.json", "chunks.json"]:
            url = vrom["files"][name.replace(".json", "")]
            download_file(url, dest / name, label=name)

        print(f"✅ {vrom['id']} saved to {dest}")


def cmd_info(args):
    """Show detailed info about a vROM."""
    registry = get_registry()
    vrom = find_vrom(registry, args.vrom_id)
    if not vrom:
        print(f"Error: vROM '{args.vrom_id}' not found.")
        sys.exit(1)

    print(f"\n🧩 {vrom['name']}")
    print(f"   ID:          {vrom['id']}")
    print(f"   Version:     {vrom['version']}")
    print(f"   Description: {vrom['description']}")
    print(f"   Vectors:     {vrom['vectors']:,}")
    print(f"   Dimensions:  {vrom['dimensions']}")
    print(f"   Tokens:      {vrom['tokens']:,}")
    print(f"   Size:        {vrom['size_mb']} MB")
    print(f"   Model:       {vrom['model']}")
    print(f"   Tags:        {', '.join(vrom.get('tags', []))}")
    print(f"   Official:    {'✓' if vrom.get('official') else '✗'}")

    # Local status
    local = VROM_HOME / "vroms" / vrom["id"]
    if (local / "index.json").exists():
        manifest = json.load(open(local / "manifest.json")) if (local / "manifest.json").exists() else {}
        print(f"\n   📁 Local: {local}")
        print(f"   Corpus hash: {manifest.get('corpus_hash', 'N/A')}")
    else:
        print(f"\n   📁 Not cached locally. Run: vrom pull {vrom['id']}")

    print(f"\n   CDN URLs:")
    for k, url in vrom["files"].items():
        print(f"     {k}: {url}")


def cmd_search(args):
    """Search a locally cached vROM."""
    local = VROM_HOME / "vroms" / args.vrom_id
    if not (local / "index.json").exists():
        print(f"Error: vROM '{args.vrom_id}' not cached. Run: vrom pull {args.vrom_id}")
        sys.exit(1)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Error: 'sentence-transformers' required for search. Install with: pip install sentence-transformers")
        sys.exit(1)

    # Load chunks for text lookup
    with open(local / "chunks.json") as f:
        chunks = json.load(f)
    chunks_by_id = {c["chunk_id"]: c for c in chunks}

    # Load manifest
    with open(local / "manifest.json") as f:
        manifest = json.load(f)

    # Load index
    print(f"Loading index ({(local / 'index.json').stat().st_size / 1e6:.1f} MB)...")
    with open(local / "index.json") as f:
        index_data = json.load(f)

    # Load embedding model
    model_name = manifest.get("embedding_spec", {}).get("model_source", "sentence-transformers/all-MiniLM-L6-v2")
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name.replace("sentence-transformers/", ""))

    # Embed query
    query_emb = model.encode([args.query], normalize_embeddings=True)[0]

    # Brute-force search (since we can't use the HNSW graph without the Rust engine)
    print(f"\nSearching {len(index_data['nodes'])} vectors for: \"{args.query}\"")

    import numpy as np
    nodes = index_data["nodes"]
    distances = []
    for i, node in enumerate(nodes):
        vec = np.array(node["vector"], dtype=np.float32)
        dist = 1.0 - np.dot(query_emb, vec)
        distances.append((dist, i))
    distances.sort(key=lambda x: x[0])

    k = args.k or 5
    print(f"\nTop {k} results:\n")
    for rank, (dist, idx) in enumerate(distances[:k], 1):
        node = nodes[idx]
        meta = json.loads(node["metadata"]) if node.get("metadata") else {}
        text = meta.get("text", "")[:200]
        heading = meta.get("section_heading", "")
        source = meta.get("source_file", "")
        url = meta.get("url", "")

        print(f"  #{rank} [d={dist:.4f}] § {heading}")
        print(f"     {text}...")
        if source:
            print(f"     📄 {source}")
        if url:
            print(f"     🔗 {url}")
        print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="vrom",
        description="vROM CLI — Manage pre-computed HNSW vector indexes for VecDB-WASM",
    )
    sub = parser.add_subparsers(dest="command")

    # list
    p_list = sub.add_parser("list", help="List available vROMs")
    p_list.add_argument("--refresh", action="store_true", help="Force refresh registry")

    # pull
    p_pull = sub.add_parser("pull", help="Download a vROM")
    p_pull.add_argument("vrom_id", nargs="?", help="vROM ID to download")
    p_pull.add_argument("--all", action="store_true", help="Download all vROMs")
    p_pull.add_argument("--force", action="store_true", help="Re-download even if cached")

    # info
    p_info = sub.add_parser("info", help="Show vROM details")
    p_info.add_argument("vrom_id", help="vROM ID")

    # search
    p_search = sub.add_parser("search", help="Search a local vROM")
    p_search.add_argument("vrom_id", help="vROM ID")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("-k", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "list": cmd_list,
        "pull": cmd_pull,
        "info": cmd_info,
        "search": cmd_search,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
