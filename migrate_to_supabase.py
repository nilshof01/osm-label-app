"""Seed Supabase with existing local SQLite decisions (run1 + run2).

Uses supabase-py (HTTPS) instead of psycopg2 — no direct DB connection needed.

  pip install supabase

Set env vars before running:
  $env:SUPABASE_URL = "https://ppxywrpoqgeguhyharen.supabase.co"
  $env:SUPABASE_KEY = "<service_role key from Settings -> API>"
  python spaces/migrate_to_supabase.py
"""
from __future__ import annotations

import base64
import os
import sqlite3
from pathlib import Path

from supabase import create_client

HERE = Path(__file__).parent
SQLITE_DBS = [
    HERE.parent / "run1" / "decisions.sqlite",
    HERE.parent / "run2" / "decisions.sqlite",
]
BATCH = 200


def read_sqlite(path: Path):
    con = sqlite3.connect(path)
    decisions = con.execute(
        "SELECT dataset_row, tile_id, decision, ts FROM decisions").fetchall()
    fixes = con.execute(
        "SELECT dataset_row, tile_id, mask_png, ts FROM fixes").fetchall()
    try:
        image_fixes = con.execute(
            "SELECT dataset_row, tile_id, image_png, ts FROM image_fixes").fetchall()
    except sqlite3.OperationalError:
        image_fixes = []
    con.close()
    return decisions, fixes, image_fixes


def push_batch(sb, table: str, rows: list[dict]) -> None:
    for i in range(0, len(rows), BATCH):
        sb.table(table).upsert(rows[i:i + BATCH]).execute()


def main():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise SystemExit("Set SUPABASE_URL and SUPABASE_KEY env vars")

    sb = create_client(url, key)
    total_d = total_f = total_i = 0

    for db_path in SQLITE_DBS:
        if not db_path.exists():
            print(f"skipping {db_path} (not found)")
            continue
        print(f"reading {db_path} ...")
        decisions, fixes, image_fixes = read_sqlite(db_path)
        print(f"  {len(decisions)} decisions  {len(fixes)} mask fixes  "
              f"{len(image_fixes)} image fixes")

        push_batch(sb, "decisions", [
            {"dataset_row": r, "tile_id": t, "decision": d, "ts": ts}
            for r, t, d, ts in decisions])

        push_batch(sb, "fixes", [
            {"dataset_row": r, "tile_id": t,
             "mask_png": base64.b64encode(bytes(p)).decode(), "ts": ts}
            for r, t, p, ts in fixes])

        push_batch(sb, "image_fixes", [
            {"dataset_row": r, "tile_id": t,
             "image_png": base64.b64encode(bytes(p)).decode(), "ts": ts}
            for r, t, p, ts in image_fixes])

        print(f"  pushed OK")
        total_d += len(decisions)
        total_f += len(fixes)
        total_i += len(image_fixes)

    print(f"\ndone — {total_d} decisions  {total_f} mask fixes  "
          f"{total_i} image fixes in Supabase")


if __name__ == "__main__":
    main()
