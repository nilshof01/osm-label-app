"""fAIr label reviewer — HuggingFace Spaces / Supabase edition.

Set these in Space secrets (Settings → Variables and secrets):
  SUPABASE_URL = https://xxxx.supabase.co
  SUPABASE_KEY = <service_role key from Settings -> API>

The run/ folder (fps_order.npy + density.parquet) must be committed to the
Space repo alongside this file.
"""
from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

import numpy as np
import streamlit as st
from datasets import load_dataset
from PIL import Image
from streamlit_shortcuts import clear_shortcuts, shortcut_button
from supabase import create_client, Client

RUN_DIR = Path(__file__).parent / "run"
DATASET = "hotosm/vhr-building-segmentation"
SPLIT   = "train"


# ---- Lazy canvas import ------------------------------------------------

def get_st_canvas():
    import streamlit.elements.image as _st_image
    try:
        from streamlit.elements.lib.image_utils import image_to_url as _itu_new
        from streamlit.elements.lib.layout_utils import LayoutConfig
        def _image_to_url(image, width, clamp, channels, output_format, image_id):
            lc = width if not isinstance(width, int) else LayoutConfig(width=width)
            return _itu_new(image, lc, clamp, channels, output_format, image_id)
        _st_image.image_to_url = _image_to_url
    except ImportError:
        pass
    from streamlit_drawable_canvas import st_canvas
    return st_canvas


# ---- Database (Supabase HTTPS) -----------------------------------------

@st.cache_resource
def open_db() -> Client:
    url = st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY")
    if not url or not key:
        st.error("Set SUPABASE_URL and SUPABASE_KEY in secrets.")
        st.stop()
    return create_client(url, key)


def _png_to_b64(img: Image.Image, mode: str) -> str:
    buf = io.BytesIO()
    img.convert(mode).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _b64_to_img(b64: str, mode: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert(mode)


def get_fix(con: Client, row: int) -> Image.Image | None:
    r = con.table("fixes").select("mask_png").eq("dataset_row", row).execute()
    return _b64_to_img(r.data[0]["mask_png"], "L") if r.data else None


def set_fix(con: Client, row: int, tile_id: str, mask_img: Image.Image) -> None:
    con.table("fixes").upsert({
        "dataset_row": row, "tile_id": tile_id,
        "mask_png": _png_to_b64(mask_img, "L"), "ts": time.time(),
    }).execute()


def clear_fix(con: Client, row: int) -> None:
    con.table("fixes").delete().eq("dataset_row", row).execute()


def get_image_fix(con: Client, row: int) -> Image.Image | None:
    r = con.table("image_fixes").select("image_png").eq("dataset_row", row).execute()
    return _b64_to_img(r.data[0]["image_png"], "RGB") if r.data else None


def set_image_fix(con: Client, row: int, tile_id: str, img: Image.Image) -> None:
    con.table("image_fixes").upsert({
        "dataset_row": row, "tile_id": tile_id,
        "image_png": _png_to_b64(img, "RGB"), "ts": time.time(),
    }).execute()


def clear_image_fix(con: Client, row: int) -> None:
    con.table("image_fixes").delete().eq("dataset_row", row).execute()


def get_decision(con: Client, row: int) -> str | None:
    r = con.table("decisions").select("decision").eq("dataset_row", row).execute()
    return r.data[0]["decision"] if r.data else None


def set_decision(con: Client, row: int, tile_id: str, dec: str) -> None:
    con.table("decisions").upsert({
        "dataset_row": row, "tile_id": tile_id,
        "decision": dec, "ts": time.time(),
    }).execute()


def _fetch_all(query) -> list[dict]:
    """Paginate through all rows (Supabase default cap is 1000)."""
    rows, page_size, offset = [], 1000, 0
    while True:
        batch = query.range(offset, offset + page_size - 1).execute().data
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def count_by(con: Client) -> dict[str, int]:
    rows = _fetch_all(con.table("decisions").select("decision"))
    counts: dict[str, int] = {}
    for row in rows:
        d = row["decision"]
        counts[d] = counts.get(d, 0) + 1
    return counts


def all_reviewed_rows(con: Client) -> set[int]:
    rows = _fetch_all(con.table("decisions").select("dataset_row"))
    return {int(row["dataset_row"]) for row in rows}


def all_kept_rows(con: Client) -> set[int]:
    kept = {int(r["dataset_row"]) for r in
            _fetch_all(con.table("decisions").select("dataset_row").eq("decision", "keep"))}
    kept |= {int(r["dataset_row"]) for r in
             _fetch_all(con.table("fixes").select("dataset_row"))}
    return kept


def _load_stats_cache(con: Client) -> None:
    """Populate session-state caches from Supabase. Called once per session open
    and after each decision so sliders/reruns don't hit the network."""
    st.session_state._stats       = count_by(con)
    st.session_state._reviewed    = all_reviewed_rows(con)
    st.session_state._kept        = all_kept_rows(con)


def _reviewed_cached() -> set[int]:
    base: set[int] = st.session_state.get("_reviewed", set())
    local: set[int] = st.session_state.get("local_reviewed", set())
    return base | local


# ---- Data --------------------------------------------------------------

@st.cache_resource(show_spinner="Loading dataset...")
def load_split(name: str, split: str):
    return load_dataset(name, split=split)


@st.cache_data(show_spinner=False)
def load_order(path: str) -> np.ndarray:
    return np.load(path)


def overlay_mask(img: Image.Image, mask: Image.Image, alpha: float = 0.4) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    if mask.mode != "L":
        mask = mask.convert("L")
    if mask.size != img.size:
        mask = mask.resize(img.size, Image.NEAREST)
    red = Image.new("RGB", img.size, (255, 0, 0))
    m = mask.point(lambda v: int(alpha * 255) if v > 0 else 0)
    return Image.composite(red, img, m)


# ---- Density buckets ---------------------------------------------------

BUCKET_EDGES  = [0,  1,  6, 21, 51]
BUCKET_LABELS = ["0 (empty)", "1-5", "6-20", "21-50", "51+"]
N_BUCKETS = len(BUCKET_LABELS)


def density_bucket(n: int) -> int:
    for i in range(len(BUCKET_EDGES) - 1, -1, -1):
        if n >= BUCKET_EDGES[i]:
            return i
    return 0


@st.cache_data(show_spinner=False)
def load_density_map(run_path: str) -> dict[int, int] | None:
    import pandas as pd
    p = Path(run_path) / "density.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    return {int(r): int(n) for r, n in zip(df["dataset_row"], df["n_instances"])}


@st.cache_data(show_spinner=False)
def build_bucket_positions(run_path: str, order_bytes: bytes) -> dict[int, list[int]]:
    density_map = load_density_map(run_path)
    order = np.frombuffer(order_bytes, dtype=np.int64)
    buckets: dict[int, list[int]] = {i: [] for i in range(N_BUCKETS)}
    if density_map is None:
        return buckets
    for pos, row in enumerate(order):
        b = density_bucket(density_map.get(int(row), 0))
        buckets[b].append(pos)
    return buckets


# ---- Navigation --------------------------------------------------------

def _mark_local(row: int) -> None:
    if "local_reviewed" not in st.session_state:
        st.session_state.local_reviewed = set()
    st.session_state.local_reviewed.add(row)


def next_unlabeled(order: np.ndarray, start: int) -> int:
    reviewed = _reviewed_cached()
    n = len(order)
    for i in range(start, n):
        if int(order[i]) not in reviewed:
            return i
    return n - 1


def advance(order: np.ndarray) -> None:
    cur = st.session_state.cursor
    st.session_state.cursor = min(next_unlabeled(order, cur + 1), len(order) - 1)


def advance_adaptive(order: np.ndarray,
                     density_map: dict[int, int] | None,
                     bucket_pos: dict[int, list[int]]) -> None:
    if density_map is None:
        advance(order)
        return

    reviewed = _reviewed_cached()
    kept: set[int] = st.session_state.get("_kept", set())

    bucket_counts = {i: 0 for i in range(N_BUCKETS)}
    for r in kept:
        bucket_counts[density_bucket(density_map.get(r, 0))] += 1

    best_bucket, best_count = None, float("inf")
    for b, positions in bucket_pos.items():
        if not positions:
            continue
        unreviewed = [p for p in positions if int(order[p]) not in reviewed]
        if unreviewed and bucket_counts[b] < best_count:
            best_count = bucket_counts[b]
            best_bucket = b

    if best_bucket is None:
        advance(order)
        return

    for pos in bucket_pos[best_bucket]:
        if int(order[pos]) not in reviewed:
            st.session_state.cursor = pos
            return

    advance(order)


def go_back() -> None:
    st.session_state.cursor = max(0, st.session_state.cursor - 1)


# ---- Fix modes ---------------------------------------------------------

def mask_from_image_data(image_data, native_size: int = 256) -> Image.Image:
    if image_data is None:
        return Image.new("L", (native_size, native_size), 0)
    arr = np.asarray(image_data)
    if arr.ndim < 3 or arr.shape[2] < 3:
        return Image.new("L", (native_size, native_size), 0)
    r = arr[..., 0].astype(int)
    g = arr[..., 1].astype(int)
    b = arr[..., 2].astype(int)
    magenta = (r > g + 30) & (b > g + 30) & (r > 100) & (b > 100)
    pil = Image.fromarray((magenta * 255).astype(np.uint8), mode="L")
    if pil.size != (native_size, native_size):
        pil = pil.resize((native_size, native_size), Image.NEAREST)
    return pil


def mask_union(a: Image.Image, b: Image.Image, native_size: int = 256) -> Image.Image:
    a_np = np.asarray(a.convert("L").resize((native_size, native_size), Image.NEAREST))
    b_np = np.asarray(b.convert("L").resize((native_size, native_size), Image.NEAREST))
    return Image.fromarray(np.maximum(a_np, b_np), mode="L")


def shift_mask(mask: Image.Image, dx: int, dy: int, native_size: int = 256) -> Image.Image:
    m = mask.convert("L")
    if m.size != (native_size, native_size):
        m = m.resize((native_size, native_size), Image.NEAREST)
    canvas = Image.new("L", m.size, 0)
    canvas.paste(m, (int(dx), int(dy)))
    return canvas


def _apply_edge_crop(arr: np.ndarray, top: int, bottom: int,
                     left: int, right: int) -> np.ndarray:
    out = arr.copy()
    if top > 0:    out[:top]     = 0
    if bottom > 0: out[-bottom:] = 0
    if left > 0:   out[:, :left]  = 0
    if right > 0:  out[:, -right:] = 0
    return out


def crop_mask(mask: Image.Image, top: int, bottom: int,
              left: int, right: int, native_size: int = 256) -> Image.Image:
    m = np.asarray(mask.convert("L").resize((native_size, native_size), Image.NEAREST))
    return Image.fromarray(_apply_edge_crop(m, top, bottom, left, right), mode="L")


def crop_image(img: Image.Image, top: int, bottom: int,
               left: int, right: int, native_size: int = 256) -> Image.Image:
    arr = np.asarray(img.convert("RGB").resize((native_size, native_size), Image.NEAREST))
    return Image.fromarray(_apply_edge_crop(arr, top, bottom, left, right), mode="RGB")


def render_crop_mode(con: Client, img: Image.Image, current_mask: Image.Image,
                     row: int, tile_id: str, size: int) -> None:
    if st.session_state.pop("crop_needs_reset", False):
        for k in ("crop_top", "crop_bottom", "crop_left", "crop_right"):
            st.session_state[k] = 0
    for k in ("crop_top", "crop_bottom", "crop_left", "crop_right"):
        if k not in st.session_state:
            st.session_state[k] = 0

    def reset() -> None:
        st.session_state.crop_needs_reset = True

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.slider("top (px)",    0, 128, key="crop_top")
    with c2: st.slider("bottom (px)", 0, 128, key="crop_bottom")
    with c3: st.slider("left (px)",   0, 128, key="crop_left")
    with c4: st.slider("right (px)",  0, 128, key="crop_right")

    t, b, l, r = (st.session_state[k] for k in
                  ("crop_top", "crop_bottom", "crop_left", "crop_right"))
    cropped_img  = crop_image(img, t, b, l, r)
    cropped_mask = crop_mask(current_mask, t, b, l, r)

    col1, col2 = st.columns(2)
    with col1: st.image(img.convert("RGB"), caption="original", width=size)
    with col2: st.image(overlay_mask(cropped_img, cropped_mask),
                        caption=f"preview  top={t}  bot={b}  left={l}  right={r}",
                        width=size)

    has_fix = get_fix(con, row) is not None or get_image_fix(con, row) is not None
    s1, s2, s3 = st.columns(3)
    with s1:
        if shortcut_button("Save crop (s)", shortcut="s", use_container_width=True):
            set_image_fix(con, row, tile_id, cropped_img)
            set_fix(con, row, tile_id, cropped_mask)
            set_decision(con, row, tile_id, "needs_fix")
            _mark_local(row); _load_stats_cache(con)
            st.session_state.crop_mode = False
            reset(); st.rerun()
    with s2:
        if shortcut_button("Cancel (c)", shortcut="c", use_container_width=True):
            st.session_state.crop_mode = False
            reset(); st.rerun()
    with s3:
        if st.button("Clear saved fix", use_container_width=True, disabled=not has_fix):
            clear_fix(con, row); clear_image_fix(con, row)
            reset(); st.rerun()
    st.caption("Sliders black out image + zero mask on each edge. **s** = save, **c** = cancel.")


def render_shift_mode(con: Client, img: Image.Image, current_mask: Image.Image,
                      row: int, tile_id: str, size: int) -> None:
    for k, v in (("shift_dx", 0), ("shift_dy", 0), ("shift_step", 2)):
        if k not in st.session_state:
            st.session_state[k] = v

    def nudge(dx: int = 0, dy: int = 0) -> None:
        st.session_state.shift_dx += dx * st.session_state.shift_step
        st.session_state.shift_dy += dy * st.session_state.shift_step

    def reset() -> None:
        st.session_state.shift_dx = 0
        st.session_state.shift_dy = 0

    st.write(f"### Shift  Δx=**{st.session_state.shift_dx}**  "
             f"Δy=**{st.session_state.shift_dy}**  px")

    c_step, c_l, c_u, c_d, c_r, c_reset = st.columns(6)
    with c_step: st.select_slider("step (px)", options=[1, 2, 5, 10], key="shift_step")
    with c_l:
        if shortcut_button("← left",    shortcut="arrowleft",  use_container_width=True):
            nudge(dx=-1); st.rerun()
    with c_u:
        if shortcut_button("↑ up",      shortcut="arrowup",    use_container_width=True):
            nudge(dy=-1); st.rerun()
    with c_d:
        if shortcut_button("↓ down",    shortcut="arrowdown",  use_container_width=True):
            nudge(dy=1);  st.rerun()
    with c_r:
        if shortcut_button("→ right",   shortcut="arrowright", use_container_width=True):
            nudge(dx=1);  st.rerun()
    with c_reset:
        if shortcut_button("Reset (0)", shortcut="0",          use_container_width=True):
            reset(); st.rerun()

    shifted = shift_mask(current_mask, st.session_state.shift_dx, st.session_state.shift_dy)
    col1, col2 = st.columns(2)
    with col1: st.image(img.convert("RGB"), caption="original", width=size)
    with col2: st.image(overlay_mask(img, shifted),
                        caption=f"shifted ({st.session_state.shift_dx}, "
                                f"{st.session_state.shift_dy})", width=size)

    s1, s2, s3 = st.columns(3)
    with s1:
        if shortcut_button("Save shift (s)", shortcut="s", use_container_width=True):
            set_fix(con, row, tile_id, shifted)
            set_decision(con, row, tile_id, "needs_fix")
            _mark_local(row); _load_stats_cache(con)
            st.session_state.shift_mode = False
            reset(); st.rerun()
    with s2:
        if shortcut_button("Cancel (c)", shortcut="c", use_container_width=True):
            st.session_state.shift_mode = False
            reset(); st.rerun()
    with s3:
        if st.button("Clear saved fix", use_container_width=True,
                     disabled=get_fix(con, row) is None):
            clear_fix(con, row); reset(); st.rerun()
    st.caption("Arrow keys to nudge. **s** = save, **c** = cancel, **0** = reset.")


def render_draw_mode(con: Client, order: np.ndarray, img: Image.Image,
                     current_mask: Image.Image, row: int, tile_id: str, size: int,
                     density_map, bucket_pos) -> None:
    st_canvas = get_st_canvas()
    if "draw_tool" not in st.session_state:
        st.session_state.draw_tool = "polygon (click vertices)"

    tool = st.radio("tool",
                    ["polygon (click vertices)", "rectangle (drag)",
                     "transform (move / rotate / scale)"],
                    horizontal=True, key="draw_tool")
    drawing_mode = {"polygon (click vertices)": "polygon",
                    "rectangle (drag)": "rect",
                    "transform (move / rotate / scale)": "transform"}[tool]

    left, right = st.columns(2)
    with left:
        st.image(overlay_mask(img, current_mask), caption="reference", width=size)
    with right:
        bg = (img.convert("RGB") if img.mode != "RGB" else img).resize(
            (size, size), Image.NEAREST)
        canvas = st_canvas(
            fill_color="rgba(220,0,220,0.40)", stroke_color="rgb(220,0,220)",
            stroke_width=2, background_image=bg, drawing_mode=drawing_mode,
            update_streamlit=True, height=size, width=size, display_toolbar=True,
            key=f"draw_canvas_{row}_{size}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("save — REPLACE mask", use_container_width=True, type="primary"):
            set_fix(con, row, tile_id, mask_from_image_data(canvas.image_data))
            set_decision(con, row, tile_id, "needs_fix")
            _mark_local(row); _load_stats_cache(con)
            st.session_state.bbox_mode = False
            advance_adaptive(order, density_map, bucket_pos); st.rerun()
    with c2:
        if st.button("save — ADD to existing", use_container_width=True):
            drawn = mask_from_image_data(canvas.image_data)
            set_fix(con, row, tile_id, mask_union(current_mask, drawn))
            set_decision(con, row, tile_id, "needs_fix")
            _mark_local(row); _load_stats_cache(con)
            st.session_state.bbox_mode = False
            advance_adaptive(order, density_map, bucket_pos); st.rerun()
    with c3:
        if st.button("cancel", use_container_width=True):
            st.session_state.bbox_mode = False; st.rerun()
    with c4:
        if st.button("clear saved fix", use_container_width=True,
                     disabled=get_fix(con, row) is None):
            clear_fix(con, row); st.rerun()


# ---- App ---------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="fAIr label reviewer", layout="wide")
    clear_shortcuts()

    order = load_order(str(RUN_DIR / "fps_order.npy"))
    ds    = load_split(DATASET, SPLIT)
    con   = open_db()

    run_path    = str(RUN_DIR)
    density_map = load_density_map(run_path)
    bucket_pos  = build_bucket_positions(run_path, order.tobytes()) if density_map else {}

    for key, val in (("cursor", None), ("img_size", 420),
                     ("bbox_mode", False), ("shift_mode", False), ("crop_mode", False)):
        if key not in st.session_state:
            st.session_state[key] = val

    # Load Supabase data once per session open (not on every slider rerun)
    if "_reviewed" not in st.session_state:
        with st.spinner("Loading labels…"):
            _load_stats_cache(con)

    if st.session_state.cursor is None:
        st.session_state.cursor = 0
        advance_adaptive(order, density_map, bucket_pos)

    cur    = st.session_state.cursor
    row    = int(order[cur])
    sample = ds[row]
    img: Image.Image       = sample["image"]
    orig_mask: Image.Image = sample["mask"]
    tile_id = str(sample.get("tile_id", row))

    fixed_mask = get_fix(con, row)
    fixed_img  = get_image_fix(con, row)
    display_mask = fixed_mask if fixed_mask is not None else orig_mask
    display_img  = fixed_img  if fixed_img  is not None else img

    def decide(d: str) -> None:
        set_decision(con, row, tile_id, d)
        _mark_local(row)
        _load_stats_cache(con)  # refresh cache after each decision
        advance_adaptive(order, density_map, bucket_pos)

    in_fix_mode = (st.session_state.bbox_mode or
                   st.session_state.shift_mode or
                   st.session_state.crop_mode)

    # --- Sidebar ---
    with st.sidebar:
        st.slider("image size (px)", 200, 700, key="img_size", step=20)
        if not in_fix_mode:
            st.divider()
            stats = st.session_state.get("_stats", {})
            st.metric("keep",      stats.get("keep", 0))
            st.metric("drop",      stats.get("drop", 0))
            st.metric("needs_fix", stats.get("needs_fix", 0))

            if density_map:
                st.divider()
                st.caption("kept chips by density bucket")
                kept_db: set[int] = st.session_state.get("_kept", set())
                bucket_counts = {i: 0 for i in range(N_BUCKETS)}
                for r in kept_db:
                    bucket_counts[density_bucket(density_map.get(r, 0))] += 1
                target = max(bucket_counts.values(), default=1) or 1
                for i, label in enumerate(BUCKET_LABELS):
                    c = bucket_counts[i]
                    st.progress(c / target, text=f"{label}: {c}")
                cur_n = density_map.get(row, 0)
                st.caption(f"this chip: **{cur_n}** instances "
                           f"({BUCKET_LABELS[density_bucket(cur_n)]})")

    # --- Header ---
    total   = len(order)
    stats   = st.session_state.get("_stats", {})
    labeled = sum(stats.values())
    st.progress(labeled / total,
                text=f"{labeled} / {total} reviewed  ·  pos {cur + 1}/{total}")

    # --- Fix modes ---
    if st.session_state.bbox_mode:
        render_draw_mode(con, order, img, display_mask, row, tile_id,
                         st.session_state.img_size, density_map, bucket_pos)
        return
    if st.session_state.shift_mode:
        render_shift_mode(con, img, display_mask, row, tile_id, st.session_state.img_size)
        return
    if st.session_state.crop_mode:
        render_crop_mode(con, img, display_mask, row, tile_id, st.session_state.img_size)
        return

    # --- Images (above buttons so they're visible without scrolling in landscape) ---
    size = st.session_state.img_size
    col1, col2 = st.columns(2)
    with col1:
        cap_img = "image (CROPPED)" if fixed_img is not None else "original"
        st.image(display_img.convert("RGB"), caption=cap_img, width=size)
    with col2:
        if fixed_mask is not None and fixed_img is not None:
            cap = "mask + image (CROP fix)"
        elif fixed_mask is not None:
            cap = "mask (FIXED)"
        else:
            cap = "mask overlay"
        st.image(overlay_mask(display_img, display_mask), caption=cap, width=size)

    # --- Action buttons (two rows for mobile friendliness) ---
    clicked: str | None = None
    b1, b2, b3 = st.columns(3)
    with b1:
        if shortcut_button("✅ Keep",      shortcut="k", use_container_width=True, type="primary"): clicked = "keep"
    with b2:
        if shortcut_button("❌ Drop",      shortcut="d", use_container_width=True): clicked = "drop"
    with b3:
        if shortcut_button("🔧 Needs fix", shortcut="n", use_container_width=True): clicked = "needs_fix"
    b4, b5, b6, b7 = st.columns(4)
    with b4:
        if shortcut_button("⬅ Back", shortcut="b", use_container_width=True):
            go_back(); st.rerun()
    with b5:
        if st.button("Shift", use_container_width=True):
            st.session_state.shift_mode = True; st.rerun()
    with b6:
        if st.button("Draw", use_container_width=True):
            st.session_state.bbox_mode = True; st.rerun()
    with b7:
        if st.button("Crop", use_container_width=True):
            st.session_state.crop_mode = True; st.rerun()

    meta_bits = [f"`tile_id`={tile_id}"] + [
        f"`{k}`={sample[k]}"
        for k in ("tile_z", "tile_x", "tile_y", "num_buildings") if k in sample
    ]
    st.caption("  ·  ".join(meta_bits))
    prev = get_decision(con, row)
    if prev:
        st.info(f"previously: **{prev}**"
                + ("  ·  fix saved" if fixed_mask is not None else ""))

    if clicked is not None:
        decide(clicked)
        st.rerun()

    with st.expander("jump"):
        j = st.number_input("position (1-based)", min_value=1, max_value=total, value=cur + 1)
        if st.button("go"):
            st.session_state.cursor = int(j) - 1; st.rerun()


if __name__ == "__main__":
    main()
