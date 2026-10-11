"""SQLite storage: image index, tags and search."""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS roots (
    id   INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL
);
CREATE TABLE IF NOT EXISTS images (
    id      INTEGER PRIMARY KEY,
    path    TEXT UNIQUE NOT NULL,
    name    TEXT NOT NULL,
    size    INTEGER NOT NULL,
    mtime   REAL NOT NULL,
    width   INTEGER,
    height  INTEGER,
    taken   TEXT,
    missing INTEGER NOT NULL DEFAULT 0,
    rotation INTEGER NOT NULL DEFAULT 0  -- degrees clockwise, set by the user; the file is not changed
);
CREATE TABLE IF NOT EXISTS tags (
    id   INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);
CREATE TABLE IF NOT EXISTS image_tags (
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    tag_id   INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (image_id, tag_id)
);
CREATE INDEX IF NOT EXISTS idx_image_tags_tag ON image_tags(tag_id);
-- tags the user removed from a photo; automatic tagging never re-adds them
CREATE TABLE IF NOT EXISTS rejected (
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    tag_id   INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (image_id, tag_id)
);
CREATE TABLE IF NOT EXISTS embeddings (
    image_id INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
    model    TEXT NOT NULL,
    mtime    REAL NOT NULL,
    vec      BLOB NOT NULL
);
-- words the model looks for: Bulgarian tag -> description the model reads
CREATE TABLE IF NOT EXISTS vocab (
    tag     TEXT PRIMARY KEY,
    prompt  TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1
);
-- text read from a photo (OCR); norm is the lowercased text used for search
CREATE TABLE IF NOT EXISTS ocr (
    image_id INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
    mtime    REAL NOT NULL,
    text     TEXT NOT NULL,
    norm     TEXT NOT NULL,
    file_mtime REAL,  -- set when the text comes from a .md file (see ocr.py), which Tesseract never overwrites
    review   INTEGER NOT NULL DEFAULT 0  -- the .md file asks for its text to be checked by hand
);
-- photos the user asked to read, whatever the OCR settings
CREATE TABLE IF NOT EXISTS ocr_requests (
    image_id INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE
);
-- "if the text contains one of these words, add this tag"
CREATE TABLE IF NOT EXISTS text_rules (
    id    INTEGER PRIMARY KEY,
    words TEXT NOT NULL,
    tag   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# source of a tag on a photo: 'manual' (set or confirmed by the user),
# 'auto' (recognised by the model or learned from the user's tags), 'folder',
# 'text' (a rule matched the text read from the photo, or the text is like that
# of photos you tagged). For automatic tags, score is the confidence shown to
# the user (0..1, NULL when there is none) and note says where the tag came from.
# A confirmed automatic tag becomes 'manual' and keeps its old source in `was`,
# so that it can be turned back into a suggestion.
AUTO_SOURCES = ("auto", "folder", "text")

# Common Bulgarian endings (articles, plurals) stripped when matching file paths,
# so that "вилата" also finds a folder called "Вила".
_BG_ENDINGS = ("ите", "ата", "ята", "ото", "ето", "ът", "ят", "та", "то", "те", "и", "а", "я", "е", "о")


def connect(db_path: str | os.PathLike) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    # SQLite's LOWER() only folds ASCII; Python folds Cyrillic too.
    conn.create_function("pylower", 1, lambda s: s.lower() if s is not None else None, deterministic=True)
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(image_tags)")}
    if "source" not in cols:
        with conn:
            conn.execute("ALTER TABLE image_tags ADD COLUMN source TEXT NOT NULL DEFAULT 'manual'")
            conn.execute("ALTER TABLE image_tags ADD COLUMN score REAL")
    if "note" not in cols:
        with conn:
            conn.execute("ALTER TABLE image_tags ADD COLUMN note TEXT")
            # score used to hold the model's raw probability; it is now a confidence
            # shown to the user, and the next run fills it in again
            conn.execute("UPDATE image_tags SET score = NULL WHERE source != 'manual'")
    if "was" not in cols:
        with conn:
            conn.execute("ALTER TABLE image_tags ADD COLUMN was TEXT")
    if "rotation" not in {r[1] for r in conn.execute("PRAGMA table_info(images)")}:
        with conn:
            conn.execute("ALTER TABLE images ADD COLUMN rotation INTEGER NOT NULL DEFAULT 0")
    if "file_mtime" not in {r[1] for r in conn.execute("PRAGMA table_info(ocr)")}:
        with conn:
            conn.execute("ALTER TABLE ocr ADD COLUMN file_mtime REAL")
            conn.execute("ALTER TABLE ocr ADD COLUMN review INTEGER NOT NULL DEFAULT 0")
    if not conn.execute("SELECT 1 FROM settings WHERE key = 'vocab_seeded'").fetchone():
        from .vocabulary import DEFAULT_VOCAB

        with conn:
            conn.executemany("INSERT OR IGNORE INTO vocab(tag, prompt) VALUES (?, ?)", DEFAULT_VOCAB)
            conn.execute("INSERT INTO settings VALUES ('vocab_seeded', '1')")


# ---------------------------------------------------------------- settings

DEFAULT_SETTINGS = {
    "auto_enabled": "1",
    "vocab_threshold": "0.01",   # SigLIP probability; see ai/reports for calibration
    "max_auto_tags": "5",
    "learn_enabled": "1",
    "learn_min_examples": "3",
    "learn_similarity": "0.85",  # cosine similarity to the nearest photo you tagged
    "folder_enabled": "1",
    "ocr_enabled": "0",          # reading text is slower, so the user switches it on
    "ocr_scope": "tags",         # 'tags': only photos with one of ocr_tags; 'all'
    "ocr_tags": "документ, фактура, скрийншот",
    "text_learn_enabled": "1",
    "text_learn_confidence": "0.5",  # 1.0 = as close to the examples as they are to each other
    "texts_dir": "",             # folder of the .md text files; "" = texts/ next to the database
}


def get_settings(conn: sqlite3.Connection) -> dict[str, str]:
    out = dict(DEFAULT_SETTINGS)
    out.update({r[0]: r[1] for r in conn.execute("SELECT key, value FROM settings") if r[0] in DEFAULT_SETTINGS})
    return out


def set_settings(conn: sqlite3.Connection, values: dict) -> None:
    with conn:
        for k, v in values.items():
            if k in DEFAULT_SETTINGS:
                conn.execute("INSERT OR REPLACE INTO settings(key, value) VALUES (?, ?)", (k, str(v)))


# ---------------------------------------------------------------- vocabulary

def vocab(conn: sqlite3.Connection) -> list[dict]:
    return [
        {"tag": r[0], "prompt": r[1], "enabled": bool(r[2])}
        for r in conn.execute("SELECT tag, prompt, enabled FROM vocab ORDER BY tag")
    ]


def set_vocab(conn: sqlite3.Connection, tag: str, prompt: str = "", enabled: bool = True) -> None:
    tag = normalize_tag(tag)
    if not tag:
        raise ValueError("Празна дума")
    prompt = " ".join(prompt.split()) or f"снимка на {tag}"
    with conn:
        conn.execute("INSERT OR REPLACE INTO vocab(tag, prompt, enabled) VALUES (?, ?, ?)", (tag, prompt, int(enabled)))


def delete_vocab(conn: sqlite3.Connection, tag: str) -> None:
    with conn:
        conn.execute("DELETE FROM vocab WHERE tag = ?", (normalize_tag(tag),))


# ---------------------------------------------------------------- tags

def normalize_tag(raw: str) -> str:
    """Lowercase, trim, collapse spaces, drop a leading '#'."""
    tag = " ".join(raw.strip().lstrip("#").split()).lower()
    return tag.strip(",;")


def split_tags(text: str) -> list[str]:
    """Split user input like 'бойлер, вила' into normalized tags."""
    out = []
    for part in re.split(r"[,;\n]+", text):
        tag = normalize_tag(part)
        if tag and tag not in out:
            out.append(tag)
    return out


def _tag_id(conn: sqlite3.Connection, name: str) -> int:
    conn.execute("INSERT OR IGNORE INTO tags(name) VALUES (?)", (name,))
    return conn.execute("SELECT id FROM tags WHERE name = ?", (name,)).fetchone()[0]


def add_tags(conn: sqlite3.Connection, image_ids: list[int], tags: list[str]) -> None:
    """Tag photos by hand. Also confirms a matching automatic tag."""
    with conn:
        for tag in {normalize_tag(t) for t in tags} - {""}:
            tid = _tag_id(conn, tag)
            rows = [(int(i), tid) for i in image_ids]
            conn.executemany(
                "INSERT INTO image_tags(image_id, tag_id, source) VALUES (?, ?, 'manual') "
                "ON CONFLICT(image_id, tag_id) DO UPDATE SET source = 'manual', score = NULL, note = NULL, was = NULL",
                rows,
            )
            conn.executemany("DELETE FROM rejected WHERE image_id = ? AND tag_id = ?", rows)


def confirm_auto(conn: sqlite3.Connection, image_ids: list[int], tags: list[str] | None = None) -> None:
    """Turn automatic tags into confirmed ones (all of them, or only `tags`)."""
    with conn:
        marks = ",".join("?" * len(image_ids))
        sql = f"UPDATE image_tags SET was = source, source = 'manual' WHERE image_id IN ({marks}) AND source != 'manual'"
        params: list = [int(i) for i in image_ids]
        if tags:
            sql += f" AND tag_id IN (SELECT id FROM tags WHERE name IN ({','.join('?' * len(tags))}))"
            params += [normalize_tag(t) for t in tags]
        conn.execute(sql, params)


def unconfirm_auto(conn: sqlite3.Connection, image_ids: list[int], tags: list[str]) -> None:
    """Turn confirmed automatic tags back into suggestions. Tags typed by hand stay."""
    with conn:
        conn.execute(
            f"UPDATE image_tags SET source = was, was = NULL WHERE image_id IN ({','.join('?' * len(image_ids))}) "
            f"AND source = 'manual' AND was IS NOT NULL "
            f"AND tag_id IN (SELECT id FROM tags WHERE name IN ({','.join('?' * len(tags))}))",
            [int(i) for i in image_ids] + [normalize_tag(t) for t in tags],
        )


def remove_tags(conn: sqlite3.Connection, image_ids: list[int], tags: list[str]) -> None:
    """Remove tags. Automatic tagging will not put them back on these photos."""
    with conn:
        for tag in {normalize_tag(t) for t in tags}:
            row = conn.execute("SELECT id FROM tags WHERE name = ?", (tag,)).fetchone()
            if row:
                pairs = [(int(i), row[0]) for i in image_ids]
                conn.executemany("DELETE FROM image_tags WHERE image_id = ? AND tag_id = ?", pairs)
                conn.executemany("INSERT OR IGNORE INTO rejected(image_id, tag_id) VALUES (?, ?)", pairs)


def rename_tag(conn: sqlite3.Connection, old: str, new: str) -> None:
    """Rename a tag; if the new name already exists, the two are merged."""
    old, new = normalize_tag(old), normalize_tag(new)
    if not old or not new or old == new:
        return
    with conn:
        row = conn.execute("SELECT id FROM tags WHERE name = ?", (old,)).fetchone()
        if not row:
            return
        new_id = _tag_id(conn, new)
        conn.execute(
            "INSERT INTO image_tags(image_id, tag_id, source, score, note, was) "
            "SELECT image_id, ?, source, score, note, was FROM image_tags WHERE tag_id = ? "
            "ON CONFLICT(image_id, tag_id) DO UPDATE SET source = "
            "CASE WHEN excluded.source = 'manual' THEN 'manual' ELSE image_tags.source END",
            (new_id, row[0]),
        )
        conn.execute(
            "INSERT OR IGNORE INTO rejected(image_id, tag_id) SELECT image_id, ? FROM rejected WHERE tag_id = ?",
            (new_id, row[0]),
        )
        conn.execute("DELETE FROM tags WHERE id = ?", (row[0],))
        if conn.execute("SELECT 1 FROM vocab WHERE tag = ?", (old,)).fetchone():
            conn.execute("INSERT OR IGNORE INTO vocab(tag, prompt, enabled) SELECT ?, prompt, enabled FROM vocab WHERE tag = ?", (new, old))
            conn.execute("DELETE FROM vocab WHERE tag = ?", (old,))


def tag_counts(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT t.name, COUNT(*) AS n, SUM(it.source != 'manual') AS auto FROM tags t "
        "JOIN image_tags it ON it.tag_id = t.id "
        "JOIN images i ON i.id = it.image_id AND i.missing = 0 "
        "GROUP BY t.id ORDER BY n DESC, t.name"
    ).fetchall()
    return [{"name": r["name"], "count": r["n"], "auto": r["auto"]} for r in rows]


def tags_for(conn: sqlite3.Connection, image_ids: list[int]) -> dict[int, dict]:
    """{image_id: {"tags": all tag names, "auto": the unconfirmed automatic ones,
    "info": {auto tag: {"source", "score", "note"}}, "confirmed": automatic tags you confirmed}}"""
    out = {i: {"tags": [], "auto": [], "info": {}, "confirmed": []} for i in image_ids}
    if not image_ids:
        return out
    marks = ",".join("?" * len(image_ids))
    for r in conn.execute(
        f"SELECT it.image_id, t.name, it.source, it.score, it.note, it.was FROM image_tags it JOIN tags t ON t.id = it.tag_id "
        f"WHERE it.image_id IN ({marks}) ORDER BY it.source = 'manual' DESC, t.name",
        image_ids,
    ):
        out[r[0]]["tags"].append(r[1])
        if r[2] != "manual":
            out[r[0]]["auto"].append(r[1])
            out[r[0]]["info"][r[1]] = {"source": r[2], "score": r[3], "note": r[4]}
        elif r[5]:
            out[r[0]]["confirmed"].append(r[1])
    return out


# ---------------------------------------------------------------- search

def parse_query(q: str) -> tuple[list[str], list[str]]:
    """Return (include, exclude) terms. Supports "quoted phrases" and -negation."""
    include, exclude = [], []
    for m in re.finditer(r'(-?)"([^"]+)"|(-?)([^\s,]+)', q):
        neg = m.group(1) or m.group(3)
        term = normalize_tag(m.group(2) or m.group(4) or "")
        if not term or term == "-":
            continue
        (exclude if neg else include).append(term)
    return include, exclude


def _stem(term: str) -> str:
    for end in _BG_ENDINGS:
        if term.endswith(end) and len(term) - len(end) >= 3:
            return term[: -len(end)]
    return term


def term_matches_tag(term: str, tag: str) -> bool:
    """Prefix match per word, tolerant to short Bulgarian endings.

    "бойл" -> "бойлер", "вилата" -> "вила", "къща" -> "стара къща".
    """
    if tag.startswith(term):
        return True
    for word in [tag, *re.split(r"[\s\-_/]+", tag)]:
        if not word:
            continue
        if word.startswith(term):
            return True
        if len(word) >= 3 and term.startswith(word) and len(term) - len(word) <= 3:
            return True
    return False


def search(
    conn: sqlite3.Connection,
    q: str = "",
    untagged: bool = False,
    review: bool = False,
    check_text: bool = False,
    sort: str = "newest",
    offset: int = 0,
    limit: int = 200,
) -> dict:
    include, exclude = parse_query(q)
    all_tags = conn.execute("SELECT id, name FROM tags").fetchall()

    where, params = ["i.missing = 0"], []

    def term_clause(term: str) -> str:
        ids = [str(r["id"]) for r in all_tags if term_matches_tag(term, r["name"])]
        stem = _stem(term)
        params.extend([f"%{stem}%"] * 2)
        # the path and the text read from the photo (OCR) match the stem: "вилата" -> "Вила"
        other = "pylower(i.path) LIKE ? OR i.id IN (SELECT image_id FROM ocr WHERE norm LIKE ?)"
        if not ids:
            return f"({other})"
        return (
            f"(i.id IN (SELECT image_id FROM image_tags WHERE tag_id IN ({','.join(ids)}))"
            f" OR {other})"
        )

    for term in include:
        where.append(term_clause(term))
    for term in exclude:
        where.append("NOT " + term_clause(term))
    if untagged:
        where.append("i.id NOT IN (SELECT image_id FROM image_tags WHERE source = 'manual')")
    if review:
        where.append("i.id IN (SELECT image_id FROM image_tags WHERE source != 'manual')")
    if check_text:
        where.append("i.id IN (SELECT image_id FROM ocr WHERE review = 1)")

    order = {
        "newest": "COALESCE(i.taken, datetime(i.mtime, 'unixepoch')) DESC, i.path",
        "oldest": "COALESCE(i.taken, datetime(i.mtime, 'unixepoch')) ASC, i.path",
        "name": "i.name COLLATE NOCASE, i.path",
        "path": "i.path",
    }.get(sort, "i.path")

    sql_where = " AND ".join(where)
    total = conn.execute(f"SELECT COUNT(*) FROM images i WHERE {sql_where}", params).fetchone()[0]
    rows = conn.execute(
        f"SELECT i.id, i.path, i.name, i.width, i.height, i.taken, i.rotation FROM images i "
        f"WHERE {sql_where} ORDER BY {order} LIMIT ? OFFSET ?",
        [*params, int(limit), int(offset)],
    ).fetchall()
    tag_map = tags_for(conn, [r["id"] for r in rows])
    items = [
        {
            "id": r["id"],
            "path": r["path"],
            "name": r["name"],
            "width": r["width"],
            "height": r["height"],
            "taken": r["taken"],
            "tags": tag_map[r["id"]]["tags"],
            "auto": tag_map[r["id"]]["auto"],
            "info": tag_map[r["id"]]["info"],
            "confirmed": tag_map[r["id"]]["confirmed"],
            "rotation": r["rotation"],
        }
        for r in rows
    ]
    return {"total": total, "items": items, "include": include, "exclude": exclude}


def rotate(conn: sqlite3.Connection, image_ids: list[int], degrees: int) -> dict[int, int]:
    """Turn photos by 90° steps (clockwise positive). Only the view changes, never
    the file. The photos are recognised again, text included if it was read."""
    ids = [int(i) for i in image_ids]
    marks = ",".join("?" * len(ids))
    with conn:
        conn.execute(f"UPDATE images SET rotation = ((rotation + ?) % 360 + 360) % 360 WHERE id IN ({marks})",
                     [int(degrees) // 90 * 90, *ids])
        conn.execute(f"DELETE FROM embeddings WHERE image_id IN ({marks})", ids)
        conn.execute(f"INSERT OR IGNORE INTO ocr_requests(image_id) SELECT image_id FROM ocr WHERE image_id IN ({marks})", ids)
    return {r[0]: r[1] for r in conn.execute(f"SELECT id, rotation FROM images WHERE id IN ({marks})", ids)}


def turn(im: Image.Image, rotation: int) -> Image.Image:
    """Apply the user's rotation to an already upright (EXIF-transposed) image."""
    return im.rotate(-rotation, expand=True) if rotation % 360 else im


def get_image(conn: sqlite3.Connection, image_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM images WHERE id = ? AND missing = 0", (image_id,)).fetchone()


# ---------------------------------------------------------------- scanning

def roots(conn: sqlite3.Connection) -> list[dict]:
    out = []
    for r in conn.execute("SELECT id, path FROM roots ORDER BY path"):
        n = conn.execute(
            "SELECT COUNT(*) FROM images WHERE missing = 0 AND path LIKE ? ESCAPE '\\'",
            (_under(r["path"]),),
        ).fetchone()[0]
        out.append({"id": r["id"], "path": r["path"], "count": n, "exists": os.path.isdir(r["path"])})
    return out


def _under(root: str) -> str:
    root = root.rstrip(os.sep)
    esc = root.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return esc + os.sep + "%"


def add_root(conn: sqlite3.Connection, path: str) -> str:
    real = str(Path(path).expanduser().resolve())
    if not os.path.isdir(real):
        raise ValueError(f"Папката не съществува: {real}")
    with conn:
        conn.execute("INSERT OR IGNORE INTO roots(path) VALUES (?)", (real,))
    return real


def remove_root(conn: sqlite3.Connection, path: str) -> None:
    """Forget a folder. Its images are hidden but their tags are kept."""
    with conn:
        conn.execute("DELETE FROM roots WHERE path = ?", (path,))
        conn.execute(
            "UPDATE images SET missing = 1 WHERE path LIKE ? ESCAPE '\\'", (_under(path),)
        )


def _read_meta(path: str) -> tuple[int | None, int | None, str | None]:
    try:
        with Image.open(path) as im:
            width, height = im.size
            taken = None
            try:
                exif = im.getexif()
                raw = exif.get_ifd(0x8769).get(0x9003) or exif.get(0x0132)  # DateTimeOriginal / DateTime
                if raw:
                    taken = datetime.strptime(str(raw).strip("\x00 "), "%Y:%m:%d %H:%M:%S").isoformat(" ")
            except (ValueError, KeyError, TypeError, AttributeError):
                pass
            if (exif_orient := im.getexif().get(0x0112)) in (5, 6, 7, 8):
                width, height = height, width
            return width, height, taken
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return None, None, None


def scan_root(conn: sqlite3.Connection, root: str, progress=None) -> dict:
    """Index images under root. Unchanged files are skipped.

    Files that disappeared are marked missing (tags kept). A new file with the
    same name and size as a missing one is treated as moved and keeps its tags.
    """
    stats = {"added": 0, "updated": 0, "moved": 0, "missing": 0, "seen": 0}
    seen: set[str] = set()
    relinked: set[int] = set()
    known = {
        r["path"]: r
        for r in conn.execute(
            "SELECT id, path, size, mtime, missing FROM images WHERE path LIKE ? ESCAPE '\\'",
            (_under(root),),
        )
    }
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for fn in sorted(filenames):
            if fn.startswith(".") or os.path.splitext(fn)[1].lower() not in IMAGE_EXTS:
                continue
            path = os.path.join(dirpath, fn)
            try:
                st = os.stat(path)
            except OSError:
                continue
            seen.add(path)
            stats["seen"] += 1
            if progress and stats["seen"] % 500 == 0:
                progress(stats["seen"])
            old = known.get(path)
            if old and old["size"] == st.st_size and old["mtime"] == st.st_mtime:
                if old["missing"]:
                    conn.execute("UPDATE images SET missing = 0 WHERE id = ?", (old["id"],))
                continue
            w, h, taken = _read_meta(path)
            if w is None:
                continue  # not a readable image
            if old:
                conn.execute(
                    "UPDATE images SET size=?, mtime=?, width=?, height=?, taken=?, missing=0 WHERE id=?",
                    (st.st_size, st.st_mtime, w, h, taken, old["id"]),
                )
                stats["updated"] += 1
                continue
            moved = next(
                (
                    r
                    for r in conn.execute(
                        "SELECT id, path, missing FROM images WHERE name = ? AND size = ?",
                        (fn, st.st_size),
                    )
                    if r["id"] not in relinked and (r["missing"] or not os.path.exists(r["path"]))
                ),
                None,
            )
            if moved:
                relinked.add(moved["id"])
                conn.execute(
                    "UPDATE images SET path=?, mtime=?, width=?, height=?, taken=?, missing=0 WHERE id=?",
                    (path, st.st_mtime, w, h, taken, moved["id"]),
                )
                stats["moved"] += 1
            else:
                conn.execute(
                    "INSERT INTO images(path, name, size, mtime, width, height, taken) VALUES (?,?,?,?,?,?,?)",
                    (path, fn, st.st_size, st.st_mtime, w, h, taken),
                )
                stats["added"] += 1
        conn.commit()
    gone = [
        r["id"]
        for p, r in known.items()
        if p not in seen and not r["missing"] and r["id"] not in relinked
    ]
    conn.executemany("UPDATE images SET missing = 1 WHERE id = ?", [(i,) for i in gone])
    stats["missing"] = len(gone)
    conn.commit()
    return stats


def scan_all(conn: sqlite3.Connection, progress=None) -> dict:
    total = {"added": 0, "updated": 0, "moved": 0, "missing": 0, "seen": 0}
    for r in roots(conn):
        if r["exists"]:
            for k, v in scan_root(conn, r["path"], progress).items():
                total[k] += v
    return total
