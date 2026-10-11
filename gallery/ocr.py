"""Text recognition (OCR) with Tesseract. Optional and off by default.

When switched on, the text of document photos is read once and stored. Search
then also finds words from that text. Two things tag photos by their text
(source 'text', shown for confirmation like the other automatic tags):

- rules like "ЕВН -> ток", when the text contains a word;
- learning: a tag you put by hand on a few documents is suggested for documents
  whose text shares their distinctive words, with a confidence.

Photos can also be read on request, one by one. Text checked by other means
(Claude, or corrected by hand in the viewer) lives in .md files and wins over
Tesseract; see "text files" below.
"""

from __future__ import annotations

import functools
import io
import math
import os
import re
import shutil
import sqlite3
import subprocess
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from . import db

LANGS = "bul+eng"
MAX_SIDE = 2000   # longer side in pixels; ~1 s per photo, larger is 3x slower for little gain
MIN_CONF = 40     # drop words Tesseract itself doubts; they are mostly noise
WORKERS = max(1, min(6, (os.cpu_count() or 2) // 2))
TIMEOUT = 30      # a few photos (textures, not text) make Tesseract run for minutes
SECONDS_PER_PHOTO = 2.0 / WORKERS  # measured on this machine with 6 workers, for the time estimate

_WORD = re.compile(r"\w", re.UNICODE)
_TOKEN = re.compile(r"[^\W\d_]{3,}")  # words of 3+ letters; numbers and OCR crumbs are noise


@functools.lru_cache(maxsize=1)
def available() -> tuple[bool, str]:
    exe = shutil.which("tesseract")
    if not exe:
        return False, "Tesseract не е инсталиран. Инсталирайте го с: sudo apt install tesseract-ocr tesseract-ocr-bul"
    try:
        langs = subprocess.run([exe, "--list-langs"], capture_output=True, text=True, timeout=20).stdout.split()
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"Tesseract не работи: {e}"
    if "bul" not in langs:
        return False, "Tesseract няма български език. Инсталирайте го с: sudo apt install tesseract-ocr-bul"
    return True, ""


def read_text(path: str, rotation: int = 0) -> str:
    """Recognise the text of one photo. Returns "" when there is none or it fails.

    Tesseract also finds text that lies sideways or upside down (--psm 1)."""
    try:
        with Image.open(path) as im:
            im.draft("L", (MAX_SIDE, MAX_SIDE))  # fast JPEG downscale
            im = db.turn(ImageOps.exif_transpose(im), rotation).convert("L")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return ""
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    im = ImageOps.autocontrast(im, cutoff=1)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    try:
        out = subprocess.run(
            ["tesseract", "stdin", "stdout", "-l", LANGS, "--psm", "1", "tsv"],
            input=buf.getvalue(), capture_output=True, timeout=TIMEOUT,
            env={**os.environ, "OMP_THREAD_LIMIT": "1"},  # we run several in parallel instead
        ).stdout.decode("utf-8", "replace")
    except (OSError, subprocess.SubprocessError):
        return ""
    return _text_from_tsv(out)


def _text_from_tsv(tsv: str) -> str:
    """Rebuild lines and paragraphs from Tesseract's word table, keeping confident words."""
    lines: dict[tuple, list[str]] = {}
    for row in tsv.splitlines()[1:]:
        f = row.split("\t")
        if len(f) != 12 or f[0] != "5":
            continue
        word = f[11].strip()
        try:
            conf = float(f[10])
        except ValueError:
            continue
        if word and conf >= MIN_CONF and _WORD.search(word):
            lines.setdefault((int(f[1]), int(f[2]), int(f[3]), int(f[4])), []).append(word)
    out, last_par = [], None
    for key, words in lines.items():
        if last_par is not None and key[:3] != last_par:
            out.append("")
        out.append(" ".join(words))
        last_par = key[:3]
    return "\n".join(out).strip()


def normalize(text: str) -> str:
    return " ".join(text.split()).lower()


# ---------------------------------------------------------------- which photos

def _scope_sql(settings: dict) -> tuple[str, list]:
    """SQL condition for the photos read automatically under these settings."""
    if settings["ocr_enabled"] != "1":
        return "0", []
    if settings["ocr_scope"] == "all":
        return "1", []
    # photos with one of the document tags, and the few tagged by hand: they are
    # the examples that learning from text needs
    tags = db.split_tags(settings["ocr_tags"])
    by_tag = f"t.name IN ({','.join('?' * len(tags))})" if tags else "0"
    manual = "it.source = 'manual'" if settings["text_learn_enabled"] == "1" else "0"
    return (
        f"i.id IN (SELECT it.image_id FROM image_tags it JOIN tags t ON t.id = it.tag_id "
        f"WHERE {by_tag} OR {manual})",
        tags,
    )


def pending(conn: sqlite3.Connection, settings: dict | None = None, limit: int = -1) -> list[sqlite3.Row]:
    """Photos to read: in scope or requested by hand, and not read since they last changed.
    Photos whose text comes from a text file are never read again."""
    scope, params = _scope_sql(settings or db.get_settings(conn))
    return conn.execute(
        "SELECT i.id, i.path, i.mtime, i.rotation FROM images i LEFT JOIN ocr o ON o.image_id = i.id "
        f"WHERE i.missing = 0 AND o.file_mtime IS NULL AND (i.id IN (SELECT image_id FROM ocr_requests) "
        f"OR (({scope}) AND (o.image_id IS NULL OR o.mtime != i.mtime))) "
        "ORDER BY i.id LIMIT ?",
        [*params, limit],
    ).fetchall()


def count_pending(conn: sqlite3.Connection, settings: dict) -> int:
    scope, params = _scope_sql(settings)
    return conn.execute(
        "SELECT COUNT(*) FROM images i LEFT JOIN ocr o ON o.image_id = i.id "
        f"WHERE i.missing = 0 AND o.file_mtime IS NULL AND (i.id IN (SELECT image_id FROM ocr_requests) "
        f"OR (({scope}) AND (o.image_id IS NULL OR o.mtime != i.mtime)))",
        params,
    ).fetchone()[0]


def request(conn: sqlite3.Connection, image_ids: list[int]) -> None:
    """Read these photos (again) at the next run, whatever the settings."""
    with conn:
        conn.executemany("INSERT OR IGNORE INTO ocr_requests(image_id) VALUES (?)", [(int(i),) for i in image_ids])


def status(conn: sqlite3.Connection, settings: dict | None = None) -> dict:
    ok, why = available()
    settings = settings or db.get_settings(conn)
    n = count_pending(conn, settings)
    return {
        "available": ok,
        "message": why,
        "read": conn.execute("SELECT COUNT(*) FROM ocr WHERE text != ''").fetchone()[0],
        "pending": n,
        "minutes": round(n * SECONDS_PER_PHOTO / 60, 1),
        "files": conn.execute("SELECT COUNT(*) FROM ocr WHERE file_mtime IS NOT NULL").fetchone()[0],
        "texts_dir": str(texts_root(conn)),
    }


def text_of(conn: sqlite3.Connection, image_id: int) -> dict:
    img = db.get_image(conn, image_id)
    if img is not None:
        f = text_file(texts_root(conn), img["path"])
        if f.is_file():
            _load_file(conn, img, f)
            meta, body = parse_md(f.read_text("utf-8"))
            return {"text": body, "queued": False, "file": str(f), "meta": meta}
    row = conn.execute("SELECT text FROM ocr WHERE image_id = ?", (image_id,)).fetchone()
    queued = conn.execute("SELECT 1 FROM ocr_requests WHERE image_id = ?", (image_id,)).fetchone()
    return {"text": row[0] if row else None, "queued": bool(queued), "file": None, "meta": {}}


# ---------------------------------------------------------------- text files
#
# Text checked outside Tesseract (by Claude, or corrected in the viewer) is kept
# in .md files under a folder the user chooses, never next to the photos. The
# folder mirrors the photo's full path: /a/b/IMG_1.jpg -> <texts>/a/b/IMG_1.md.
# Such text wins: it is what search and the text tags use, and Tesseract never
# reads that photo again.

_FRONT = re.compile(r"\A---\n(.*?)\n---[ \t]*\n?", re.S)


def texts_root(conn: sqlite3.Connection) -> Path:
    chosen = db.get_settings(conn)["texts_dir"].strip()
    if chosen:
        return Path(chosen).expanduser()
    main = conn.execute("PRAGMA database_list").fetchone()[2]
    return Path(main).parent / "texts"


def text_file(root: Path, image_path: str) -> Path:
    p = Path(image_path)
    return root / p.relative_to(p.anchor).with_suffix(".md")


def parse_md(raw: str) -> tuple[dict[str, str], str]:
    """Front matter as plain "key: value" lines, and the text below it."""
    m = _FRONT.match(raw.replace("\r\n", "\n"))
    if not m:
        return {}, raw.strip()
    meta = {}
    for line in m.group(1).splitlines():
        k, sep, v = line.partition(":")
        if sep and k.strip() and not line[:1].isspace():
            meta[k.strip()] = v.strip().strip("'\"")
    return meta, raw.replace("\r\n", "\n")[m.end():].strip()


def _set_front(raw_front: str, values: dict[str, str]) -> str:
    """Change or add these keys, keeping every other line as it was."""
    lines, left = [], dict(values)
    for line in raw_front.splitlines():
        k = line.partition(":")[0].strip()
        if k in left and not line[:1].isspace():
            lines.append(f"{k}: {left.pop(k)}")
        else:
            lines.append(line)
    lines += [f"{k}: {v}" for k, v in left.items()]
    return "\n".join(lines)


def save_text(conn: sqlite3.Connection, image_id: int, text: str, reviewed: bool) -> dict:
    """Write the corrected text of a photo to its .md file and use it from now on."""
    img = db.get_image(conn, image_id)
    if img is None:
        raise ValueError("Няма такава снимка")
    f = text_file(texts_root(conn), img["path"])
    now = datetime.now().isoformat(timespec="seconds")
    values = {"edited": "true", "edited_at": now}
    if reviewed:
        values["review"] = "false"
    if f.is_file():
        raw = f.read_text("utf-8").replace("\r\n", "\n")
        m = _FRONT.match(raw)
        front = _set_front(m.group(1) if m else "", values)
    else:
        front = _set_front("", {"source": Path(img["path"]).name, "source_path": img["path"],
                                "method": "manual", **values})
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".md.tmp")
    tmp.write_text(f"---\n{front}\n---\n\n{text.strip()}\n", "utf-8")
    os.replace(tmp, f)
    _load_file(conn, img, f)
    return text_of(conn, image_id)


def _load_file(conn: sqlite3.Connection, img, f: Path) -> bool:
    """Take the text of one .md file into the database, if it changed."""
    mtime = f.stat().st_mtime
    row = conn.execute("SELECT file_mtime FROM ocr WHERE image_id = ?", (img["id"],)).fetchone()
    if row and row[0] == mtime:
        return False
    meta, body = parse_md(f.read_text("utf-8"))
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO ocr(image_id, mtime, text, norm, file_mtime, review) VALUES (?, ?, ?, ?, ?, ?)",
            (img["id"], img["mtime"], body, normalize(body), mtime, meta.get("review", "").lower() == "true"),
        )
        conn.execute("DELETE FROM ocr_requests WHERE image_id = ?", (img["id"],))
    return True


def sync_files(conn: sqlite3.Connection) -> int:
    """Take new and changed .md files into the database. Text of a file that is
    gone is dropped, so Tesseract reads that photo again. Returns the changes."""
    root = texts_root(conn)
    changed = 0
    for img in conn.execute("SELECT id, path, mtime FROM images WHERE missing = 0").fetchall():
        f = text_file(root, img["path"])
        if f.is_file():
            changed += _load_file(conn, img, f)
    gone = [
        r[0] for r in conn.execute(
            "SELECT o.image_id, i.path FROM ocr o JOIN images i ON i.id = o.image_id WHERE o.file_mtime IS NOT NULL"
        ) if not text_file(root, r[1]).is_file()
    ]
    if gone:
        with conn:
            conn.executemany("DELETE FROM ocr WHERE image_id = ?", [(i,) for i in gone])
    return changed + len(gone)


def run_pending(conn, progress=None, stop: threading.Event | None = None) -> int:
    """Read all pending photos. Settings are re-checked as results come in, so
    switching OCR off stops the run. Returns how many were read."""
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        # all photos are queued at once so that one slow page never idles the workers
        while rows := pending(conn):
            futures = {pool.submit(read_text, r["path"], r["rotation"]): r for r in rows}
            ready: list[tuple[sqlite3.Row, str]] = []
            for f in as_completed(futures):
                if f.cancelled():
                    continue
                ready.append((futures[f], f.result()))
                if len(ready) >= WORKERS * 2:
                    done += _save(conn, ready)
                    ready = []
                    still = {r["id"] for r in pending(conn)}
                    if stop is not None and stop.is_set():
                        still = set()
                    for g, r in futures.items():
                        if r["id"] not in still:
                            g.cancel()
                    if progress:
                        progress(done, done + len(still))
            done += _save(conn, ready)
            if stop is not None and stop.is_set():
                break
    return done


def _save(conn, results: list[tuple[sqlite3.Row, str]]) -> int:
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO ocr(image_id, mtime, text, norm) VALUES (?, ?, ?, ?)",
            [(r["id"], r["mtime"], t, normalize(t)) for r, t in results],
        )
        conn.executemany("DELETE FROM ocr_requests WHERE image_id = ?", [(r["id"],) for r, _ in results])
    return len(results)


# ---------------------------------------------------------------- rules: text -> tag

def _pattern(words: str) -> re.Pattern | None:
    """'ЕВН, електро енергия' -> matches any of them at the start of a word."""
    alts = [normalize(w) for w in re.split(r"[,;\n]+", words)]
    alts = [re.escape(a) for a in alts if a]
    return re.compile(r"(?<!\w)(?:" + "|".join(alts) + ")") if alts else None


def rules(conn: sqlite3.Connection, with_counts: bool = True) -> list[dict]:
    out = [
        {"id": r[0], "words": r[1], "tag": r[2]}
        for r in conn.execute("SELECT id, words, tag FROM text_rules ORDER BY tag, id")
    ]
    if with_counts:
        texts = [r[0] for r in conn.execute("SELECT norm FROM ocr o JOIN images i ON i.id = o.image_id WHERE i.missing = 0 AND norm != ''")]
        for r in out:
            p = _pattern(r["words"])
            r["count"] = sum(1 for t in texts if p.search(t)) if p else 0
    return out


def set_rule(conn: sqlite3.Connection, words: str, tag: str, rule_id: int | None = None) -> None:
    words = ", ".join(w.strip() for w in re.split(r"[,;\n]+", words) if w.strip())
    tag = db.normalize_tag(tag)
    if not words:
        raise ValueError("Въведете дума, която да се търси в текста")
    if not tag:
        raise ValueError("Въведете таг")
    with conn:
        if rule_id:
            conn.execute("UPDATE text_rules SET words = ?, tag = ? WHERE id = ?", (words, tag, int(rule_id)))
        else:
            conn.execute("INSERT INTO text_rules(words, tag) VALUES (?, ?)", (words, tag))


def delete_rule(conn: sqlite3.Connection, rule_id: int) -> None:
    with conn:
        conn.execute("DELETE FROM text_rules WHERE id = ?", (int(rule_id),))


def text_tags(conn: sqlite3.Connection) -> dict[tuple[int, str], tuple[None, str]]:
    compiled = [(p, r["tag"]) for r in rules(conn, with_counts=False) if (p := _pattern(r["words"]))]
    if not compiled:
        return {}
    out = {}
    for image_id, norm in conn.execute(
        "SELECT o.image_id, o.norm FROM ocr o JOIN images i ON i.id = o.image_id WHERE i.missing = 0 AND o.norm != ''"
    ):
        for p, tag in compiled:
            if m := p.search(norm):
                out[(image_id, tag)] = (None, f"в текста има „{m.group(0)}“")
    return out


# ---------------------------------------------------------------- learning from text

def _vectors(norms: list[str]) -> tuple[list[dict[str, float]], dict[str, str]]:
    """TF-IDF of word stems, normalised; words in one document only or in all are dropped."""
    docs, surface = [], {}
    for norm in norms:
        counts: Counter = Counter()
        for word in _TOKEN.findall(norm):
            stem = db._stem(word)
            counts[stem] += 1
            surface.setdefault(stem, word)
        docs.append(counts)
    n = len(docs)
    df = Counter(t for d in docs for t in d)
    idf = {t: math.log(n / k) for t, k in df.items() if 2 <= k < n}
    vecs = []
    for d in docs:
        v = {t: (1 + math.log(c)) * idf[t] for t, c in d.items() if t in idf}
        length = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({t: x / length for t, x in v.items()})
    return vecs, surface


def _dot(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(t, 0.0) for t, x in a.items())


def _centroid(vs: list[dict[str, float]]) -> dict[str, float]:
    c: Counter = Counter()
    for v in vs:
        c.update(v)
    length = math.sqrt(sum(x * x for x in c.values())) or 1.0
    return {t: x / length for t, x in c.items()}


def learned_text_tags(conn: sqlite3.Connection, settings: dict | None = None) -> dict[tuple[int, str], tuple[float, str]]:
    """Suggest a hand-made tag for documents whose text is like that of the tagged ones.

    Confidence 1.0 means "as close to the examples as they are to each other";
    a document is tagged from settings["text_learn_confidence"] up.
    """
    settings = settings or db.get_settings(conn)
    min_n = int(settings["learn_min_examples"])
    threshold = float(settings["text_learn_confidence"])
    rows = conn.execute(
        "SELECT o.image_id, o.norm FROM ocr o JOIN images i ON i.id = o.image_id "
        "WHERE i.missing = 0 AND o.norm != '' ORDER BY o.image_id"
    ).fetchall()
    if len(rows) < min_n + 1:
        return {}
    ids = [r[0] for r in rows]
    row_of = {i: k for k, i in enumerate(ids)}
    vecs, surface = _vectors([r[1] for r in rows])
    examples: dict[str, list[int]] = {}
    for image_id, tag in conn.execute(
        "SELECT it.image_id, t.name FROM image_tags it JOIN tags t ON t.id = it.tag_id WHERE it.source = 'manual'"
    ):
        if image_id in row_of and vecs[row_of[image_id]]:
            examples.setdefault(tag, []).append(row_of[image_id])

    out = {}
    for tag, pos in examples.items():
        if len(pos) < min_n:
            continue
        # how close each example is to the others: the yardstick for "100 %"
        own = sorted(_dot(vecs[p], _centroid([vecs[q] for q in pos if q != p])) for p in pos)
        ref = own[len(own) // 2]
        if ref < 0.05:
            continue  # the examples share no words: nothing to learn
        centroid = _centroid([vecs[p] for p in pos])
        # words of the tag's own name that the examples contain ("благоевград данъци")
        # must be in the text too: without "благоевград" it is someone else's tax
        key = {db._stem(w) for w in _TOKEN.findall(tag)}
        key = {t for t in key if sum(t in vecs[p] for p in pos) * 2 >= len(pos)}
        taken = set(pos)
        for k, v in enumerate(vecs):
            if k in taken or not v:
                continue
            conf = min(1.0, _dot(v, centroid) / ref)
            if key:
                conf *= sum(t in v for t in key) / len(key)
            if conf >= threshold:
                shared = sorted(v, key=lambda t: -v[t] * centroid.get(t, 0.0))[:4]
                words = ", ".join(surface[t] for t in shared if centroid.get(t))
                out[(ids[k], tag)] = (round(conf, 3), f"текстът прилича на примерите ви, общи думи: {words}")
    return out
