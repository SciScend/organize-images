"""Automatic tagging: image embeddings -> tags from three sources.

1. Vocabulary: the model scores every photo against each description in the
   vocabulary ("бойлер" -> "a photo of an electric water heater ...").
2. Learned: a tag you put on at least a few photos by hand is suggested for
   photos that look very similar to one of them ("вила").
3. Folder names: "…/Вила 2023/IMG_1.jpg" gets the tag "вила 2023".
4. Text (OCR, see ocr.py): a rule like "ЕВН -> ток", or a tag you put on a few
   documents by hand is suggested for documents with similar text.

Automatic tags are stored with source 'auto', 'folder' or 'text'. They are recomputed
from scratch on every run, never override your own tags, and never come back on
a photo you removed them from.
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import db, ocr

BATCH = 16

_SKIP_DIR = re.compile(
    r"^(dcim|camera|camera ?roll|100andro|\d{3}apple|\d{3}media|pictures?|photos?|images?|img|"
    r"снимки|картинки|screenshots?|whatsapp.*|viber.*|telegram.*|downloads?|сваляния|backups?|"
    r"media|sent|private|misc|other|new folder|нова папка|untitled|exports?|.*cameraexport|"
    r"thumbnails?|\.thumbnails|originals?|edited)$",
    re.IGNORECASE,
)
_LEADING_DATE = re.compile(r"^[\d._\-\s]+(?=\D)")


# ---------------------------------------------------------------- embeddings

def pending(conn: sqlite3.Connection, model_name: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT i.id, i.path, i.mtime, i.rotation FROM images i "
        "LEFT JOIN embeddings e ON e.image_id = i.id "
        "WHERE i.missing = 0 AND (e.image_id IS NULL OR e.model != ? OR e.mtime != i.mtime) "
        "ORDER BY i.id",
        (model_name,),
    ).fetchall()


def embed_pending(conn, model, progress=None, stop: threading.Event | None = None) -> int:
    """Compute embeddings for new or changed photos. Returns how many were done."""
    from .ai import to_blob

    rows = pending(conn, model.name)
    done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, len(rows), BATCH):
            if stop is not None and stop.is_set():
                break
            batch = rows[start:start + BATCH]
            images = list(pool.map(lambda r: _load(model, r), batch))
            ok = [(r, im) for r, im in zip(batch, images) if im is not None]
            vecs = model.embed_images([im for _, im in ok]) if ok else []
            blobs = {r["id"]: to_blob(v) for (r, _), v in zip(ok, vecs)}
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings(image_id, model, mtime, vec) VALUES (?, ?, ?, ?)",
                # unreadable photos get an empty vector so they are not retried every run
                [(r["id"], model.name, r["mtime"], blobs.get(r["id"], b"")) for r in batch],
            )
            conn.commit()
            done += len(batch)
            if progress:
                progress(done, len(rows))
    return done


def _load(model, row):
    im = model.load_image(row["path"])
    return db.turn(im, row["rotation"]) if im is not None else None


def load_embeddings(conn, model_name: str) -> tuple[np.ndarray, np.ndarray]:
    from .ai import from_blob

    rows = conn.execute(
        "SELECT e.image_id, e.vec FROM embeddings e JOIN images i ON i.id = e.image_id "
        "WHERE i.missing = 0 AND e.model = ? AND length(e.vec) > 0 ORDER BY e.image_id",
        (model_name,),
    ).fetchall()
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    return ids, np.stack([from_blob(r[1]) for r in rows])


# ---------------------------------------------------------------- tag sources

def vocab_tags(conn, model, ids, X, settings) -> dict[tuple[int, str], float]:
    entries = [v for v in db.vocab(conn) if v["enabled"]]
    if not entries or not len(ids):
        return {}
    T = model.embed_texts([v["prompt"] for v in entries])
    thr = float(settings["vocab_threshold"])
    k = int(settings["max_auto_tags"])
    out = {}
    for start in range(0, len(ids), 4096):
        P = model.probs(X[start:start + 4096], T)
        for r, row in enumerate(P):
            hits = np.flatnonzero(row >= thr)
            for j in hits[np.argsort(-row[hits])][:k]:
                out[(int(ids[start + r]), entries[j]["tag"])] = float(row[j])
    return out


def learned_tags(conn, ids, X, settings) -> dict[tuple[int, str], float]:
    """Suggest a hand-made tag for photos very similar to one already tagged."""
    if not len(ids):
        return {}
    min_n = int(settings["learn_min_examples"])
    sim_thr = float(settings["learn_similarity"])
    row_of = {int(i): r for r, i in enumerate(ids)}
    manual: dict[str, list[int]] = {}
    for image_id, tag in conn.execute(
        "SELECT it.image_id, t.name FROM image_tags it JOIN tags t ON t.id = it.tag_id WHERE it.source = 'manual'"
    ):
        if image_id in row_of:
            manual.setdefault(tag, []).append(row_of[image_id])
    rejected: dict[str, list[int]] = {}
    for image_id, tag in conn.execute(
        "SELECT r.image_id, t.name FROM rejected r JOIN tags t ON t.id = r.tag_id"
    ):
        if image_id in row_of:
            rejected.setdefault(tag, []).append(row_of[image_id])

    def best_sim(rows: list[int]) -> np.ndarray:
        best = np.full(len(ids), -1.0, dtype=np.float32)
        for s in range(0, len(rows), 1024):
            best = np.maximum(best, (X @ X[rows[s:s + 1024]].T).max(axis=1))
        return best

    out = {}
    for tag, pos in manual.items():
        if len(pos) < min_n:
            continue
        sim = best_sim(pos)
        sim[pos] = -1.0  # already tagged
        if tag in rejected:
            # closer to a photo where you removed this tag -> don't suggest
            sim[sim <= best_sim(rejected[tag])] = -1.0
        for r in np.flatnonzero(sim >= sim_thr):
            out[(int(ids[r]), tag)] = float(sim[r])
    return out


def folder_tag_names(path: str, roots: list[str]) -> list[str]:
    root = max((r for r in roots if path.startswith(r.rstrip(os.sep) + os.sep)), key=len, default=None)
    if root is None:
        return []
    rel = Path(path).relative_to(root).parent.parts
    out = []
    for part in (Path(root).name, *rel):
        name = _LEADING_DATE.sub("", part).strip(" _-.")
        tag = db.normalize_tag(name.replace("_", " "))
        if not tag or len(tag) > 40 or len(tag) < 2 or _SKIP_DIR.match(tag) or not re.search(r"[^\W\d_]", tag):
            continue
        if tag not in out:
            out.append(tag)
    return out


def folder_tags(conn) -> dict[tuple[int, str], None]:
    roots = [r["path"] for r in db.roots(conn)]
    out = {}
    for image_id, path in conn.execute("SELECT id, path FROM images WHERE missing = 0"):
        for tag in folder_tag_names(path, roots):
            out[(image_id, tag)] = None
    return out


Found = dict[tuple[int, str], tuple[float | None, str | None]]  # (image, tag) -> (confidence, note)


# when two sources suggest the same tag for a photo, the earlier one is kept:
# a rule or the text is the most specific reason, the model's guess the least
PRIORITY = ("text", "folder", "auto")


def _replace(conn, source: str, pairs: Found) -> int:
    """Swap all tags of one automatic source for a fresh set.

    A tag that a lower-priority source already put on the photo is taken over,
    so the result does not depend on which source ran last time.
    """
    weaker = PRIORITY[PRIORITY.index(source) + 1:]
    tag_ids: dict[str, int] = {}
    with conn:
        conn.execute("DELETE FROM image_tags WHERE source = ?", (source,))
        for (_, tag) in pairs:
            if tag not in tag_ids:
                tag_ids[tag] = db._tag_id(conn, tag)
        conn.executemany(
            "INSERT INTO image_tags(image_id, tag_id, source, score, note) "
            "SELECT ?, ?, ?, ?, ? WHERE NOT EXISTS (SELECT 1 FROM rejected WHERE image_id = ? AND tag_id = ?) "
            "ON CONFLICT(image_id, tag_id) DO UPDATE SET source = excluded.source, score = excluded.score, "
            f"note = excluded.note WHERE image_tags.source IN ({','.join('?' * len(weaker)) or 'NULL'})",
            [(i, tag_ids[t], source, s, n, i, tag_ids[t], *weaker) for (i, t), (s, n) in pairs.items()],
        )
    return len(pairs)


def retag(conn, model=None) -> dict:
    """Recompute automatic tags. Without a model only folder tags are refreshed."""
    settings = db.get_settings(conn)
    stats = {"folder": 0, "auto": 0, "text": 0}
    # in PRIORITY order, so that a stronger source takes over the tags of a weaker one
    text: Found = ocr.learned_text_tags(conn, settings) if settings["text_learn_enabled"] == "1" else {}
    text.update(ocr.text_tags(conn))  # a rule is certain, so it wins
    stats["text"] = _replace(conn, "text", text)
    folders = folder_tags(conn) if settings["folder_enabled"] == "1" else {}
    stats["folder"] = _replace(conn, "folder", {k: (None, "от името на папката") for k in folders})
    if model is not None:
        ids, X = load_embeddings(conn, model.name)
        auto: Found = {}
        if settings["auto_enabled"] == "1":
            # SigLIP's probabilities are not calibrated, so no confidence is shown for them
            auto = {k: (None, "разпознато от модела") for k in vocab_tags(conn, model, ids, X, settings)}
        if settings["learn_enabled"] == "1":
            for key, sim in learned_tags(conn, ids, X, settings).items():
                auto[key] = (sim, "прилича на снимки, които сте тагнали така")
        stats["auto"] = _replace(conn, "auto", auto)
    return stats


# ---------------------------------------------------------------- background worker

class AutoTagger:
    """Runs embedding + retagging in a background thread, one job at a time."""

    def __init__(self, db_path, cache_dir, model_name: str | None = None):
        from .ai import DEFAULT_MODEL, available

        self.db_path = db_path
        self.cache_dir = cache_dir
        self.model_name = model_name or DEFAULT_MODEL
        self.ok, why = available()
        self.status = {"state": "idle" if self.ok else "off", "done": 0, "total": 0,
                       "message": why, "updated": time.time()}
        self._wake = threading.Event()
        self._full = False
        self._due = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="autotag", daemon=True)

    def start(self):
        if self.ok:
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._wake.set()

    def request(self, full: bool = True, delay: float = 0.0):
        """full: embed new photos first; otherwise only recompute tags (after edits)."""
        self._full = self._full or full
        self._due = max(self._due, time.time() + delay)
        self._wake.set()

    def _set(self, **kw):
        self.status.update(kw, updated=time.time())

    def _loop(self):
        from .ai import get_model

        while not self._stop.is_set():
            self._wake.wait()
            while time.time() < self._due and not self._stop.is_set():
                time.sleep(0.2)
            if self._stop.is_set():
                return
            self._wake.clear()
            full, self._full = self._full, False
            conn = db.connect(self.db_path)
            try:
                model = None
                if db.get_settings(conn)["auto_enabled"] == "1":
                    if self.status["state"] in ("idle", "off", "disabled", "error"):
                        self._set(state="loading", message="Зареждане на модела…")
                    model = get_model(self.model_name, self.cache_dir)
                    if full:
                        self._set(state="embedding", done=0, total=len(pending(conn, model.name)),
                                  message="Разпознаване на снимките…")
                        embed_pending(conn, model, lambda d, t: self._set(done=d, total=t), self._stop)
                ocr.sync_files(conn)  # .md files written by Claude or by hand come first
                if ocr.available()[0] and ocr.pending(conn, limit=1):
                    # after embedding: new photos may just have been tagged "документ"
                    self._set(state="ocr", done=0, total=ocr.count_pending(conn, db.get_settings(conn)),
                              message="Разпознаване на текста…")
                    ocr.run_pending(conn, lambda d, t: self._set(done=d, total=t), self._stop)
                self._set(state="tagging", message="Поставяне на тагове…")
                stats = retag(conn, model)
                if model is None:
                    self._set(state="disabled", message="Разпознаването на снимките е изключено")
                else:
                    self._set(state="ready", message=f"Готово: {stats['auto']} автоматични тага"
                              + (f", {stats['text']} по текста" if stats["text"] else ""))
            except Exception as e:  # keep the server alive; show the error in the UI
                self._set(state="error", message=f"Грешка при автоматичното тагване: {e}")
            finally:
                conn.close()
