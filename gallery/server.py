"""Local HTTP server: JSON API, thumbnails and the single-page UI."""

from __future__ import annotations

import json
import mimetypes
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageOps

from . import db, ocr, verify

STATIC = Path(__file__).parent / "static"
THUMB_SIZE = 480
FULL_SIZE = 3200  # a turned photo in the viewer; enough to read a document
_thumb_lock = threading.Lock()
_scan_lock = threading.Lock()


def make_thumb(src: str, dest: Path, mtime: float, rotation: int = 0, size: int = THUMB_SIZE) -> bool:
    if dest.exists() and dest.stat().st_mtime >= mtime:
        return True
    try:
        with Image.open(src) as im:
            im.draft("RGB", (size * 2, size * 2))  # fast JPEG downscale
            im = db.turn(ImageOps.exif_transpose(im), rotation)
            im.thumbnail((size, size))
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(f".{threading.get_ident()}.tmp")
            im.save(tmp, "JPEG", quality=82 if size == THUMB_SIZE else 92)
            os.replace(tmp, dest)
        return True
    except (OSError, ValueError, Image.DecompressionBombError):
        return False


def make_handler(db_path: Path, thumbs: Path, tagger=None):
    def kick(full: bool, delay: float = 0.0):
        if tagger is not None:
            tagger.request(full=full, delay=delay)

    class Handler(BaseHTTPRequestHandler):
        server_version = "Gallery/0.1"

        def log_message(self, fmt, *args):  # keep the terminal quiet
            pass

        # ------------------------------------------------ helpers
        def _conn(self):
            return db.connect(db_path)

        def _json(self, data, status=HTTPStatus.OK):
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, msg, status=HTTPStatus.BAD_REQUEST):
            self._json({"error": msg}, status)

        def _file(self, path: str | Path, ctype: str | None = None, cache: bool = False):
            try:
                data = Path(path).read_bytes()
            except OSError:
                return self._error("Файлът не може да се прочете", HTTPStatus.NOT_FOUND)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype or mimetypes.guess_type(str(path))[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "private, max-age=86400" if cache else "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8"))

        def _image_row(self, raw_id: str):
            try:
                image_id = int(raw_id)
            except ValueError:
                return None
            conn = self._conn()
            try:
                return db.get_image(conn, image_id)
            finally:
                conn.close()

        # ------------------------------------------------ GET
        def do_GET(self):
            url = urlparse(self.path)
            qs = {k: v[-1] for k, v in parse_qs(url.query).items()}
            parts = url.path.strip("/").split("/")

            if url.path in ("/", "/index.html"):
                return self._file(STATIC / "index.html", "text/html; charset=utf-8")

            if parts[0] in ("thumb", "image") and len(parts) == 2:
                row = self._image_row(parts[1])
                if row is None:
                    return self._error("Няма такава снимка", HTTPStatus.NOT_FOUND)
                rot = row["rotation"]
                if parts[0] == "image" and not rot:
                    return self._file(row["path"], cache=True)
                # a turned photo is served as a turned copy; the name changes with the rotation
                name = f"{row['id']}_{rot}.jpg" if rot else f"{row['id']}.jpg"
                if parts[0] == "image":
                    dest = thumbs / "turned" / name
                    ok = make_thumb(row["path"], dest, row["mtime"], rot, FULL_SIZE)
                else:
                    dest = thumbs / f"{row['id'] // 1000}" / name
                    ok = make_thumb(row["path"], dest, row["mtime"], rot)
                if not ok:
                    return self._error("Миниатюрата не може да се създаде", HTTPStatus.INTERNAL_SERVER_ERROR)
                return self._file(dest, "image/jpeg", cache=True)

            conn = self._conn()
            try:
                if url.path == "/api/images":
                    return self._json(
                        db.search(
                            conn,
                            q=qs.get("q", ""),
                            untagged=qs.get("untagged") == "1",
                            review=qs.get("review") == "1",
                            check_text=qs.get("check_text") == "1",
                            sort=qs.get("sort", "newest"),
                            offset=int(qs.get("offset", 0)),
                            limit=min(int(qs.get("limit", 200)), 1000),
                        )
                    )
                if url.path == "/api/tags":
                    return self._json(db.tag_counts(conn))
                if url.path == "/api/roots":
                    return self._json(db.roots(conn))
                if url.path == "/api/auto":
                    status = dict(tagger.status) if tagger else {"state": "off", "message": "Изключено"}
                    settings = db.get_settings(conn)
                    return self._json({"status": status, "settings": settings, "ocr": ocr.status(conn, settings)})
                if url.path == "/api/ocr/estimate":
                    # how many photos other OCR settings would read, before saving them
                    settings = {**db.get_settings(conn), "ocr_enabled": "1",
                                **{k: qs[k] for k in ("ocr_scope", "ocr_tags") if k in qs}}
                    return self._json(ocr.status(conn, settings))
                if parts[:2] == ["api", "text"] and len(parts) == 3 and parts[2].isdigit():
                    image_id = int(parts[2])
                    return self._json({**ocr.text_of(conn, image_id), "verify": verify.status(image_id),
                                       "can_verify": verify.available()[0]})
                if url.path == "/api/rules":
                    return self._json(ocr.rules(conn))
                if url.path == "/api/vocab":
                    return self._json(db.vocab(conn))
            finally:
                conn.close()
            self._error("Няма такъв адрес", HTTPStatus.NOT_FOUND)

        # ------------------------------------------------ POST
        def do_POST(self):
            try:
                body = self._body()
            except (ValueError, UnicodeDecodeError):
                return self._error("Невалидни данни")
            path = urlparse(self.path).path
            conn = self._conn()
            try:
                if path in ("/api/tags/add", "/api/tags/remove"):
                    ids = [int(i) for i in body.get("ids", [])]
                    tags = body.get("tags", [])
                    if isinstance(tags, str):
                        tags = db.split_tags(tags)
                    if not ids or not tags:
                        return self._error("Изберете снимки и въведете таг")
                    (db.add_tags if path.endswith("add") else db.remove_tags)(conn, ids, tags)
                    kick(full=False, delay=3)  # learn from the new hand-made tags
                    return self._json({"ok": True, "tags": db.tags_for(conn, ids)})
                if path == "/api/tags/confirm":
                    ids = [int(i) for i in body.get("ids", [])]
                    if not ids:
                        return self._error("Изберете снимки")
                    db.confirm_auto(conn, ids, body.get("tags") or None)
                    kick(full=False, delay=3)
                    return self._json({"ok": True, "tags": db.tags_for(conn, ids)})
                if path == "/api/tags/unconfirm":
                    ids = [int(i) for i in body.get("ids", [])]
                    tags = body.get("tags") or []
                    if not ids or not tags:
                        return self._error("Изберете снимки и таг")
                    db.unconfirm_auto(conn, ids, tags)
                    kick(full=False, delay=3)
                    return self._json({"ok": True, "tags": db.tags_for(conn, ids)})
                if path == "/api/rotate":
                    ids = [int(i) for i in body.get("ids", [])]
                    if not ids:
                        return self._error("Изберете снимки")
                    rotation = db.rotate(conn, ids, int(body.get("degrees", 90)))
                    kick(full=True, delay=2)  # recognise again, after a burst of turns
                    return self._json({"ok": True, "rotation": rotation})
                if path == "/api/tags/rename":
                    db.rename_tag(conn, body.get("old", ""), body.get("new", ""))
                    kick(full=False)
                    return self._json({"ok": True})
                if path == "/api/roots":
                    try:
                        root = db.add_root(conn, body.get("path", ""))
                    except ValueError as e:
                        return self._error(str(e))
                    with _scan_lock:
                        stats = db.scan_root(conn, root)
                    kick(full=True)
                    return self._json({"ok": True, "root": root, "stats": stats})
                if path == "/api/roots/remove":
                    db.remove_root(conn, body.get("path", ""))
                    kick(full=False)
                    return self._json({"ok": True})
                if path == "/api/scan":
                    with _scan_lock:
                        stats = db.scan_all(conn)
                    stats["texts"] = ocr.sync_files(conn)
                    kick(full=True)
                    return self._json({"ok": True, "stats": stats})
                if path == "/api/auto/run":
                    kick(full=True)
                    return self._json({"ok": True})
                if path == "/api/ocr/run":
                    ok, why = ocr.available()
                    if not ok:
                        return self._error(why)
                    if tagger is None or not tagger.ok:
                        return self._error("Фоновата обработка е изключена. Пуснете галерията без --no-auto.")
                    ids = [int(i) for i in body.get("ids", [])]
                    if not ids:
                        return self._error("Изберете снимки")
                    ocr.request(conn, ids)
                    kick(full=False)
                    return self._json({"ok": True})
                if path.startswith("/api/text/") and path[len("/api/text/"):].isdigit():
                    text = body.get("text")
                    if not isinstance(text, str):
                        return self._error("Няма текст")
                    try:
                        result = ocr.save_text(conn, int(path[len("/api/text/"):]), text, bool(body.get("reviewed")))
                    except ValueError as e:
                        return self._error(str(e))
                    except OSError as e:
                        return self._error(f"Текстът не може да се запише: {e}", HTTPStatus.INTERNAL_SERVER_ERROR)
                    kick(full=False, delay=3)  # the text tags follow the new text
                    return self._json({"ok": True, **result})
                if path.startswith("/api/verify/") and path[len("/api/verify/"):].isdigit():
                    try:
                        job = verify.start(conn, int(path[len("/api/verify/"):]), db_path)
                    except ValueError as e:
                        return self._error(str(e))
                    return self._json({"ok": True, "verify": job})
                if path == "/api/rules":
                    try:
                        ocr.set_rule(conn, body.get("words", ""), body.get("tag", ""), body.get("id"))
                    except ValueError as e:
                        return self._error(str(e))
                    kick(full=False)
                    return self._json({"ok": True, "rules": ocr.rules(conn)})
                if path == "/api/rules/remove":
                    ocr.delete_rule(conn, body.get("id", 0))
                    kick(full=False)
                    return self._json({"ok": True, "rules": ocr.rules(conn)})
                if path == "/api/settings":
                    db.set_settings(conn, body)
                    if "texts_dir" in body:
                        ocr.sync_files(conn)
                    kick(full=True)
                    return self._json({"ok": True, "settings": db.get_settings(conn)})
                if path == "/api/vocab":
                    try:
                        db.set_vocab(conn, body.get("tag", ""), body.get("prompt", ""), bool(body.get("enabled", True)))
                    except ValueError as e:
                        return self._error(str(e))
                    kick(full=False)
                    return self._json({"ok": True, "vocab": db.vocab(conn)})
                if path == "/api/vocab/remove":
                    db.delete_vocab(conn, body.get("tag", ""))
                    kick(full=False)
                    return self._json({"ok": True, "vocab": db.vocab(conn)})
            finally:
                conn.close()
            self._error("Няма такъв адрес", HTTPStatus.NOT_FOUND)

    return Handler


def serve(db_path: Path, thumbs: Path, host: str, port: int, tagger=None) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), make_handler(db_path, thumbs, tagger))
    httpd.daemon_threads = True
    return httpd
