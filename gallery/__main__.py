"""Command line entry point: `python -m gallery [FOLDER ...]`."""

from __future__ import annotations

import argparse
import errno
import os
import sys
import threading
import webbrowser
from pathlib import Path

from . import db
from .server import serve

# Next to the code: the model alone is 1.5 GB, and /data has room while /home does not.
DEFAULT_DATA = Path(os.environ.get("GALLERY_DATA", Path(__file__).resolve().parent.parent / "data"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="gallery",
        description="Проста галерия със снимки, тагове и търсене. Отваря се в браузъра.",
    )
    p.add_argument("folders", nargs="*", help="папки със снимки, които да се добавят и сканират")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help=f"къде се пазят базата и миниатюрите (по подразбиране {DEFAULT_DATA})")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="не отваряй браузъра автоматично")
    p.add_argument("--no-scan", action="store_true", help="не пресканирай папките при стартиране")
    p.add_argument("--no-auto", action="store_true", help="без автоматично тагване")
    p.add_argument("--tag-only", action="store_true", help="само сканирай и тагни автоматично (с прогрес), без да пускаш галерията")
    p.add_argument("--model", default=None, help="модел от Hugging Face (по подразбиране google/siglip2-base-patch16-224)")
    args = p.parse_args(argv)

    args.data.mkdir(parents=True, exist_ok=True)
    db_path = args.data / "gallery.db"
    conn = db.connect(db_path)
    for folder in args.folders:
        try:
            print("Добавена папка:", db.add_root(conn, folder))
        except ValueError as e:
            print(e, file=sys.stderr)
    if not args.no_scan:
        print("Сканиране…", flush=True)
        stats = db.scan_all(conn, progress=lambda n: print(f"  {n} файла…", flush=True))
        print(f"Готово: {stats['seen']} снимки, нови {stats['added']}, променени {stats['updated']}, "
              f"преместени {stats['moved']}, изчезнали {stats['missing']}.", flush=True)
    models = args.data / "models"
    if args.tag_only:
        return _tag_only(conn, models, args.model)
    conn.close()

    tagger = None
    if not args.no_auto:
        from .autotag import AutoTagger

        tagger = AutoTagger(db_path, models, args.model)

    httpd, port = None, args.port
    for port in range(args.port, args.port + 20):
        try:
            httpd = serve(db_path, args.data / "thumbs", args.host, port, tagger)
            break
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
            if _is_gallery(args.host, port):
                url = f"http://{args.host}:{port}/"
                print(f"Галерията вече работи на {url}. Отварям нея.")
                if not args.no_browser:
                    webbrowser.open(url)
                return 0
            print(f"Порт {port} е зает от друга програма, опитвам {port + 1}…", flush=True)
    if httpd is None:
        print(f"Няма свободен порт между {args.port} и {port}. Задайте друг с --port.", file=sys.stderr)
        return 1

    if tagger is not None:
        tagger.start()
        if tagger.ok:
            tagger.request(full=True)
            print("Автоматичното тагване работи във фон. Прогресът се вижда в галерията.", flush=True)
        else:
            print(tagger.status["message"], file=sys.stderr)
    url = f"http://{args.host}:{port}/"
    print(f"Галерията работи на {url}  (Ctrl+C за спиране)", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nСпряно.")
    finally:
        if tagger:
            tagger.stop()
        httpd.server_close()
    return 0


def _is_gallery(host: str, port: int) -> bool:
    """True when the program on this port is another copy of this gallery."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/auto", timeout=2) as r:
            return "status" in json.loads(r.read())
    except Exception:
        return False


def _tag_only(conn, models: Path, model_name: str | None) -> int:
    import time

    from . import ocr
    from .ai import DEFAULT_MODEL, get_model
    from .autotag import embed_pending, retag

    print("Зареждане на модела (първия път се изтегля около 1,5 GB)…", flush=True)
    model = get_model(model_name or DEFAULT_MODEL, models)
    t0 = time.time()

    def progress(done, total):
        if done % 160 == 0 or done == total:
            rate = done / max(time.time() - t0, 1e-6)
            left = (total - done) / rate / 60 if rate else 0
            print(f"  {done}/{total} снимки, остават ~{left:.0f} мин", flush=True)

    n = embed_pending(conn, model, progress)
    ocr.sync_files(conn)
    if ocr.available()[0] and (total := ocr.count_pending(conn, db.get_settings(conn))):
        print(f"Разпознаване на текста на {total} снимки…", flush=True)
        ocr.run_pending(conn, lambda d, t: print(f"  {d}/{t} снимки", flush=True) if d % 24 == 0 or d == t else None)
    stats = retag(conn, model)
    print(f"Обработени {n} снимки. Автоматични тагове: {stats['auto']}, от папки: {stats['folder']}, "
          f"по текста: {stats['text']}.")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
