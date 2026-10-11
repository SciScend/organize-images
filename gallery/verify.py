"""Check the text of one document with Claude (Opus), on request.

Claude Code runs in the background (`claude -p`), looks at the photo in two
halves at high resolution and corrects only the wrong places in the .md file.
It is opt-in per document, from the viewer: a check costs far more than
reading with Tesseract, so it is never run on a whole archive.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from PIL import Image, ImageOps

from . import db, ocr

MODEL = "opus"
TIMEOUT = 15 * 60
SIDE = 2400  # longer side of each half; Claude scales images down beyond ~2000 px anyway

_jobs: dict[int, dict] = {}
_lock = threading.Lock()

PROMPT = """Проверяваш и поправяш текст, прочетен от снимка на документ, най-често на български.
Намери и поправи грешките. Не преписвай текста наново.

1. Прочети с Read файла {md}.
2. Прочети с Read двете половини на снимката: {part1} и {part2}. Те се застъпват с около 10%.
3. Сравни текста със снимката ред по ред. Най-внимателно гледай цифрите, сумите, датите, телефоните,
   ЕГН, IBAN, номерата на документи и имената.
4. Поправяй с Edit, само на грешните места. Не пренаписвай целия файл с Write.
   - Грешна дума или цифра: поправи я.
   - `[?]` или `[нечетливо]`, които вече можеш да прочетеш ясно: замени ги с текста.
   - Място, което и ти не можеш да прочетеш със сигурност: сложи `[?]` или `[нечетливо]`. Никога не допълвай с догадки.
   - Пропуснат ред или блок: добави го на мястото му.
5. Във front matter (между редовете ---) допиши към `extracted_by:` текста `, checked by Opus-5`
   или добави `extracted_by: checked by Opus-5`, ако го няма. Сложи `vision_confidence:` (high, medium
   или low) и `unclear:` (броят на [?] и [нечетливо]) според твоята оценка, и `corrections: <брой поправки>`.
   Останалите редове на front matter не ги пипай.

Не отваряй други файлове и не пускай команди. Накрая отговори с един ред: броят на поправките."""


def available() -> tuple[bool, str]:
    if not shutil.which("claude"):
        return False, "Claude Code не е инсталиран (няма команда claude)."
    return True, ""


def status(image_id: int) -> dict | None:
    with _lock:
        job = _jobs.get(image_id)
        return dict(job) if job else None


def _halves(src: str, rotation: int, work: Path) -> list[Path]:
    work.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(src).stem}_{rotation}"
    parts = [work / f"{stem}_part1.jpg", work / f"{stem}_part2.jpg"]
    if all(p.exists() and p.stat().st_mtime >= os.stat(src).st_mtime for p in parts):
        return parts
    with Image.open(src) as im:
        im = db.turn(ImageOps.exif_transpose(im), rotation).convert("RGB")
    w, h = im.size
    boxes = [(0, 0, w, int(h * .55)), (0, int(h * .45), w, h)] if h >= w else \
            [(0, 0, int(w * .55), h), (int(w * .45), 0, w, h)]
    for p, box in zip(parts, boxes):
        q = im.crop(box)
        q.thumbnail((SIDE, SIDE))
        q.save(p, quality=90)
    return parts


def start(conn, image_id: int, db_path: Path) -> dict:
    """Start checking one photo's text. The text must be in a .md file already."""
    ok, why = available()
    if not ok:
        raise ValueError(why)
    img = db.get_image(conn, image_id)
    if img is None:
        raise ValueError("Няма такава снимка")
    md = ocr.text_file(ocr.texts_root(conn), img["path"])
    if not md.is_file():
        raise ValueError("Първо запазете текста")
    with _lock:
        if (_jobs.get(image_id) or {}).get("state") == "running":
            return dict(_jobs[image_id])
        _jobs[image_id] = {"state": "running", "started": time.time(), "message": "Opus проверява текста…"}
    work = Path(db_path).parent / "ocr-work" / "img"
    args = (img["path"], img["rotation"], md, work, image_id, db_path)
    threading.Thread(target=_run, args=args, daemon=True).start()
    return status(image_id)


def _run(src: str, rotation: int, md: Path, work: Path, image_id: int, db_path: Path) -> None:
    def done(state: str, message: str):
        with _lock:
            _jobs[image_id] = {"state": state, "finished": time.time(), "message": message}

    try:
        part1, part2 = _halves(src, rotation, work)
        prompt = PROMPT.format(md=md, part1=part1, part2=part2)
        before = md.stat().st_mtime
        r = subprocess.run(
            # the prompt goes on stdin: --add-dir takes several values and would swallow it
            ["claude", "-p", "--model", MODEL, "--allowedTools", "Read,Edit",
             "--permission-mode", "acceptEdits", "--add-dir", str(md.parent), "--add-dir", str(work)],
            input=prompt, capture_output=True, text=True, timeout=TIMEOUT, cwd=str(md.parent),
        )
    except subprocess.TimeoutExpired:
        return done("error", "Проверката отне твърде дълго и беше спряна.")
    except (OSError, ValueError, Image.DecompressionBombError) as e:
        return done("error", f"Проверката не можа да започне: {e}")
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or ["неизвестна грешка"]
        return done("error", f"Claude върна грешка: {tail[0][:300]}")
    conn = db.connect(db_path)
    try:
        img = db.get_image(conn, image_id)
        if img is not None and md.is_file():
            ocr._load_file(conn, img, md)
    finally:
        conn.close()
    changed = md.is_file() and md.stat().st_mtime != before
    answer = r.stdout.strip().splitlines()[-1:] or [""]
    done("done", f"Opus провери текста. Поправки: {answer[0][:80]}" if changed else "Opus провери текста и не промени нищо.")
