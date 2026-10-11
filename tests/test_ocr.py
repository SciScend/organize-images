import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from unittest import mock

from PIL import Image, ImageDraw, ImageFont

from gallery import autotag, db, ocr
from gallery.server import serve

FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

# what the fake OCR "reads" from each file
TEXTS = {
    "fak.jpg": "ЕВН България Електроснабдяване\nФактура № 123 за електроенергия",
    "vik.jpg": "Водоснабдяване и канализация\nсметка за вода",
    "reka.jpg": "Табела: поток на реката",
    "kuche.jpg": "",
}


def fake_read(path, rotation=0):
    return TEXTS[os.path.basename(path)]


def make(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), "white").save(path)


class OcrTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.pics = base / "pics"
        for name in TEXTS:
            make(self.pics / name)
        self.conn = db.connect(base / "g.db")
        self.root = db.add_root(self.conn, str(self.pics))
        db.scan_root(self.conn, self.root)
        db.set_settings(self.conn, {"folder_enabled": "0"})  # keep "pics" out of the tags
        patcher = mock.patch.object(ocr, "read_text", side_effect=fake_read)
        self.read = patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def id_of(self, name):
        return self.conn.execute("SELECT id FROM images WHERE name = ?", (name,)).fetchone()[0]

    def pending_names(self):
        return {r["path"].rsplit("/", 1)[1] for r in ocr.pending(self.conn)}

    def names(self, q):
        return {it["name"] for it in db.search(self.conn, q)["items"]}

    def read_all(self):
        db.set_settings(self.conn, {"ocr_enabled": "1", "ocr_scope": "all"})
        return ocr.run_pending(self.conn)

    def test_off_by_default(self):
        self.assertEqual(db.get_settings(self.conn)["ocr_enabled"], "0")
        self.assertEqual(self.pending_names(), set())
        self.assertEqual(ocr.run_pending(self.conn), 0)
        self.read.assert_not_called()

    def test_only_photos_with_document_tags(self):
        db.set_settings(self.conn, {"ocr_enabled": "1"})
        self.assertEqual(self.pending_names(), set())
        db.add_tags(self.conn, [self.id_of("fak.jpg")], ["фактура"])
        db.add_tags(self.conn, [self.id_of("kuche.jpg")], ["куче"])
        # photos tagged by hand are read too: learning from text needs them as examples
        self.assertEqual(self.pending_names(), {"fak.jpg", "kuche.jpg"})
        db.set_settings(self.conn, {"text_learn_enabled": "0"})
        self.assertEqual(self.pending_names(), {"fak.jpg"})
        # an automatic tag counts too: new photos are read after the model tags them
        with self.conn:
            self.conn.execute(
                "INSERT INTO image_tags(image_id, tag_id, source) VALUES (?, ?, 'auto')",
                (self.id_of("vik.jpg"), db._tag_id(self.conn, "документ")),
            )
        self.assertEqual(self.pending_names(), {"fak.jpg", "vik.jpg"})
        self.assertEqual(ocr.count_pending(self.conn, db.get_settings(self.conn)), 2)

    def test_each_photo_read_once_until_it_changes(self):
        self.assertEqual(self.read_all(), 4)
        self.assertEqual(self.pending_names(), set())
        self.assertEqual(ocr.run_pending(self.conn), 0)
        p = self.pics / "vik.jpg"
        os.utime(p, (1, 1))
        db.scan_root(self.conn, self.root)
        self.assertEqual(self.pending_names(), {"vik.jpg"})

    def test_read_on_request_even_when_off(self):
        ocr.request(self.conn, [self.id_of("fak.jpg")])
        self.assertEqual(self.pending_names(), {"fak.jpg"})
        self.assertTrue(ocr.text_of(self.conn, self.id_of("fak.jpg"))["queued"])
        self.assertEqual(ocr.run_pending(self.conn), 1)
        t = ocr.text_of(self.conn, self.id_of("fak.jpg"))
        self.assertEqual((t["queued"], t["text"].split("\n")[0]), (False, "ЕВН България Електроснабдяване"))
        self.assertIsNone(ocr.text_of(self.conn, self.id_of("vik.jpg"))["text"])
        # asking again reads it again
        ocr.request(self.conn, [self.id_of("fak.jpg")])
        self.assertEqual(ocr.run_pending(self.conn), 1)

    def test_search_finds_words_in_text(self):
        self.read_all()
        self.assertEqual(self.names("евн"), {"fak.jpg"})
        self.assertEqual(self.names("фактурата"), {"fak.jpg"})     # Bulgarian ending tolerated
        self.assertEqual(self.names('"сметка за вода"'), {"vik.jpg"})
        self.assertEqual(self.names("снабдяване -евн"), {"vik.jpg"})

    def test_rules_tag_by_text(self):
        self.read_all()
        ocr.set_rule(self.conn, "ЕВН, електроенергия", "Ток")
        ocr.set_rule(self.conn, "ток", "токов")
        ocr.set_rule(self.conn, "вода", "вик")
        autotag.retag(self.conn)
        tags = db.tags_for(self.conn, [self.id_of(n) for n in TEXTS])
        self.assertEqual(tags[self.id_of("fak.jpg")], {"tags": ["ток"], "auto": ["ток"], "info": {"ток": {"source": "text", "score": None, "note": "в текста има „евн“"}}, "confirmed": []})
        self.assertEqual(tags[self.id_of("vik.jpg")]["tags"], ["вик"])
        # a word must start a word of the text: "ток" does not match "поток"
        self.assertEqual(tags[self.id_of("reka.jpg")]["tags"], [])
        counts = {r["tag"]: r["count"] for r in ocr.rules(self.conn)}
        self.assertEqual(counts, {"ток": 1, "токов": 0, "вик": 1})

    def test_removed_text_tag_stays_removed_and_rules_can_go(self):
        self.read_all()
        ocr.set_rule(self.conn, "вода", "вик")
        autotag.retag(self.conn)
        vik = self.id_of("vik.jpg")
        db.remove_tags(self.conn, [vik], ["вик"])
        autotag.retag(self.conn)
        self.assertEqual(db.tags_for(self.conn, [vik])[vik]["tags"], [])
        ocr.set_rule(self.conn, "канализация", "вода и канал")
        autotag.retag(self.conn)
        self.assertEqual(db.tags_for(self.conn, [vik])[vik]["tags"], ["вода и канал"])
        rid = next(r["id"] for r in ocr.rules(self.conn) if r["tag"] == "вода и канал")
        ocr.delete_rule(self.conn, rid)
        autotag.retag(self.conn)
        self.assertEqual(db.tags_for(self.conn, [vik])[vik]["tags"], [])

    def test_rule_validation(self):
        with self.assertRaises(ValueError):
            ocr.set_rule(self.conn, " , ", "таг")
        with self.assertRaises(ValueError):
            ocr.set_rule(self.conn, "дума", "  ")
        ocr.set_rule(self.conn, "ЕВН ;  ток ", "#Ток")
        self.assertEqual([(r["words"], r["tag"]) for r in ocr.rules(self.conn)], [("ЕВН, ток", "ток")])

    def md_of(self, name):
        return ocr.text_file(ocr.texts_root(self.conn), str(self.pics / name))

    def test_text_files_win_over_tesseract(self):
        self.read_all()
        f = self.md_of("fak.jpg")
        self.assertEqual(f, Path(self.tmp.name) / "texts" / str(self.pics / "fak.md").lstrip("/"))
        f.parent.mkdir(parents=True)
        f.write_text("---\nsource: fak.jpg\nmethod: vision\nreview: true\n---\n\nЕВН фактура, прочетена от Claude\n", "utf-8")
        self.assertEqual(ocr.sync_files(self.conn), 1)
        self.assertEqual(ocr.sync_files(self.conn), 0)  # unchanged file: nothing to do
        self.assertEqual(self.names("claude"), {"fak.jpg"})
        self.assertEqual({it["name"] for it in db.search(self.conn, check_text=True)["items"]}, {"fak.jpg"})
        t = ocr.text_of(self.conn, self.id_of("fak.jpg"))
        self.assertEqual((t["text"], t["meta"]["method"], t["file"]), ("ЕВН фактура, прочетена от Claude", "vision", str(f)))
        # Tesseract never reads it again, not even on request or after a turn
        ocr.request(self.conn, [self.id_of("fak.jpg")])
        db.rotate(self.conn, [self.id_of("fak.jpg")], 90)
        self.assertNotIn("fak.jpg", self.pending_names())
        # the file is gone: back to Tesseract
        f.unlink()
        ocr.sync_files(self.conn)
        self.assertIn("fak.jpg", self.pending_names())

    def test_save_text_keeps_front_matter(self):
        f = self.md_of("vik.jpg")
        f.parent.mkdir(parents=True)
        f.write_text("---\nsource: vik.jpg\nextracted_by: Sonnet-5\nreview: true\nnotes:\n  - избледняло\n---\n\nсмтка\n", "utf-8")
        t = ocr.save_text(self.conn, self.id_of("vik.jpg"), "сметка за вода\n", reviewed=True)
        self.assertEqual(t["text"], "сметка за вода")
        self.assertEqual((t["meta"]["extracted_by"], t["meta"]["review"], t["meta"]["edited"]), ("Sonnet-5", "false", "true"))
        self.assertIn("  - избледняло", f.read_text("utf-8"))
        self.assertEqual(self.names("сметка"), {"vik.jpg"})
        self.assertEqual(db.search(self.conn, check_text=True)["total"], 0)

    def test_texts_folder_can_be_chosen(self):
        other = Path(self.tmp.name) / "моите текстове"
        db.set_settings(self.conn, {"texts_dir": str(other)})
        ocr.save_text(self.conn, self.id_of("reka.jpg"), "река", reviewed=False)
        self.assertTrue((other / str(self.pics / "reka.md").lstrip("/")).is_file())
        self.assertFalse((self.pics / "reka.md").exists())  # never next to the photos
        # another folder has no text for it: it is dropped, and read again
        db.set_settings(self.conn, {"texts_dir": str(Path(self.tmp.name) / "празна")})
        ocr.sync_files(self.conn)
        self.assertIsNone(ocr.text_of(self.conn, self.id_of("reka.jpg"))["text"])

    def test_opus_check_runs_claude_on_the_text_file(self):
        from gallery import verify
        iid = self.id_of("fak.jpg")
        f = Path(ocr.save_text(self.conn, iid, "ЕВН фактура 9-17ч.", reviewed=False)["file"])

        def fake_claude(cmd, input, **kw):
            self.assertEqual(cmd[:4], ["claude", "-p", "--model", "opus"])
            self.assertIn(str(f), input)
            f.write_text(f.read_text("utf-8").replace("9-17", "8-17"), "utf-8")
            return mock.Mock(returncode=0, stdout="1\n", stderr="")

        with mock.patch.object(verify, "available", return_value=(True, "")), \
             mock.patch.object(verify.subprocess, "run", side_effect=fake_claude):
            verify.start(self.conn, iid, Path(self.tmp.name) / "g.db")
            for _ in range(100):
                if verify.status(iid)["state"] != "running":
                    break
                threading.Event().wait(0.05)
        self.assertEqual(verify.status(iid)["state"], "done")
        self.assertEqual(self.names("8-17ч"), {"fak.jpg"})  # the corrected text is searched

    def test_text_from_tsv(self):
        head = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        rows = [
            "5\t1\t1\t1\t1\t1\t0\t0\t1\t1\t91\tФактура",
            "5\t1\t1\t1\t1\t2\t0\t0\t1\t1\t12\tжх",     # doubtful word dropped
            "5\t1\t1\t1\t1\t3\t0\t0\t1\t1\t88\t№",      # no letters or digits
            "5\t1\t1\t1\t2\t1\t0\t0\t1\t1\t90\t123",
            "5\t1\t2\t1\t1\t1\t0\t0\t1\t1\t90\tЕВН",
        ]
        self.assertEqual(ocr._text_from_tsv(head + "\n".join(rows)), "Фактура\n123\n\nЕВН")


class TextLearningTest(unittest.TestCase):
    TEXTS = [
        "община благоевград дирекция местни данъци и такси съобщение за данък сгради 2024 партида",
        "община благоевград местни данъци и такси данък върху недвижими имоти за 2025 година партида",
        "община благоевград отдел местни данъци такса битови отпадъци данък 2023 партида номер",
        "община благоевград местни данъци съобщение данък моторни превозни средства 2026 партида",
        "евн българия електроснабдяване фактура за електроенергия сума за плащане клиентски номер",
        "евн българия фактура електроенергия период клиентски номер сума",
        "вик благоевград фактура за вода период клиентски номер сума за плащане",
        "община сандански местни данъци и такси данък сгради партида",
    ]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = db.connect(Path(self.tmp.name) / "g.db")
        with self.conn:
            for i, t in enumerate(self.TEXTS, 1):
                self.conn.execute("INSERT INTO images(id, path, name, size, mtime) VALUES (?, ?, ?, 0, 0)", (i, f"/x/{i}.jpg", f"{i}.jpg"))
                self.conn.execute("INSERT INTO ocr(image_id, mtime, text, norm) VALUES (?, 0, ?, ?)", (i, t, ocr.normalize(t)))

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_learns_from_three_examples_with_confidence(self):
        db.add_tags(self.conn, [1, 2, 3], ["благоевград данъци"])
        found = ocr.learned_text_tags(self.conn)
        self.assertEqual(set(found), {(4, "благоевград данъци")})
        conf, note = found[(4, "благоевград данъци")]
        self.assertGreater(conf, 0.7)
        self.assertIn("благоевград", note)
        # the Sandanski tax notice is similar, but lacks a word of the tag's name
        db.set_settings(self.conn, {"text_learn_confidence": "0.1"})
        self.assertLess(ocr.learned_text_tags(self.conn)[(8, "благоевград данъци")][0], 0.5)
        # stored with the confidence, shown for confirmation
        autotag.retag(self.conn)
        info = db.tags_for(self.conn, [4])[4]["info"]["благоевград данъци"]
        self.assertEqual((info["source"], info["score"]), ("text", conf))

    def test_text_wins_over_a_suggestion_from_the_photo(self):
        db.add_tags(self.conn, [1, 2, 3], ["благоевград данъци"])
        # left over from the last run: the photo looks like the tagged ones too
        with self.conn:
            self.conn.execute(
                "INSERT INTO image_tags(image_id, tag_id, source, note) VALUES (4, ?, 'auto', 'прилича на снимки')",
                (db._tag_id(self.conn, "благоевград данъци"),),
            )
        autotag.retag(self.conn)
        info = db.tags_for(self.conn, [4])[4]["info"]["благоевград данъци"]
        self.assertEqual(info["source"], "text")
        self.assertIn("текстът", info["note"])

    def test_needs_enough_examples_and_can_be_off(self):
        db.add_tags(self.conn, [1, 2], ["благоевград данъци"])
        self.assertEqual(ocr.learned_text_tags(self.conn), {})
        db.add_tags(self.conn, [3], ["благоевград данъци"])
        db.set_settings(self.conn, {"text_learn_enabled": "0"})
        autotag.retag(self.conn)
        self.assertEqual(db.tags_for(self.conn, [4])[4]["tags"], [])

    def test_removed_suggestion_stays_removed(self):
        db.add_tags(self.conn, [1, 2, 3], ["благоевград данъци"])
        autotag.retag(self.conn)
        self.assertEqual(db.tags_for(self.conn, [4])[4]["auto"], ["благоевград данъци"])
        db.remove_tags(self.conn, [4], ["благоевград данъци"])
        autotag.retag(self.conn)
        self.assertEqual(db.tags_for(self.conn, [4])[4]["tags"], [])


@unittest.skipUnless(ocr.available()[0] and FONT.exists(), "tesseract with Bulgarian not installed")
class RealTesseractTest(unittest.TestCase):
    def test_reads_bulgarian(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "doc.jpg"
            im = Image.new("RGB", (900, 260), "white")
            d = ImageDraw.Draw(im)
            font = ImageFont.truetype(str(FONT), 44)
            d.text((40, 40), "ФАКТУРА за електроенергия", fill="black", font=font)
            d.text((40, 140), "Сандански 2026", fill="black", font=font)
            im.save(p, quality=95)
            text = ocr.normalize(ocr.read_text(str(p)))
        self.assertIn("фактура", text)
        self.assertIn("сандански", text)


class OcrServerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        make(base / "pics" / "fak.jpg")
        conn = db.connect(base / "g.db")
        db.scan_root(conn, db.add_root(conn, str(base / "pics")))
        conn.close()
        self.httpd = serve(base / "g.db", base / "thumbs", "127.0.0.1", 0)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def call(self, path, body=None):
        req = urllib.request.Request(self.url + path)
        if body is not None:
            req.data = json.dumps(body).encode()
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    def test_endpoints(self):
        auto = self.call("/api/auto")
        self.assertEqual(auto["settings"]["ocr_enabled"], "0")
        self.assertEqual(auto["ocr"]["pending"], 0)
        est = self.call("/api/ocr/estimate?" + urllib.parse.urlencode({"ocr_scope": "all", "ocr_tags": ""}))
        self.assertEqual(est["pending"], 1)
        iid = self.call("/api/images")["items"][0]["id"]
        t = self.call(f"/api/text/{iid}")
        self.assertEqual({k: t[k] for k in ("text", "queued", "file", "meta", "verify")},
                         {"text": None, "queued": False, "file": None, "meta": {}, "verify": None})
        with self.assertRaises(urllib.error.HTTPError):  # nothing saved yet: nothing to check
            self.call(f"/api/verify/{iid}", {})
        saved = self.call(f"/api/text/{iid}", {"text": "Поправен текст", "reviewed": True})
        self.assertEqual((saved["text"], saved["meta"]["method"], saved["meta"]["edited"]), ("Поправен текст", "manual", "true"))
        self.assertTrue(Path(saved["file"]).is_file())
        with self.assertRaises(urllib.error.HTTPError):
            self.call(f"/api/text/{iid}", {})
        rules = self.call("/api/rules", {"words": "ЕВН", "tag": "ток"})["rules"]
        self.assertEqual([(r["words"], r["tag"], r["count"]) for r in rules], [("ЕВН", "ток", 0)])
        with self.assertRaises(urllib.error.HTTPError):
            self.call("/api/rules", {"words": "", "tag": "ток"})
        self.assertEqual(self.call("/api/rules/remove", {"id": rules[0]["id"]})["rules"], [])
        # without the background worker a request cannot be served, and says so
        with self.assertRaises(urllib.error.HTTPError) as err:
            self.call("/api/ocr/run", {"ids": [iid]})
        self.assertIn("error", json.loads(err.exception.read()))


if __name__ == "__main__":
    unittest.main()
