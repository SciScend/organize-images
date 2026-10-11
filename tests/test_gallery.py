import json
import os
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

from PIL import Image

from gallery import db
from gallery.server import serve


def make_img(path: Path, color="red", size=(64, 48)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


class GalleryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.pics = base / "Снимки"
        make_img(self.pics / "Вила" / "IMG_001.jpg")
        make_img(self.pics / "Вила" / "IMG_002.png", "blue")
        make_img(self.pics / "Апартамент" / "баня.jpg", "green")
        make_img(self.pics / ".скрита" / "x.jpg")
        (self.pics / "бележка.txt").write_text("не е снимка")
        (self.pics / "счупена.jpg").write_bytes(b"not an image")
        self.conn = db.connect(base / "g.db")
        self.root = db.add_root(self.conn, str(self.pics))
        self.stats = db.scan_root(self.conn, self.root)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def ids(self, q="", **kw):
        return {it["name"] for it in db.search(self.conn, q, **kw)["items"]}

    def id_of(self, name):
        return self.conn.execute("SELECT id FROM images WHERE name = ?", (name,)).fetchone()[0]

    def test_scan_skips_hidden_and_broken(self):
        self.assertEqual(self.stats["added"], 3)
        self.assertEqual(self.ids(), {"IMG_001.jpg", "IMG_002.png", "баня.jpg"})

    def test_rescan_is_idempotent(self):
        s = db.scan_root(self.conn, self.root)
        self.assertEqual((s["added"], s["updated"], s["missing"]), (0, 0, 0))

    def test_tag_search_with_bulgarian_forms(self):
        db.add_tags(self.conn, [self.id_of("баня.jpg")], ["Бойлер", " #баня "])
        self.assertEqual(self.ids("бойлер"), {"баня.jpg"})
        self.assertEqual(self.ids("БОЙЛ"), {"баня.jpg"})       # prefix, case-insensitive
        self.assertEqual(self.ids("бойлерът"), {"баня.jpg"})   # definite article
        self.assertEqual(self.ids("бойлери"), {"баня.jpg"})    # plural

    def test_folder_name_is_searchable(self):
        # "снимки на вилата": the folder is called "Вила", no tags needed
        self.assertEqual(self.ids("вилата"), {"IMG_001.jpg", "IMG_002.png"})

    def test_and_and_exclusion(self):
        db.add_tags(self.conn, [self.id_of("IMG_001.jpg"), self.id_of("баня.jpg")], ["бойлер"])
        self.assertEqual(self.ids("бойлер вила"), {"IMG_001.jpg"})
        self.assertEqual(self.ids("бойлер -вила"), {"баня.jpg"})

    def test_phrase_tag(self):
        db.add_tags(self.conn, [self.id_of("IMG_002.png")], ["стара къща"])
        self.assertEqual(self.ids('"стара къща"'), {"IMG_002.png"})
        self.assertEqual(self.ids("къщата"), {"IMG_002.png"})

    def test_untagged_filter(self):
        db.add_tags(self.conn, [self.id_of("IMG_001.jpg")], ["вила"])
        self.assertEqual(self.ids(untagged=True), {"IMG_002.png", "баня.jpg"})

    def test_remove_and_rename(self):
        a, b = self.id_of("IMG_001.jpg"), self.id_of("IMG_002.png")
        db.add_tags(self.conn, [a], ["вилла"])
        db.add_tags(self.conn, [b], ["вила"])
        db.rename_tag(self.conn, "вилла", "вила")  # merge a typo into the real tag
        self.assertEqual(db.tag_counts(self.conn), [{"name": "вила", "count": 2, "auto": 0}])
        db.remove_tags(self.conn, [a, b], ["вила"])
        self.assertEqual(db.tag_counts(self.conn), [])

    def test_moved_file_keeps_tags(self):
        img = self.id_of("баня.jpg")
        db.add_tags(self.conn, [img], ["бойлер"])
        os.rename(self.pics / "Апартамент" / "баня.jpg", self.pics / "Вила" / "баня.jpg")
        s = db.scan_root(self.conn, self.root)
        self.assertEqual(s["moved"], 1)
        hit = db.search(self.conn, "бойлер")["items"]
        self.assertEqual([(h["id"], h["path"].endswith("Вила/баня.jpg")) for h in hit], [(img, True)])

    def test_deleted_file_hidden(self):
        os.remove(self.pics / "Вила" / "IMG_001.jpg")
        self.assertEqual(db.scan_root(self.conn, self.root)["missing"], 1)
        self.assertNotIn("IMG_001.jpg", self.ids())


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.pics = base / "pics"
        make_img(self.pics / "a.jpg", size=(1200, 800))
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
            return r.headers.get("Content-Type"), r.read()

    def test_end_to_end(self):
        ctype, html = self.call("/")
        self.assertIn("text/html", ctype)
        self.assertIn("Галерия".encode(), html)
        _, raw = self.call("/api/roots", {"path": str(self.pics)})
        self.assertEqual(json.loads(raw)["stats"]["added"], 1)
        items = json.loads(self.call("/api/images")[1])["items"]
        iid = items[0]["id"]
        self.call("/api/tags/add", {"ids": [iid], "tags": "бойлер, вила"})
        found = json.loads(self.call("/api/images?q=" + urllib.parse.quote("вилата"))[1])
        self.assertEqual(found["items"][0]["tags"], ["бойлер", "вила"])
        self.assertEqual(found["items"][0]["auto"], [])
        ctype, thumb = self.call(f"/thumb/{iid}")
        self.assertEqual(ctype, "image/jpeg")
        from io import BytesIO
        self.assertLessEqual(max(Image.open(BytesIO(thumb)).size), 480)
        ctype, _ = self.call(f"/image/{iid}")
        self.assertEqual(ctype, "image/jpeg")
        # turning serves a turned copy and thumbnail; the file itself is untouched
        before = (self.pics / "a.jpg").read_bytes()
        _, raw = self.call("/api/rotate", {"ids": [iid], "degrees": 90})
        self.assertEqual(json.loads(raw)["rotation"], {str(iid): 90})
        self.assertEqual(Image.open(BytesIO(self.call(f"/image/{iid}")[1])).size, (800, 1200))
        self.assertEqual(Image.open(BytesIO(self.call(f"/thumb/{iid}")[1])).size, (320, 480))
        self.assertEqual((self.pics / "a.jpg").read_bytes(), before)
        items = json.loads(self.call("/api/images")[1])["items"]
        self.assertEqual(items[0]["rotation"], 90)
        with self.assertRaises(urllib.error.HTTPError):
            self.call("/image/999")


import urllib.error  # noqa: E402
import urllib.parse  # noqa: E402

if __name__ == "__main__":
    unittest.main()
