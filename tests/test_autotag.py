import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from gallery import autotag, db

MODELS = Path(__file__).resolve().parent.parent / "data" / "models"
COLORS = {"red": (255, 0, 0), "green": (0, 160, 0), "blue": (0, 0, 255)}


class FakeModel:
    """Embeds a photo as its average colour; 'a photo of red' matches red photos."""

    name = "fake"
    scale, bias = 10.0, -8.0

    def load_image(self, path):
        try:
            return Image.open(path).convert("RGB")
        except OSError:
            return None

    def embed_images(self, images):
        v = np.array([np.asarray(im, dtype=np.float32).mean(axis=(0, 1)) + 1 for im in images])
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def embed_texts(self, texts):
        v = np.array([np.array(COLORS[t.split()[-1]], dtype=np.float32) + 1 for t in texts])
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def probs(self, iv, tv):
        return 1 / (1 + np.exp(-(iv @ tv.T * self.scale + self.bias)))


def make(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), COLORS[color]).save(path)


class AutoTagTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.pics = base / "Снимки"
        make(self.pics / "Вила" / "r1.jpg", "red")
        make(self.pics / "Вила" / "r2.jpg", "red")
        make(self.pics / "DCIM" / "Camera" / "2023-07-01 Море" / "g1.jpg", "green")
        make(self.pics / "DCIM" / "Camera" / "b1.jpg", "blue")
        make(self.pics / "DCIM" / "Camera" / "b2.jpg", "blue")
        make(self.pics / "DCIM" / "Camera" / "b3.jpg", "blue")
        make(self.pics / "DCIM" / "Camera" / "b4.jpg", "blue")
        self.conn = db.connect(base / "g.db")
        with self.conn:
            self.conn.execute("DELETE FROM vocab")
        db.set_vocab(self.conn, "червено", "a photo of red")
        db.set_vocab(self.conn, "зелено", "a photo of green")
        db.set_settings(self.conn, {"vocab_threshold": "0.5", "learn_similarity": "0.99"})
        self.root = db.add_root(self.conn, str(self.pics))
        db.scan_root(self.conn, self.root)
        self.model = FakeModel()
        self.assertEqual(autotag.embed_pending(self.conn, self.model), 7)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def id_of(self, name):
        return self.conn.execute("SELECT id FROM images WHERE name = ?", (name,)).fetchone()[0]

    def tags(self, name):
        t = db.tags_for(self.conn, [self.id_of(name)])[self.id_of(name)]
        return set(t["tags"]), set(t["auto"])

    def test_embeddings_only_once(self):
        self.assertEqual(autotag.pending(self.conn, "fake"), [])
        self.assertEqual(autotag.embed_pending(self.conn, self.model), 0)

    def test_folder_tag_names(self):
        roots = [self.root]
        self.assertEqual(autotag.folder_tag_names(str(self.pics / "Вила" / "r1.jpg"), roots), ["вила"])
        # camera folders are skipped, a leading date is stripped
        self.assertEqual(
            autotag.folder_tag_names(str(self.pics / "DCIM" / "Camera" / "2023-07-01 Море" / "g1.jpg"), roots),
            ["море"],
        )

    def test_vocab_and_folder_tags(self):
        autotag.retag(self.conn, self.model)
        self.assertEqual(self.tags("r1.jpg"), ({"червено", "вила"}, {"червено", "вила"}))
        self.assertEqual(self.tags("g1.jpg"), ({"зелено", "море"}, {"зелено", "море"}))
        self.assertEqual(self.tags("b1.jpg"), (set(), set()))
        # automatic tags are searchable like any other
        self.assertEqual({i["name"] for i in db.search(self.conn, "червеното")["items"]}, {"r1.jpg", "r2.jpg"})
        self.assertEqual(len(db.search(self.conn, review=True)["items"]), 3)

    def test_learned_from_manual_tags(self):
        blues = [self.id_of(f"b{i}.jpg") for i in (1, 2, 3)]
        db.add_tags(self.conn, blues, ["басейн"])
        autotag.retag(self.conn, self.model)
        self.assertEqual(self.tags("b4.jpg"), ({"басейн"}, {"басейн"}))
        self.assertNotIn("басейн", self.tags("r1.jpg")[0])

    def test_learning_needs_enough_examples(self):
        db.add_tags(self.conn, [self.id_of("b1.jpg"), self.id_of("b2.jpg")], ["басейн"])
        autotag.retag(self.conn, self.model)
        self.assertEqual(self.tags("b4.jpg"), (set(), set()))

    def test_removed_auto_tag_stays_removed(self):
        autotag.retag(self.conn, self.model)
        db.remove_tags(self.conn, [self.id_of("r1.jpg")], ["червено"])
        autotag.retag(self.conn, self.model)
        self.assertEqual(self.tags("r1.jpg")[0], {"вила"})
        self.assertIn("червено", self.tags("r2.jpg")[0])
        # tagging by hand again lifts the block
        db.add_tags(self.conn, [self.id_of("r1.jpg")], ["червено"])
        self.assertEqual(self.tags("r1.jpg"), ({"червено", "вила"}, {"вила"}))

    def test_confirm_and_manual_wins(self):
        autotag.retag(self.conn, self.model)
        rid = self.id_of("r1.jpg")
        db.confirm_auto(self.conn, [rid], ["червено"])
        self.assertEqual(self.tags("r1.jpg"), ({"червено", "вила"}, {"вила"}))
        autotag.retag(self.conn, self.model)  # a rerun must not demote a confirmed tag
        self.assertEqual(self.tags("r1.jpg"), ({"червено", "вила"}, {"вила"}))
        db.confirm_auto(self.conn, [rid])
        self.assertEqual(self.tags("r1.jpg")[1], set())

    def test_confirm_toggles_back_to_suggestion(self):
        autotag.retag(self.conn, self.model)
        rid = self.id_of("r1.jpg")
        db.confirm_auto(self.conn, [rid], ["червено"])
        t = db.tags_for(self.conn, [rid])[rid]
        self.assertEqual((t["auto"], t["confirmed"]), (["вила"], ["червено"]))
        db.unconfirm_auto(self.conn, [rid], ["червено"])
        t = db.tags_for(self.conn, [rid])[rid]
        self.assertEqual(set(t["auto"]), {"червено", "вила"})
        self.assertEqual(t["confirmed"], [])
        self.assertEqual(t["info"]["червено"]["source"], "auto")
        # a tag typed by hand is not a suggestion and cannot be turned into one
        db.add_tags(self.conn, [rid], ["вила", "бойлер"])
        db.unconfirm_auto(self.conn, [rid], ["вила", "бойлер"])
        t = db.tags_for(self.conn, [rid])[rid]
        self.assertEqual((set(t["tags"]) - set(t["auto"]), t["confirmed"]), ({"вила", "бойлер"}, []))

    def test_rotation_only_changes_the_view_and_re_recognises(self):
        rid = self.id_of("r1.jpg")
        self.assertEqual(db.rotate(self.conn, [rid], 90), {rid: 90})
        self.assertEqual(db.rotate(self.conn, [rid], -180), {rid: 270})
        self.assertEqual(db.rotate(self.conn, [rid], 90), {rid: 0})
        self.assertEqual([r["id"] for r in autotag.pending(self.conn, "fake")], [rid])
        im = Image.new("RGB", (40, 20))
        self.assertEqual(db.turn(im, 90).size, (20, 40))
        self.assertEqual(db.turn(im, 180).size, (40, 20))

    def test_untagged_means_no_confirmed_tags(self):
        autotag.retag(self.conn, self.model)
        self.assertEqual(len(db.search(self.conn, untagged=True)["items"]), 7)

    def test_disabled_sources(self):
        db.set_settings(self.conn, {"folder_enabled": "0", "auto_enabled": "0"})
        autotag.retag(self.conn, self.model)
        self.assertEqual(self.tags("r1.jpg"), (set(), set()))

    def test_changed_file_is_re_embedded(self):
        p = self.pics / "Вила" / "r1.jpg"
        make(p, "green")
        os.utime(p, (1, 1))
        db.scan_root(self.conn, self.root)
        self.assertEqual(len(autotag.pending(self.conn, "fake")), 1)


@unittest.skipUnless((MODELS / "models--google--siglip2-base-patch16-224").exists(), "model not downloaded")
class RealModelTest(unittest.TestCase):
    def test_siglip_scores(self):
        from gallery.ai import get_model

        m = get_model(cache_dir=MODELS)
        img = Image.new("RGB", (224, 224), (255, 255, 255))
        for x in range(60, 164):
            for y in range(40, 190):
                img.putpixel((x, y), (200, 0, 0))
        iv = m.embed_images([img])
        tv = m.embed_texts(["a photo of a red rectangle", "a photo of a dog"])
        p = m.probs(iv, tv)[0]
        self.assertGreater(p[0], p[1])
        self.assertAlmostEqual(float(np.linalg.norm(iv[0])), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
