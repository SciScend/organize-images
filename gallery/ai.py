"""Local image model (SigLIP 2) used for automatic tagging.

Runs on the CPU. Nothing leaves the computer: the model is downloaded once from
Hugging Face into the data folder and then used offline.
"""

from __future__ import annotations

import os
import threading

import numpy as np
from PIL import Image, ImageOps

DEFAULT_MODEL = "google/siglip2-base-patch16-224"


class ImageModel:
    """Thin wrapper: normalized image/text embeddings plus SigLIP's sigmoid score."""

    def __init__(self, name: str = DEFAULT_MODEL, cache_dir: str | os.PathLike | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor

        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
        self.torch = torch
        self.name = name
        try:  # offline once downloaded
            self.model = AutoModel.from_pretrained(name, cache_dir=cache_dir, local_files_only=True)
            self.processor = AutoProcessor.from_pretrained(name, cache_dir=cache_dir, local_files_only=True)
        except OSError:
            self.model = AutoModel.from_pretrained(name, cache_dir=cache_dir)
            self.processor = AutoProcessor.from_pretrained(name, cache_dir=cache_dir)
        self.model.eval()
        self.scale = float(self.model.logit_scale.detach().exp())
        self.bias = float(self.model.logit_bias.detach())
        self._text_cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    @staticmethod
    def load_image(path: str, size: int = 448) -> Image.Image | None:
        try:
            with Image.open(path) as im:
                im.draft("RGB", (size, size))
                im = ImageOps.exif_transpose(im)
                im.thumbnail((size, size))
                return im.convert("RGB")
        except (OSError, ValueError, Image.DecompressionBombError):
            return None

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        with self._lock, self.torch.inference_mode():
            inputs = self.processor(images=images, return_tensors="pt")
            feats = self.model.get_image_features(**inputs)
            feats = getattr(feats, "pooler_output", feats)
        return _normalize(feats.float().numpy())

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embeddings for descriptions, cached (the vocabulary rarely changes)."""
        todo = [t for t in dict.fromkeys(texts) if t not in self._text_cache]
        if todo:
            for t, v in zip(todo, self._embed_texts(todo)):
                self._text_cache[t] = v
        return np.stack([self._text_cache[t] for t in texts]) if texts else np.zeros((0, 1), np.float32)

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        with self._lock, self.torch.inference_mode():
            # SigLIP was trained with max_length padding; other padding hurts accuracy.
            inputs = self.processor(text=texts, padding="max_length", max_length=64,
                                    truncation=True, return_tensors="pt")
            feats = self.model.get_text_features(**inputs)
            feats = getattr(feats, "pooler_output", feats)
        return _normalize(feats.float().numpy())

    def probs(self, image_vecs: np.ndarray, text_vecs: np.ndarray) -> np.ndarray:
        """Independent per-label probabilities, shape (images, texts)."""
        return 1.0 / (1.0 + np.exp(-(image_vecs @ text_vecs.T * self.scale + self.bias)))


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


_models: dict[str, ImageModel] = {}
_models_lock = threading.Lock()


def get_model(name: str = DEFAULT_MODEL, cache_dir: str | os.PathLike | None = None) -> ImageModel:
    with _models_lock:
        if name not in _models:
            _models[name] = ImageModel(name, cache_dir)
        return _models[name]


def to_blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float16).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float16).astype(np.float32)


def available() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        return False, f"Липсва библиотека за автоматично тагване: {e.name}"
    return True, ""

