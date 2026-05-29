#!/usr/bin/env python3
"""Vision/multimodal tests for ollamaquery2.

Downloads test images, resizes them with ImageMagick (if available),
and tests single/multi-image queries against vision-capable models.

Requires a live backend with a vision model loaded (qwen3-vl, qwen2.5vl, etc.).

Usage:
  python3 -m unittest tests.test_vision -v

  # With specific vision model
  VISION_MODEL=qwen3-vl:4b python3 -m unittest tests.test_vision -v
"""

import os
import re
import sys
import json
import time
import shutil
import subprocess
import tempfile
import unittest
from urllib.request import Request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ollamaquery2 as q

def _ensure_url(host):
    host = host.strip()
    if not host.startswith('http'):
        host = 'http://' + host
    return host

OLLAMA_HOST = _ensure_url(os.environ.get('OLLAMA_HOST', 'http://192.168.1.20:11434'))
LLAMACPP_HOST = _ensure_url(os.environ.get('LLAMACPP_HOST', 'http://127.0.0.1:8080'))
VISION_MODEL = os.environ.get('VISION_MODEL', None)

IMAGE_URLS = {
    "cat1": "https://live.staticflickr.com/1204/1090235720_da0ca9dc95.jpg",
    "cat2": "https://live.staticflickr.com/3649/3551252588_b6ab64855e.jpg",
    "dog1": "https://live.staticflickr.com/3940/15473596487_5ed985dd35_b.jpg",
    "dog2": "https://live.staticflickr.com/3014/2848494407_431429e588_b.jpg",
}

BACKEND = None
BASE_URL = None
MODEL = None
BACKEND_AVAILABLE = False
VISION_AVAILABLE = False
IMAGES_DIR = None
IMAGE_PATHS = {}
HAVE_CONVERT = shutil.which("convert") is not None


def _check_backend(url, label):
    try:
        req = Request(url, method='HEAD')
        q.urlopen(req, timeout=2)
        try:
            tags = json.loads(q.urlopen(q.Request(f"{url}/api/tags"), timeout=3).read())
            models = [m['name'] for m in tags.get('models', [])]
            vision_keywords = ['vl', 'vision', 'llava', 'gemma3', 'glm4v']
            vision_models = [m for m in models if any(k in m.lower() for k in vision_keywords)]
            if vision_models:
                preferred = VISION_MODEL if VISION_MODEL in models else None
                if not preferred:
                    for m in vision_models:
                        if '4b' in m:
                            preferred = m
                            break
                if not preferred:
                    preferred = vision_models[0]
                return True, preferred
            return True, models[0] if models else None
        except Exception:
            pass
        return True, None
    except Exception:
        return False, None


def _download_and_resize(img_dir):
    import shutil
    have_convert = shutil.which("convert")
    paths = {}
    for name, url in IMAGE_URLS.items():
        dest = os.path.join(img_dir, f"{name}.jpg")
        try:
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with q.urlopen(req, timeout=15) as r:
                data = r.read()
            with open(dest, 'wb') as f:
                f.write(data)
            if have_convert:
                subprocess.run(['convert', dest, '-resize', '500x500', dest],
                               capture_output=True, timeout=15)
            paths[name] = dest
        except Exception:
            paths[name] = None
    return paths


def setUpModule():
    global BACKEND, BASE_URL, MODEL, BACKEND_AVAILABLE, VISION_AVAILABLE, IMAGES_DIR, IMAGE_PATHS

    for url, label in [(OLLAMA_HOST, "ollama"), (LLAMACPP_HOST, "llamacpp")]:
        ok, model = _check_backend(url, label)
        if ok:
            BACKEND = label
            BASE_URL = url
            MODEL = model or VISION_MODEL
            BACKEND_AVAILABLE = True
            VISION_AVAILABLE = bool(model and ('vl' in model.lower() or 'vision' in model.lower() or 'llava' in model.lower()))
            break

    if not BACKEND_AVAILABLE:
        return

    IMAGES_DIR = tempfile.mkdtemp(prefix="vision_test_")
    IMAGE_PATHS = _download_and_resize(IMAGES_DIR)


def tearDownModule():
    if IMAGES_DIR and os.path.isdir(IMAGES_DIR):
        import shutil
        shutil.rmtree(IMAGES_DIR, ignore_errors=True)


def _run_query(prompt, images=None, model=None):
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}]
    handler = q.ModelQuery(context=q.CommandContext())
    handler.ctx.base_url = BASE_URL
    handler.ctx.backend = BACKEND
    handler.ctx.shell_timeout = 5
    return handler.query_sync(messages, model or MODEL, images=images)


def _run_single_image(image_path, prompt="Describe this image in 5 words or fewer"):
    img_data = q.prepare_image_data(image_path)
    if not img_data:
        return None
    response = _run_query(prompt, images=[img_data])
    return response.get('message', {}).get('content', '') if isinstance(response, dict) else str(response)


class _VisionTestBase(unittest.TestCase):
    """Base class that skips tests if no vision-capable backend is available."""

    def setUp(self):
        if not BACKEND_AVAILABLE:
            self.skipTest("No backend available")
        if not IMAGE_PATHS or not all(IMAGE_PATHS.values()):
            self.skipTest("Could not download test images")


class TestVisionDetection(_VisionTestBase):
    """Verify vision model is available for testing."""

    def test_backend_reachable(self):
        self.assertTrue(BACKEND_AVAILABLE)
        self.assertIsNotNone(BASE_URL)

    def test_model_selected(self):
        self.assertIsNotNone(MODEL, "No model selected")
        print(f"\n[Backend: {BACKEND} @ {BASE_URL}]")
        print(f"[Model: {MODEL}]")


class TestVisionSingleImage(_VisionTestBase):
    """Test single image queries."""

    def test_cat1(self):
        result = _run_single_image(IMAGE_PATHS["cat1"])
        self.assertIsNotNone(result)
        self.assertGreater(len(result.strip()), 0)
        print(f"\n[cat1] {result}")

    def test_cat2(self):
        result = _run_single_image(IMAGE_PATHS["cat2"])
        self.assertIsNotNone(result)
        self.assertGreater(len(result.strip()), 0)
        print(f"\n[cat2] {result}")

    def test_dog1(self):
        result = _run_single_image(IMAGE_PATHS["dog1"])
        self.assertIsNotNone(result)
        self.assertGreater(len(result.strip()), 0)
        print(f"\n[dog1] {result}")

    def test_dog2(self):
        result = _run_single_image(IMAGE_PATHS["dog2"])
        self.assertIsNotNone(result)
        self.assertGreater(len(result.strip()), 0)
        print(f"\n[dog2] {result}")


class TestVisionMultiImage(_VisionTestBase):
    """Test multiple images in one query."""

    def test_two_images(self):
        imgs = [q.prepare_image_data(IMAGE_PATHS["cat1"]),
                q.prepare_image_data(IMAGE_PATHS["dog1"])]
        imgs = [i for i in imgs if i]
        self.assertEqual(len(imgs), 2)
        response = _run_query("What animals? Reply: X and Y", images=imgs)
        content = response.get('message', {}).get('content', '') if isinstance(response, dict) else str(response)
        self.assertGreater(len(content.strip()), 0)
        print(f"\n[cat+dog] {content}")

    def test_four_images(self):
        imgs = [q.prepare_image_data(IMAGE_PATHS[name]) for name in ["cat1", "cat2", "dog1", "dog2"]]
        imgs = [i for i in imgs if i]
        self.assertEqual(len(imgs), 4)
        response = _run_query("How many cats and dogs? Reply: X cats, Y dogs", images=imgs)
        content = response.get('message', {}).get('content', '') if isinstance(response, dict) else str(response)
        self.assertGreater(len(content.strip()), 0)
        print(f"\n[4 images] {content}")
