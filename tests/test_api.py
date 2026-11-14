"""Local HTTP API (FastAPI): upload an image or PDF, get JSON/text/hOCR/ALTO/PDF back."""

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
NASKH = ROOT / "data" / "fonts" / "notonaskharabic" / "NotoNaskhArabic[wght].ttf"
needs_models = pytest.mark.skipif(
    not (NASKH.exists() and (MODELS / "detector-v2.onnx").exists()), reason="fonts/models"
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from arabic_ocr.api import create_app

    return TestClient(create_app())


@pytest.fixture(scope="module")
def page_png() -> bytes:
    from training.render import render_line

    ink = render_line("بسم الله الرحمن الرحيم", NASKH, 40)
    page = np.full((ink.shape[0] + 80, ink.shape[1] + 80), 255, np.uint8)
    page[40:-40, 40:-40] = 255 - ink
    buf = io.BytesIO()
    Image.fromarray(page).save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_rejects_unsupported_files(client):
    r = client.post("/ocr", files={"file": ("notes.docx", b"x")})
    assert r.status_code == 415


@needs_models
def test_ocr_json_text_and_xml(client, page_png):
    r = client.post("/ocr", files={"file": ("page.png", page_png)})
    assert r.status_code == 200
    doc = r.json()
    assert doc["file"] == "page.png" and "الرحمن" in doc["pages"][0]["text"]

    r = client.post("/ocr?format=txt&digits=western", files={"file": ("page.png", page_png)})
    assert r.status_code == 200 and "الرحمن" in r.text

    r = client.post("/ocr?format=alto", files={"file": ("page.png", page_png)})
    assert r.status_code == 200 and r.text.startswith("<?xml")


@needs_models
def test_ocr_searchable_pdf(client, page_png):
    r = client.post("/ocr?format=pdf", files={"file": ("page.png", page_png)})
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf" and r.content.startswith(b"%PDF")


@needs_models
def test_bad_image_is_a_client_error(client):
    r = client.post("/ocr", files={"file": ("broken.png", b"not an image")})
    assert r.status_code == 422
    assert json.loads(r.text)["detail"]
