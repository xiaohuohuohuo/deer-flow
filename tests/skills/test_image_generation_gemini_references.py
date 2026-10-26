"""Exercise Gemini reference metadata with real image files and offline HTTP."""

import base64
import builtins
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from skill_loader import FakeResp, load

img = load("image-generation")


def generated_image_response():
    return FakeResp(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "data": base64.b64encode(b"generated").decode()
                                }
                            }
                        ]
                    }
                }
            ]
        }
    )


@pytest.mark.parametrize(
    ("image_format", "filename", "mime_type"),
    [
        ("PNG", "reference.png", "image/png"),
        ("PNG", "reference.PNG", "image/png"),
        ("PNG", "reference.jpg", "image/png"),
        ("PNG", "reference", "image/png"),
        ("WEBP", "reference.webp", "image/webp"),
        ("GIF", "reference.gif", "image/gif"),
        ("JPEG", "reference.jpg", "image/jpeg"),
        ("JPEG", "reference.jpeg", "image/jpeg"),
    ],
)
def test_gemini_reference_mime_matches_image_bytes(
    monkeypatch, tmp_path, image_format, filename, mime_type
):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("IMAGE_GENERATION_PROVIDER", "gemini")
    reference = tmp_path / filename
    Image.new("RGB", (2, 2), "red").save(reference, format=image_format)
    prompt = tmp_path / "prompt.json"
    prompt.write_text('{"prompt": "use the reference"}', encoding="utf-8")
    output = tmp_path / "output.png"
    captured = {}

    def fake_post(url, headers=None, json=None, **kwargs):
        captured["json"] = json
        return generated_image_response()

    monkeypatch.setattr(img.requests, "post", fake_post)
    result = img.generate_image(str(prompt), [str(reference)], str(output), "1:1")

    parts = captured["json"]["contents"][0]["parts"]
    inline = parts[0]["inlineData"]
    assert inline["mimeType"] == mime_type
    assert base64.b64decode(inline["data"]) == reference.read_bytes()
    assert parts[1] == {"text": prompt.read_text(encoding="utf-8")}
    assert output.read_bytes() == b"generated"
    assert "Successfully generated image" in result


def test_gemini_mixed_reference_formats_keep_order_and_skip_corrupt_files(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    references = []
    for image_format in ("PNG", "JPEG", "WEBP", "GIF"):
        reference = tmp_path / f"{image_format}.image"
        Image.new("RGB", (2, 2), "blue").save(reference, format=image_format)
        references.append(reference)
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    captured = {}

    def fake_post(url, headers=None, json=None, **kwargs):
        captured["json"] = json
        return generated_image_response()

    monkeypatch.setattr(img.requests, "post", fake_post)
    img._generate_image_gemini(
        "use the references",
        [str(references[0]), str(corrupt), *(str(p) for p in references[1:])],
        str(tmp_path / "output.png"),
        "1:1",
    )

    parts = captured["json"]["contents"][0]["parts"]
    assert [part["inlineData"]["mimeType"] for part in parts[:-1]] == [
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
    ]
    assert [base64.b64decode(part["inlineData"]["data"]) for part in parts[:-1]] == [
        p.read_bytes() for p in references
    ]
    assert parts[-1] == {"text": "use the references"}


def test_gemini_without_references_does_not_require_pillow(monkeypatch, tmp_path):
    real_import = builtins.__import__

    def import_without_pillow(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("Pillow unavailable for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_pillow)
    without_pillow = load("image-generation")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        without_pillow.requests,
        "post",
        lambda *args, **kwargs: generated_image_response(),
    )
    output = tmp_path / "output.png"

    result = without_pillow._generate_image_gemini("a deer", [], str(output), "1:1")

    assert output.read_bytes() == b"generated"
    assert "Successfully generated image" in result
