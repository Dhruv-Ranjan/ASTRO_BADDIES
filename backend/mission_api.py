"""Local inference service for the AURORA mission-control demo.

This service exposes the existing SpectralNetV10 checkpoint only. It does not
pretend that an uploaded image is a satellite observation or create drought
labels from a single frame.
"""

from __future__ import annotations

import base64
import cgi
import io
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent / "FINAL_PROJECT_NASA"
SRC = PROJECT_ROOT / "src"
CHECKPOINT = PROJECT_ROOT / "artifacts" / "spectral_realworld_production.pt"
sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from model_defs import SpectralNetV10  # noqa: E402


HOST = "127.0.0.1"
PORT = 8000
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL: SpectralNetV10 | None = None


def load_model() -> SpectralNetV10:
    global MODEL
    if MODEL is None:
        if not CHECKPOINT.exists():
            raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
        model = SpectralNetV10().to(DEVICE)
        state = torch.load(CHECKPOINT, map_location=DEVICE)
        weights = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
        model.load_state_dict(weights)
        model.eval()
        MODEL = model
    return MODEL


def json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, allow_nan=False).encode("utf-8")


def preview_data_url(values: torch.Tensor) -> str:
    image = values.detach().cpu().clamp(0, 1).mul(255).byte().numpy()
    output = io.BytesIO()
    Image.fromarray(image, mode="L").resize((256, 256)).save(output, format="PNG", optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def reconstruct(image_bytes: bytes) -> dict[str, Any]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = image.size
    width = max(16, min(512, image.width - (image.width % 16)))
    height = max(16, min(512, image.height - (image.height % 16)))
    if (width, height) != image.size:
        image = image.resize((width, height), Image.Resampling.BILINEAR)

    pixels = torch.frombuffer(image.tobytes(), dtype=torch.uint8).reshape(height, width, 3)
    rgb = pixels.permute(2, 0, 1).float().div(255).unsqueeze(0).to(DEVICE)
    model = load_model()
    with torch.inference_mode():
        spectral = model(rgb)[0].cpu()

    bands = []
    for index, name in enumerate(("B02 blue", "B03 green", "B04 red", "B08 NIR", "B11 SWIR", "B12 SWIR")):
        band = spectral[index]
        bands.append(
            {
                "name": name,
                "minimum": round(float(band.min()), 6),
                "maximum": round(float(band.max()), 6),
                "mean": round(float(band.mean()), 6),
                "preview": preview_data_url(band),
            }
        )
    return {
        "status": "ok",
        "model": "SpectralNetV10",
        "device": str(DEVICE),
        "checkpoint": str(CHECKPOINT),
        "original_size": list(original_size),
        "processed_size": [width, height],
        "bands": bands,
        "note": "Model output from the uploaded RGB image. This is not a new satellite observation.",
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict[str, Any], content_type: str = "application/json") -> None:
        body = payload["body"] if isinstance(payload.get("body"), bytes) else json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, {"ok": True, "checkpoint": CHECKPOINT.exists(), "device": str(DEVICE)})
            return
        if self.path == "/ai_drought_snapshot.json":
            snapshot = ROOT / "ai_drought_snapshot.json"
            if snapshot.exists():
                body = snapshot.read_bytes()
                self._send(200, {"body": body}, "application/json")
                return
            self._send(404, {"error": "Forecast snapshot not found"})
            return
        if self.path == "/demo_assets/test_rgb_sample.png":
            image_path = ROOT / "demo_assets" / "test_rgb_sample.png"
            if image_path.exists():
                body = image_path.read_bytes()
                self._send(200, {"body": body}, "image/png")
                return
            self._send(404, {"error": "Demo sample not found"})
            return
        if self.path in ("/", "/mission-control.html"):
            body = (ROOT / "mission-control.html").read_bytes()
            self._send(200, {"body": body}, "text/html; charset=utf-8")
            return
        self._send(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/api/reconstruct":
            self._send(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_UPLOAD_BYTES:
                raise ValueError("Upload must be between 1 byte and 10 MB.")
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                raise ValueError("Send an image as multipart/form-data with field name 'image'.")
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
            field = form["image"] if "image" in form else None
            if field is None or not getattr(field, "file", None):
                raise ValueError("The multipart field 'image' is required.")
            self._send(200, reconstruct(field.file.read()))
        except (ValueError, OSError, RuntimeError) as error:
            self._send(400, {"error": str(error)})
        except Exception as error:
            self._send(500, {"error": f"Inference failed: {error}"})

    def log_message(self, format: str, *args: Any) -> None:
        return


if __name__ == "__main__":
    load_model()
    print(f"AURORA inference service: http://{HOST}:{PORT}/mission-control.html")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
