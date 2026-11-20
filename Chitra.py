"""
CHITRA - Single-File Image Generation API & Web Application

A self-contained Flask application that provides both a REST API and a web dashboard
for generating images using the official OpenAI Image API.

Requirements:
    pip install flask openai

Environment Variables:
    OPENAI_API_KEY      - Your OpenAI API key (required)
    CHITRA_API_KEY      - Optional API key for protecting CHITRA's endpoints
    CHITRA_HOST         - Host to bind (default: 127.0.0.1)
    CHITRA_PORT         - Port to bind (default: 5050)
    CHITRA_OUTPUT_DIR   - Directory for storing generated images (default: ./chitra_output)
    CHITRA_MODEL        - Default image model (default: gpt-image-1)
    CHITRA_RATE_LIMIT   - Requests per minute per IP (default: 10)

Run:
    python chitra.py
"""

import os
import io
import json
import base64
import uuid
import hashlib
import hmac
import time
import threading
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, Dict, Any

from flask import Flask, request, jsonify, render_template_string, send_file, abort, Response
from openai import OpenAI, OpenAIError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read an environment variable, returning default if unset."""
    return os.environ.get(key, default)

OPENAI_API_KEY = get_env("OPENAI_API_KEY")
CHITRA_API_KEY = get_env("CHITRA_API_KEY")  # Optional protection for CHITRA itself
HOST = get_env("CHITRA_HOST", "127.0.0.1")
PORT = int(get_env("CHITRA_PORT", "5050"))
OUTPUT_DIR = Path(get_env("CHITRA_OUTPUT_DIR", "./chitra_output")).resolve()
DEFAULT_MODEL = get_env("CHITRA_MODEL", "gpt-image-1")
RATE_LIMIT_PER_MINUTE = int(get_env("CHITRA_RATE_LIMIT", "10"))

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_FILE = OUTPUT_DIR / "_metadata.json"

# In-memory rate limiter (per-process, suitable for local use)
_rate_buckets: Dict[str, list] = defaultdict(list)
_rate_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def check_rate_limit(identifier: str) -> bool:
    """Return True if request is allowed under the per-minute rate limit."""
    now = time.time()
    window = 60.0
    with _rate_lock:
        bucket = _rate_buckets[identifier]
        # Remove entries older than the window
        bucket[:] = [t for t in bucket if now - t < window]
        if len(bucket) >= RATE_LIMIT_PER_MINUTE:
            return False
        bucket.append(now)
        return True

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def require_auth(f):
    """Decorator: require CHITRA_API_KEY if it is configured."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if CHITRA_API_KEY is None:
            # No API key configured — local mode, allow everything
            return f(*args, **kwargs)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or malformed Authorization header"}), 401

        token = auth_header[len("Bearer "):]
        if not hmac.compare_digest(token.encode(), CHITRA_API_KEY.encode()):
            return jsonify({"error": "Invalid API key"}), 401

        return f(*args, **kwargs)
    return decorated

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

def get_openai_client() -> Optional[OpenAI]:
    """Create an OpenAI client if the API key is available."""
    if not OPENAI_API_KEY:
        return None
    return OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# Metadata management
# ---------------------------------------------------------------------------

_metadata_lock = threading.Lock()

def load_metadata() -> Dict[str, Any]:
    """Load generation history from the metadata file."""
    if not METADATA_FILE.exists():
        return {"images": []}
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "images" not in data:
            return {"images": []}
        return data
    except (json.JSONDecodeError, OSError):
        return {"images": []}

def save_metadata(data: Dict[str, Any]) -> None:
    """Persist generation history to disk atomically."""
    tmp = METADATA_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(METADATA_FILE)

def add_image_record(record: Dict[str, Any]) -> None:
    """Append a new image record to metadata."""
    with _metadata_lock:
        data = load_metadata()
        data["images"].insert(0, record)  # newest first
        save_metadata(data)

def get_image_record(image_id: str) -> Optional[Dict[str, Any]]:
    """Look up a single image record by ID."""
    data = load_metadata()
    for img in data["images"]:
        if img.get("id") == image_id:
            return img
    return None

def remove_image_record(image_id: str) -> Optional[Dict[str, Any]]:
    """Remove and return an image record by ID."""
    with _metadata_lock:
        data = load_metadata()
        for i, img in enumerate(data["images"]):
            if img.get("id") == image_id:
                removed = data["images"].pop(i)
                save_metadata(data)
                return removed
        return None

# ---------------------------------------------------------------------------
# File safety
# ---------------------------------------------------------------------------

def safe_filename(image_id: str, ext: str) -> str:
    """Generate a safe filename from an image ID and extension."""
    # Only allow hex IDs produced by uuid4()
    if not all(c in "0123456789abcdef-" for c in image_id):
        raise ValueError("Invalid image ID format")
    # Validate extension against allowlist
    if ext not in ("png", "jpeg", "jpg", "webp"):
        raise ValueError("Unsupported image format")
    return f"{image_id}.{ext}"

def resolve_image_path(image_id: str, ext: str) -> Path:
    """Resolve an image path, ensuring it stays inside OUTPUT_DIR."""
    filename = safe_filename(image_id, ext)
    path = (OUTPUT_DIR / filename).resolve()
    if not str(path).startswith(str(OUTPUT_DIR)):
        raise ValueError("Path traversal detected")
    return path

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Model-specific supported parameters (verified against current OpenAI docs)
MODEL_CAPABILITIES = {
    "gpt-image-1": {
        "sizes": ["1024x1024", "1536x1024", "1024x1536", "auto"],
        "qualities": ["low", "medium", "high", "auto"],
        "formats": ["png", "jpeg", "webp"],
        "backgrounds": ["transparent", "opaque", "auto"],
    },
    "gpt-image-1-mini": {
        "sizes": ["1024x1024", "1536x1024", "1024x1536", "auto"],
        "qualities": ["low", "medium", "high", "auto"],
        "formats": ["png", "jpeg", "webp"],
        "backgrounds": ["transparent", "opaque", "auto"],
    },
}

def get_supported_models() -> list:
    """Return the list of model IDs that CHITRA exposes."""
    return list(MODEL_CAPABILITIES.keys())

def validate_generation_request(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Validate and normalize a generation request.
    Returns (normalized_params, error_message).
    """
    prompt = data.get("prompt")
    if not prompt or not isinstance(prompt, str):
        return {}, "Field 'prompt' is required and must be a string."
    prompt = prompt.strip()
    if len(prompt) < 1:
        return {}, "Prompt cannot be empty."
    if len(prompt) > 32000:
        return {}, "Prompt exceeds maximum length of 32000 characters."

    model = data.get("model", DEFAULT_MODEL)
    if model not in MODEL_CAPABILITIES:
        supported = ", ".join(MODEL_CAPABILITIES.keys())
        return {}, f"Unsupported model '{model}'. Supported models: {supported}"

    caps = MODEL_CAPABILITIES[model]

    size = data.get("size", "1024x1024")
    if size not in caps["sizes"]:
        return {}, f"Unsupported size '{size}' for model '{model}'. Supported: {caps['sizes']}"

    quality = data.get("quality", "auto")
    if quality not in caps["qualities"]:
        return {}, f"Unsupported quality '{quality}' for model '{model}'. Supported: {caps['qualities']}"

    output_format = data.get("output_format", "png")
    if output_format not in caps["formats"]:
        return {}, f"Unsupported output_format '{output_format}'. Supported: {caps['formats']}"

    background = data.get("background", "auto")
    if background not in caps["backgrounds"]:
        return {}, f"Unsupported background '{background}'. Supported: {caps['backgrounds']}"

    # Background transparency requires png or webp
    if background == "transparent" and output_format not in ("png", "webp"):
        return {}, "Transparent background requires output_format to be 'png' or 'webp'."

    # n parameter (number of images)
    n = data.get("n", 1)
    if not isinstance(n, int) or n < 1 or n > 10:
        return {}, "Field 'n' must be an integer between 1 and 10."

    return {
        "prompt": prompt,
        "model": model,
        "size": size,
        "quality": quality,
        "output_format": output_format,
        "background": background,
        "n": n,
    }, None

# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def generate_images(params: Dict[str, Any]) -> Tuple[Optional[list], Optional[str], Optional[str]]:
    """
    Call the OpenAI Images API and save results locally.
    Returns (list_of_records, error_message, status_code).
    """
    client = get_openai_client()
    if client is None:
        return None, "OPENAI_API_KEY is not configured on the server.", 503

    try:
        # Build kwargs — only include parameters supported by the model
        kwargs = {
            "model": params["model"],
            "prompt": params["prompt"],
            "size": params["size"],
            "quality": params["quality"],
            "output_format": params["output_format"],
            "background": params["background"],
            "n": params["n"],
        }

        response = client.images.generate(**kwargs)

    except OpenAIError as e:
        # Distinguish common error types without leaking secrets
        err_msg = str(e).lower()
        if "invalid api key" in err_msg or "incorrect api key" in err_msg:
            return None, "OpenAI API key is invalid.", 401
        if "quota" in err_msg or "insufficient" in err_msg:
            return None, "OpenAI API quota exceeded. Please check your billing.", 402
        if "rate limit" in err_msg:
            return None, "OpenAI rate limit reached. Please try again later.", 429
        if "model" in err_msg and ("not found" in err_msg or "does not exist" in err_msg):
            return None, f"Model '{params['model']}' is not available via the OpenAI API.", 400
        return None, f"OpenAI API error: {str(e)}", 502
    except Exception as e:
        return None, f"Unexpected error calling OpenAI: {str(e)}", 500

    if not response or not hasattr(response, "data") or not response.data:
        return None, "OpenAI returned an empty response.", 502

    records = []
    timestamp = datetime.now(timezone.utc).isoformat()

    for item in response.data:
        # gpt-image models always return b64_json; handle both for robustness
        image_bytes = None
        if hasattr(item, "b64_json") and item.b64_json:
            image_bytes = base64.b64decode(item.b64_json)
        elif hasattr(item, "url") and item.url:
            # URL-based responses are not expected for gpt-image-1, but handle gracefully
            return None, "URL-based responses are not supported by this endpoint. Use a GPT Image model.", 502
        else:
            continue

        image_id = str(uuid.uuid4())
        ext = params["output_format"]
        filename = safe_filename(image_id, ext)

        try:
            path = resolve_image_path(image_id, ext)
            with open(path, "wb") as f:
                f.write(image_bytes)
        except (OSError, ValueError) as e:
            return None, f"Failed to save image: {str(e)}", 500

        # Determine dimensions from the requested size (OpenAI may vary slightly with 'auto')
        width, height = None, None
        size_str = params["size"]
        if "x" in size_str:
            try:
                w, h = size_str.split("x")
                width, height = int(w), int(h)
            except ValueError:
                pass

        record = {
            "id": image_id,
            "filename": filename,
            "model": params["model"],
            "prompt": params["prompt"],
            "size": params["size"],
            "quality": params["quality"],
            "output_format": ext,
            "background": params["background"],
            "width": width,
            "height": height,
            "created_at": timestamp,
            "url": f"/v1/images/{image_id}",
        }
        add_image_record(record)
        records.append(record)

    return records, None, 200

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB request limit

# ---------------------------------------------------------------------------
# Routes: Health & Models
# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    """Return service health and configuration status."""
    return jsonify({
        "status": "ok",
        "service": "CHITRA",
        "version": "1.0.0",
        "openai_configured": OPENAI_API_KEY is not None,
        "auth_required": CHITRA_API_KEY is not None,
        "default_model": DEFAULT_MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

@app.route("/v1/models")
@require_auth
def list_models():
    """Return supported image models and their capabilities."""
    models = []
    for model_id, caps in MODEL_CAPABILITIES.items():
        models.append({
            "id": model_id,
            "object": "model",
            "owned_by": "openai",
            "capabilities": caps,
        })
    return jsonify({"object": "list", "data": models})

# ---------------------------------------------------------------------------
# Routes: Image generation
# ---------------------------------------------------------------------------

@app.route("/v1/images/generations", methods=["POST"])
@require_auth
def create_generation():
    """Generate images from a prompt via the OpenAI Image API."""
    client_ip = request.remote_addr or "unknown"
    if not check_rate_limit(client_ip):
        return jsonify({
            "error": "Rate limit exceeded",
            "detail": f"Maximum {RATE_LIMIT_PER_MINUTE} requests per minute.",
            "status": 429,
        }), 429

    # Parse JSON body
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    try:
        data = request.get_json()
    except Exception:
        return jsonify({"error": "Invalid JSON in request body"}), 400

    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    # Validate
    params, err = validate_generation_request(data)
    if err:
        return jsonify({"error": err}), 400

    # Generate
    records, err, status_code = generate_images(params)
    if err:
        return jsonify({"error": err}), status_code

    return jsonify({
        "created": int(time.time()),
        "data": [
            {
                "id": r["id"],
                "url": r["url"],
                "model": r["model"],
                "prompt": r["prompt"],
                "size": r["size"],
                "quality": r["quality"],
                "output_format": r["output_format"],
                "width": r["width"],
                "height": r["height"],
                "created_at": r["created_at"],
            }
            for r in records
        ],
    }), 201

# ---------------------------------------------------------------------------
# Routes: Image listing, retrieval, deletion
# ---------------------------------------------------------------------------

@app.route("/v1/images", methods=["GET"])
@require_auth
def list_images():
    """Return the generation history."""
    data = load_metadata()
    images = data.get("images", [])
    return jsonify({
        "object": "list",
        "total": len(images),
        "data": images,
    })

@app.route("/v1/images/<image_id>", methods=["GET"])
@require_auth
def get_image(image_id: str):
    """Serve a generated image file."""
    record = get_image_record(image_id)
    if not record:
        return jsonify({"error": "Image not found"}), 404

    try:
        path = resolve_image_path(image_id, record["output_format"])
    except ValueError:
        return jsonify({"error": "Invalid image identifier"}), 400

    if not path.exists():
        return jsonify({"error": "Image file missing from storage"}), 404

    mime = {
        "png": "image/png",
        "jpeg": "image/jpeg",
        "jpg": "image/jpeg",
        "webp": "image/webp",
    }.get(record["output_format"], "application/octet-stream")

    return send_file(path, mimetype=mime, as_attachment=False, download_name=record["filename"])

@app.route("/v1/images/<image_id>", methods=["DELETE"])
@require_auth
def delete_image(image_id: str):
    """Delete a generated image and its metadata."""
    record = remove_image_record(image_id)
    if not record:
        return jsonify({"error": "Image not found"}), 404

    # Delete the file
    try:
        path = resolve_image_path(image_id, record["output_format"])
        if path.exists():
            path.unlink()
    except (OSError, ValueError):
        # Metadata already removed; file deletion failure is logged but not fatal
        pass

    return jsonify({"deleted": True, "id": image_id}), 200

# ---------------------------------------------------------------------------
# Routes: Documentation
# ---------------------------------------------------------------------------

@app.route("/v1/docs")
def api_docs():
    """Return JSON API documentation."""
    base_url = request.host_url.rstrip("/")
    return jsonify({
        "name": "CHITRA Image Generation API",
        "version": "1.0.0",
        "base_url": base_url,
        "authentication": {
            "type": "Bearer token" if CHITRA_API_KEY else "none (local mode)",
            "header": "Authorization: Bearer <CHITRA_API_KEY>",
            "note": "Set CHITRA_API_KEY env var to enable authentication.",
        },
        "endpoints": [
            {"method": "GET", "path": "/health", "description": "Health check"},
            {"method": "GET", "path": "/v1/models", "description": "List supported models"},
            {"method": "POST", "path": "/v1/images/generations", "description": "Generate images"},
            {"method": "GET", "path": "/v1/images", "description": "List generation history"},
            {"method": "GET", "path": "/v1/images/<id>", "description": "Retrieve an image"},
            {"method": "DELETE", "path": "/v1/images/<id>", "description": "Delete an image"},
            {"method": "GET", "path": "/v1/docs", "description": "This documentation"},
        ],
        "generation_request": {
            "content_type": "application/json",
            "fields": {
                "prompt": "string (required, max 32000 chars)",
                "model": f"string (default: {DEFAULT_MODEL})",
                "size": "string (default: 1024x1024)",
                "quality": "string (default: auto)",
                "output_format": "string (default: png)",
                "background": "string (default: auto)",
                "n": "integer (1-10, default: 1)",
            },
        },
        "errors": {
            "400": "Invalid request / validation error",
            "401": "Missing or invalid API key",
            "402": "OpenAI quota exceeded",
            "404": "Image not found",
            "415": "Unsupported content type",
            "429": "Rate limit exceeded",
            "500": "Internal server error",
            "502": "OpenAI API error",
            "503": "OpenAI not configured",
        },
    })

# ---------------------------------------------------------------------------
# Dashboard HTML (inline, single-file)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CHITRA — Image Generation Platform</title>
<!-- Fonts: using system stack for zero external requests; premium feel via weight & spacing -->
<style>
/* ============================================================
   CHITRA Dashboard — Dark glassmorphism, purple/cyan accents
   Font stack: system-ui for zero-latency, no external deps.
   ============================================================ */
:root {
  --bg: #0a0a12;
  --surface: rgba(20, 20, 35, 0.7);
  --glass: rgba(30, 30, 55, 0.45);
  --glass-border: rgba(140, 120, 255, 0.18);
  --purple: #8b5cf6;
  --purple-dim: #6d28d9;
  --cyan: #22d3ee;
  --text: #e8e6f0;
  --text-dim: #9490a8;
  --danger: #f87171;
  --success: #34d399;
  --radius: 16px;
  --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, Roboto, 'Helvetica Neue', sans-serif;
  --mono: 'SF Mono', 'Fira Code', 'Cascadia Code', Consolas, monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: var(--font);
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}
/* Subtle radial glow behind content */
body::before {
  content: '';
  position: fixed; inset: 0; z-index: -1;
  background:
    radial-gradient(ellipse 80% 60% at 20% 10%, rgba(139,92,246,0.12), transparent),
    radial-gradient(ellipse 60% 50% at 80% 90%, rgba(34,211,238,0.08), transparent);
  pointer-events: none;
}
.app { max-width: 1280px; margin: 0 auto; padding: 24px 20px 60px; }

/* --- Header --- */
header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 20px 0 32px; flex-wrap: wrap; gap: 16px;
}
.brand { display: flex; align-items: center; gap: 14px; }
.brand-mark {
  width: 44px; height: 44px; border-radius: 14px;
  background: linear-gradient(135deg, var(--purple), var(--cyan));
  display: flex; align-items: center; justify-content: center;
  font-size: 22px; font-weight: 800; color: #0a0a12;
  letter-spacing: -1px; box-shadow: 0 0 24px rgba(139,92,246,0.35);
}
.brand h1 { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.5px; }
.brand h1 span { color: var(--purple); }
.badge {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.5px; padding: 4px 12px; border-radius: 20px;
  background: var(--glass); border: 1px solid var(--glass-border);
  color: var(--text-dim);
}
.badge.online { color: var(--success); border-color: rgba(52,211,153,0.3); }
.badge.offline { color: var(--danger); border-color: rgba(248,113,113,0.3); }
.dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }

/* --- Stats row --- */
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 32px; }
.stat-card {
  background: var(--glass); backdrop-filter: blur(16px);
  border: 1px solid var(--glass-border); border-radius: var(--radius);
  padding: 20px 22px;
}
.stat-card .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.8px; color: var(--text-dim); font-weight: 600; }
.stat-card .value { font-size: 1.8rem; font-weight: 700; margin-top: 6px; letter-spacing: -1px; }
.stat-card .value.purple { color: var(--purple); }
.stat-card .value.cyan { color: var(--cyan); }
.stat-card .sub { font-size: 0.8rem; color: var(--text-dim); margin-top: 4px; }

/* --- Tabs --- */
.tabs { display: flex; gap: 4px; margin-bottom: 28px; border-bottom: 1px solid var(--glass-border); padding-bottom: 0; overflow-x: auto; }
.tab-btn {
  background: none; border: none; color: var(--text-dim);
  padding: 12px 22px; font-size: 0.9rem; font-weight: 600;
  cursor: pointer; border-bottom: 2px solid transparent;
  white-space: nowrap; transition: all 0.2s; font-family: var(--font);
}
.tab-btn:hover { color: var(--text); }
.tab-btn.active { color: var(--purple); border-bottom-color: var(--purple); }

/* --- Panels --- */
.panel { display: none; animation: fadeIn 0.25s ease; }
.panel.active { display: block; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }

/* --- Cards & forms --- */
.card {
  background: var(--glass); backdrop-filter: blur(16px);
  border: 1px solid var(--glass-border); border-radius: var(--radius);
  padding: 28px; margin-bottom: 24px;
}
.card h2 { font-size: 1.15rem; font-weight: 700; margin-bottom: 20px; letter-spacing: -0.3px; }
.card h2 .accent { color: var(--cyan); }
.form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.form-grid .full { grid-column: 1 / -1; }
@media (max-width: 640px) { .form-grid { grid-template-columns: 1fr; } }
label { display: block; font-size: 0.8rem; font-weight: 600; color: var(--text-dim); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.4px; }
textarea, input, select {
  width: 100%; padding: 12px 14px; border-radius: 10px;
  background: rgba(10,10,20,0.6); border: 1px solid var(--glass-border);
  color: var(--text); font-family: var(--font); font-size: 0.92rem;
  transition: border-color 0.2s, box-shadow 0.2s; outline: none;
}
textarea:focus, input:focus, select:focus {
  border-color: var(--purple); box-shadow: 0 0 0 3px rgba(139,92,246,0.15);
}
textarea { resize: vertical; min-height: 100px; font-family: var(--mono); font-size: 0.85rem; }
select { cursor: pointer; }

/* --- Buttons --- */
.btn {
  display: inline-flex; align-items: center; justify-content: center; gap: 8px;
  padding: 12px 24px; border-radius: 10px; border: none;
  font-family: var(--font); font-weight: 600; font-size: 0.9rem;
  cursor: pointer; transition: all 0.2s; letter-spacing: 0.2px;
}
.btn-primary {
  background: linear-gradient(135deg, var(--purple), #7c3aed);
  color: #fff; box-shadow: 0 4px 20px rgba(139,92,246,0.3);
}
.btn-primary:hover:not(:disabled) { box-shadow: 0 6px 28px rgba(139,92,246,0.45); transform: translateY(-1px); }
.btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-ghost {
  background: transparent; border: 1px solid var(--glass-border); color: var(--text-dim);
}
.btn-ghost:hover { border-color: var(--purple); color: var(--text); }
.btn-danger { background: rgba(248,113,113,0.15); border: 1px solid rgba(248,113,113,0.3); color: var(--danger); }
.btn-danger:hover { background: rgba(248,113,113,0.25); }
.btn-sm { padding: 6px 14px; font-size: 0.8rem; border-radius: 8px; }

/* --- Loading spinner --- */
.spinner {
  width: 20px; height: 20px; border: 2px solid rgba(255,255,255,0.2);
  border-top-color: #fff; border-radius: 50%;
  animation: spin 0.7s linear infinite; display: inline-block;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* --- Gallery --- */
.gallery-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 20px; }
.gallery-item {
  background: var(--glass); border: 1px solid var(--glass-border);
  border-radius: 14px; overflow: hidden; transition: border-color 0.2s;
}
.gallery-item:hover { border-color: rgba(139,92,246,0.4); }
.gallery-item img { width: 100%; aspect-ratio: 1; object-fit: cover; display: block; background: #111; }
.gallery-meta { padding: 14px 16px; }
.gallery-meta .prompt-text {
  font-size: 0.82rem; color: var(--text-dim); line-height: 1.4;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
  margin-bottom: 8px; min-height: 2.3em;
}
.gallery-meta .info-row { display: flex; justify-content: space-between; align-items: center; font-size: 0.72rem; color: var(--text-dim); margin-bottom: 10px; }
.gallery-actions { display: flex; gap: 8px; }
.gallery-actions .btn { flex: 1; }
.empty-state { text-align: center; padding: 60px 20px; color: var(--text-dim); }
.empty-state .icon { font-size: 3rem; margin-bottom: 12px; opacity: 0.5; }

/* --- API Playground --- */
.playground-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
@media (max-width: 800px) { .playground-grid { grid-template-columns: 1fr; } }
.code-block {
  background: rgba(5,5,15,0.8); border: 1px solid var(--glass-border);
  border-radius: 12px; padding: 16px; font-family: var(--mono);
  font-size: 0.82rem; overflow-x: auto; line-height: 1.5;
  white-space: pre-wrap; word-break: break-word; color: #c9c4e0;
}
.code-block .key { color: var(--cyan); }
.code-block .str { color: #a5f3a5; }
.code-block .num { color: #fbbf24; }
.status-badge {
  display: inline-block; padding: 4px 12px; border-radius: 8px;
  font-size: 0.78rem; font-weight: 700; font-family: var(--mono);
}
.status-200, .status-201 { background: rgba(52,211,153,0.15); color: var(--success); }
.status-400, .status-401, .status-404, .status-429 { background: rgba(248,113,113,0.15); color: var(--danger); }
.status-500, .status-502, .status-503 { background: rgba(251,191,36,0.15); color: #fbbf24; }

/* --- Docs --- */
.docs-content h3 { font-size: 1rem; margin: 24px 0 10px; color: var(--cyan); }
.docs-content h3:first-child { margin-top: 0; }
.docs-content p, .docs-content li { font-size: 0.88rem; color: var(--text-dim); line-height: 1.7; }
.docs-content ul { padding-left: 20px; margin: 8px 0; }
.docs-content code { font-family: var(--mono); background: rgba(139,92,246,0.12); padding: 2px 7px; border-radius: 5px; font-size: 0.82rem; color: var(--purple); }
.docs-content pre {
  background: rgba(5,5,15,0.8); border: 1px solid var(--glass-border);
  border-radius: 10px; padding: 16px; overflow-x: auto; margin: 10px 0;
  font-family: var(--mono); font-size: 0.8rem; color: #c9c4e0; line-height: 1.5;
}

/* --- Toast --- */
.toast {
  position: fixed; bottom: 28px; right: 28px; z-index: 1000;
  padding: 14px 22px; border-radius: 12px; font-size: 0.88rem; font-weight: 600;
  background: var(--glass); backdrop-filter: blur(20px);
  border: 1px solid var(--glass-border);
  box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  transform: translateY(80px); opacity: 0; transition: all 0.3s ease;
  pointer-events: none; max-width: 360px;
}
.toast.show { transform: translateY(0); opacity: 1; }
.toast.success { border-color: rgba(52,211,153,0.4); color: var(--success); }
.toast.error { border-color: rgba(248,113,113,0.4); color: var(--danger); }
.toast.info { border-color: rgba(34,211,238,0.4); color: var(--cyan); }

/* --- Preview image --- */
.preview-container { text-align: center; margin-top: 20px; display: none; }
.preview-container.show { display: block; }
.preview-container img {
  max-width: 100%; max-height: 480px; border-radius: 12px;
  border: 1px solid var(--glass-border);
}

/* --- Settings rows --- */
.setting-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 14px 0; border-bottom: 1px solid rgba(140,120,255,0.08);
  flex-wrap: wrap; gap: 8px;
}
.setting-row:last-child { border-bottom: none; }
.setting-row .k { font-size: 0.88rem; color: var(--text-dim); font-weight: 500; }
.setting-row .v { font-size: 0.88rem; font-weight: 600; font-family: var(--mono); color: var(--text); }

/* --- Scrollbar --- */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.25); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(139,92,246,0.4); }
</style>
</head>
<body>
<div class="app">

  <!-- Header -->
  <header>
    <div class="brand">
      <div class="brand-mark">C</div>
      <div>
        <h1>CHITRA<span>.</span></h1>
        <div style="font-size:0.75rem;color:var(--text-dim);letter-spacing:1px;text-transform:uppercase;">Image Generation Platform</div>
      </div>
    </div>
    <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
      <span class="badge" id="auth-badge">checking...</span>
      <span class="badge" id="status-badge"><span class="dot"></span> connecting...</span>
    </div>
  </header>

  <!-- Stats -->
  <div class="stats">
    <div class="stat-card">
      <div class="label">Total Images</div>
      <div class="value purple" id="stat-total">—</div>
      <div class="sub">generated locally</div>
    </div>
    <div class="stat-card">
      <div class="label">Default Model</div>
      <div class="value cyan" id="stat-model" style="font-size:1.1rem;margin-top:10px;">—</div>
      <div class="sub">configurable</div>
    </div>
    <div class="stat-card">
      <div class="label">API Status</div>
      <div class="value" id="stat-api" style="font-size:1.1rem;margin-top:10px;">—</div>
      <div class="sub">OpenAI connectivity</div>
    </div>
  </div>

  <!-- Tabs -->
  <div class="tabs">
    <button class="tab-btn active" data-tab="generate">Generate</button>
    <button class="tab-btn" data-tab="gallery">Gallery</button>
    <button class="tab-btn" data-tab="playground">API Playground</button>
    <button class="tab-btn" data-tab="docs">Documentation</button>
    <button class="tab-btn" data-tab="settings">Settings</button>
  </div>

  <!-- ==================== GENERATE ==================== -->
  <div class="panel active" id="panel-generate">
    <div class="card">
      <h2>Generate <span class="accent">Image</span></h2>
      <div class="form-grid">
        <div class="full">
          <label for="gen-prompt">Prompt</label>
          <textarea id="gen-prompt" placeholder="Describe the image you want to create..." maxlength="32000"></textarea>
        </div>
        <div>
          <label for="gen-model">Model</label>
          <select id="gen-model"></select>
        </div>
        <div>
          <label for="gen-size">Size</label>
          <select id="gen-size"></select>
        </div>
        <div>
          <label for="gen-quality">Quality</label>
          <select id="gen-quality"></select>
        </div>
        <div>
          <label for="gen-format">Output Format</label>
          <select id="gen-format"></select>
        </div>
        <div>
          <label for="gen-background">Background</label>
          <select id="gen-background"></select>
        </div>
        <div>
          <label for="gen-n">Number of Images</label>
          <input type="number" id="gen-n" value="1" min="1" max="10">
        </div>
        <div class="full" style="display:flex;gap:12px;align-items:center;margin-top:8px;">
          <button class="btn btn-primary" id="gen-btn" onclick="generateImage()">Generate</button>
          <span id="gen-status" style="font-size:0.85rem;color:var(--text-dim);"></span>
        </div>
      </div>
      <div class="preview-container" id="gen-preview">
        <img id="gen-preview-img" src="" alt="Generated image">
      </div>
    </div>
  </div>

  <!-- ==================== GALLERY ==================== -->
  <div class="panel" id="panel-gallery">
    <div class="card">
      <h2>Image <span class="accent">Gallery</span></h2>
      <div id="gallery-grid" class="gallery-grid"></div>
      <div class="empty-state" id="gallery-empty" style="display:none;">
        <div class="icon">🖼️</div>
        <p>No images generated yet. Start creating!</p>
      </div>
    </div>
  </div>

  <!-- ==================== API PLAYGROUND ==================== -->
  <div class="panel" id="panel-playground">
    <div class="card">
      <h2>API <span class="accent">Playground</span></h2>
      <p style="font-size:0.85rem;color:var(--text-dim);margin-bottom:18px;">Send a real request to the CHITRA generation endpoint.</p>
      <div class="playground-grid">
        <div>
          <label>Endpoint</label>
          <div class="code-block" style="margin-bottom:14px;padding:10px 14px;">POST /v1/images/generations</div>
          <label>Request Body (JSON)</label>
          <textarea id="play-body" style="min-height:200px;" spellcheck="false">{
  "prompt": "A serene mountain lake at sunset, photorealistic",
  "model": "gpt-image-1",
  "size": "1024x1024",
  "quality": "high",
  "output_format": "png"
}</textarea>
          <div style="margin-top:14px;display:flex;gap:10px;">
            <button class="btn btn-primary" onclick="playgroundSend()">Send Request</button>
            <button class="btn btn-ghost btn-sm" onclick="copyPlaygroundExample()">Copy Example</button>
          </div>
        </div>
        <div>
          <label>Response</label>
          <div id="play-status" style="margin-bottom:10px;min-height:28px;"></div>
          <div class="code-block" id="play-response" style="min-height:200px;">// Response will appear here</div>
        </div>
      </div>
    </div>
  </div>

  <!-- ==================== DOCS ==================== -->
  <div class="panel" id="panel-docs">
    <div class="card docs-content">
      <h2>Documentation</h2>

      <h3>Installation</h3>
      <pre>pip install flask openai
export OPENAI_API_KEY="sk-..."
python chitra.py</pre>

      <h3>Authentication</h3>
      <p>By default CHITRA runs in <strong>local mode</strong> with no API key required. To protect the API, set the <code>CHITRA_API_KEY</code> environment variable:</p>
      <pre>export CHITRA_API_KEY="your-secret-token"</pre>
      <p>Then include the header in every request:</p>
      <pre>Authorization: Bearer your-secret-token</pre>

      <h3>Generate an Image</h3>
      <pre>curl -X POST http://127.0.0.1:5050/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_CHITRA_API_KEY" \
  -d '{
    "prompt": "A futuristic city at night",
    "model": "gpt-image-1",
    "size": "1024x1024",
    "quality": "high",
    "output_format": "png"
  }'</pre>

      <h3>List Images</h3>
      <pre>curl http://127.0.0.1:5050/v1/images \
  -H "Authorization: Bearer YOUR_CHITRA_API_KEY"</pre>

      <h3>Retrieve an Image</h3>
      <pre>curl http://127.0.0.1:5050/v1/images/IMAGE_ID \
  -H "Authorization: Bearer YOUR_CHITRA_API_KEY" \
  --output image.png</pre>

      <h3>Delete an Image</h3>
      <pre>curl -X DELETE http://127.0.0.1:5050/v1/images/IMAGE_ID \
  -H "Authorization: Bearer YOUR_CHITRA_API_KEY"</pre>

      <h3>Error Codes</h3>
      <ul>
        <li><code>400</code> — Invalid request or validation error</li>
        <li><code>401</code> — Missing or invalid API key</li>
        <li><code>402</code> — OpenAI quota exceeded</li>
        <li><code>404</code> — Image not found</li>
        <li><code>415</code> — Unsupported content type</li>
        <li><code>429</code> — Rate limit exceeded</li>
        <li><code>502</code> — OpenAI API error</li>
        <li><code>503</code> — OpenAI not configured on server</li>
      </ul>

      <h3>Using from Another Application</h3>
      <p>CHITRA exposes a standard REST API. Point your HTTP client at <code>http://YOUR_HOST:5050</code>, include your CHITRA API key (if configured), and send JSON. The generated image is stored server-side and referenced by <code>id</code> and <code>url</code>.</p>
    </div>
  </div>

  <!-- ==================== SETTINGS ==================== -->
  <div class="panel" id="panel-settings">
    <div class="card">
      <h2>Settings</h2>
      <div class="setting-row">
        <span class="k">OpenAI API Key</span>
        <span class="v" id="set-openai">—</span>
      </div>
      <div class="setting-row">
        <span class="k">CHITRA API Key (auth)</span>
        <span class="v" id="set-auth">—</span>
      </div>
      <div class="setting-row">
        <span class="k">Default Model</span>
        <span class="v" id="set-model">—</span>
      </div>
      <div class="setting-row">
        <span class="k">Output Directory</span>
        <span class="v" id="set-output" style="font-size:0.8rem;">—</span>
      </div>
      <div class="setting-row">
        <span class="k">Rate Limit</span>
        <span class="v" id="set-rate">—</span>
      </div>
      <div class="setting-row">
        <span class="k">Server</span>
        <span class="v" id="set-server">—</span>
      </div>
    </div>
  </div>

</div>

<div class="toast" id="toast"></div>

<script>
// ============================================================
// CHITRA Dashboard JavaScript
// ============================================================

const state = {
  health: null,
  models: [],
  images: [],
  authToken: null,  // Set via localStorage if needed for auth mode
};

// --- Utilities ---
function $(id) { return document.getElementById(id); }

function toast(msg, type) {
  type = type || 'info';
  const el = $('toast');
  el.textContent = msg;
  el.className = 'toast ' + type;
  // Force reflow then show
  void el.offsetWidth;
  el.classList.add('show');
  clearTimeout(el._timer);
  el._timer = setTimeout(function() { el.classList.remove('show'); }, 3500);
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

async function apiFetch(url, opts) {
  opts = opts || {};
  const headers = Object.assign({}, opts.headers || {});
  if (state.authToken) {
    headers['Authorization'] = 'Bearer ' + state.authToken;
  }
  const res = await fetch(url, Object.assign({}, opts, { headers: headers }));
  return res;
}

// --- Health check ---
async function loadHealth() {
  try {
    const res = await fetch('/health');
    const data = await res.json();
    state.health = data;

    // Update status badge
    const badge = $('status-badge');
    if (data.openai_configured) {
      badge.className = 'badge online';
      badge.innerHTML = '<span class="dot"></span> API Ready';
    } else {
      badge.className = 'badge offline';
      badge.innerHTML = '<span class="dot"></span> No OpenAI Key';
    }

    const authBadge = $('auth-badge');
    authBadge.textContent = data.auth_required ? '🔒 Protected' : '🔓 Local Mode';

    // Stats
    $('stat-total').textContent = '0';
    $('stat-model').textContent = data.default_model || '—';
    $('stat-api').textContent = data.openai_configured ? '✅ Connected' : '❌ Not configured';
    $('stat-api').style.fontSize = '1rem';

    // Settings
    $('set-openai').textContent = data.openai_configured ? '✅ Configured' : '❌ Missing';
    $('set-openai').style.color = data.openai_configured ? 'var(--success)' : 'var(--danger)';
    $('set-auth').textContent = data.auth_required ? '🔒 Enabled' : '🔓 Disabled';
    $('set-model').textContent = data.default_model || '—';
    $('set-rate').textContent = '—';  // filled below via docs or not exposed
    $('set-server').textContent = window.location.host;

    // Load models
    await loadModels();
    // Load images
    await loadImages();

  } catch (e) {
    $('status-badge').className = 'badge offline';
    $('status-badge').innerHTML = '<span class="dot"></span> Offline';
    toast('Cannot reach CHITRA server', 'error');
  }
}

async function loadModels() {
  try {
    const res = await apiFetch('/v1/models');
    if (!res.ok) throw new Error('Models endpoint returned ' + res.status);
    const data = await res.json();
    state.models = data.data || [];
    populateSelectors();
  } catch (e) {
    // If auth required and no token, models may fail — that's fine
    state.models = [];
    populateSelectors();
  }
}

function populateSelectors() {
  const modelSel = $('gen-model');
  modelSel.innerHTML = '';
  if (state.models.length === 0) {
    // Fallback: show default
    const opt = document.createElement('option');
    opt.value = state.health ? state.health.default_model : 'gpt-image-1';
    opt.textContent = opt.value;
    modelSel.appendChild(opt);
  } else {
    state.models.forEach(function(m) {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = m.id;
      if (state.health && m.id === state.health.default_model) opt.selected = true;
      modelSel.appendChild(opt);
    });
  }
  updateCapabilities();
}

function updateCapabilities() {
  const modelSel = $('gen-model');
  const modelId = modelSel.value;
  const model = state.models.find(function(m) { return m.id === modelId; });
  const caps = model ? model.capabilities : { sizes: ['1024x1024'], qualities: ['auto'], formats: ['png'], backgrounds: ['auto'] };

  fillSelect('gen-size', caps.sizes, '1024x1024');
  fillSelect('gen-quality', caps.qualities, 'auto');
  fillSelect('gen-format', caps.formats, 'png');
  fillSelect('gen-background', caps.backgrounds, 'auto');
}

function fillSelect(id, options, selected) {
  const sel = $(id);
  sel.innerHTML = '';
  options.forEach(function(o) {
    const opt = document.createElement('option');
    opt.value = o;
    opt.textContent = o;
    if (o === selected) opt.selected = true;
    sel.appendChild(opt);
  });
}

// --- Image generation ---
async function generateImage() {
  const prompt = $('gen-prompt').value.trim();
  if (!prompt) {
    toast('Please enter a prompt', 'error');
    return;
  }

  const btn = $('gen-btn');
  const status = $('gen-status');
  const preview = $('gen-preview');
  const previewImg = $('gen-preview-img');

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Generating...';
  status.textContent = 'Calling OpenAI...';
  preview.classList.remove('show');

  const body = {
    prompt: prompt,
    model: $('gen-model').value,
    size: $('gen-size').value,
    quality: $('gen-quality').value,
    output_format: $('gen-format').value,
    background: $('gen-background').value,
    n: parseInt($('gen-n').value) || 1,
  };

  try {
    const res = await apiFetch('/v1/images/generations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || 'HTTP ' + res.status);
    }

    const images = data.data || [];
    status.textContent = '✅ ' + images.length + ' image(s) generated';

    if (images.length > 0) {
      previewImg.src = images[0].url;
      preview.classList.add('show');
    }

    toast('Image generated successfully', 'success');
    await loadImages();  // refresh gallery and stats

  } catch (e) {
    status.textContent = '❌ ' + e.message;
    toast(e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Generate';
  }
}

// --- Gallery ---
async function loadImages() {
  try {
    const res = await apiFetch('/v1/images');
    if (!res.ok) {
      if (res.status === 401) {
        // Auth required — show notice
        $('gallery-grid').innerHTML = '';
        $('gallery-empty').style.display = 'block';
        $('gallery-empty').querySelector('p').textContent = 'Authentication required. Set CHITRA_API_KEY in localStorage or disable auth.';
        return;
      }
      throw new Error('HTTP ' + res.status);
    }
    const data = await res.json();
    state.images = data.data || [];
    $('stat-total').textContent = state.images.length;
    renderGallery();
  } catch (e) {
    // silent
  }
}

function renderGallery() {
  const grid = $('gallery-grid');
  const empty = $('gallery-empty');

  if (state.images.length === 0) {
    grid.innerHTML = '';
    empty.style.display = 'block';
    empty.querySelector('p').textContent = 'No images generated yet. Start creating!';
    return;
  }

  empty.style.display = 'none';
  grid.innerHTML = state.images.map(function(img) {
    const date = img.created_at ? new Date(img.created_at).toLocaleDateString() : '—';
    return '<div class="gallery-item">' +
      '<img src="' + escapeHtml(img.url) + '" alt="' + escapeHtml(img.prompt) + '" loading="lazy" onerror="this.style.display=\'none\'">' +
      '<div class="gallery-meta">' +
        '<div class="prompt-text" title="' + escapeHtml(img.prompt) + '">' + escapeHtml(img.prompt) + '</div>' +
        '<div class="info-row"><span>' + escapeHtml(img.model || '') + '</span><span>' + date + '</span></div>' +
        '<div class="gallery-actions">' +
          '<a class="btn btn-ghost btn-sm" href="' + escapeHtml(img.url) + '" download>Download</a>' +
          '<button class="btn btn-danger btn-sm" onclick="deleteImage(\'' + img.id + '\')">Delete</button>' +
        '</div>' +
      '</div>' +
    '</div>';
  }).join('');
}

async function deleteImage(id) {
  if (!confirm('Delete this image permanently?')) return;
  try {
    const res = await apiFetch('/v1/images/' + id, { method: 'DELETE' });
    if (!res.ok) {
      const data = await res.json();
      throw new Error(data.error || 'HTTP ' + res.status);
    }
    toast('Image deleted', 'success');
    await loadImages();
  } catch (e) {
    toast(e.message, 'error');
  }
}

// --- Playground ---
async function playgroundSend() {
  const bodyEl = $('play-body');
  const statusEl = $('play-status');
  const respEl = $('play-response');

  let body;
  try {
    body = JSON.parse(bodyEl.value);
  } catch (e) {
    statusEl.innerHTML = '<span class="status-badge status-400">Invalid JSON</span>';
    respEl.textContent = e.message;
    return;
  }

  statusEl.innerHTML = '<span class="spinner"></span> Sending...';
  respEl.textContent = 'Waiting for response...';

  try {
    const res = await apiFetch('/v1/images/generations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const text = await res.text();
    let json;
    try { json = JSON.parse(text); } catch (e) { json = { raw: text }; }

    statusEl.innerHTML = '<span class="status-badge status-' + res.status + '">HTTP ' + res.status + '</span>';
    respEl.textContent = JSON.stringify(json, null, 2);

    if (res.ok) {
      toast('Request successful', 'success');
      await loadImages();
    } else {
      toast('Request failed: ' + (json.error || res.status), 'error');
    }
  } catch (e) {
    statusEl.innerHTML = '<span class="status-badge status-500">Error</span>';
    respEl.textContent = e.message;
    toast(e.message, 'error');
  }
}

function copyPlaygroundExample() {
  const text = $('play-body').value;
  navigator.clipboard.writeText(text).then(function() {
    toast('Copied to clipboard', 'success');
  }).catch(function() {
    toast('Copy failed', 'error');
  });
}

// --- Tabs ---
document.querySelectorAll('.tab-btn').forEach(function(btn) {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.tab-btn').forEach(function(b) { b.classList.remove('active'); });
    document.querySelectorAll('.panel').forEach(function(p) { p.classList.remove('active'); });
    btn.classList.add('active');
    $('panel-' + btn.dataset.tab).classList.add('active');
    if (btn.dataset.tab === 'gallery') loadImages();
    if (btn.dataset.tab === 'settings') loadHealth();
  });
});

// --- Init ---
document.addEventListener('DOMContentLoaded', function() {
  // Model change → update capabilities
  $('gen-model').addEventListener('change', updateCapabilities);

  // Optional: read auth token from localStorage for protected instances
  try {
    const stored = localStorage.getItem('chitra_auth_token');
    if (stored) state.authToken = stored;
  } catch (e) { /* ignore */ }

  loadHealth();
  // Refresh health + gallery every 30s
  setInterval(loadHealth, 30000);
});
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Dashboard route
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    """Serve the CHITRA web dashboard."""
    return render_template_string(DASHBOARD_HTML)

# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/v1/") or request.path == "/health":
        return jsonify({"error": "Not found"}), 404
    return render_template_string(DASHBOARD_HTML), 200  # SPA fallback to dashboard

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  CHITRA — Image Generation Platform")
    print("=" * 60)
    print(f"  Dashboard:  http://{HOST}:{PORT}")
    print(f"  API:        http://{HOST}:{PORT}/v1")
    print(f"  Models:     http://{HOST}:{PORT}/v1/models")
    print(f"  Docs:       http://{HOST}:{PORT}/v1/docs")
    print(f"  Output:     {OUTPUT_DIR}")
    print(f"  OpenAI key: {'✅ configured' if OPENAI_API_KEY else '❌ MISSING — set OPENAI_API_KEY'}")
    print(f"  Auth:       {'🔒 CHITRA_API_KEY set' if CHITRA_API_KEY else '🔓 local mode (no auth)'}")
    print("=" * 60)

    if not OPENAI_API_KEY:
        print("\n⚠️  WARNING: OPENAI_API_KEY is not set.")
        print("   Image generation will fail until you set it.\n")

    # debug=False in all cases to avoid exposing debugger
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
