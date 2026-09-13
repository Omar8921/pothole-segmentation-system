import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field


# =============================================================================
# Configuration
# =============================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NEEDS_FIX_FILENAME = "images_need_fix.json"
IMAGES_TO_DELETE_FILENAME = "images_to_delete.json"
REVIEW_LOG_FILENAME = "review_log.csv"

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "workspace_config.json"


# =============================================================================
# Schemas
# =============================================================================

class WorkspaceUpdate(BaseModel):
    """Paths used by the local review tool."""

    images_dir: str | None = None
    labels_dir: str | None = None
    output_dir: str | None = None


class WorkspaceState(BaseModel):
    """Current workspace folder paths."""

    images_dir: str | None = None
    labels_dir: str | None = None
    output_dir: str | None = None


class SampleInfo(BaseModel):
    """Image sample shown in the review UI."""

    filename: str
    stem: str
    label_filename: str
    has_label: bool


class Point(BaseModel):
    """Normalized polygon point in YOLO segmentation format."""

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class PolygonLabel(BaseModel):
    """One segmentation polygon object."""

    class_id: int = Field(ge=0)
    points: list[Point]


class LabelPayload(BaseModel):
    """Full label file payload used only for viewing overlays."""

    polygons: list[PolygonLabel]


class ReviewUpdate(BaseModel):
    """Request body for marking one image into a review bucket or unmarking it."""

    filename: str
    reason: str | None = None


class ReviewState(BaseModel):
    """Current lists of image files marked for fixing or deletion."""

    images_need_fix: list[str]
    labels_need_fix: list[str]
    images_to_delete: list[str]
    labels_to_delete: list[str]
    need_fix_count: int
    delete_count: int


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    """Support both Pydantic v1 and v2 when saving models to JSON."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


# =============================================================================
# Workspace service
# =============================================================================

def get_workspace() -> WorkspaceState:
    if not CONFIG_PATH.exists():
        return WorkspaceState()

    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return WorkspaceState(**data)


def update_workspace(update: WorkspaceUpdate) -> WorkspaceState:
    current = model_to_dict(get_workspace())
    if hasattr(update, "model_dump"):
        incoming = update.model_dump(exclude_none=True)
    else:
        incoming = update.dict(exclude_none=True)

    current.update(incoming)
    workspace = WorkspaceState(**current)
    CONFIG_PATH.write_text(json.dumps(model_to_dict(workspace), indent=2), encoding="utf-8")
    return workspace


def require_existing_dir(path_value: str | None, name: str) -> Path:
    if not path_value:
        raise ValueError(f"{name} is not selected")

    path = Path(path_value).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise ValueError(f"{name} does not exist or is not a directory: {path}")

    return path


def list_samples() -> list[SampleInfo]:
    workspace = get_workspace()
    images_dir = require_existing_dir(workspace.images_dir, "Images folder")
    labels_dir = require_existing_dir(workspace.labels_dir, "Labels folder")

    samples: list[SampleInfo] = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_filename = f"{image_path.stem}.txt"
        label_path = labels_dir / label_filename
        samples.append(
            SampleInfo(
                filename=image_path.name,
                stem=image_path.stem,
                label_filename=label_filename,
                has_label=label_path.exists(),
            )
        )

    return samples


def get_image_path(filename: str) -> Path:
    workspace = get_workspace()
    images_dir = require_existing_dir(workspace.images_dir, "Images folder")
    safe_filename = Path(filename).name
    image_path = (images_dir / safe_filename).resolve()

    if images_dir not in image_path.parents and image_path != images_dir:
        raise ValueError("Invalid image path")

    if not image_path.exists() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise FileNotFoundError(f"Image not found: {safe_filename}")

    return image_path


def get_label_path_for_image(filename: str) -> Path:
    workspace = get_workspace()
    labels_dir = require_existing_dir(workspace.labels_dir, "Labels folder")
    safe_stem = Path(filename).stem
    return labels_dir / f"{safe_stem}.txt"


def get_output_dir() -> Path:
    workspace = get_workspace()
    return require_existing_dir(workspace.output_dir, "Output folder")


# =============================================================================
# Polygon label service
# =============================================================================

def read_yolo_segmentation_label(label_path: Path) -> LabelPayload:
    """
    Read a YOLO segmentation label file.

    Expected line format:
        class_id x1 y1 x2 y2 x3 y3 ...

    Coordinates are normalized in the range [0, 1].
    """
    if not label_path.exists():
        return LabelPayload(polygons=[])

    polygons: list[PolygonLabel] = []

    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        values = line.split()
        if len(values) < 7:
            raise ValueError(f"Invalid label line {line_number}: expected class id plus at least 3 points")

        class_id = int(values[0])
        coords = [float(value) for value in values[1:]]

        if len(coords) % 2 != 0:
            raise ValueError(f"Invalid label line {line_number}: polygon coordinate count must be even")

        points: list[Point] = []
        for index in range(0, len(coords), 2):
            x = min(max(coords[index], 0.0), 1.0)
            y = min(max(coords[index + 1], 0.0), 1.0)
            points.append(Point(x=x, y=y))

        if len(points) < 3:
            raise ValueError(f"Invalid label line {line_number}: polygon must contain at least 3 points")

        polygons.append(PolygonLabel(class_id=class_id, points=points))

    return LabelPayload(polygons=polygons)


# =============================================================================
# Review service
# =============================================================================

def get_images_need_fix_path() -> Path:
    return get_output_dir() / NEEDS_FIX_FILENAME


def get_images_to_delete_path() -> Path:
    return get_output_dir() / IMAGES_TO_DELETE_FILENAME


def get_review_log_path() -> Path:
    return get_output_dir() / REVIEW_LOG_FILENAME


def safe_image_filename(filename: str) -> str:
    safe_name = Path(filename).name.strip()
    if not safe_name:
        raise ValueError("Filename cannot be empty")
    return safe_name


def label_filename_for_image(filename: str) -> str:
    return f"{Path(filename).stem}.txt"


def _read_filename_list(path: Path, preferred_key: str) -> list[str]:
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        values = data
    else:
        values = data.get(preferred_key, [])

    return sorted({safe_image_filename(value) for value in values})


def _write_filename_payload(path: Path, images: list[str], image_key: str, label_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique_images = sorted({safe_image_filename(filename) for filename in images})
    payload = {
        "count": len(unique_images),
        image_key: unique_images,
        label_key: [label_filename_for_image(filename) for filename in unique_images],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_state(images_need_fix: list[str], images_to_delete: list[str]) -> ReviewState:
    fix_images = sorted(set(images_need_fix))
    delete_images = sorted(set(images_to_delete))

    return ReviewState(
        images_need_fix=fix_images,
        labels_need_fix=[label_filename_for_image(filename) for filename in fix_images],
        images_to_delete=delete_images,
        labels_to_delete=[label_filename_for_image(filename) for filename in delete_images],
        need_fix_count=len(fix_images),
        delete_count=len(delete_images),
    )


def read_review_state() -> ReviewState:
    images_need_fix = _read_filename_list(get_images_need_fix_path(), "images_need_fix")
    images_to_delete = _read_filename_list(get_images_to_delete_path(), "images_to_delete")
    return build_state(images_need_fix, images_to_delete)


def write_review_state(state: ReviewState) -> None:
    _write_filename_payload(
        get_images_need_fix_path(),
        state.images_need_fix,
        image_key="images_need_fix",
        label_key="labels_need_fix",
    )
    _write_filename_payload(
        get_images_to_delete_path(),
        state.images_to_delete,
        image_key="images_to_delete",
        label_key="labels_to_delete",
    )


def append_review_log(filename: str, action: str, reason: str | None = None) -> None:
    log_path = get_review_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()

    with log_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["timestamp_utc", "image_filename", "label_filename", "action", "reason"],
        )
        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "image_filename": filename,
                "label_filename": label_filename_for_image(filename),
                "action": action,
                "reason": reason or "",
            }
        )


def mark_image_need_fix(filename: str, reason: str | None = None) -> ReviewState:
    safe_name = safe_image_filename(filename)
    state = read_review_state()

    fix_images = set(state.images_need_fix)
    delete_images = set(state.images_to_delete)

    fix_images.add(safe_name)
    delete_images.discard(safe_name)

    updated_state = build_state(list(fix_images), list(delete_images))
    write_review_state(updated_state)
    append_review_log(safe_name, "mark_need_fix", reason)
    return updated_state


def mark_image_for_deletion(filename: str, reason: str | None = None) -> ReviewState:
    safe_name = safe_image_filename(filename)
    state = read_review_state()

    fix_images = set(state.images_need_fix)
    delete_images = set(state.images_to_delete)

    delete_images.add(safe_name)
    fix_images.discard(safe_name)

    updated_state = build_state(list(fix_images), list(delete_images))
    write_review_state(updated_state)
    append_review_log(safe_name, "mark_delete", reason)
    return updated_state


def unmark_image(filename: str, reason: str | None = None) -> ReviewState:
    safe_name = safe_image_filename(filename)
    state = read_review_state()

    fix_images = set(state.images_need_fix)
    delete_images = set(state.images_to_delete)

    fix_images.discard(safe_name)
    delete_images.discard(safe_name)

    updated_state = build_state(list(fix_images), list(delete_images))
    write_review_state(updated_state)
    append_review_log(safe_name, "mark_keep", reason)
    return updated_state


# =============================================================================
# Frontend assets embedded in Python
# =============================================================================

INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Pothole Review Tool</title>
    <link rel="stylesheet" href="/styles.css" />
  </head>
  <body>
    <header>
      <h1>Pothole Review Tool</h1>
      <p>Review image-mask overlays and mark each sample as need fix, need deletion, or keep.</p>
    </header>

    <main>
      <aside>
        <section class="panel">
          <h2>Workspace</h2>
          <button id="selectImagesBtn">Select images folder</button>
          <button id="selectLabelsBtn">Select labels folder</button>
          <button id="selectOutputBtn">Select output folder</button>
          <button id="loadSamplesBtn" class="primary">Load samples</button>
          <div id="workspaceInfo" class="small"></div>
        </section>

        <section class="panel">
          <h2>Samples</h2>
          <input id="searchInput" type="search" placeholder="Search filename..." />
          <div id="reviewSummary" class="small"></div>
          <div id="sampleList"></div>
        </section>
      </aside>

      <section class="viewer">
        <div class="toolbar">
          <button id="prevBtn">Previous (A)</button>
          <button id="nextBtn">Next (D)</button>
          <button id="markFixBtn" class="warning">Need fix (F)</button>
          <button id="markDeleteBtn" class="danger">Need deletion (X)</button>
          <button id="markKeepBtn">Keep / unmark (G)</button>
        </div>

        <div class="overlay-controls">
          <label for="opacitySlider">Overlay opacity</label>
          <input id="opacitySlider" type="range" min="0" max="1" step="0.05" value="0.35" />
        </div>

        <div id="status"></div>
        <div id="canvasContainer">
          <canvas id="overlayCanvas"></canvas>
        </div>
      </section>
    </main>

    <script src="/app.js"></script>
  </body>
</html>
"""

STYLES_CSS = """* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: Arial, sans-serif;
  background: #f5f5f5;
  color: #1f2937;
}

header {
  padding: 16px 24px;
  background: #111827;
  color: white;
}

header h1 {
  margin: 0 0 4px;
  font-size: 22px;
}

header p {
  margin: 0;
  color: #d1d5db;
}

main {
  display: grid;
  grid-template-columns: 320px 1fr;
  gap: 16px;
  padding: 16px;
}

aside,
.viewer {
  min-height: calc(100vh - 110px);
}

.panel,
.viewer {
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 14px;
}

.panel {
  margin-bottom: 16px;
}

.panel h2 {
  margin: 0 0 10px;
  font-size: 16px;
}

button {
  display: inline-block;
  border: 1px solid #d1d5db;
  background: white;
  color: #111827;
  padding: 8px 10px;
  border-radius: 8px;
  cursor: pointer;
  margin: 4px 4px 4px 0;
}

button:hover {
  background: #f3f4f6;
}

button.primary {
  background: #111827;
  color: white;
  border-color: #111827;
}

button.primary:hover {
  background: #374151;
}

button.warning {
  background: #9a3412;
  color: white;
  border-color: #9a3412;
}

button.warning:hover {
  background: #c2410c;
}

button.danger {
  background: #991b1b;
  color: white;
  border-color: #991b1b;
}

button.danger:hover {
  background: #7f1d1d;
}

input[type="search"] {
  width: 100%;
  padding: 8px;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  margin-bottom: 10px;
}

input[type="range"] {
  width: 220px;
}

#sampleList {
  max-height: 520px;
  overflow-y: auto;
}

.sample-item {
  padding: 8px;
  border-radius: 8px;
  cursor: pointer;
  border: 1px solid transparent;
  font-size: 13px;
}

.sample-item:hover {
  background: #f3f4f6;
}

.sample-item.active {
  background: #e5e7eb;
  border-color: #9ca3af;
}

.sample-item.missing-label {
  color: #b91c1c;
}

.sample-item.marked-fix {
  background: #ffedd5;
  border-color: #fdba74;
}

.sample-item.marked-fix.active {
  background: #fed7aa;
  border-color: #f97316;
}

.sample-item.marked-delete {
  background: #fee2e2;
  border-color: #fecaca;
}

.sample-item.marked-delete.active {
  background: #fecaca;
  border-color: #ef4444;
}

.toolbar {
  margin-bottom: 10px;
}

.overlay-controls {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
  font-size: 14px;
}

#status {
  min-height: 24px;
  margin-bottom: 10px;
  font-size: 14px;
  color: #374151;
}

#canvasContainer {
  display: inline-block;
  max-width: 100%;
  border: 1px solid #d1d5db;
  background: #111827;
  overflow: auto;
}

#overlayCanvas {
  display: block;
}

.small {
  font-size: 12px;
  color: #4b5563;
  word-break: break-word;
  margin-top: 8px;
  margin-bottom: 8px;
  line-height: 1.4;
}
"""

APP_JS = """const API_BASE = "";

let samples = [];
let filteredSamples = [];
let currentIndex = -1;
let currentImage = null;
let currentPolygons = [];
let imagesNeedFix = new Set();
let imagesToDelete = new Set();
let displayWidth = 0;
let displayHeight = 0;

const workspaceInfo = document.getElementById("workspaceInfo");
const sampleList = document.getElementById("sampleList");
const reviewSummary = document.getElementById("reviewSummary");
const statusBox = document.getElementById("status");
const searchInput = document.getElementById("searchInput");
const opacitySlider = document.getElementById("opacitySlider");
const canvas = document.getElementById("overlayCanvas");
const context = canvas.getContext("2d");

function setStatus(message) {
  statusBox.textContent = message;
}

async function apiFetch(url, options = {}) {
  const response = await fetch(`${API_BASE}${url}`, options);
  if (!response.ok) {
    const errorPayload = await response.json().catch(() => ({}));
    throw new Error(errorPayload.detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

async function refreshWorkspace() {
  const workspace = await apiFetch("/api/workspace");
  workspaceInfo.innerHTML = `
    <strong>Images:</strong> ${workspace.images_dir || "not selected"}<br>
    <strong>Labels:</strong> ${workspace.labels_dir || "not selected"}<br>
    <strong>Output:</strong> ${workspace.output_dir || "not selected"}<br>
    <span>Output files: <code>images_need_fix.json</code>, <code>images_to_delete.json</code>, and <code>review_log.csv</code></span>
  `;
}

async function refreshReviewState() {
  const review = await apiFetch("/api/review");
  imagesNeedFix = new Set(review.images_need_fix);
  imagesToDelete = new Set(review.images_to_delete);
  reviewSummary.textContent = `${review.need_fix_count} image(s) need fix | ${review.delete_count} image(s) need deletion.`;
}

async function selectFolder(kind) {
  setStatus(`Opening ${kind} folder picker...`);
  await apiFetch(`/api/workspace/select-folder?kind=${kind}`, { method: "POST" });
  await refreshWorkspace();
  setStatus(`${kind} folder selected.`);
}

async function loadSamples() {
  await refreshReviewState();

  const payload = await apiFetch("/api/workspace/samples");
  samples = payload.samples;
  filteredSamples = samples;
  currentIndex = samples.length > 0 ? 0 : -1;
  renderSampleList();

  if (currentIndex >= 0) {
    await loadCurrentSample();
  } else {
    setStatus("No images found in the selected images folder.");
  }
}

function sampleReviewStatus(sample) {
  if (imagesToDelete.has(sample.filename)) return "NEED DELETION";
  if (imagesNeedFix.has(sample.filename)) return "NEED FIX";
  return "not marked";
}

function renderSampleList() {
  sampleList.innerHTML = "";

  filteredSamples.forEach((sample) => {
    const originalIndex = samples.findIndex((item) => item.filename === sample.filename);
    const item = document.createElement("div");
    item.className = "sample-item";

    if (originalIndex === currentIndex) item.classList.add("active");
    if (!sample.has_label) item.classList.add("missing-label");
    if (imagesNeedFix.has(sample.filename)) item.classList.add("marked-fix");
    if (imagesToDelete.has(sample.filename)) item.classList.add("marked-delete");

    const statusParts = [];
    if (!sample.has_label) statusParts.push("missing label");
    if (imagesNeedFix.has(sample.filename)) statusParts.push("NEED FIX");
    if (imagesToDelete.has(sample.filename)) statusParts.push("NEED DELETION");

    item.textContent = statusParts.length > 0 ? `${sample.filename}  (${statusParts.join(", ")})` : sample.filename;
    item.onclick = async () => {
      currentIndex = originalIndex;
      await loadCurrentSample();
      renderSampleList();
    };

    sampleList.appendChild(item);
  });
}

async function loadCurrentSample() {
  if (currentIndex < 0 || currentIndex >= samples.length) return;

  const sample = samples[currentIndex];
  setStatus(`Loading ${sample.filename}...`);

  const image = new Image();
  image.onload = async () => {
    currentImage = image;

    const maxWidth = Math.max(600, window.innerWidth - 390);
    const maxHeight = Math.max(400, window.innerHeight - 230);
    const scale = Math.min(maxWidth / image.naturalWidth, maxHeight / image.naturalHeight, 1);

    displayWidth = Math.round(image.naturalWidth * scale);
    displayHeight = Math.round(image.naturalHeight * scale);

    canvas.width = displayWidth;
    canvas.height = displayHeight;

    const labelPayload = await apiFetch(`/api/labels/${encodeURIComponent(sample.filename)}`);
    currentPolygons = labelPayload.polygons;

    drawOverlay();
    updateCurrentStatus();
    renderSampleList();
  };

  image.onerror = () => setStatus(`Failed to load image: ${sample.filename}`);
  image.src = `/api/images/${encodeURIComponent(sample.filename)}?t=${Date.now()}`;
}

function updateCurrentStatus() {
  if (currentIndex < 0 || currentIndex >= samples.length) return;

  const sample = samples[currentIndex];
  const reviewStatus = sampleReviewStatus(sample);
  setStatus(`${sample.filename} | ${currentPolygons.length} polygon(s) | ${reviewStatus} | ${currentIndex + 1}/${samples.length}`);
}

function drawOverlay() {
  if (!currentImage) return;

  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(currentImage, 0, 0, displayWidth, displayHeight);

  const opacity = Number(opacitySlider.value);

  currentPolygons.forEach((polygon) => {
    if (!polygon.points || polygon.points.length < 3) return;

    context.beginPath();
    polygon.points.forEach((point, index) => {
      const x = point.x * displayWidth;
      const y = point.y * displayHeight;
      if (index === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    });
    context.closePath();

    context.fillStyle = `rgba(0, 255, 102, ${opacity})`;
    context.strokeStyle = "rgb(0, 255, 102)";
    context.lineWidth = 2;
    context.fill();
    context.stroke();
  });
}

async function markCurrentSampleNeedFix() {
  if (currentIndex < 0 || currentIndex >= samples.length) return;

  const sample = samples[currentIndex];
  const payload = { filename: sample.filename };

  await apiFetch("/api/review/fix", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  await refreshReviewState();
  renderSampleList();
  updateCurrentStatus();
}

async function markCurrentSampleForDeletion() {
  if (currentIndex < 0 || currentIndex >= samples.length) return;

  const sample = samples[currentIndex];
  const payload = { filename: sample.filename };

  await apiFetch("/api/review/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  await refreshReviewState();
  renderSampleList();
  updateCurrentStatus();
}

async function unmarkCurrentSample() {
  if (currentIndex < 0 || currentIndex >= samples.length) return;

  const sample = samples[currentIndex];
  const payload = { filename: sample.filename };

  await apiFetch("/api/review/keep", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  await refreshReviewState();
  renderSampleList();
  updateCurrentStatus();
}

async function previousSample() {
  if (samples.length === 0) return;
  currentIndex = Math.max(0, currentIndex - 1);
  await loadCurrentSample();
}

async function nextSample() {
  if (samples.length === 0) return;
  currentIndex = Math.min(samples.length - 1, currentIndex + 1);
  await loadCurrentSample();
}

searchInput.addEventListener("input", () => {
  const query = searchInput.value.toLowerCase().trim();
  filteredSamples = samples.filter((sample) => sample.filename.toLowerCase().includes(query));
  renderSampleList();
});

opacitySlider.addEventListener("input", drawOverlay);

document.getElementById("selectImagesBtn").onclick = () => selectFolder("images").catch((error) => setStatus(error.message));
document.getElementById("selectLabelsBtn").onclick = () => selectFolder("labels").catch((error) => setStatus(error.message));
document.getElementById("selectOutputBtn").onclick = () => selectFolder("output").catch((error) => setStatus(error.message));
document.getElementById("loadSamplesBtn").onclick = () => loadSamples().catch((error) => setStatus(error.message));
document.getElementById("prevBtn").onclick = () => previousSample().catch((error) => setStatus(error.message));
document.getElementById("nextBtn").onclick = () => nextSample().catch((error) => setStatus(error.message));
document.getElementById("markFixBtn").onclick = () => markCurrentSampleNeedFix().catch((error) => setStatus(error.message));
document.getElementById("markDeleteBtn").onclick = () => markCurrentSampleForDeletion().catch((error) => setStatus(error.message));
document.getElementById("markKeepBtn").onclick = () => unmarkCurrentSample().catch((error) => setStatus(error.message));

document.addEventListener("keydown", async (event) => {
  if (event.target.tagName === "INPUT") return;

  if (event.key.toLowerCase() === "a") await previousSample();
  if (event.key.toLowerCase() === "d") await nextSample();
  if (event.key.toLowerCase() === "f") await markCurrentSampleNeedFix();
  if (event.key.toLowerCase() === "g") await unmarkCurrentSample();
  if (event.key.toLowerCase() === "x") await markCurrentSampleForDeletion();
});

refreshWorkspace().catch((error) => setStatus(error.message));
refreshReviewState().catch(() => {});
"""


# =============================================================================
# Routes
# =============================================================================

frontend_router = APIRouter(tags=["frontend"])
workspace_router = APIRouter(prefix="/api/workspace", tags=["workspace"])
images_router = APIRouter(prefix="/api/images", tags=["images"])
labels_router = APIRouter(prefix="/api/labels", tags=["labels"])
review_router = APIRouter(prefix="/api/review", tags=["review"])


@frontend_router.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@frontend_router.get("/styles.css")
def styles() -> Response:
    return Response(STYLES_CSS, media_type="text/css")


@frontend_router.get("/app.js")
def javascript() -> Response:
    return Response(APP_JS, media_type="application/javascript")


@images_router.get("/{filename}")
def read_image(filename: str):
    try:
        image_path = get_image_path(filename)
        return FileResponse(image_path)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@labels_router.get("/{filename}", response_model=LabelPayload)
def read_label(filename: str) -> LabelPayload:
    try:
        label_path = get_label_path_for_image(filename)
        return read_yolo_segmentation_label(label_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@review_router.get("", response_model=ReviewState)
def get_review() -> ReviewState:
    try:
        return read_review_state()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@review_router.post("/fix", response_model=ReviewState)
def mark_fix(payload: ReviewUpdate) -> ReviewState:
    try:
        return mark_image_need_fix(payload.filename, payload.reason)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@review_router.post("/delete", response_model=ReviewState)
def mark_delete(payload: ReviewUpdate) -> ReviewState:
    try:
        return mark_image_for_deletion(payload.filename, payload.reason)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@review_router.post("/keep", response_model=ReviewState)
def mark_keep(payload: ReviewUpdate) -> ReviewState:
    try:
        return unmark_image(payload.filename, payload.reason)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def open_folder_dialog(title: str) -> str:
    """
    Open a local OS folder picker.

    This works best when FastAPI is running on the same desktop OS as your browser.
    If it is running inside WSL without GUI support, use Windows Python instead.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected_path = filedialog.askdirectory(title=title)
        root.destroy()
    except Exception as exc:
        raise RuntimeError(f"Folder dialog failed: {exc}") from exc

    if not selected_path:
        raise RuntimeError("No folder selected")

    return selected_path


@workspace_router.get("", response_model=WorkspaceState)
def read_workspace() -> WorkspaceState:
    return get_workspace()


@workspace_router.post("", response_model=WorkspaceState)
def save_workspace(update: WorkspaceUpdate) -> WorkspaceState:
    return update_workspace(update)


@workspace_router.post("/select-folder", response_model=WorkspaceState)
def select_folder(kind: str = Query(pattern="^(images|labels|output)$")) -> WorkspaceState:
    titles = {
        "images": "Select images folder",
        "labels": "Select labels folder",
        "output": "Select output folder for deletion review files",
    }

    try:
        selected_path = open_folder_dialog(titles[kind])
        if kind == "images":
            return update_workspace(WorkspaceUpdate(images_dir=selected_path))
        if kind == "labels":
            return update_workspace(WorkspaceUpdate(labels_dir=selected_path))
        return update_workspace(WorkspaceUpdate(output_dir=selected_path))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@workspace_router.get("/samples")
def read_samples() -> dict[str, list[SampleInfo]]:
    try:
        return {"samples": list_samples()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# =============================================================================
# App factory and CLI entry point
# =============================================================================

def create_app() -> FastAPI:
    fastapi_app = FastAPI(title="Pothole Review Tool")

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    fastapi_app.include_router(workspace_router)
    fastapi_app.include_router(images_router)
    fastapi_app.include_router(labels_router)
    fastapi_app.include_router(review_router)
    fastapi_app.include_router(frontend_router)
    return fastapi_app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the single-file Pothole Review Tool.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to.")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind the server to.")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload.")
    args = parser.parse_args()

    import uvicorn

    if args.reload:
        module_name = Path(__file__).resolve().stem
        uvicorn.run(
            f"{module_name}:app",
            host=args.host,
            port=args.port,
            reload=True,
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
