# Databricks notebook source
# DBTITLE 1,P2 (COCO) — Soup model + PARALLELIZED image writes, on large COCO images
# MAGIC %md
# MAGIC P2 pattern (one soup model, one GPU forward pass, thread-pool parallel annotate+write)
# MAGIC run on the first 3,925 **COCO val2017** images (~159 KB avg, ~20x Imagenette) to match
# MAGIC P5's dataset. Isolates whether parallel writes hide the large-image FUSE write cost that
# MAGIC dominated P5 (serial write = 508s / 72% of wall). Serverless GPU. Labels not meaningful.

# COMMAND ----------

# DBTITLE 1,Config + flat COCO dataset
import os, time, json, glob
import torch
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import mlflow
from mlflow.tracking import MlflowClient

CATALOG = "serverless_stable_r4umw1_catalog"
SCHEMA = "cv"
BACKEND = "p2_coco_parallel_io"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.vit_imagenette_soup"
N_IMAGES = 3925                       # match P5
WRITE_WORKERS = int(os.environ.get("WRITE_WORKERS", "32"))

COCO_VAL = f"/Volumes/{CATALOG}/{SCHEMA}/coco2017/val2017"
BASE_MODEL_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/vit_models/vit_imagenette/base_model"
OUTPUT_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/inference_out_p2_coco"
RESULTS_DIR = f"{OUTPUT_DIR}/results"

IMAGENETTE_LABELS = {
    "n01440764": "tench", "n02102040": "English springer", "n02979186": "cassette player",
    "n03000684": "chain saw", "n03028079": "church", "n03394916": "French horn",
    "n03417042": "garbage truck", "n03425413": "gas pump", "n03445777": "golf ball",
    "n03888257": "parachute",
}
CLASS_ORDER = ["n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
               "n03394916", "n03417042", "n03425413", "n03445777", "n03888257"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | Backend: {BACKEND} | write_workers={WRITE_WORKERS}")
os.makedirs(RESULTS_DIR, exist_ok=True)
SCRIPT_START = time.time()

_t = time.time()
paths = sorted(glob.glob(f"{COCO_VAL}/*.jpg"))[:N_IMAGES]
in_bytes = sum(os.path.getsize(p) for p in paths)
processor = AutoImageProcessor.from_pretrained(BASE_MODEL_DIR)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

class FlatImageDataset(Dataset):
    def __init__(self, paths, tf): self.paths, self.tf = paths, tf
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), i

ds = FlatImageDataset(paths, val_transform)
loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=6,
                    pin_memory=True, persistent_workers=True)
dataset_prep_time = time.time() - _t
n_img = len(paths)
print(f"COCO images: {n_img} | avg input {in_bytes/n_img/1024:.0f} KB | dataset_prep={dataset_prep_time:.2f}s")

# COMMAND ----------

# DBTITLE 1,Load soup model + ONE forward pass
mlflow.set_registry_uri("databricks-uc")
_c = MlflowClient(registry_uri="databricks-uc")
_ver = max(int(v.version) for v in _c.search_model_versions(f"name = '{MODEL_NAME}'"))
_t = time.time()
components = mlflow.transformers.load_model(f"models:/{MODEL_NAME}/{_ver}", return_type="components")
model = components["model"].to(device).eval()
model_read_time = time.time() - _t

all_logits, all_idx = [], []
_t = time.time()
with torch.no_grad():
    for images, idx in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(pixel_values=images).logits
        all_logits.append(logits.float().cpu()); all_idx.append(idx)
inference_time = time.time() - _t
logits_t = torch.cat(all_logits); idx_t = torch.cat(all_idx)
preds = logits_t.argmax(dim=-1); conf = torch.softmax(logits_t, dim=-1).max(dim=-1).values
print(f"model_read={model_read_time:.2f}s inference={inference_time:.2f}s (labels not meaningful for COCO)")

# COMMAND ----------

# DBTITLE 1,Annotate + write — PARALLEL thread pool
print(f"Annotating COCO (parallel, {WRITE_WORKERS} workers)...")
output_subdir = f"{OUTPUT_DIR}/annotated_val"
os.makedirs(output_subdir, exist_ok=True)
order = idx_t.tolist(); pl = preds.tolist(); cl = conf.tolist()
jobs = [(order[k], pl[k], cl[k]) for k in range(n_img)]

def annotate_one(job):
    i, pred_idx, cf = job
    img = Image.open(paths[i]).convert("RGB")
    draw = ImageDraw.Draw(img); class_id = CLASS_ORDER[pred_idx]
    text = f"{IMAGENETTE_LABELS[class_id]} ({cf:.1%})"
    bbox = draw.textbbox((0, 0), text); tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    draw.rectangle([(2, 2), (tw+10, th+10)], fill="black"); draw.text((6, 4), text, fill="white")
    out = f"{output_subdir}/{i:06d}.jpg"
    img.save(out, "JPEG", quality=90)
    return os.path.getsize(out)

_t_ann = time.time()
with ThreadPoolExecutor(max_workers=WRITE_WORKERS) as ex:
    out_sizes = list(ex.map(annotate_one, jobs))
annotate_write_time = time.time() - _t_ann
out_bytes = sum(out_sizes)
print(f"Annotated {n_img} in {annotate_write_time:.1f}s ({1000*annotate_write_time/n_img:.1f} ms/img eff, "
      f"{WRITE_WORKERS}w) | avg out {out_bytes/n_img/1024:.0f} KB")

# COMMAND ----------

# DBTITLE 1,Results JSON + MLflow telemetry
now = datetime.now(timezone.utc).isoformat()
run_id = os.environ.get("DATABRICKS_RUN_ID", "job")
total_wall = time.time() - SCRIPT_START
summary = {
    "backend": BACKEND, "checkpoint": "phase_summary", "epoch": -1,
    "dataset": "coco2017_val", "write_workers": WRITE_WORKERS,
    "avg_input_kb": round(in_bytes/n_img/1024, 1), "avg_output_kb": round(out_bytes/n_img/1024, 1),
    "dataset_prep_s": round(dataset_prep_time, 3),
    "model_read_total_s": round(model_read_time, 3),
    "inference_total_s": round(inference_time, 3),
    "annotate_write_s": round(annotate_write_time, 3),
    "annotate_per_img_ms": round(1000*annotate_write_time/max(n_img, 1), 3),
    "total_wall_s": round(total_wall, 3), "num_images": n_img, "num_checkpoints": 10,
    "run_timestamp": now, "job_run_id": run_id,
}
results_file = f"{RESULTS_DIR}/results_{BACKEND}_{now[:19].replace(':', '-')}.json"
with open(results_file, "w") as f:
    json.dump([summary], f, indent=2)
try:
    with mlflow.start_run(run_name=f"timing_{BACKEND}"):
        mlflow.log_params({"backend": BACKEND, "dataset": "coco2017_val",
                           "write_workers": WRITE_WORKERS, "num_images": n_img})
        mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float)) and k not in ("epoch", "write_workers")})
    print("[telemetry] logged to MLflow")
except Exception as e:
    print("[telemetry] skipped:", e)
print(f"\nP2 COCO phases: prep={dataset_prep_time:.1f}s read={model_read_time:.1f}s infer={inference_time:.1f}s "
      f"annotate={annotate_write_time:.1f}s ({WRITE_WORKERS}w) wall={total_wall:.1f}s")
print("Done.")
