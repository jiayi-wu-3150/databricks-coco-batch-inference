#!/usr/bin/env python3
"""P7 — Ray Data batch inference on Serverless GPU (via AI Runtime CLI `air`).

Adapts the AIR Ray batch-inference template (Ray Data map_batches) to our ViT soup
model. Ray Data streams blocks through a GPU actor, overlapping image read/decode with
GPU compute (unlike the plain DataLoader pipeline in P1). One 1xA10 node, one GPU actor.
Tracks phase timing to compare with P1-P5.
"""
import os, time, json, glob
import numpy as np
import torch
from datetime import datetime, timezone
from PIL import Image, ImageDraw
from transformers import ViTForImageClassification, AutoImageProcessor
import ray

CATALOG = "serverless_stable_r4umw1_catalog"
SCHEMA = "cv"
BACKEND = "p7_ray_batch_inference"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.vit_imagenette_soup"
DATA_VAL = f"/Volumes/{CATALOG}/{SCHEMA}/vit_models/vit_imagenette/data/val"
BASE_MODEL_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/vit_models/vit_imagenette/base_model"
OUTPUT_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/inference_out_p7"
ANN_DIR = f"{OUTPUT_DIR}/annotated_val"
RESULTS_DIR = f"{OUTPUT_DIR}/results"
MODEL_LOCAL = f"{OUTPUT_DIR}/_soup_model"

CLASS_ORDER = ["n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
               "n03394916", "n03417042", "n03425413", "n03445777", "n03888257"]
IMAGENETTE_LABELS = {
    "n01440764": "tench", "n02102040": "English springer", "n02979186": "cassette player",
    "n03000684": "chain saw", "n03028079": "church", "n03394916": "French horn",
    "n03417042": "garbage truck", "n03425413": "gas pump", "n03445777": "golf ball",
    "n03888257": "parachute",
}

os.makedirs(ANN_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)
SCRIPT_START = time.time()

# ── Materialize soup model to a volume on the head (actors load via from_pretrained) ──
_t = time.time()
import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_registry_uri("databricks-uc")
VER = max(int(v.version) for v in MlflowClient(registry_uri="databricks-uc").search_model_versions(f"name = '{MODEL_NAME}'"))
comp = mlflow.transformers.load_model(f"models:/{MODEL_NAME}/{VER}", return_type="components")
comp["model"].save_pretrained(MODEL_LOCAL)
AutoImageProcessor.from_pretrained(BASE_MODEL_DIR).save_pretrained(MODEL_LOCAL)
model_read_time = time.time() - _t
print(f"materialized soup model in {model_read_time:.1f}s")

# ── Ray Data pipeline ──
ray.init(ignore_reinit_error=True)
paths = sorted(glob.glob(f"{DATA_VAL}/*/*"))
n_img = len(paths)
ds = ray.data.from_items([{"path": p} for p in paths])

class Classifier:
    def __init__(self):
        import torchvision.transforms as T
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViTForImageClassification.from_pretrained(MODEL_LOCAL).to(self.device).eval()
        proc = AutoImageProcessor.from_pretrained(MODEL_LOCAL)
        self.tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                             T.Normalize(mean=proc.image_mean, std=proc.image_std)])

    def __call__(self, batch):
        paths_b = list(batch["path"])
        imgs = [Image.open(p).convert("RGB") for p in paths_b]
        x = torch.stack([self.tf(im) for im in imgs]).to(self.device)
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                enabled=(self.device.type == "cuda")):
                logits = self.model(pixel_values=x).logits.float()
        preds = logits.argmax(-1).tolist()
        confs = torch.softmax(logits, -1).max(-1).values.tolist()
        op, oc, of = [], [], []
        for p, im, pr, cf in zip(paths_b, imgs, preds, confs):
            cid = CLASS_ORDER[pr]; text = f"{IMAGENETTE_LABELS[cid]} ({cf:.1%})"
            d = ImageDraw.Draw(im); bb = d.textbbox((0, 0), text); tw, th = bb[2]-bb[0], bb[3]-bb[1]
            d.rectangle([(2, 2), (tw+10, th+10)], fill="black"); d.text((6, 4), text, fill="white")
            im.save(f"{ANN_DIR}/{os.path.basename(p)}", "JPEG", quality=90)
            op.append(p); oc.append(cid); of.append(float(cf))
        return {"path": op, "pred_class": oc, "confidence": of}

_t = time.time()
res = ds.map_batches(Classifier, batch_size=64, concurrency=1, num_gpus=1)
res.write_parquet(f"{OUTPUT_DIR}/predictions")
inference_time = time.time() - _t

total_wall = time.time() - SCRIPT_START
now = datetime.now(timezone.utc).isoformat()
summary = {
    "backend": BACKEND, "checkpoint": "phase_summary", "epoch": -1,
    "compute": "serverless_gpu_ray_1xA10",
    "model_read_total_s": round(model_read_time, 3),
    "inference_total_s": round(inference_time, 3),
    "total_wall_s": round(total_wall, 3),
    "num_images": n_img, "num_checkpoints": 10,
    "images_per_sec": round(n_img / max(inference_time, 1e-9), 2),
    "run_timestamp": now, "job_run_id": os.environ.get("DATABRICKS_RUN_ID", "air"),
}
with open(f"{RESULTS_DIR}/results_{BACKEND}_{now[:19].replace(':', '-')}.json", "w") as f:
    json.dump([summary], f, indent=2)
print(f"P7 Ray: materialize={model_read_time:.1f}s map_batches(read+infer+write)={inference_time:.1f}s "
      f"wall={total_wall:.1f}s ({n_img} imgs, {summary['images_per_sec']} img/s)")
print("Done.")
