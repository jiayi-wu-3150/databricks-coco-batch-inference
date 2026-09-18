# Databricks notebook source
# DBTITLE 1,P7 v2 (COCO Job) — Ray Data staged batch inference + annotated writes
# MAGIC %md
# MAGIC Corrected P7 on **COCO** as a Databricks Job (serverless GPU, `databricks_ai_v5`),
# MAGIC following the Databricks industry-solutions Ray reference, with THREE Ray Data stages:
# MAGIC 1. **CPU** decode + preprocess (`.map`, autoscaled)
# MAGIC 2. **GPU** inference (`.map_batches`, `num_gpus=1`, model loaded once)
# MAGIC 3. **CPU** annotate + write one JPEG per image (`.map`, autoscaled)
# MAGIC
# MAGIC Writes stay OUT of the GPU actor (the v1 mistake), but P7 now produces annotated
# MAGIC images like P1–P6, so total time is comparable. 3,925 COCO val2017 images.

# COMMAND ----------

import os, time, json, glob
import numpy as np
import torch
from datetime import datetime, timezone
from PIL import Image, ImageDraw
from transformers import ViTForImageClassification, AutoImageProcessor
import ray

CATALOG = "serverless_stable_r4umw1_catalog"
SCHEMA = "cv"
BACKEND = "p7_ray_v2_coco"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.vit_imagenette_soup"
COCO_VAL = f"/Volumes/{CATALOG}/{SCHEMA}/coco2017/val2017"
N_IMAGES = 3925  # match P1-P4 COCO
BASE_MODEL_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/vit_models/vit_imagenette/base_model"
OUTPUT_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/inference_out_p7"
RESULTS_DIR = f"{OUTPUT_DIR}/results"
ANN_DIR = f"{OUTPUT_DIR}/annotated_val_v2_coco"
PRED_DIR = f"{OUTPUT_DIR}/predictions_v2_coco"
MODEL_LOCAL = f"{OUTPUT_DIR}/_soup_model_v2"
CPU_CONCURRENCY = (2, 16)
GPU_BATCH_SIZE = 64
CLASS_ORDER = ["n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
               "n03394916", "n03417042", "n03425413", "n03445777", "n03888257"]
IMAGENETTE_LABELS = {
    "n01440764": "tench", "n02102040": "English springer", "n02979186": "cassette player",
    "n03000684": "chain saw", "n03028079": "church", "n03394916": "French horn",
    "n03417042": "garbage truck", "n03425413": "gas pump", "n03445777": "golf ball",
    "n03888257": "parachute",
}
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(ANN_DIR, exist_ok=True)
SCRIPT_START = time.time()

# COMMAND ----------

# DBTITLE 1,Materialize soup model to a Volume once
_t = time.time()
import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_registry_uri("databricks-uc")
VER = max(int(v.version) for v in MlflowClient(registry_uri="databricks-uc").search_model_versions(f"name = '{MODEL_NAME}'"))
comp = mlflow.transformers.load_model(f"models:/{MODEL_NAME}/{VER}", return_type="components")
comp["model"].save_pretrained(MODEL_LOCAL)
_proc = AutoImageProcessor.from_pretrained(BASE_MODEL_DIR)
_proc.save_pretrained(MODEL_LOCAL)
MEAN = [float(x) for x in _proc.image_mean]
STD = [float(x) for x in _proc.image_std]
model_read_time = time.time() - _t
print(f"materialized soup model in {model_read_time:.1f}s")

# COMMAND ----------

# DBTITLE 1,Ray Data 3-stage pipeline: CPU preproc -> GPU infer -> CPU annotate+write
ray.init(ignore_reinit_error=True)
paths = sorted(glob.glob(f"{COCO_VAL}/*.jpg"))[:N_IMAGES]
n_img = len(paths)
ds = ray.data.from_items([{"path": p} for p in paths])
print(f"images={n_img} | CPU concurrency={CPU_CONCURRENCY} | GPU batch_size={GPU_BATCH_SIZE}")

class Preprocess:
    def __init__(self):
        import torchvision.transforms as T
        self.tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                             T.Normalize(mean=MEAN, std=STD)])
    def __call__(self, row):
        img = Image.open(row["path"]).convert("RGB")
        return {"path": row["path"], "pixel_values": self.tf(img).numpy()}

class GPUClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViTForImageClassification.from_pretrained(MODEL_LOCAL).to(self.device).eval()
    def __call__(self, batch):
        x = torch.from_numpy(np.asarray(batch["pixel_values"])).to(self.device)
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                enabled=(self.device.type == "cuda")):
                logits = self.model(pixel_values=x).logits.float()
        preds = logits.argmax(-1).cpu().numpy()
        confs = torch.softmax(logits, -1).max(-1).values.cpu().numpy()
        return {"path": np.asarray(batch["path"]),
                "pred_class": np.array([CLASS_ORDER[int(p)] for p in preds]),
                "confidence": confs.astype(np.float64)}

class AnnotateWrite:
    """CPU stage AFTER the GPU actor: re-open the image, draw label+score, write JPEG.
    Keeps per-image FUSE writes off the GPU actor (Ray autoscales these CPU tasks)."""
    def __call__(self, row):
        cid = str(row["pred_class"]); conf = float(row["confidence"])
        img = Image.open(row["path"]).convert("RGB")
        text = f"{IMAGENETTE_LABELS[cid]} ({conf:.1%})"
        d = ImageDraw.Draw(img); bb = d.textbbox((0, 0), text)
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
        d.rectangle([(2, 2), (tw+10, th+10)], fill="black"); d.text((6, 4), text, fill="white")
        outp = f"{ANN_DIR}/{os.path.basename(row['path'])}"
        img.save(outp, "JPEG", quality=90)
        return {"path": row["path"], "pred_class": cid, "confidence": conf, "output_path": outp}

_t = time.time()
out = (ds
       .map(Preprocess, concurrency=CPU_CONCURRENCY, num_cpus=1)
       .map_batches(GPUClassifier, concurrency=1, num_gpus=1, batch_size=GPU_BATCH_SIZE)
       .map(AnnotateWrite, concurrency=CPU_CONCURRENCY, num_cpus=1))
out.write_parquet(PRED_DIR)     # predictions incl. output_path; annotated JPEGs are in ANN_DIR
pipeline_time = time.time() - _t

# COMMAND ----------

# DBTITLE 1,Results JSON
total_wall = time.time() - SCRIPT_START
now = datetime.now(timezone.utc).isoformat()
summary = {
    "backend": BACKEND, "checkpoint": "phase_summary", "epoch": -1,
    "compute": "serverless_gpu_ray_1xA10_JOB", "dataset": "coco2017_val",
    "pipeline": "cpu_preproc -> gpu_infer -> cpu_annotate_write (3-stage, annotated images)",
    "cpu_concurrency": list(CPU_CONCURRENCY), "gpu_batch_size": GPU_BATCH_SIZE,
    "model_read_total_s": round(model_read_time, 3),
    "pipeline_total_s": round(pipeline_time, 3),
    "total_wall_s": round(total_wall, 3),
    "num_images": n_img, "num_checkpoints": 10,
    "images_per_sec": round(n_img / max(pipeline_time, 1e-9), 2),
    "ann_dir": ANN_DIR, "run_timestamp": now, "job_run_id": os.environ.get("DATABRICKS_RUN_ID", "job"),
}
with open(f"{RESULTS_DIR}/results_{BACKEND}_{now[:19].replace(':', '-')}.json", "w") as f:
    json.dump([summary], f, indent=2)
print(f"P7 v2 COCO JOB: materialize={model_read_time:.1f}s pipeline(preproc+infer+annotate_write)={pipeline_time:.1f}s "
      f"wall={total_wall:.1f}s ({n_img} imgs, {summary['images_per_sec']} img/s)")
print("Done.")
