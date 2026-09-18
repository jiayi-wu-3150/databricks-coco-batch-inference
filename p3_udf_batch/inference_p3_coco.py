# Databricks notebook source
# DBTITLE 1,P3 (COCO) — Distributed batch inference with a Spark UDF (mapInPandas)
# MAGIC %md
# MAGIC P3 pattern (path DataFrame -> repartition -> mapInPandas: lazy model load + CPU inference
# MAGIC + annotate + distributed write) on the first 3,925 **COCO val2017** images (~159 KB avg)
# MAGIC to match P5. Distributed writes should avoid the single-node FUSE write bottleneck that
# MAGIC dominated P5 (508s serial). Plain serverless Spark (CPU, autoscaled). Labels not meaningful.

# COMMAND ----------

# DBTITLE 1,Config
import os, glob, time
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

CATALOG = "serverless_stable_r4umw1_catalog"
SCHEMA = "cv"
BACKEND = "p3_coco_udf_batch"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.vit_imagenette_soup"
BASE_MODEL_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/vit_models/vit_imagenette/base_model"
COCO_VAL = f"/Volumes/{CATALOG}/{SCHEMA}/coco2017/val2017"
OUTPUT_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/inference_out_p3_coco"
ANN_DIR = f"{OUTPUT_DIR}/annotated_val"
RESULTS_DIR = f"{OUTPUT_DIR}/results"
PRED_TABLE = f"{CATALOG}.{SCHEMA}.p3_coco_predictions"
N_IMAGES = 3925
NUM_PARTITIONS = 128
CHUNK = 8

CLASS_ORDER = ["n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
               "n03394916", "n03417042", "n03425413", "n03445777", "n03888257"]
IMAGENETTE_LABELS = {
    "n01440764": "tench", "n02102040": "English springer", "n02979186": "cassette player",
    "n03000684": "chain saw", "n03028079": "church", "n03394916": "French horn",
    "n03417042": "garbage truck", "n03425413": "gas pump", "n03445777": "golf ball",
    "n03888257": "parachute",
}

mlflow.set_registry_uri("databricks-uc")
VER = max(int(v.version) for v in MlflowClient(registry_uri="databricks-uc").search_model_versions(f"name = '{MODEL_NAME}'"))
MODEL_URI = f"models:/{MODEL_NAME}/{VER}"
MODEL_LOCAL = f"{OUTPUT_DIR}/_soup_model"
os.makedirs(ANN_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)

# Materialize the soup model to a Volume ONCE on the driver, so workers load via from_pretrained.
_comp = mlflow.transformers.load_model(MODEL_URI, return_type="components")
_comp["model"].save_pretrained(MODEL_LOCAL)
from transformers import AutoImageProcessor as _AIP
_AIP.from_pretrained(BASE_MODEL_DIR).save_pretrained(MODEL_LOCAL)
print("materialized soup model ->", MODEL_LOCAL)

# COMMAND ----------

# DBTITLE 1,Path DataFrame (first N COCO val images)
SCRIPT_START = time.time()
paths = sorted(glob.glob(f"{COCO_VAL}/*.jpg"))[:N_IMAGES]
n_img = len(paths)
df = spark.createDataFrame(pd.DataFrame({"path": paths})).repartition(NUM_PARTITIONS)
print(f"images={n_img} partitions={NUM_PARTITIONS} model={MODEL_URI}")

# COMMAND ----------

# DBTITLE 1,mapInPandas UDF: lazy model load + batched CPU inference + annotate/write
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
out_schema = StructType([
    StructField("path", StringType()), StructField("pred_class", StringType()),
    StructField("confidence", DoubleType()),
])

def infer_partition(itr):
    import torch, os
    from PIL import Image, ImageDraw
    import torchvision.transforms as T
    from transformers import AutoImageProcessor
    g = globals()
    if "_P3C_MODEL" not in g:
        from transformers import ViTForImageClassification
        g["_P3C_MODEL"] = ViTForImageClassification.from_pretrained(MODEL_LOCAL).eval()
        _proc = AutoImageProcessor.from_pretrained(MODEL_LOCAL)
        g["_P3C_TF"] = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                                  T.Normalize(mean=_proc.image_mean, std=_proc.image_std)])
    model, tf = g["_P3C_MODEL"], g["_P3C_TF"]
    for pdf in itr:
        paths_all = list(pdf["path"])
        for s in range(0, len(paths_all), CHUNK):
            chunk = paths_all[s:s + CHUNK]
            imgs = [Image.open(p).convert("RGB") for p in chunk]
            x = torch.stack([tf(im) for im in imgs])
            with torch.no_grad():
                logits = model(pixel_values=x).logits
            preds = logits.argmax(-1).tolist()
            confs = torch.softmax(logits, -1).max(-1).values.tolist()
            out = []
            for p, im, pr, cf in zip(chunk, imgs, preds, confs):
                cid = CLASS_ORDER[pr]; text = f"{IMAGENETTE_LABELS[cid]} ({cf:.1%})"
                d = ImageDraw.Draw(im); bb = d.textbbox((0, 0), text)
                tw, th = bb[2]-bb[0], bb[3]-bb[1]
                d.rectangle([(2, 2), (tw+10, th+10)], fill="black"); d.text((6, 4), text, fill="white")
                im.save(f"{ANN_DIR}/{os.path.basename(p)}", "JPEG", quality=90)
                out.append((p, cid, float(cf)))
            del imgs, x, logits
            yield pd.DataFrame(out, columns=["path", "pred_class", "confidence"])

res = df.mapInPandas(infer_partition, schema=out_schema)
res.write.mode("overwrite").saveAsTable(PRED_TABLE)
n_pred = spark.table(PRED_TABLE).count()
total_wall = time.time() - SCRIPT_START
print(f"P3 COCO done: {n_pred} predictions | wall={total_wall:.1f}s | partitions={NUM_PARTITIONS}")

# COMMAND ----------

# DBTITLE 1,Results JSON + MLflow telemetry
import json
from datetime import datetime, timezone
now = datetime.now(timezone.utc).isoformat()
summary = {
    "backend": BACKEND, "checkpoint": "phase_summary", "epoch": -1,
    "dataset": "coco2017_val", "compute": "serverless_spark_cpu", "num_partitions": NUM_PARTITIONS,
    "total_wall_s": round(total_wall, 3), "num_images": n_pred, "num_checkpoints": 10,
    "images_per_sec": round(n_pred / max(total_wall, 1e-9), 2),
    "pred_table": PRED_TABLE, "run_timestamp": now,
}
with open(f"{RESULTS_DIR}/results_{BACKEND}_{now[:19].replace(':', '-')}.json", "w") as f:
    json.dump([summary], f, indent=2)
try:
    with mlflow.start_run(run_name=f"timing_{BACKEND}"):
        mlflow.log_params({"backend": BACKEND, "dataset": "coco2017_val",
                           "compute": "serverless_spark_cpu", "num_partitions": NUM_PARTITIONS})
        mlflow.log_metrics({"total_wall_s": total_wall, "num_images": n_pred,
                            "images_per_sec": n_pred / max(total_wall, 1e-9)})
    print("[telemetry] logged")
except Exception as e:
    print("[telemetry] skipped:", e)
print(f"P3 COCO wall={total_wall:.1f}s ({n_pred/max(total_wall,1e-9):.1f} img/s across {NUM_PARTITIONS} partitions)")
print("Done.")
