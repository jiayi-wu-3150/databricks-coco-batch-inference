# Databricks notebook source
# DBTITLE 1,P4 (COCO) — Auto Loader -> binary Delta table -> batch inference
# MAGIC %md
# MAGIC P4 pattern on COCO. (1) **Auto Loader** (`cloudFiles`, `binaryFile`, `availableNow`)
# MAGIC ingests COCO val2017 JPEGs into a Delta table with a `content` binary column.
# MAGIC (2) A `mapInPandas` UDF decodes the bytes, runs the soup model (CPU), annotates + writes,
# MAGIC and lands predictions. Inference is capped at 3,925 rows to match P5. Serverless Spark.
# MAGIC Larger COCO images stress both the ingest (bigger binary rows) and the write phase.

# COMMAND ----------

# DBTITLE 1,Config
import os, time, json
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime, timezone

CATALOG = "serverless_stable_r4umw1_catalog"
SCHEMA = "cv"
BACKEND = "p4_coco_autoloader_binary"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.vit_imagenette_soup"
BASE_MODEL_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/vit_models/vit_imagenette/base_model"
COCO_VAL = f"/Volumes/{CATALOG}/{SCHEMA}/coco2017/val2017"
OUTPUT_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/inference_out_p4_coco"
ANN_DIR = f"{OUTPUT_DIR}/annotated_val"
RESULTS_DIR = f"{OUTPUT_DIR}/results"
CHK = f"{OUTPUT_DIR}/_autoloader"
BINARY_TABLE = f"{CATALOG}.{SCHEMA}.coco_val_binary"
PRED_TABLE = f"{CATALOG}.{SCHEMA}.p4_coco_predictions"
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

# Materialize soup model to a Volume once (driver) so executors load via from_pretrained.
_comp = mlflow.transformers.load_model(MODEL_URI, return_type="components")
_comp["model"].save_pretrained(MODEL_LOCAL)
from transformers import AutoImageProcessor as _AIP
_AIP.from_pretrained(BASE_MODEL_DIR).save_pretrained(MODEL_LOCAL)
print("materialized soup model ->", MODEL_LOCAL)

# COMMAND ----------

# DBTITLE 1,Stage 1 — Auto Loader ingest COCO images as binary -> Delta table
SCRIPT_START = time.time()
_t = time.time()
(spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .option("cloudFiles.schemaLocation", f"{CHK}/schema")
    .option("pathGlobFilter", "*.jpg")
    .load(COCO_VAL)
    .writeStream
    .option("checkpointLocation", f"{CHK}/chk")
    .trigger(availableNow=True)
    .toTable(BINARY_TABLE)).awaitTermination()
ingest_time = time.time() - _t
n_binary = spark.table(BINARY_TABLE).count()
print(f"Auto Loader ingested {n_binary} images -> {BINARY_TABLE} in {ingest_time:.1f}s")

# COMMAND ----------

# DBTITLE 1,Stage 2 — batch inference reading the binary table (capped at N_IMAGES)
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
out_schema = StructType([
    StructField("path", StringType()), StructField("pred_class", StringType()),
    StructField("confidence", DoubleType()),
])

def infer_partition(itr):
    import torch, os, io
    from PIL import Image, ImageDraw
    import torchvision.transforms as T
    from transformers import AutoImageProcessor
    g = globals()
    if "_P4C_MODEL" not in g:
        from transformers import ViTForImageClassification
        g["_P4C_MODEL"] = ViTForImageClassification.from_pretrained(MODEL_LOCAL).eval()
        _proc = AutoImageProcessor.from_pretrained(MODEL_LOCAL)
        g["_P4C_TF"] = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                                  T.Normalize(mean=_proc.image_mean, std=_proc.image_std)])
    model, tf = g["_P4C_MODEL"], g["_P4C_TF"]
    for pdf in itr:
        rows = list(zip(pdf["path"], pdf["content"]))
        for s in range(0, len(rows), CHUNK):
            chunk = rows[s:s + CHUNK]
            imgs = [Image.open(io.BytesIO(c)).convert("RGB") for _, c in chunk]
            x = torch.stack([tf(im) for im in imgs])
            with torch.no_grad():
                logits = model(pixel_values=x).logits
            preds = logits.argmax(-1).tolist(); confs = torch.softmax(logits, -1).max(-1).values.tolist()
            out = []
            for (path, _), im, pr, cf in zip(chunk, imgs, preds, confs):
                cid = CLASS_ORDER[pr]; text = f"{IMAGENETTE_LABELS[cid]} ({cf:.1%})"
                d = ImageDraw.Draw(im); bb = d.textbbox((0, 0), text); tw, th = bb[2]-bb[0], bb[3]-bb[1]
                d.rectangle([(2, 2), (tw+10, th+10)], fill="black"); d.text((6, 4), text, fill="white")
                im.save(f"{ANN_DIR}/{os.path.basename(path)}", "JPEG", quality=90)
                out.append((path, cid, float(cf)))
            del imgs, x, logits
            yield pd.DataFrame(out, columns=["path", "pred_class", "confidence"])

_t = time.time()
src = (spark.table(BINARY_TABLE).select("path", "content")
       .orderBy("path").limit(N_IMAGES)
       .repartition(NUM_PARTITIONS))
res = src.mapInPandas(infer_partition, schema=out_schema)
res.write.mode("overwrite").saveAsTable(PRED_TABLE)
infer_time = time.time() - _t
n_pred = spark.table(PRED_TABLE).count()
total_wall = time.time() - SCRIPT_START
print(f"P4 COCO inference {n_pred} rows in {infer_time:.1f}s | total_wall={total_wall:.1f}s")

# COMMAND ----------

# DBTITLE 1,Results JSON + MLflow telemetry
now = datetime.now(timezone.utc).isoformat()
summary = {
    "backend": BACKEND, "checkpoint": "phase_summary", "epoch": -1,
    "dataset": "coco2017_val", "compute": "serverless_spark_cpu", "num_partitions": NUM_PARTITIONS,
    "autoloader_ingest_s": round(ingest_time, 3), "num_ingested": n_binary,
    "inference_s": round(infer_time, 3),
    "total_wall_s": round(total_wall, 3),
    "num_images": n_pred, "num_checkpoints": 10,
    "images_per_sec": round(n_pred / max(total_wall, 1e-9), 2),
    "binary_table": BINARY_TABLE, "pred_table": PRED_TABLE, "run_timestamp": now,
}
with open(f"{RESULTS_DIR}/results_{BACKEND}_{now[:19].replace(':', '-')}.json", "w") as f:
    json.dump([summary], f, indent=2)
try:
    with mlflow.start_run(run_name=f"timing_{BACKEND}"):
        mlflow.log_params({"backend": BACKEND, "dataset": "coco2017_val",
                           "compute": "serverless_spark_cpu", "num_partitions": NUM_PARTITIONS})
        mlflow.log_metrics({"autoloader_ingest_s": ingest_time, "num_ingested": n_binary,
                            "inference_s": infer_time, "total_wall_s": total_wall, "num_images": n_pred})
    print("[telemetry] logged")
except Exception as e:
    print("[telemetry] skipped:", e)
print(f"P4 COCO: ingest={ingest_time:.1f}s ({n_binary} imgs) infer={infer_time:.1f}s wall={total_wall:.1f}s ({n_pred} imgs)")
print("Done.")
