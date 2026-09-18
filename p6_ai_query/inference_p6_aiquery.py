# Databricks notebook source
# DBTITLE 1,P6 — ai_query() batch inference against GPU endpoint + write annotated images (COCO)
# MAGIC %md
# MAGIC End-to-end batch inference via **`ai_query`** against the GPU serving endpoint
# MAGIC `vit-soup-aiquery` (inference-only). Everything else is **serverless CPU (Spark)** and
# MAGIC ALL of it counts toward end-to-end time:
# MAGIC 1. **read raw COCO JPEGs → base64 table** (`binaryFile` + `base64(content)`),
# MAGIC 2. `ai_query` each row → GPU endpoint → `{label, score}`,
# MAGIC 3. **annotate + write one JPEG per prediction** (distributed `mapInPandas`, decodes the
# MAGIC    base64 it already has — self-contained, no pre-built binary table needed).
# MAGIC
# MAGIC GPU endpoint does inference only; the CPU job does image→base64, `ai_query`, and writes.

# COMMAND ----------

# DBTITLE 1,Config
import os, io, base64, time, json
import pandas as pd
from datetime import datetime, timezone
from pyspark.sql import functions as F

CATALOG = "serverless_stable_r4umw1_catalog"
SCHEMA = "cv"
ENDPOINT = "vit-soup-aiquery"
COCO_VAL = f"/Volumes/{CATALOG}/{SCHEMA}/coco2017/val2017"   # raw JPEGs (same source as P1-P5)
B64_TABLE = f"{CATALOG}.{SCHEMA}.coco_val_b64"               # path, image_b64
PRED_TABLE = f"{CATALOG}.{SCHEMA}.p6_coco_predictions"       # path, prediction struct
OUT_TABLE = f"{CATALOG}.{SCHEMA}.p6_coco_predictions_written"
OUTPUT_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/inference_out_p6_coco"
ANN_DIR = f"{OUTPUT_DIR}/annotated_val"
RESULTS_DIR = f"{OUTPUT_DIR}/results"
N_IMAGES = 3925
NUM_PARTITIONS = 128
os.makedirs(ANN_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)
SCRIPT_START = time.time()

# COMMAND ----------

# DBTITLE 1,Stage 1 (CPU) — read raw COCO JPEGs and convert to a base64 table
_t = time.time()
(spark.read.format("binaryFile").option("pathGlobFilter", "*.jpg").load(COCO_VAL)
     .withColumn("path", F.regexp_replace("path", "^dbfs:", ""))
     .selectExpr("path", "base64(content) AS image_b64")
     .orderBy("path").limit(N_IMAGES)
     .write.mode("overwrite").saveAsTable(B64_TABLE))
n_b64 = spark.table(B64_TABLE).count()
b64_build_s = time.time() - _t
print(f"read raw JPEGs -> base64 table {B64_TABLE}: {n_b64} rows in {b64_build_s:.1f}s")

# COMMAND ----------

# DBTITLE 1,Stage 2 (CPU) — ai_query against the GPU endpoint -> predictions table
_t = time.time()
spark.sql(f"""
  CREATE OR REPLACE TABLE {PRED_TABLE} AS
  SELECT
    path,
    ai_query(
      '{ENDPOINT}',
      named_struct('image', image_b64),
      returnType => 'STRUCT<label STRING, score DOUBLE>'
    ) AS prediction
  FROM {B64_TABLE}
""")
n_pred = spark.table(PRED_TABLE).count()
aiquery_s = time.time() - _t
print(f"ai_query wrote {n_pred} predictions to {PRED_TABLE} in {aiquery_s:.1f}s "
      f"({n_pred/max(aiquery_s,1e-9):.1f} img/s)")

# COMMAND ----------

# DBTITLE 1,Stage 3 (CPU, distributed) — decode base64, annotate + write an image per prediction
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
out_schema = StructType([
    StructField("path", StringType()), StructField("label", StringType()),
    StructField("score", DoubleType()), StructField("output_path", StringType()),
])

def annotate_write(itr):
    import io, os, base64
    from PIL import Image, ImageDraw
    for pdf in itr:
        rows = []
        for _, r in pdf.iterrows():
            im = Image.open(io.BytesIO(base64.b64decode(r["image_b64"]))).convert("RGB")
            text = f'{r["label"]} ({float(r["score"]):.1%})'
            d = ImageDraw.Draw(im); bb = d.textbbox((0, 0), text)
            tw, th = bb[2]-bb[0], bb[3]-bb[1]
            d.rectangle([(2, 2), (tw+10, th+10)], fill="black"); d.text((6, 4), text, fill="white")
            outp = f"{ANN_DIR}/{os.path.basename(r['path'])}"
            im.save(outp, "JPEG", quality=90)
            rows.append((r["path"], r["label"], float(r["score"]), outp))
        yield pd.DataFrame(rows, columns=["path", "label", "score", "output_path"])

_t = time.time()
src = (spark.table(PRED_TABLE).selectExpr("path", "prediction.label AS label", "prediction.score AS score")
       .join(spark.table(B64_TABLE).select("path", "image_b64"), "path")
       .repartition(NUM_PARTITIONS))
written = src.mapInPandas(annotate_write, schema=out_schema)
written.write.mode("overwrite").saveAsTable(OUT_TABLE)
n_written = spark.table(OUT_TABLE).count()
write_s = time.time() - _t
print(f"annotated+wrote {n_written} images -> {ANN_DIR} in {write_s:.1f}s")

# COMMAND ----------

# DBTITLE 1,Results JSON — end-to-end includes image->base64, ai_query, annotate+write
total_wall = time.time() - SCRIPT_START
now = datetime.now(timezone.utc).isoformat()
summary = {
    "backend": "p6_ai_query_gpu_endpoint", "checkpoint": "phase_summary",
    "dataset": "coco2017_val", "endpoint": ENDPOINT, "endpoint_workload": "GPU_SMALL",
    "compute_caller": "serverless_spark_cpu", "num_partitions": NUM_PARTITIONS,
    "img_to_b64_s": round(b64_build_s, 3), "aiquery_s": round(aiquery_s, 3),
    "annotate_write_s": round(write_s, 3), "total_wall_s": round(total_wall, 3),
    "num_images": n_written, "aiquery_images_per_sec": round(n_pred / max(aiquery_s, 1e-9), 2),
    "end_to_end_images_per_sec": round(n_written / max(total_wall, 1e-9), 2),
    "pred_table": PRED_TABLE, "written_table": OUT_TABLE, "ann_dir": ANN_DIR,
    "run_timestamp": now,
}
with open(f"{RESULTS_DIR}/results_p6_{now[:19].replace(':', '-')}.json", "w") as f:
    json.dump([summary], f, indent=2)
display(spark.sql(f"SELECT path, label, score, output_path FROM {OUT_TABLE} LIMIT 10"))
print(json.dumps(summary, indent=2))
print("Done.")
