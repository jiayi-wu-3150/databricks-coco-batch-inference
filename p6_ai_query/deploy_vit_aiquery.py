# Databricks notebook source
# DBTITLE 1,P6 — Deploy ViT soup as an INFERENCE-ONLY endpoint for ai_query()
# MAGIC %md
# MAGIC Deploys the ViT soup model as a custom `pyfunc` serving endpoint that does **inference
# MAGIC only** — it receives base64-encoded image bytes in the request, runs the model, and
# MAGIC returns `{label, score}`. **No Volume reads/writes inside the endpoint** (the caller
# MAGIC supplies image bytes via `ai_query` and persists results). **GPU_SMALL** endpoint,
# MAGIC scale-to-zero. (This deploy notebook itself runs on serverless CPU; only the endpoint
# MAGIC is GPU.) The base64 image prep + ai_query + result writes run on serverless CPU.
# MAGIC
# MAGIC This is the deployment shape `ai_query` needs — the earlier `vit-soup-endpoint` failed
# MAGIC because it served a raw `mlflow.transformers` model with no request-friendly signature.

# COMMAND ----------

# DBTITLE 1,Config + materialize soup model locally
import os, io, base64, json, time
import pandas as pd
import torch
import mlflow
from mlflow.tracking import MlflowClient

CATALOG = "serverless_stable_r4umw1_catalog"
SCHEMA = "cv"
SRC_MODEL = f"{CATALOG}.{SCHEMA}.vit_imagenette_soup"
PYFUNC_MODEL = f"{CATALOG}.{SCHEMA}.vit_soup_aiquery"      # new registered pyfunc
ENDPOINT = "vit-soup-aiquery"
BASE_MODEL_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/vit_models/vit_imagenette/base_model"
LOCAL_MODEL = "/tmp/vit_soup_aiquery_model"

CLASS_ORDER = ["n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
               "n03394916", "n03417042", "n03425413", "n03445777", "n03888257"]
IMAGENETTE_LABELS = {
    "n01440764": "tench", "n02102040": "English springer", "n02979186": "cassette player",
    "n03000684": "chain saw", "n03028079": "church", "n03394916": "French horn",
    "n03417042": "garbage truck", "n03425413": "gas pump", "n03445777": "golf ball",
    "n03888257": "parachute",
}

mlflow.set_registry_uri("databricks-uc")
_c = MlflowClient(registry_uri="databricks-uc")
VER = max(int(v.version) for v in _c.search_model_versions(f"name = '{SRC_MODEL}'"))
comp = mlflow.transformers.load_model(f"models:/{SRC_MODEL}/{VER}", return_type="components")
comp["model"].save_pretrained(LOCAL_MODEL)
from transformers import AutoImageProcessor
AutoImageProcessor.from_pretrained(BASE_MODEL_DIR).save_pretrained(LOCAL_MODEL)
print("materialized soup model ->", LOCAL_MODEL)

# COMMAND ----------

# DBTITLE 1,Inference-only pyfunc wrapper (base64 image in -> label/score out)
class ViTInferenceOnly(mlflow.pyfunc.PythonModel):
    """Decodes base64 image bytes from the request, runs the ViT soup model, returns
    label + score. No file I/O — pure inference, so the endpoint needs no Volume auth."""
    def __init__(self, class_order, label_map):
        self.class_order = class_order
        self.label_map = label_map

    def load_context(self, context):
        import torch, torchvision.transforms as T
        from transformers import ViTForImageClassification, AutoImageProcessor
        md = context.artifacts["model_dir"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViTForImageClassification.from_pretrained(md).to(self.device).eval()
        proc = AutoImageProcessor.from_pretrained(md)
        self.tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                             T.Normalize(mean=proc.image_mean, std=proc.image_std)])

    def predict(self, context, model_input, params=None):
        import io, base64, torch
        from PIL import Image
        b64s = list(model_input["image"])
        imgs = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in b64s]
        x = torch.stack([self.tf(im) for im in imgs]).to(self.device)
        with torch.no_grad():
            logits = self.model(pixel_values=x).logits.float()
        preds = logits.argmax(-1).tolist()
        confs = torch.softmax(logits, -1).max(-1).values.tolist()
        return pd.DataFrame({
            "label": [self.label_map[self.class_order[p]] for p in preds],
            "score": [float(c) for c in confs],
        })

# COMMAND ----------

# DBTITLE 1,Smoke-test the wrapper locally, then log + register
from mlflow.types.schema import Schema, ColSpec
from mlflow.models.signature import ModelSignature

# quick local check on one base64 image
import glob
_sample = sorted(glob.glob(f"/Volumes/{CATALOG}/{SCHEMA}/vit_models/vit_imagenette/data/val/*/*"))[0]
_b64 = base64.b64encode(open(_sample, "rb").read()).decode()
_wrap = ViTInferenceOnly(CLASS_ORDER, IMAGENETTE_LABELS)

class _Ctx:  # minimal context for local test
    artifacts = {"model_dir": LOCAL_MODEL}
_wrap.load_context(_Ctx())
print("local test:", _wrap.predict(None, pd.DataFrame({"image": [_b64]})).to_dict("records"))

input_schema = Schema([ColSpec("string", "image")])
output_schema = Schema([ColSpec("string", "label"), ColSpec("double", "score")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
input_example = pd.DataFrame({"image": [_b64]})

with mlflow.start_run(run_name="vit_soup_aiquery_inference_only") as run:
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=ViTInferenceOnly(CLASS_ORDER, IMAGENETTE_LABELS),
        artifacts={"model_dir": LOCAL_MODEL},
        pip_requirements=["torch==2.5.1", "torchvision==0.20.1", "transformers==4.49.0", "Pillow"],
        input_example=input_example,
        signature=signature,
    )
    run_id = run.info.run_id
reg = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=PYFUNC_MODEL)
print("registered", PYFUNC_MODEL, "version", reg.version)

# COMMAND ----------

# DBTITLE 1,Deploy to a GPU serving endpoint (GPU_SMALL, scale-to-zero)
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (ServedEntityInput, EndpointCoreConfigInput,
                                            ServingModelWorkloadType)
from datetime import timedelta

cfg = EndpointCoreConfigInput(
    name=ENDPOINT,
    served_entities=[ServedEntityInput(
        entity_name=PYFUNC_MODEL,
        entity_version=reg.version,
        workload_type=ServingModelWorkloadType.GPU_SMALL,   # enum, NOT the string "GPU_SMALL"
        workload_size="Small",
        scale_to_zero_enabled=True,
    )],
)
w = WorkspaceClient()
existing = next((e for e in w.serving_endpoints.list() if e.name == ENDPOINT), None)
if existing is None:
    print(f"Creating endpoint {ENDPOINT} (GPU_SMALL) ...")
    w.serving_endpoints.create_and_wait(name=ENDPOINT, config=cfg, timeout=timedelta(minutes=45))
else:
    print(f"Updating endpoint {ENDPOINT} ...")
    w.serving_endpoints.update_config_and_wait(served_entities=cfg.served_entities, name=ENDPOINT, timeout=timedelta(minutes=45))
print("endpoint ready:", ENDPOINT)
