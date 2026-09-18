# Databricks COCO Batch Inference — Pattern Comparison (P1–P7)

Six batch-inference patterns for running a ViT image classifier over **COCO val2017**
images on **Databricks serverless compute**, benchmarked head-to-head. The goal is to
compare *compute strategies* (single-node GPU, distributed CPU, Ray, model-serving) and
*I/O strategies* (serial vs. parallel vs. distributed writes) on the same workload.

All patterns run the same weight-averaged "soup" ViT model over the first **3,925 COCO
val2017 images** (~159 KB avg — ~20× larger than Imagenette) and write annotated JPEGs
back to a Unity Catalog Volume. Since the model is Imagenette-trained, labels are not
meaningful on COCO — **these experiments measure timing/throughput, not accuracy.**
(P5 = the "P1-on-COCO" baseline itself, so it isn't a separate entry here.)

## The patterns

| Dir | Pattern | Compute | I/O strategy |
|-----|---------|---------|--------------|
| [`p1_soup_serial/`](p1_soup_serial/inference_p1_coco.py) | Single soup model, one forward pass | Serverless GPU (1×A10) | **Serial** annotate + write |
| [`p2_soup_parallel_io/`](p2_soup_parallel_io/inference_p2_coco.py) | Single soup model, one forward pass | Serverless GPU (1×A10) | **32-thread** parallel writes |
| [`p3_udf_batch/`](p3_udf_batch/inference_p3_coco.py) | `mapInPandas` Spark UDF | Serverless Spark (CPU, 128 partitions) | **Distributed** writes across executors |
| [`p4_autoloader_binary/`](p4_autoloader_binary/inference_p4_coco.py) | Auto Loader → binary Delta table → UDF | Serverless Spark (CPU, 128 partitions) | Distributed ingest + writes |
| [`p6_ai_query/`](p6_ai_query/) | `ai_query` → **GPU model-serving endpoint** (inference-only) | GPU endpoint + CPU caller | CPU does base64 + `ai_query` + distributed writes |
| [`p7_ray_staged/`](p7_ray_staged/) | **Ray Data** 3-stage: CPU decode → GPU infer → CPU write | Serverless GPU (Ray, 1×A10) | Distributed CPU write actors (off the GPU) |

## Results (3,925 COCO images, all write-inclusive)

| Rank | Pattern | Compute | **Total wall** | Notes |
|:----:|---------|---------|:--------------:|-------|
| 🥇 1 | **P6** — `ai_query` → GPU endpoint + CPU write | GPU endpoint + CPU | **255s** | inference offloaded to endpoint; 100% match, 0 failures |
| 🥈 2 | **P7** — Ray Data 3-stage | Serverless GPU | **345s** | staged CPU→GPU→CPU-write; fixes v1 (was 3,804s) |
| 🥉 3 | **P2** — GPU + 32-thread writes | Serverless GPU | **402s** | write 303s |
| 4 | **P3** — Spark UDF (CPU) | Serverless CPU | **507s** | distributed |
| 5 | **P4** — Auto Loader + UDF (CPU) | Serverless CPU | **591s** | + ingest |
| 6 | **P1** — GPU + serial writes | Serverless GPU | **741s** | write 546s = 74% |

### Key findings
1. **This workload is I/O-bound on writes, not compute.** GPU inference is only ~78s;
   P1's serial annotated-JPEG write to the UC Volume is 546s = 74% of its runtime.
2. **Parallelizing writes is the biggest single win.** P2's thread pool cut the write
   phase 546s → 303s and total time 741s → 402s (**1.8× faster**) on identical hardware.
3. **Distributing to CPU (P3/P4) beats serial-GPU but not parallel-GPU** for this size —
   it avoids single-node FUSE write contention but pays CPU inference + Spark overhead.
4. **Write cost is per-file, not per-byte.** P1's serial write barely grew vs. the tiny
   Imagenette baseline despite COCO files being ~20× larger — the cost is FUSE metadata
   round-trips per file.

> ⚠️ **The winner is not always the same — it depends on image size.** On *tiny* images
> (Imagenette, ~7.8 KB), P2's 32-thread writes **backfire** (severe FUSE metadata
> contention from many concurrent small-file creates) and the distributed-CPU P3 wins
> instead. On *large* images (COCO, ~159 KB), parallel-GPU writes (P2) win. Same 3,925
> images both times:
>
> | Pattern | Imagenette wall | COCO wall |
> |---------|:---------------:|:---------:|
> | P1 — GPU serial | 633s | 741s |
> | P2 — GPU parallel | 1,872s ⚠️ | **402s** 🥇 |
> | P3 — CPU UDF | **431s** 🥇 | 507s |
> | P4 — CPU autoloader | 711s | 591s |
>
> **Winner flips: P3 on Imagenette, P2 on COCO.** These four scripts are provided as
> **reference templates and a benchmark harness**, not a one-size-fits-all recommendation.

### What determines the best pattern

Image size is only one axis. The right choice depends on many factors — benchmark on
*your* workload:

- **Do you write images at all?** These examples annotate + write a JPEG per image, which
  is the dominant cost. If you only need predictions to a Delta table (no image output),
  the write bottleneck disappears and GPU inference dominates — a completely different
  ranking.
- **Image size & count** — large files favor parallel-GPU writes (P2); many tiny files
  favor distributed writes (P3/P4) and punish thread-pool writes (FUSE contention).
- **Parallelism knobs** — P2's `WRITE_WORKERS` (thread count) and P3/P4's `NUM_PARTITIONS`
  are tunable and materially change results; the values here are starting points.
- **Ensemble orchestration** — a single weight-averaged *soup* model = one load + one
  forward pass (what these scripts use). A *logit ensemble* over N checkpoints = N loads +
  N passes, which shifts the balance heavily toward `model_read`/inference cost.
- **Where the model lives / how it loads** — each path has different load cost and
  executor-distribution implications:
  - **Registered in UC** (`models:/<catalog>.<schema>.<model>/<version>`) — clean
    versioning; each executor needs UC auth to pull, so these scripts materialize it to a
    Volume once on the driver.
  - **Artifacts on a UC Volume** — executors load with plain `from_pretrained(volume_path)`,
    no MLflow/auth on workers (how P3/P4 distribute).
  - **Loaded from an MLflow experiment run** — convenient during dev, but run-artifact
    reads can be the slowest load path (the original backend comparison saw MLflow model
    read dominate).

## P6 — `ai_query` against a GPU model-serving endpoint

Batch inference by calling a **model-serving endpoint** from SQL with `ai_query`. Split of
labor: the **GPU endpoint does inference only** (base64 image in → `{label, score}` out),
and **serverless CPU** does everything else — read raw JPEGs → base64 table, `ai_query`,
then distributed annotate + write. End-to-end (255s) = base64 54s · `ai_query` 103s
(38 img/s) · annotate+write 97s.

- **Deployment shape is what matters, not GPU vs CPU.** A raw `mlflow.transformers` model
  served directly fails to load. The fix is a custom `pyfunc` wrapper with a real
  `ModelSignature` (`image` string in → `label,score` out). See
  [`deploy_vit_aiquery.py`](p6_ai_query/deploy_vit_aiquery.py).
- **`workload_type` gotcha:** the SDK's `ServedEntityInput.workload_type` wants the
  **`ServingModelWorkloadType.GPU_SMALL` enum**, not the string `"GPU_SMALL"` (the REST
  API accepts the string, the Python SDK does not → `AttributeError: 'str' … 'value'`).
- **Inference-only endpoint = writes are decoupled.** Annotated images are written by the
  CPU caller *after* `ai_query`, so a 429 can never produce a half/bad image. `ai_query`
  auto-retries throttling; with default `failOnError=true` a request is either retried to a
  200 (image written normally) or the whole query fails (nothing written). Validate on the
  **output** (completeness + correctness), not the 429 count — retried 429s are invisible.
- **Correctness:** predictions matched the direct-inference pattern (P3) **100%** across all
  3,925 rows. Concurrency left at the endpoint default (`workload_size=Small`).

## P7 — Ray Data staged pipeline

**Ray Data** on serverless GPU, following the Databricks industry-solutions reference:
three stages — `.map` CPU decode (autoscaled) → `.map_batches` GPU inference
(`num_gpus=1`, model loaded once) → `.map` CPU annotate+write. Writes stay **off** the GPU
actor. See [`inference_p7_ray_staged.py`](p7_ray_staged/inference_p7_ray_staged.py); launch
via the `air` CLI ([`train_p7_ray_v2.yaml`](p7_ray_staged/train_p7_ray_v2.yaml)) or as a
Jobs-API GPU notebook task (same config as P1/P2 + `ray[data]`).

> ⚠️ **Anti-pattern (kept as [`ray_v1_broken_reference.py`](p7_ray_staged/ray_v1_broken_reference.py)):**
> the first version did decode **and** a per-image FUSE write **inside a single GPU actor**
> (`concurrency=1`). The GPU sat at **0% utilization** and it took **3,804s** (1.04 img/s)
> on 3,925 images. Separating CPU decode/write into their own autoscaled Ray stages (and
> keeping the GPU actor pure inference) took it to **345s** with real GPU usage — a ~11×
> speedup, and the fastest write-inclusive GPU-native pattern here.

### Cost (AWS us-east-1 / N. Virginia, est.)
GPU billed at $2.50/A10-GPU-hr; CPU at $0.45/DBU (Enterprise). Region rates vary.

| Pattern | Compute | ~Cost | Notes |
|---------|---------|:-----:|-------|
| P2 | GPU | **$0.35** | fastest *and* cheapest |
| P1 | GPU | $0.59 | |
| P4 | CPU | $2.35 | autoscaled workers → higher $/hr |
| P3 | CPU | $4.29 | ~50 DBU/hr effective |

**GPU wins on both speed and cost** here — the distributed-CPU patterns cost 7–12× more
for slower results on this write-bound workload.

## Running these

Each script is a Databricks notebook-source `.py`. Upload to a workspace and run as a
notebook job. Example job specs are in [`jobs/`](jobs/). Two compute profiles are used:

**Serverless GPU (P1, P2)** — task-level accelerator + AI Runtime base environment:
```json
{
  "tasks": [{
    "task_key": "infer",
    "notebook_task": {"notebook_path": "/Workspace/.../p1_coco_single_model", "source": "WORKSPACE"},
    "environment_key": "aiv5",
    "compute": {"hardware_accelerator": "GPU_1xA10"}
  }],
  "environments": [{"environment_key": "aiv5", "spec": {"base_environment": "databricks_ai_v5"}}]
}
```

**Serverless Spark CPU with torch on executors (P3, P4)** — dependencies propagate to
`mapInPandas` workers:
```json
{
  "tasks": [{
    "task_key": "infer",
    "notebook_task": {"notebook_path": "/Workspace/.../p3_coco_udf_batch", "source": "WORKSPACE"},
    "environment_key": "spk"
  }],
  "environments": [{"environment_key": "spk",
                    "spec": {"client": "3", "dependencies": ["torch", "torchvision", "transformers"]}}]
}
```

### Prerequisites & gotchas
- **Serverless GPU jobs (P1/P2) require a Beta preview:** enable **"Serverless workspace
  base environment support in Jobs"** (Beta) in the workspace preview settings. Without it,
  the `environments[].spec.base_environment: databricks_ai_v5` + `compute.hardware_accelerator`
  config is rejected / silently downgraded to a CPU environment and the GPU task fails.
  GPU serverless also supports only `notebook_task` and `python_wheel_task` (not
  `spark_python_task`), and `for_each` is unsupported on GPU jobs.
- A registered UC model (the ViT "soup" model) and COCO val2017 JPEGs on a UC Volume.
- **Output dirs under `/Volumes/{catalog}/{schema}/` must be registered UC Volumes.**
  `os.makedirs()` on a non-Volume path fails with `OSError [Errno 95] Operation not
  supported`. Create them first: `databricks volumes create {cat} {schema} {name} MANAGED`.
- Catalog/schema/model names are hard-coded near the top of each script — edit for your
  workspace.

## Notes
- Timings measured end-to-end in-script (read + inference + write); serverless startup and
  environment install are excluded (they add ~1.5–3 min, reflected in billed run time).
- Built and benchmarked on Databricks serverless (AWS us-east-1).
