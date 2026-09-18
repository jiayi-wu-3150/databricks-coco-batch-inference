# Databricks COCO Batch Inference — Pattern Comparison (P1–P4)

Four batch-inference patterns for running a ViT image classifier over **COCO val2017**
images on **Databricks serverless compute**, benchmarked head-to-head. The goal is to
compare *compute strategies* (single-node GPU vs. distributed CPU) and *I/O strategies*
(serial vs. parallel vs. distributed writes) on the same workload.

All four run the same weight-averaged "soup" ViT model over the first **3,925 COCO
val2017 images** (~159 KB avg — ~20× larger than Imagenette) and write annotated JPEGs
back to a Unity Catalog Volume. Since the model is Imagenette-trained, labels are not
meaningful on COCO — **these experiments measure timing/throughput, not accuracy.**

## The four patterns

| Dir | Pattern | Compute | I/O strategy |
|-----|---------|---------|--------------|
| [`p1_soup_serial/`](p1_soup_serial/inference_p1_coco.py) | Single soup model, one forward pass | Serverless GPU (1×A10) | **Serial** annotate + write |
| [`p2_soup_parallel_io/`](p2_soup_parallel_io/inference_p2_coco.py) | Single soup model, one forward pass | Serverless GPU (1×A10) | **32-thread** parallel writes |
| [`p3_udf_batch/`](p3_udf_batch/inference_p3_coco.py) | `mapInPandas` Spark UDF | Serverless Spark (CPU, 128 partitions) | **Distributed** writes across executors |
| [`p4_autoloader_binary/`](p4_autoloader_binary/inference_p4_coco.py) | Auto Loader → binary Delta table → UDF | Serverless Spark (CPU, 128 partitions) | Distributed ingest + writes |

## Results (3,925 COCO images)

| Rank | Pattern | Inference | Write phase | **Total wall** | Throughput |
|:----:|---------|:---------:|:-----------:|:--------------:|:----------:|
| 🥇 1 | **P2** — GPU + 32-thread writes | 78.0s | 303.4s | **402s** | 9.8 img/s |
| 🥈 2 | **P3** — Spark UDF (CPU) | — | *(distributed)* | **507s** | 7.75 img/s |
| 🥉 3 | **P4** — Auto Loader + UDF (CPU) | 467s | + 122s ingest (5,000 imgs) | **591s** | 6.65 img/s |
| 4 | **P1** — GPU + serial writes | 78.6s | 546.1s (74% of wall) | **741s** | 5.3 img/s |

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

> ⚠️ On *tiny* images (Imagenette), P2's 32-thread writes instead **backfire** (severe
> FUSE metadata contention from many concurrent small-file creates). Parallel writes help
> large files but hurt many tiny ones.

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
