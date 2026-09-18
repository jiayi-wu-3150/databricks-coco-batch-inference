# Clean Batch-Inference Benchmark (P1–P6) — MLflow-native, user-iterable

Six batch-inference patterns for a ViT image classifier, benchmarked on the same clean
UC model and dataset so you can **swap in your own model + images and re-run**. Everything
is MLflow-native and parameterized via **job params / widgets** (no code edits to re-target).

- **Model:** `…cv.bench_vit@prod` (transformers) — batch patterns load `models:/…bench_vit@prod`.
  `…cv.bench_vit_serving@prod` (pyfunc, base64→`{label,score}`) backs the P6 endpoint.
- **Labels** auto-derived from the model's `id2label` (works for any HF classifier).
- **Outputs** namespaced per dataset (`bench_out_pX/<dataset>/…`); phase timings logged to a
  shared MLflow experiment.
- **Setup:** run [`00_setup_register_model.py`](00_setup_register_model.py) once (registers the
  two models, creates volumes, and — optionally — deploys the GPU serving endpoint for P6).

## Summary — COCO val2017, 3,925 images (~159 KB avg)

| # | Pattern | Compute | Total job time (billed) | Script wall | Time breakdown |
|---|---------|---------|:-----------------------:|:-----------:|----------------|
| **P1** | Single model, **serial** writes | Serverless GPU (1×A10) | ~848s | 740s | model_read 29s · inference 75s · **write 635s** |
| **P2** | Single model, **32-thread parallel** writes | Serverless GPU (1×A10) | ~460s | **358s (cold)** · 155s (warm) | model_read 26s · inference ~50s · **write 288s (cold)** / 66s (warm) |
| **P3** | Spark UDF `mapInPandas` (distributed) | Serverless Spark CPU (128p) | ~457s | 387s | distributed read+infer+write (not split) |
| **P4** | Auto Loader binary → UDF (distributed) | Serverless Spark CPU (128p) | ~513s | 438s | ingest 68s · infer+write 368s |
| **P5** | Ray Data 3-stage (CPU→GPU→CPU) | Serverless GPU (Ray, 1×A10) | ~441s | 350s | model_read 30s · pipeline 308s |
| **P6** | `ai_query` → GPU serving endpoint + CPU write | GPU endpoint + Serverless CPU | ~405s | 374s | base64 55s · ai_query 104s · **write 214s** |

**Two timing caveats (read before ranking):**
1. **Billed job time = script wall + a fixed ~70–110s** of serverless startup + environment
   install (torch / `databricks_ai_v5` / Ray). Compare *patterns* with **script wall**; plan
   *cost/wall-clock* with **billed**.
2. **Write-bound patterns (P1, P2, P3, P4, P6) are cold/warm sensitive.** Writing thousands
   of annotated JPEGs to a UC Volume is object-store (S3) I/O: a fresh prefix throttles
   (`503 SlowDown` → retries), a warmed one flies. Measured on P2: **write 288s cold → 66s
   warm** (total 358s → 155s), while GPU compute stayed flat. **Serial writes (P1) stay under
   the throttle limit → consistent but slow; parallel writes (P2) burst over it → fast but
   variable.** Distributed-CPU (P3/P4) additionally swing ±25% with autoscaled worker count.
   **Every pattern above is reported at its *cold* single-run number so the comparison is
   apples-to-apples** — including P2 (358s cold). P2's 155s warm run is its best-case upside,
   not its comparable figure.

**Correctness:** P6 (endpoint) vs P3 (direct inference) match **99.95%** (3923/3925; 2 borderline
argmax flips from GPU fp16 vs CPU fp32) — the served model is equivalent to direct inference.

## Summary — Imagenette val, 3,925 images (small JPEGs, nested class dirs)

Same 3,925-image count as COCO, so the only variables are **image size** (Imagenette is much
smaller per file) and **layout** (nested `class/*.JPEG` vs COCO's flat, larger `.jpg`). All
**cold single runs**, script wall.

| # | Pattern | Compute | Script wall | Time breakdown |
|---|---------|---------|:-----------:|----------------|
| **P1** | serial writes | Serverless GPU (1×A10) | 629s | model_read 25s · inference 43s · **write 560s** |
| **P2** | 32-thread parallel | Serverless GPU (1×A10) | **345s** | model_read 27s · inference 34s · **write 283s** |
| **P3** | Spark UDF (distributed) | Serverless Spark CPU (128p) | 417s | distributed read+infer+write |
| **P4** | Auto Loader → UDF | Serverless Spark CPU (128p) | 522s | ingest 97s · infer+write 424s |
| **P5** | Ray Data 3-stage | Serverless GPU (Ray, 1×A10) | **350s** | model_read 39s · pipeline 299s |
| **P6** | `ai_query` → GPU endpoint | GPU endpoint + Serverless CPU | 441s | base64 39s · **ai_query 195s** · write 206s |

## COCO vs Imagenette — the winner is *not* always the same

Both runs are 3,925 images, cold, single-run — so this isolates image size + file layout.

| Pattern | COCO (≈159 KB `.jpg`, flat) | Imagenette (small `.JPEG`, nested) | What changed |
|---------|:---------------------------:|:----------------------------------:|--------------|
| **P1** serial | 740s | 629s | smaller files → faster per-write, but still write-bound & slowest |
| **P2** parallel | 358s | **345s** | ~flat; fastest-tier on both (cold) |
| **P3** Spark UDF | 387s | 417s | +8% |
| **P4** Auto Loader | 438s | 522s | **+19%** — nested-dir listing + many tiny files cost more at ingest |
| **P5** Ray staged | 350s | **350s** | **identical — GPU-compute-bound, so dataset-size-independent** |
| **P6** ai_query | 374s | 441s | **+18%** — `ai_query` phase 104s→195s (more per-image endpoint round-trips) |

**Takeaways:**
- **No single winner.** COCO-cold fastest tier is P5 (350) ≈ P2 (358); Imagenette-cold it's
  P2 (345) ≈ P5 (350). P2 *warm* (~155s) would win either — but it's the least consistent.
- **P5 (Ray) is the most dataset-stable** — 350s on both, because time is dominated by the
  single GPU inference stage, not by file size or count.
- **File *layout and count* matter as much as total bytes.** The patterns that lean on
  directory listing / per-file endpoint calls (P4 ingest, P6 `ai_query`) are the ones that
  slow down on Imagenette's many small nested files, even though the total data is smaller.

## Comparison across dimensions

Legend: 🟢 strong · 🟡 moderate · 🔴 weak (relative to the others, this workload).

| Pattern | Latency (warm) | Consistency | Ease / simplicity | Cost | Scalability | Reuse / governance |
|---------|:--------------:|:-----------:|:-----------------:|:----:|:-----------:|:------------------:|
| **P1** GPU serial | 🔴 740s (slowest) | 🟢 stable (serial stays under S3 throttle) | 🟢 simplest | 🟡 GPU but ~14 min | 🔴 single node, serial I/O | 🔴 one-off job |
| **P2** GPU parallel | 🟡 358s cold (155s warm) | 🔴 2.3× cold/warm swing | 🟢 simple (+ thread pool) | 🟢 short GPU run | 🟡 single node; tune `write_workers` | 🔴 one-off job |
| **P3** Spark UDF (CPU) | 🟡 387s | 🟡 ±25% (autoscale) | 🟡 UDF + model materialize | 🔴 autoscaled CPU workers pricey | 🟢 horizontal | 🔴 one-off job |
| **P4** Auto Loader (CPU) | 🟡 438s | 🟡 ±25% (autoscale) | 🟡 two-stage (ingest+infer) | 🔴 autoscaled CPU + ingest | 🟢 horizontal + incremental ingest | 🟡 reusable binary table |
| **P5** Ray staged (GPU) | 🟢 350s | 🟢 GPU-bound, stable | 🔴 Ray/AIR + staged pipeline | 🟢 GPU ~7 min | 🟢 scales to multi-GPU | 🔴 one-off job |
| **P6** ai_query endpoint | 🟡 374s | 🟡 write + endpoint concurrency (429 retries) | 🔴 deploy endpoint + pyfunc wrapper | 🟡 CPU job + GPU endpoint (scale-to-zero) | 🟢 endpoint concurrency + SQL fan-out | 🟢 governed, reusable, SQL-callable |

## Notes per pattern

- **P1 — GPU serial.** The baseline. Dead simple (load model, one forward pass, write in a
  loop) and the *most consistent* because serial writes never exceed S3's per-prefix rate
  limit — but that same seriality makes it the slowest (write = 86% of wall). Use as the
  correctness reference, not for speed.

- **P2 — GPU parallel.** Same as P1 with a `write_workers`-thread pool over the writes. On a
  **cold prefix (its comparable number) it lands at 358s** — right in the pack with P5/P3/P6 —
  because the parallel write burst is exactly what trips S3 throttling. Warm the prefix and it
  becomes the **fastest pattern (~155s)**, but that's a best case, not the comparable figure.
  So P2 is the **least consistent** (2.3× swing): a great default *if* you warm the target or
  tolerate variance; tune `write_workers`. (On Imagenette's tiny nested files it held ~345s —
  no backfire in the clean run — so its cold cost tracks COCO.)

- **P3 — Spark UDF (CPU, distributed).** No GPU needed; fans inference + writes across
  autoscaled Spark workers, so writes are spread over many prefixes/tasks (less single-node
  contention). Costs more (autoscaled CPU DBUs) and swings ±25% with how many workers spin
  up. Good when you have no GPU or want pure horizontal scale.

- **P4 — Auto Loader → UDF (CPU, distributed).** P3 plus an incremental **binary-ingest**
  stage (`cloudFiles`), which decouples ingestion from inference and enables reprocessing /
  streaming. Pays an extra ingest phase (~68s here) and the same CPU inference cost. Best
  when ingestion is ongoing or you want a reusable binary table.

- **P5 — Ray Data staged (GPU).** Three Ray stages: autoscaled CPU decode → single GPU
  inference actor (model loaded once) → autoscaled CPU annotate+write. Keeps the GPU fed and
  writes off the GPU actor, so it's fast (~350s) and GPU-stable. The trade is complexity
  (Ray/AI-Runtime) and scales cleanly to multi-GPU. *(Anti-pattern to avoid: doing decode +
  per-image writes inside the GPU actor pins the GPU at 0% — that was ~11× slower.)*

- **P6 — `ai_query` → GPU serving endpoint.** Inference is offloaded to a **reusable, governed
  GPU endpoint** callable from **SQL** (`ai_query`); the CPU Spark job does base64 + the
  `ai_query` fan-out + distributed writes. Most complex to stand up (pyfunc wrapper w/
  signature, endpoint deploy — do it in setup so its build time isn't in P6). Unique wins:
  the endpoint is shared/versioned and usable outside this job, and throughput scales with
  endpoint concurrency (which auto-retries 429s). Costs two compute resources at once (CPU +
  GPU endpoint), though scale-to-zero limits idle cost. The base64 round-trip (~55s) is a
  serialization tax intrinsic to crossing the endpoint boundary.

## Run it on your own model / images

1. Run `00_setup_register_model.py` with your `source_model` (and `deploy_endpoint=true` for P6).
2. Run any pattern job with params: `catalog, schema, dataset, image_dir, image_glob`
   (`*.jpg` flat vs `*/*` nested), `file_glob` (P4/P6 pathGlobFilter), `n_images`,
   `num_partitions` / `write_workers`. GPU patterns (P1/P2/P5) need the *"Serverless workspace
   base environment support in Jobs"* Beta; P5 also adds `ray[data]`; CPU patterns add
   `torch/torchvision/transformers` (P3/P4) or `Pillow` (P6).
3. Compare in the MLflow experiment; annotated images + prediction tables land under
   `bench_out_pX/<dataset>/`.
