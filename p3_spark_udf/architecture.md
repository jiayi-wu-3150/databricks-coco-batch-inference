# P3 — Spark UDF `mapInPandas` (distributed CPU) (architecture)

**Compute:** Serverless Spark, CPU, 128 partitions (`torch`/`torchvision`/`transformers`
propagated to executors)
**Model load:** `bench_vit` materialized once to a UC Volume (`_model`), then every
executor does `from_pretrained(volume_path)` — no MLflow/UC auth needed on workers.
**Knob:** `num_partitions` (default 128).

```
  UC Volume        ┌──────────────  Spark executors (autoscaled CPU)  ──────────────┐
  (JPEGs) ─binary─► │  partition 1: from_pretrained ─► infer(chunk=8) ─► annotate+write│─► UC Vol
                    │  partition 2: from_pretrained ─► infer(chunk=8) ─► annotate+write│─► + Delta
       repartition  │  ...                                                            │    table
       (128) ─────► │  partition N: from_pretrained ─► infer(chunk=8) ─► annotate+write│
                    └────────────────────────────────────────────────────────────────┘
                       inference + writes spread across many prefixes/tasks
```

**Why it's shaped this way:** no GPU required. `mapInPandas` fans both inference *and*
writes across autoscaled executors, so writes hit many S3 prefixes at once (less
single-node FUSE contention than P1/P2).

**Trade-off:** costs more (autoscaled CPU DBUs, and CPU inference is slower per image) and
swings ±25% with how many workers spin up. Best when you have no GPU or want pure
horizontal scale.

**Timing (3,925 imgs, cold, script wall):** COCO **387s** · Imagenette **417s**.
