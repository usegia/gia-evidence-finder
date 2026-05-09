# SOTA Progress Event Log

Date: 2026-05-08

## Resume State

- Phase: reviewed benchmark hardening, baseline comparison, and first Modal
  fine-tuning probe.
- Main artifacts:
  - `docs/sota-progress-2026-05-08.md`
  - `docs/sota-progress-2026-05-08-event-log.md`
  - `docs/model-baseline-results.md`
  - `scripts/modal_train_reranker.py`
- Current best measured ranking: seeded 4-epoch MiniLM cross-encoder fine-tune
  with typed first-stage candidates.
- Current best measured decision: deterministic typed default and calibrated
  typed-decision hybrid tie at `0.9886`.
- Current product candidate: calibrated typed first-stage support labels
  (`0.30` dev-selected threshold) plus trained MiniLM reranker ordering.
- Next planned check: expand the benchmark beyond README documents and batch
  extraction-time reranker scoring.

## Log

1. Added BM25 as a real corpus-level baseline over document spans.
   - Files: `src/gia_evidence_finder/model_baselines.py`,
     `src/gia_evidence_finder/experiments.py`, tests.
   - Observation: BM25 is much stronger than plain keyword overlap on README
     claims and must remain in the scoreboard.

2. Added external reviewed-series benchmark commands.
   - Commands added:
     - `benchmark-reviewed-series`
     - `benchmark-reviewed-series-calibrated`
     - `explain-reviewed-case`
     - `export-reviewed-training-jsonl`
   - Reason: reviewed JSONL in `.tmp` can now be benchmarked, explained, and
     exported without hard-coding the data into the package.

3. Ran deterministic calibrated reviewed-series benchmark.
   - Command:
     `uv run gia-evidence-finder benchmark-reviewed-series-calibrated --source-dir .tmp/readme-sources --reviewed-jsonl .tmp/reviewed-cases-final.jsonl --series-id evidence_benchmark_v3_model_adjudicated_final`
   - Result: default test `MRR 0.9536`, `R@1 0.9259`, `R@3 0.9877`,
     `Decision 0.9886`, `Forbidden supported 0.0000`.
   - Result: BM25 test `MRR 0.9362`, `R@1 0.9012`, `Decision 0.9205`.

4. Ran local model smoke baselines.
   - Command:
     `uv run --extra models gia-evidence-finder benchmark-reviewed-series-calibrated ... --baseline-profile local_smoke`
   - Result: MiniLM embedding and MS MARCO MiniLM cross-encoder did not beat
     BM25 or the typed default.

5. Probed Jina reranker compatibility.
   - Command:
     `uv run --extra models gia-evidence-finder benchmark-reviewed-series ... --cross-encoder-model jinaai/jina-reranker-v2-base-multilingual`
   - Result: non-fatal unavailable output after wrapping model load failures.
   - Open issue: Jina requires `trust_remote_code=True`; do not execute without
     explicit approval.

6. Ran Qwen3 0.6B calibrated probe.
   - Command:
     `uv run --extra models gia-evidence-finder benchmark-reviewed-series-calibrated ... --cross-encoder-model Qwen/Qwen3-Reranker-0.6B`
   - Result: test `MRR 0.8395`, `R@1 0.6914`, `R@3 0.9877`,
     `Decision 0.9886`, `Forbidden supported 0.0000`.
   - Interpretation: it can tie decision after high-threshold calibration but
     loses heavily on ranking.

7. Added training-data parsing contract.
   - File: `src/gia_evidence_finder/training.py`
   - Contract: exported pair JSONL maps `supports` to `1.0` and all special
     negatives to `0.0` for binary reranker fine-tuning.
   - Test added for parsing exported JSONL.

8. Added Modal H200 training script.
   - File: `scripts/modal_train_reranker.py`
   - Initial run found missing `datasets`; added it.
   - Second run found missing `accelerate`; added it.
   - Third run trained successfully but reload failed; changed evaluation to
     use the in-memory trained model and call `model.save`.

9. Generated reviewed training/dev pairs.
   - Commands:
     - `uv run gia-evidence-finder export-reviewed-training-jsonl ... --split train --negatives-per-case 5 > .tmp/reviewed-train-pairs.jsonl`
     - `uv run gia-evidence-finder export-reviewed-training-jsonl ... --split dev --negatives-per-case 5 > .tmp/reviewed-dev-pairs.jsonl`
   - Counts: `785` train pairs, `412` dev pairs.

10. Ran Modal fine-tunes.
    - 2 epochs: improved MiniLM cross-encoder, but did not beat typed default.
    - 4 epochs: ranking beat typed default, decision still lower.
    - Added seed control for reproducible future runs.

11. Added first-stage label controls to `CrossEncoderRerankExtractor`.
    - `require_first_stage_support`: prevents reranker-only support when the
      first stage did not support the span.
    - `preserve_first_stage_labels`: uses reranker only for ordering while
      preserving first-stage labels.
    - Tests cover both controls.

12. Ran seeded 4-epoch Modal fine-tune.
    - Command:
      `uv run --extra training modal run scripts/modal_train_reranker.py --train-jsonl-path .tmp/reviewed-train-pairs.jsonl --dev-jsonl-path .tmp/reviewed-dev-pairs.jsonl --reviewed-jsonl-path .tmp/reviewed-cases-final.jsonl --source-dir .tmp/readme-sources --series-id evidence_benchmark_v3_model_adjudicated_final --run-name reviewed-msmarco-minilm-20260508-e4-seed13 --epochs 4 --batch-size 16 --learning-rate 2e-5 --warmup-ratio 0.1 --seed 13`
    - Best pair threshold: `0.05`, pair dev `F1 0.7826`.
    - Typed first-stage extraction test: `MRR 0.9774`, `R@1 0.9630`,
      `R@3 1.0000`, `Top-1 support 0.9506`, `Decision 0.9773`,
      `Forbidden supported 0.0000`.

13. Fixed calibrated holdout evaluation policy.
    - File: `src/gia_evidence_finder/experiments.py`
    - Problem: `run_calibrated_holdout` always wrapped test extractors in
      `CandidateScoreThresholdExtractor`, so `preserve_first_stage_labels`
      experiments were silently relabeled by reranker score.
    - Fix: added explicit `HoldoutDecisionPolicy` with `score_threshold` and
      `extractor_labels`.
    - Test: added a high-score reject extractor proving extractor-owned labels
      are preserved under holdout evaluation.

14. Promoted the production hybrid boundary.
    - File: `src/gia_evidence_finder/model_baselines.py`
    - Added `TypedDecisionRerankExtractor`, which uses a pair reranker only for
      ordering and preserves first-stage labels.
    - Added first-stage feature/reason propagation into reranked candidates so
      traces explain both the deterministic decision and learned ordering.

15. Added benchmark failure-report tooling and local-document curation.
    - Files: `src/gia_evidence_finder/failure_analysis.py`,
      `src/gia_evidence_finder/curation_queue.py`, CLI.
    - New command:
      `uv run gia-evidence-finder failure-report-reviewed-series ... --apply-dev-calibration`
    - New command:
      `uv run gia-evidence-finder prepare-document-curation-queue --input docs --output-jsonl .tmp/docs-curation.jsonl`
    - Purpose: diagnose exact remaining cases and grow the benchmark beyond
      README sources without committing raw upstream docs.

16. Re-ran deterministic reviewed-series holdout after the policy fix.
    - Command:
      `uv run gia-evidence-finder benchmark-reviewed-series-calibrated --source-dir .tmp/readme-sources --reviewed-jsonl .tmp/reviewed-cases-final.jsonl --series-id evidence_benchmark_v3_model_adjudicated_final`
    - Result: unchanged deterministic/BM25/keyword metrics.
    - Default test: `MRR 0.9536`, `R@1 0.9259`, `R@3 0.9877`,
      `Decision 0.9886`.

17. Re-evaluated the saved seeded MiniLM model on Modal without retraining.
    - Command:
      `uv run --extra training modal run scripts/modal_train_reranker.py --evaluate-run-name reviewed-msmarco-minilm-20260508-e4-seed13 --reviewed-jsonl-path .tmp/reviewed-cases-final.jsonl --source-dir .tmp/readme-sources --series-id evidence_benchmark_v3_model_adjudicated_final`
    - Corrected raw label-preserve result: `MRR 0.9774`, `R@1 0.9630`,
      `R@3 1.0000`, `Decision 0.6364`.
    - Interpretation: raw typed labels are too conservative after reranking.
    - Added calibrated typed-decision hybrid with first-stage threshold `0.30`.
    - Calibrated hybrid result: `MRR 0.9774`, `R@1 0.9630`, `R@3 1.0000`,
      `Top-1 support 0.9506`, `Abstain 0.8571`, `Decision 0.9886`,
      `Forbidden supported 0.0000`.

## Blockers / Manual Decisions

- Jina reranker requires `trust_remote_code=True`; this should be a manual
  approval decision.
- Final public SOTA claims still need a larger model-adjudicated benchmark with
  repeated judging and disagreement tracking. Human audit is no longer a
  project blocker per current direction, but curation source must remain
  explicit as `model_adjudicated_*`.
- Extraction-level Modal evaluation is correct but slow. Batch/cached reranker
  scoring is a near-term engineering improvement before larger probes.
