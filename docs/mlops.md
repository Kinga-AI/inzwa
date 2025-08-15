## MLOps and Data Lifecycle: Inzwa

### Objectives

- Reproducible data pipelines, traceable model lineage, objective evaluations, safe rollouts.

### Data Sources and Governance

- ASR: self‑recorded Shona speech, Common Voice, academic partners; maintain manifests with speaker meta (age, gender, region) and consent flags.
- LLM: Shona Wikipedia, news, curated Q&A; store provenance and licenses.
- TTS: single‑speaker clean studio recordings; transcripts aligned.
- PII: redact at collection; store only necessary metadata encrypted.

### Tooling

- Versioning: DVC or Git‑LFS for large artifacts; HF Datasets for splits (free hosting on HF Hub)
- Experiment tracking: Weights & Biases (free tier) or MLflow Tracking (self-hosted free)
- Registry: MLflow Model Registry or Hugging Face model repos (free public/private)
- Evaluation: custom scripts + JiWER (WER/CER), MOSNet or subjective MOS, latency probes; LLM-as-judge for response coherence (e.g., via free Grok API or local model)

### Pipelines (Training)

- ASR fine‑tune (Whisper/faster‑whisper)
  - Preprocess to 16 kHz mono WAV
  - Augment (noise, reverb) with sox/torchaudio
  - Train on GPU (Colab/Kaggle/AWS spot) 10–20 h; early stopping on CER

- LLM LoRA
  - Tokenize Shona corpus; build conversational pairs
  - LoRA rank 8–16, lr ~2e‑4, bf16; eval on held‑out QA; safety tests

- TTS VITS‑lite
  - Normalize text; phonemize if needed; alignments check
  - Train 5–10 h, eval MOS, intelligibility, prosody metrics

### Validation and QA Gates

- ASR: CER ≤ 12% on curated test; robust to code‑switching; real-time WER proxy via streaming evals
- LLM: response relevance ≥ 0.8 (human eval or LLM-judge), harmlessness pass rate ≥ 99%, coherence score ≥ 0.85
- TTS: MOS ≥ 3.8; intelligibility ≥ 95%

### Release Process

- Train → Evaluate → Package → Sign → Push to Registry
- Canary rollout to 5% sessions; monitor regressions; automatic rollback on SLO breach

### Monitoring in Production

- Passive eval: opt‑in anonymized metrics (WER proxy via confidence), latency, TTFW, errors
- Active eval: scheduled probes with fixed test utterances

### Bias and Fairness

- Ensure demographic balance in ASR/TTS datasets; measure WER/MOS per subgroup
- Document limitations; publish model cards with intended use and risks


