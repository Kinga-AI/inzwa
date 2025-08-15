## Data Strategy and Governance

### Sources

- ASR: self‑recorded Shona, Common Voice, universities (UZ, NUST); crowdsourced via social media/GitHub (free)
- LLM: Shona Wikipedia, Kwayedza, Herald Shona, curated Q&A; synthetic pairs via free ChatGPT or local LLM
- TTS: single‑speaker studio recordings (5–10 h), clean transcripts; bootstrap with free synthetic voices

### Schemas

- ASR sample
  - `audio_path`, `sample_rate`, `duration_s`, `transcript`, `speaker_id`, `consent`, `split`
- LLM pair
  - `id`, `prompt_sn`, `response_sn`, `domain`, `safety_tags`
- TTS item
  - `audio_path`, `text_sn`, `phonemes` (optional), `speaker_id`, `quality_score`

### Processing

- Normalize audio to 16 kHz mono WAV; trim silence; denoise (rnnoise, free)
- Text normalization for Shona: punctuation, numerals, diacritics policy
- Augmentation for ASR: noise, speed perturb, RIR (free torchaudio/sox)
- Synthetic data: Use free TTS (e.g., gTTS) to generate audio from text, loop with ASR for self-supervised pairs

### Labeling and QA

- Double‑pass transcription verification; inter‑annotator agreement checks (use free LabelStudio for annotation)
- Prosody review for TTS; reject noisy samples

### Licensing and Consent

- Track licenses per sample; only permissive content
- Signed speaker consent for TTS; allow revocation

### Storage and Versioning

- DVC/Hub datasets; immutable versions; manifest JSON/CSV
- Access controls; encryption at rest for any opt‑in user data


