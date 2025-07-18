# Inzwa

**Inzwa turns any phone line into a multilingual, bandwidth-savvy AI hotline.**
Ingest PSTN/SIP calls, detect language, transcribe (ASR), fetch trusted answers
from **Kweli**, apply compliance overlays from **Kinga**, and respond via TTS or
recorded prompts — even on basic feature phones.

- ☎ Telephony ingress adapters (SIP, PSTN gateway, callback)
- 🔊 Streaming LID → ASR pipeline w/ low-bandwidth codecs
- 🧭 IVR / flow builder (menu + LLM fallback)
- 🌍 Language coverage driven by `kinga-core` registry (v1.1)
- 🔐 Emits schema-validated PromptLog events for compliance (Kinga)

**License:** Apache-2.0 (code; codec deps may vary).  
**Contributing:** DCO required. Language contributions welcome — see `/langpacks/`.

> Pilot geos: Nigeria · Kenya · Zimbabwe. Initial langs: en, fr, sw, ha, am, zu, sn, nd.

