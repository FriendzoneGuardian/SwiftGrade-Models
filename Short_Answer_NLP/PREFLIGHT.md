# ✈️ Module C — NLP Pre-Flight Checklist
**SwiftGrade / Short Answer Scoring Pipeline**
*Run this on the ML-capable machine — not the dev workstation.*

---

## 1 · System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA GTX 1650 (4 GB VRAM) | RTX 3060 / higher |
| **CUDA** | 12.1 | 12.1 or 12.4 |
| **cuDNN** | 8.x | 8.9+ |
| **RAM** | 8 GB | 16 GB |
| **Disk (free)** | 5 GB | 10 GB (models + datasets) |
| **Python** | 3.10 | 3.10 or 3.11 |
| **OS** | Windows 10/11 or Ubuntu 22.04 | Ubuntu 22.04 LTS |

> **GTX 1650 note:** 4 GB VRAM is tight. Use `batch_size=8` max during training.
> DistilBERT fits comfortably. BERT Base is borderline. RoBERTa needs CPU offloading.

---

## 2 · Pre-Flight Sequence

Follow these steps **in order**. Do not skip ahead.

---

### STEP 0 — Verify CUDA Before Anything Else

```powershell
# Check NVIDIA driver and CUDA version
nvidia-smi

# Expected output includes:
# CUDA Version: 12.x
# GPU: NVIDIA GeForce GTX 1650
```

If `nvidia-smi` fails → install/update the NVIDIA driver first.
Driver download: https://www.nvidia.com/Download/index.aspx

---

### STEP 1 — Create a Virtual Environment

```powershell
# Navigate to project root
cd C:\path\to\SwiftGrade-Models

# Create isolated environment
python -m venv .venv

# Activate it (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate it (Windows CMD)
.\.venv\Scripts\activate.bat

# Activate it (Linux/macOS)
source .venv/bin/activate

# Confirm Python version (should be 3.10 or 3.11)
python --version
```

---

### STEP 2 — Install PyTorch WITH CUDA

> ⚠️ **Critical:** Do NOT run plain `pip install torch`.
> That installs the CPU-only build. You must use the CUDA wheel.

```powershell
# For CUDA 12.1 (GTX 1650 / most modern NVIDIA drivers)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU is detected immediately after:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA GeForce GTX 1650
```

If you have CUDA 11.8 instead:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### STEP 3 — Install NLP Dependencies

```powershell
# From the Short_Answer_NLP folder
cd Short_Answer_NLP

pip install -r requirements_aes.txt
```

This installs (in one shot):
- `transformers==4.40.0` — BERT / DistilBERT / RoBERTa
- `tokenizers==0.19.1` — fast tokenizer backend
- `sentence-transformers==2.6.1` — semantic similarity
- `huggingface-hub>=0.20.0` — model cache manager
- `accelerate>=0.20.0` — mixed-precision / multi-GPU
- `nltk>=3.8.0` — tokenizer support + stop words
- `spacy>=3.0.0` — linguistic features
- `gradio>=4.0.0` — Gradio interface
- `scikit-learn`, `xgboost`, `pandas`, `numpy`, etc.

---

### STEP 4 — Download All Models

```powershell
# Still inside Short_Answer_NLP/, with .venv active
python download_models.py
```

This script downloads and caches:

| Model | Size | Purpose |
|-------|------|---------|
| `bert-base-uncased` | ~440 MB | Primary model (Table 2.3.3) |
| `distilbert-base-uncased` | ~268 MB | Lightweight variant |
| `roberta-base` | ~500 MB | Experimental (richer embeddings) |
| `all-mpnet-base-v2` | ~420 MB | Sentence similarity (rubric scoring) |
| `all-MiniLM-L6-v2` | ~80 MB | Lightweight similarity fallback |
| NLTK punkt / stopwords / wordnet | ~30 MB | Tokenizer data |
| spaCy `en_core_web_sm` | ~12 MB | Linguistic pipeline |

**Total download: ~1.75 GB** (one-time, then cached locally)

Cache locations (Windows):
```
C:\Users\<you>\.cache\huggingface\hub\         ← HF models
C:\Users\<you>\.cache\torch\sentence_transformers\  ← ST models
C:\Users\<you>\nltk_data\                      ← NLTK
```

---

### STEP 5 — Verify Everything Loaded

```python
# Run this in Python or a Jupyter cell
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import nltk, spacy

# 1. GPU check
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

# 2. BERT Base
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
mdl = AutoModel.from_pretrained("bert-base-uncased")
print("BERT Base: OK — params:", sum(p.numel() for p in mdl.parameters()) // 1_000_000, "M")

# 3. DistilBERT
tok2 = AutoTokenizer.from_pretrained("distilbert-base-uncased")
mdl2 = AutoModel.from_pretrained("distilbert-base-uncased")
print("DistilBERT: OK — params:", sum(p.numel() for p in mdl2.parameters()) // 1_000_000, "M")

# 4. RoBERTa
tok3 = AutoTokenizer.from_pretrained("roberta-base")
mdl3 = AutoModel.from_pretrained("roberta-base")
print("RoBERTa: OK — params:", sum(p.numel() for p in mdl3.parameters()) // 1_000_000, "M")

# 5. Sentence-Transformers
st = SentenceTransformer("all-mpnet-base-v2")
emb = st.encode(["Test sentence for rubric scoring."])
print("SentenceTransformer: OK — embedding dim:", emb.shape[1])

# 6. NLTK
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stops = set(stopwords.words("english"))
tokens = [w for w in word_tokenize("The student answered the question correctly.") if w not in stops]
print("NLTK filtered tokens:", tokens)

# 7. spaCy
nlp = spacy.load("en_core_web_sm")
doc = nlp("The student answered correctly.")
print("spaCy POS:", [(t.text, t.pos_) for t in doc])

print("\n✅ All systems green — Module C NLP pipeline ready.")
```

---

## 3 · Common Failure Points

| Error | Cause | Fix |
|-------|-------|-----|
| `CUDA out of memory` | VRAM exceeded on GTX 1650 | Reduce `batch_size` to 4 or 8; use DistilBERT |
| `torch.cuda.is_available() = False` | CPU-only PyTorch installed | Reinstall with CUDA wheel (Step 2) |
| `OSError: bert-base-uncased not found` | No internet / proxy block | Run on a machine with HuggingFace access |
| `[E050] Can't find model 'en_core_web_sm'` | spaCy model not downloaded | `python -m spacy download en_core_web_sm` |
| `LookupError: punkt` | NLTK data missing | `python -c "import nltk; nltk.download('all')"` |
| `transformers version mismatch` | Wrong version installed | `pip install transformers==4.40.0 --force-reinstall` |

---

## 4 · Offline / Air-Gapped Option

If the ML machine has **no internet**, pre-download models on a machine that does,
then transfer the cache folder:

```powershell
# On the internet machine — download to a specific folder
python -c "
from transformers import AutoTokenizer, AutoModel
import os
os.makedirs('./offline_models/bert-base-uncased', exist_ok=True)
AutoTokenizer.from_pretrained('bert-base-uncased').save_pretrained('./offline_models/bert-base-uncased')
AutoModel.from_pretrained('bert-base-uncased').save_pretrained('./offline_models/bert-base-uncased')
"

# On the air-gapped machine — load from local path
python -c "
from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained('./offline_models/bert-base-uncased', local_files_only=True)
mdl = AutoModel.from_pretrained('./offline_models/bert-base-uncased', local_files_only=True)
print('Loaded from local cache:', tok)
"
```

Transfer the `./offline_models/` folder via USB drive or network share.

---

## 5 · VRAM Budget (GTX 1650 — 4 GB)

| Model | Inference VRAM | Fine-tune VRAM | Fits 4 GB? |
|-------|---------------|----------------|-----------|
| DistilBERT + Dense | ~600 MB | ~1.5 GB | ✅ Yes |
| BERT Base + Class Head | ~1.3 GB | ~3.5 GB | ✅ Borderline (batch=4) |
| RoBERTa + Attention | ~1.5 GB | ~4.0 GB | ⚠️ CPU offload needed |

For RoBERTa fine-tuning, add this to your training config:
```python
from accelerate import Accelerator
accelerator = Accelerator(mixed_precision="fp16")  # Halves VRAM usage
```

---

## 6 · What Comes After Pre-Flight

Once all models are verified (Step 5 passes), next steps are:

1. **Load ASAP-AES dataset** → `src/data_loader.py`
2. **Run BERT fine-tuning notebook** → `notebooks/Train_Essay_Scoring_Model.ipynb`
3. **Evaluate QWK** → `evaluate.py --model bert-base-uncased`
4. **Compare three variants** per Table 2.3.3
5. **Integrate rubric scoring** via `sentence-transformers`

---

*Pre-flight prepared: May 9, 2026 — SwiftGrade Module C*
