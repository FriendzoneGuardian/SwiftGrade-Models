"""
download_models.py — SwiftGrade NLP Model Downloader
=====================================================
Downloads and caches all transformer models required for Module C:
  - bert-base-uncased        (primary model, 440 MB)
  - distilbert-base-uncased  (lightweight variant, 268 MB)
  - roberta-base             (experimental, 500 MB)
  - sentence-transformers    (semantic similarity, ~420 MB)
  - NLTK corpora             (punkt, stopwords, averaged_perceptron_tagger)
  - spaCy en_core_web_sm     (tokenizer support, ~12 MB)

Run once before training:
    python download_models.py

All models are cached in:
    ~/.cache/huggingface/hub/     (HF models)
    ~/.cache/torch/sentence_transformers/  (sentence-transformers)
    ~/nltk_data/                  (NLTK)
"""

import sys
import os
import shutil


# ─── Disk Space Guard ──────────────────────────────────────────────────────────

def check_disk_space(required_gb: float = 2.5):
    """Warn if free disk space is less than required_gb."""
    cache_drive = os.path.expanduser("~")
    total, used, free = shutil.disk_usage(cache_drive)
    free_gb = free / (1024 ** 3)
    print(f"  💾 Free disk space: {free_gb:.1f} GB  (need ~{required_gb} GB)")
    if free_gb < required_gb:
        print(f"  ⚠️  WARNING: Low disk space. Downloads may fail.")
    else:
        print(f"  ✅ Disk space OK")
    return free_gb


# ─── Section 1: Hugging Face Transformers ─────────────────────────────────────

def download_hf_model(model_name: str, model_type: str = "both"):
    """
    Download a Hugging Face model + tokenizer and verify it loads.
    model_type: 'tokenizer', 'model', or 'both'
    """
    from transformers import AutoTokenizer, AutoModel
    print(f"\n  📥 Downloading: {model_name}")
    try:
        if model_type in ("both", "tokenizer"):
            print(f"     → Tokenizer...", end=" ", flush=True)
            AutoTokenizer.from_pretrained(model_name)
            print("✅")
        if model_type in ("both", "model"):
            print(f"     → Model weights...", end=" ", flush=True)
            AutoModel.from_pretrained(model_name)
            print("✅")
        print(f"  ✔  {model_name} ready.")
        return True
    except Exception as e:
        print(f"\n  ❌ Failed to download {model_name}: {e}")
        return False


def download_transformer_models():
    """Download all three BERT-family models defined in Table 2.3.3."""
    print("\n" + "=" * 60)
    print("  STEP 1 — Hugging Face Transformer Models")
    print("=" * 60)
    print("  Target models (Table 2.3.3):")
    print("    • bert-base-uncased        ~440 MB  [PRIMARY]")
    print("    • distilbert-base-uncased  ~268 MB  [LIGHTWEIGHT]")
    print("    • roberta-base             ~500 MB  [EXPERIMENTAL]")

    models = [
        "bert-base-uncased",
        "distilbert-base-uncased",
        "roberta-base",
    ]

    results = {}
    for model_name in models:
        results[model_name] = download_hf_model(model_name, model_type="both")

    return results


# ─── Section 2: Sentence-Transformers ─────────────────────────────────────────

def download_sentence_transformers():
    """
    Download the sentence-transformers model for semantic similarity scoring.
    Uses all-mpnet-base-v2 as the default (best quality/size tradeoff).
    Also downloads all-MiniLM-L6-v2 as a lightweight fallback.
    """
    print("\n" + "=" * 60)
    print("  STEP 2 — Sentence-Transformers (Semantic Similarity)")
    print("=" * 60)

    from sentence_transformers import SentenceTransformer

    # Primary: best quality for rubric scoring
    st_models = [
        ("all-mpnet-base-v2",  "Primary — best quality for rubric similarity   ~420 MB"),
        ("all-MiniLM-L6-v2",  "Fallback — lightweight, faster inference        ~80 MB"),
    ]

    results = {}
    for model_name, desc in st_models:
        print(f"\n  📥 {desc}")
        print(f"     Downloading: {model_name}...", end=" ", flush=True)
        try:
            SentenceTransformer(model_name)
            print("✅")
            results[model_name] = True
        except Exception as e:
            print(f"\n  ❌ Failed: {e}")
            results[model_name] = False

    return results


# ─── Section 3: NLTK Corpora ──────────────────────────────────────────────────

def download_nltk_data():
    """
    Download NLTK corpora needed for tokenization and stop word removal.
    These are lightweight (~30 MB total).
    """
    print("\n" + "=" * 60)
    print("  STEP 3 — NLTK Corpora (Tokenizer Support)")
    print("=" * 60)

    import nltk

    corpora = [
        ("punkt",                       "Sentence/word tokenizer"),
        ("punkt_tab",                   "Punkt tokenizer tables (NLTK 3.8+)"),
        ("stopwords",                   "English stop words (a, an, the, but…)"),
        ("averaged_perceptron_tagger",  "POS tagger"),
        ("averaged_perceptron_tagger_eng", "POS tagger (English, NLTK 3.9+)"),
        ("wordnet",                     "WordNet lexical database"),
        ("omw-1.4",                     "Open Multilingual WordNet"),
    ]

    results = {}
    for corpus_id, desc in corpora:
        print(f"  📥 {corpus_id:40s} {desc}...", end=" ", flush=True)
        try:
            nltk.download(corpus_id, quiet=True)
            print("✅")
            results[corpus_id] = True
        except Exception as e:
            print(f"❌ ({e})")
            results[corpus_id] = False

    return results


# ─── Section 4: spaCy Model ───────────────────────────────────────────────────

def download_spacy_model():
    """
    Download the spaCy English model for linguistic feature extraction.
    en_core_web_sm is ~12 MB and sufficient for tokenization.
    en_core_web_trf uses a BERT backbone (~500 MB) — optional.
    """
    print("\n" + "=" * 60)
    print("  STEP 4 — spaCy Language Model")
    print("=" * 60)

    import subprocess

    spacy_models = [
        ("en_core_web_sm",  "Small English pipeline (dep, NER, POS)  ~12 MB  [REQUIRED]"),
        # ("en_core_web_trf", "Transformer-based pipeline              ~500 MB [OPTIONAL]"),
    ]

    results = {}
    for model_name, desc in spacy_models:
        print(f"  📥 {desc}")
        print(f"     Running: python -m spacy download {model_name}...", end=" ", flush=True)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "spacy", "download", model_name],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                print("✅")
                results[model_name] = True
            else:
                print(f"❌\n     {result.stderr.strip()}")
                results[model_name] = False
        except Exception as e:
            print(f"❌ ({e})")
            results[model_name] = False

    return results


# ─── Section 5: Verification ──────────────────────────────────────────────────

def verify_imports():
    """Quick import verification — confirms all libraries are accessible."""
    print("\n" + "=" * 60)
    print("  STEP 5 — Import Verification")
    print("=" * 60)

    checks = [
        ("transformers",        "Hugging Face Transformers v4.40.0+"),
        ("tokenizers",          "Hugging Face Tokenizers v0.19.1+"),
        ("sentence_transformers","Sentence-Transformers v2.6.1+"),
        ("torch",               "PyTorch (GPU backend)"),
        ("nltk",                "NLTK"),
        ("spacy",               "spaCy"),
        ("sklearn",             "scikit-learn"),
    ]

    all_ok = True
    for module_name, desc in checks:
        print(f"  {'import ' + module_name:35s}", end=" ")
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "?")
            print(f"✅  v{version}")
        except ImportError as e:
            print(f"❌  NOT INSTALLED — {e}")
            all_ok = False

    return all_ok


def print_cache_locations():
    """Print where models are cached on this machine."""
    import torch
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    st_cache = os.path.expanduser("~/.cache/torch/sentence_transformers")
    nltk_data = os.path.expanduser("~/nltk_data")

    print("\n" + "=" * 60)
    print("  📁 Model Cache Locations")
    print("=" * 60)

    locations = [
        ("Hugging Face Hub",        hf_cache),
        ("Sentence-Transformers",   st_cache),
        ("NLTK Data",               nltk_data),
    ]

    for label, path in locations:
        exists = "✅ exists" if os.path.exists(path) else "⚠️  not yet created"
        print(f"  {label:25s} → {path}")
        print(f"  {'':25s}   [{exists}]")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  SwiftGrade — Module C NLP Model Downloader")
    print("  Table 2.3.3: BERT / DistilBERT / RoBERTa Pipeline")
    print("=" * 60)
    print(f"\n  Python: {sys.version}")
    print(f"  Working dir: {os.getcwd()}")

    # Disk space check (~2.0 GB needed for all models)
    print("\n  Checking disk space...")
    check_disk_space(required_gb=2.0)

    # GPU availability check
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"  🎮 GPU: {gpu_name}  ({vram:.1f} GB VRAM)")
        else:
            print("  ⚠️  No GPU detected — will use CPU (slower but functional)")
    except ImportError:
        print("  ⚠️  PyTorch not installed — install it first!")

    print_cache_locations()

    # Run all downloads
    results = {}

    # Step 1: Transformer models
    results["transformers"] = download_transformer_models()

    # Step 2: Sentence-transformers
    results["sentence_transformers"] = download_sentence_transformers()

    # Step 3: NLTK
    results["nltk"] = download_nltk_data()

    # Step 4: spaCy
    results["spacy"] = download_spacy_model()

    # Step 5: Verify imports
    all_imports_ok = verify_imports()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  DOWNLOAD SUMMARY")
    print("=" * 60)

    all_ok = True

    for category, cat_results in results.items():
        if isinstance(cat_results, dict):
            for model, ok in cat_results.items():
                status = "✅ Ready" if ok else "❌ FAILED"
                print(f"  {status}  {model}")
                if not ok:
                    all_ok = False
        else:
            status = "✅ OK" if cat_results else "❌ FAILED"
            print(f"  {status}  {category}")

    print()
    if all_ok and all_imports_ok:
        print("  🎉 All models downloaded and verified!")
        print("  You can now run training notebooks.")
    else:
        print("  ⚠️  Some downloads failed. Check errors above.")
        print("  Re-run this script to retry failed downloads.")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
