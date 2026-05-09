"""
metrics.py — OCR Evaluation Metrics (Module B)
==============================================
Implements WER, CER, and Drop Rate calculations.
"""

import re
import string

def normalize_text(text):
    """
    Normalizes text for fair comparison:
    - Lowercase
    - Remove punctuation
    - Normalize whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def calculate_wer(reference, hypothesis):
    """Calculates Word Error Rate using jiwer."""
    try:
        import jiwer
        ref_norm = normalize_text(reference)
        hyp_norm = normalize_text(hypothesis)
        
        # If both are empty, error rate is 0
        if not ref_norm and not hyp_norm:
            return 0.0
        # If reference is empty but hyp is not, technically infinite, but we cap at 1.0 or count as 100%
        if not ref_norm:
            return 1.0
            
        return jiwer.wer(ref_norm, hyp_norm)
    except ImportError:
        print("⚠️ jiwer not installed. Cannot calculate WER.")
        return None

def calculate_cer(reference, hypothesis):
    """Calculates Character Error Rate using jiwer."""
    try:
        import jiwer
        ref_norm = normalize_text(reference)
        hyp_norm = normalize_text(hypothesis)
        
        if not ref_norm and not hyp_norm:
            return 0.0
        if not ref_norm:
            return 1.0
            
        return jiwer.cer(ref_norm, hyp_norm)
    except ImportError:
        print("⚠️ jiwer not installed. Cannot calculate CER.")
        return None

def is_dropped(hypothesis):
    """
    Determines if an OCR output constitutes a 'Drop'.
    A drop is an output that is completely unreadable or missing,
    blocking the NLP stage.
    """
    if not hypothesis:
        return True
    
    # If text contains no alphanumeric characters, it's garbage
    if not any(c.isalnum() for c in hypothesis):
        return True
        
    return False

def calculate_drop_rate(hypotheses_list):
    """
    Calculates the Drop Rate across a dataset.
    Drop Rate = (Dropped Documents / Total Documents) * 100
    """
    if not hypotheses_list:
        return 0.0
        
    dropped_count = sum(1 for hyp in hypotheses_list if is_dropped(hyp))
    return (dropped_count / len(hypotheses_list)) * 100.0
