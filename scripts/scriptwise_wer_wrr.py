import json
import os
import re
import string
from collections import defaultdict
import jiwer

# --- CONFIGURATION ---
TARGET_LANGUAGES = {
    "hindi", "english", "bengali", "gujarati", "kannada",
    "malayalam", "marathi", "odia", "punjabi", "tamil", "telugu"
}

# Mapping for scripts that share Unicode blocks or evaluation categories
SCRIPT_MAPPING = {
    "assamese": "bengali",
    "marathi": "hindi"
}

UNICODE_RANGES = {
    "english": (0x0000, 0x007F),
    "bengali": (0x0980, 0x09FF),
    "gujarati": (0x0A80, 0x0AFF),
    "hindi": (0x0900, 0x097F),
    "kannada": (0x0C80, 0x0CFF),
    "malayalam": (0x0D00, 0x0D7F),
    "odia": (0x0B00, 0x0B7F),
    "punjabi": (0x0A00, 0x0A7F),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F)
}

PUNC_RE = re.compile(f'[{re.escape(string.punctuation)}]')

# --- UTILS ---

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_text(text):
    """Removes punctuation, strips whitespace, and lowercases."""
    if not text:
        return ""
    text = PUNC_RE.sub('', text)
    return text.strip().lower()

def normalize_filename(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def detect_language(word):
    if not word:
        return "unknown"
    first_char_code = ord(word[0])
    for lang, (start, end) in UNICODE_RANGES.items():
        if start <= first_char_code <= end:
            return SCRIPT_MAPPING.get(lang, lang)
    return "unknown"

# --- CORE LOGIC ---

def process_data(gt_json, pred_json):
    """Filters and cleans GT and Pred data in one pass."""
    pred_keys = {normalize_filename(k) for k in pred_json.keys()}
    scriptwise_gt = defaultdict(lambda: defaultdict(list))
    pred_cleaned = {}

    # Clean Prediction Data
    for filename, annotations in pred_json.items():
        key = normalize_filename(filename)
        pred_cleaned[key] = [clean_text(v.get("text", "")) for v in annotations.values() if v.get("text")]

    # Clean Ground Truth Data (Only for matched files)
    for key, gt_entry in gt_json.items():
        if key not in pred_keys:
            continue
        
        for ann in gt_entry.get("annotations", {}).values():
            text = clean_text(ann.get("text", ""))
            lang = ann.get("script_language", "UNK").lower()
            lang = SCRIPT_MAPPING.get(lang, lang)

            if not text or text == "unk" or lang not in TARGET_LANGUAGES:
                continue
            
            scriptwise_gt[key][lang].append(text)
    
    return scriptwise_gt, pred_cleaned

def calculate_metrics(scriptwise_gt, pred_cleaned):
    lang_stats = defaultdict(lambda: {"total_wer": 0, "count": 0, "total_words": 0})

    for img_key, lang_dict in scriptwise_gt.items():
        # Group predictions by detected language
        pred_texts = pred_cleaned.get(img_key, [])
        pred_by_lang = defaultdict(list)
        for word in pred_texts:
            pred_by_lang[detect_language(word)].append(word)

        for lang, gt_list in lang_dict.items():
            pred_list = pred_by_lang[lang]
            
            # Bag of words WER
            gt_str = " ".join(sorted(gt_list))
            pred_str = " ".join(sorted(pred_list))
            
            # Calculate WER (capped at 1.0)
            wer = min(1.0, jiwer.wer(gt_str, pred_str)) if gt_str else 0
            
            lang_stats[lang]["total_wer"] += wer
            lang_stats[lang]["count"] += 1
            lang_stats[lang]["total_words"] += len(gt_list)

    print_results(lang_stats)

def print_results(stats):
    print(f"\n{'Language':<12} | {'WER':<6} | {'GT Words':<10}")
    print("-" * 35)
    for lang in sorted(stats.keys()):
        s = stats[lang]
        avg_wer = s["total_wer"] / s["count"] if s["count"] > 0 else 0
        print(f"{lang.capitalize():<12} | {avg_wer:.2f}   | {s['total_words']:<10}")

    #print WRR also
    print(f"\n{'Language':<12} | {'WRR':<6}")
    print("-" * 25)
    for lang in sorted(stats.keys()):
        s = stats[lang]
        avg_wer = s["total_wer"] / s["count"] if s["count"] > 0 else 0
        wrr = (1 - avg_wer) * 100
        print(f"{lang.capitalize():<12} | {wrr:.2f}%")

    print(f"\nOverall Average WER: {sum(s['total_wer'] for s in stats.values()) / sum(s['count'] for s in stats.values()):.2f}")
    print(f"Overall Average WRR: {(1 - (sum(s['total_wer'] for s in stats.values()) / sum(s['count'] for s in stats.values()))) * 100:.2f}%")

# --- MAIN ---

if __name__ == "__main__":
    # Update these paths as needed
    GT_PATH = "BSTD_v17.57.json"
    PRED_PATH = "IndicPhotoOCR_sliced_v2m.json"

    gt_data = load_json(GT_PATH)
    pred_data = load_json(PRED_PATH)

    gt_filtered, pred_filtered = process_data(gt_data, pred_data)
    calculate_metrics(gt_filtered, pred_filtered)