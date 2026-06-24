
import json
import os
import re
import numpy as np
from collections import defaultdict, Counter
import jiwer
import csv
import string
from word_prf import calculate_metrics

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def normalize_filename(filename):
    return os.path.splitext(filename)[0]

def remove_punctuation(text):
    """Removes all standard punctuation from a string."""
    # Use re.sub to replace any punctuation character with an empty string
    # We escape the punctuation characters in the pattern
    punctuation_pattern = '[' + re.escape(string.punctuation) + ']'
    return re.sub(punctuation_pattern, '', text)

def extract_gt_data(gt_json, pred_json):
    """
    Extracts ground truth data and filters it to include only files
    present in the prediction JSON.
    """
    scriptwise_gt = defaultdict(lambda: defaultdict(list))
    matched_filenames = set()
    
    # Define the list of target languages
    target_languages = {
        "hindi", "english", "assamese", "bengali", "gujarati", "kannada",
        "malayalam", "marathi", "odia", "punjabi", "tamil", "telugu"
    }

    for pred_filename in pred_json:
        image_key = normalize_filename(pred_filename)
        if image_key not in gt_json:
            continue

        matched_filenames.add(image_key)
        gt_entry = gt_json[image_key]
        annotations = gt_entry.get("annotations", {})

        for ann in annotations.values():
            text = ann.get("text", "").strip()

            # --- MODIFICATION: Remove ALL punctuation from GT text ---
            text = remove_punctuation(text)

            lang = ann.get("script_language", "UNK")

            # Skip if the language is not in the target list
            if lang not in target_languages:
                continue

            if text.upper() == "UNK" or lang.upper() == "UNK" or not text:
                continue
            
            if lang=="assamese":
                lang = "bengali"
                
            if lang=="marathi":
                lang = "hindi"
                
            scriptwise_gt[image_key][lang].append(text.lower())
            
    print(f"# image_names match: {len(matched_filenames)}")
    return scriptwise_gt, matched_filenames

def extract_pred_data(pred_json):
    pred_cleaned = {}
    for image_filename, annotations in pred_json.items():
        image_key = normalize_filename(image_filename)
        pred_cleaned[image_key] = {
            # k: v["text"].strip().lower() for k, v in annotations.items() if v.get("text", "").strip()
            k: remove_punctuation(v["text"].strip()).lower() for k, v in annotations.items() if v.get("text", "").strip()
        }
    return pred_cleaned

def detect_language(word):
    # Define Unicode ranges for each language
    unicode_ranges = {
        "english": (0x0000, 0x007F),
        # "assamese": (0x0980, 0x09FF),
        "bengali": (0x0980, 0x09FF),
        "gujarati": (0x0A80, 0x0AFF),
        "hindi": (0x0900, 0x097F),
        # "marathi": (0x0900, 0x097F),
        "kannada": (0x0C80, 0x0CFF),
        "malayalam": (0x0D00, 0x0D7F),
        "odia": (0x0B00, 0x0B7F),
        "punjabi": (0x0A00, 0x0A7F),
        "tamil": (0x0B80, 0x0BFF),
        "telugu": (0x0C00, 0x0C7F)
    }

    # If the word is empty, return unknown
    if not word:
        return "unknown"

    # Check the first character's Unicode range
    first_char_code = ord(word[0])

    for lang, (start, end) in unicode_ranges.items():
        if start <= first_char_code <= end:
            return lang  # Return the detected language

    return "unknown"  # If no match is found

def map_pred_text_using_unicode(list_of_words):
    lang_dict = defaultdict(list)
    for word in list_of_words:
        lang_dict[detect_language(word)].append(word)
        
    return lang_dict

def bag_of_words_wer(gt_text, rec_text, show=None):
    # Split words, sort, and join back
    gt_sorted = " ".join(sorted(gt_text))
    rec_sorted = " ".join(sorted(rec_text))
    if show:
        print("gt:",gt_sorted)
        print("pred:",rec_sorted)

    # Compute standard WER on sorted words
    return jiwer.wer(gt_sorted, rec_sorted), gt_sorted, rec_sorted

def calculate_scriptwise_WER(scriptwise_gt, pred_cleaned):
    language_counts = defaultdict(int)
    language_p = defaultdict(int)
    language_r = defaultdict(int)
    language_f = defaultdict(int)


    # Collect all data rows for CSV writing
    csv_rows = []
    
    for image_key, lang_dict in scriptwise_gt.items():
        pred_texts = list(pred_cleaned.get(image_key, {}).values())
        # print(image_key)
        # print(lang_dict)
        # print(pred_texts)
        pred_dict = map_pred_text_using_unicode(pred_texts)
        # print(pred_dict)
        # break

        for lang, gt_texts in lang_dict.items():
            gt_set = list(gt_texts)
            pred_set = list(pred_dict[lang])
            # matched = [word for word in pred_texts if word in gt_set]
            
            result = calculate_metrics(gt_set, pred_set)
            # if image_key == "L_image_35":    
            #     # print(gt_set)
            #     # print(pred_set)
            #     wer, _, _ = bag_of_words_wer(gt_set, pred_set, show=True)
            #     print(wer)
            # wer = min(1, wer)
            language_counts[lang] += 1
            language_p[lang] += result["precision"]
            language_r[lang] += result["recall"]
            language_f[lang] += result["f1_score"]
                        # Store data for this image-script pair
            csv_rows.append({
                "image_key": image_key,
                "script": lang,
                "gt_sorted": sorted(gt_set),
                "pred_sorted": sorted(pred_set),
                "PRF": {round(result["precision"], 2), round(result["recall"], 2), round(result["f1_score"], 2)}
            })
        # break
    #    # Write all results into a CSV
    # with open("indicphotoOCR_PRF.csv", mode="w", newline="", encoding="utf-8") as csvfile:
    #     fieldnames = ["image_key", "script", "gt_sorted", "pred_sorted", "PRF"]
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     writer.writerows(csv_rows)
        
    print("Language, Precision, Recall, F1-score")
    for lang in sorted(language_counts):
        total = language_counts[lang]
        total_p = language_p[lang]
        total_r = language_r[lang]
        total_f = language_f[lang]
        # acc = (total_wer / total) * 100 if total > 0 else 0
        avg_p = (total_p / total)
        avg_r = (total_r / total)
        avg_f = (total_f / total)
        
        print(f"{lang}, {avg_p:.2f}, {avg_r:.2f}, {avg_f:.2f}")
    print(f"Average Precision: {sum(language_p.values()) / sum(language_counts.values()):.2f}")
    print(f"Average Recall: {sum(language_r.values()) / sum(language_counts.values()):.2f}")
    print(f"Average F1-score: {sum(language_f.values()) / sum(language_counts.values()):.2f}")

gt_path = "BSTD_v17.57.json"
pred_path = "Your_JSON_Path_Here"

gt_json = load_json(gt_path)
pred_json = load_json(pred_path)
scriptwise_gt, matched_filenames = extract_gt_data(gt_json, pred_json)
pred_cleaned = extract_pred_data(pred_json)
calculate_scriptwise_WER(scriptwise_gt, pred_cleaned)


