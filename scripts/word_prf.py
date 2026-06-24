import re

def calculate_metrics(ground_truth_words, predicted_words):
    """
    Calculates Precision, Recall, and F1 Score based on set intersection
    of ground truth and predicted words.

    It handles case where lists may contain duplicates by converting inputs
    to sets first, ensuring each unique word counts only once.

    Args:
        ground_truth_words (list): List of words that should have been predicted (Ground Truth).
        predicted_words (list): List of words that were actually predicted (Prediction).

    Returns:
        dict: A dictionary containing 'precision', 'recall', and 'f1_score'.
    """

    # 1. Convert lists to sets for efficient intersection and difference calculation.
    gt_set = set(ground_truth_words)
    pred_set = set(predicted_words)

    # 2. Calculate True Positives, False Positives, and False Negatives sizes
    
    # True Positives (TP): Words in both sets (correctly predicted).
    true_positives = len(gt_set.intersection(pred_set))

    # False Positives (FP): Words in prediction but not in ground truth (incorrect predictions).
    false_positives = len(pred_set.difference(gt_set))

    # False Negatives (FN): Words in ground truth but not in prediction (missed predictions).
    false_negatives = len(gt_set.difference(pred_set))

    # print(f"--- Intermediate Results ---")
    # print(f"True Positives (TP): {true_positives}")
    # print(f"False Positives (FP): {false_positives}")
    # print(f"False Negatives (FN): {false_negatives}")
    # print("----------------------------")

    # 3. Calculate Precision
    # Precision = TP / (TP + FP) -> proportion of positive predictions that were correct
    try:
        precision = true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        # If no words were predicted, precision is 0.
        precision = 0.0

    # 4. Calculate Recall
    # Recall = TP / (TP + FN) -> proportion of actual positives that were correctly identified
    try:
        recall = true_positives / (true_positives + false_negatives)
    except ZeroDivisionError:
        # If no ground truth words exist, recall is technically 1 (vacuously true)
        # or defined as 0. We'll use 0.0 as it's common practice when evaluating.
        recall = 0.0

    # 5. Calculate F1 Score
    # F1 Score = 2 * (Precision * Recall) / (Precision + Recall) -> harmonic mean of P & R
    try:
        f1_score = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        # If both precision and recall are 0, F1 is 0.
        f1_score = 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

# # --- Example Usage ---

# # Example 1: High Precision, Low Recall (Predicted only one correct word, missed many others)
# gt_1 = ["apple", "banana", "cherry", "date"]
# pred_1 = ["apple", "grape"]
# print("--- Example 1: High Precision, Low Recall ---")
# results_1 = calculate_metrics(gt_1, pred_1)
# print(f"Ground Truth: {gt_1}")
# print(f"Prediction:   {pred_1}")
# print(f"Results: {results_1}\n")
# # TP=1 (apple), FP=1 (grape), FN=3 (banana, cherry, date)
# # P = 1 / (1+1) = 0.5
# # R = 1 / (1+3) = 0.25

# # Example 2: Low Precision, High Recall (Predicted many correct words, but also many wrong ones)
# gt_2 = ["car", "bus", "train"]
# pred_2 = ["car", "bus", "train", "plane", "bike"]
# print("--- Example 2: Low Precision, High Recall ---")
# results_2 = calculate_metrics(gt_2, pred_2)
# print(f"Ground Truth: {gt_2}")
# print(f"Prediction:   {pred_2}")
# print(f"Results: {results_2}\n")
# # TP=3 (car, bus, train), FP=2 (plane, bike), FN=0
# # P = 3 / (3+2) = 0.6
# # R = 3 / (3+0) = 1.0

# # Example 3: Perfect Match
# gt_3 = ["data", "science"]
# pred_3 = ["data", "science"]
# print("--- Example 3: Perfect Match ---")
# results_3 = calculate_metrics(gt_3, pred_3)
# print(f"Ground Truth: {gt_3}")
# print(f"Prediction:   {pred_3}")
# print(f"Results: {results_3}\n")
# # TP=2, FP=0, FN=0
# # P=1.0, R=1.0, F1=1.0

# # Example 4: No Match
# gt_4 = ["dog"]
# pred_4 = ["cat"]
# print("--- Example 4: No Match ---")
# results_4 = calculate_metrics(gt_4, pred_4)
# print(f"Ground Truth: {gt_4}")
# print(f"Prediction:   {pred_4}")
# print(f"Results: {results_4}\n")
# # TP=0, FP=1, FN=1
# # P=0.0, R=0.0, F1=0.0

# # Example 5: Handling empty ground truth (ZeroDivisionError in Recall denominator handled)
# gt_5 = []
# pred_5 = ["cat", "dog"]
# print("--- Example 5: Empty Ground Truth ---")
# results_5 = calculate_metrics(gt_5, pred_5)
# print(f"Ground Truth: {gt_5}")
# print(f"Prediction:   {pred_5}")
# print(f"Results: {results_5}\n")
# # TP=0, FP=2, FN=0
# # P = 0 / (0+2) = 0.0
# # R = 0 / (0+0) = 0.0 (Handled by try/except block)

# # Example 6: Handling empty prediction (ZeroDivisionError in Precision denominator handled)
# gt_6 = ["cat", "dog"]
# pred_6 = []
# print("--- Example 6: Empty Prediction ---")
# results_6 = calculate_metrics(gt_6, pred_6)
# print(f"Ground Truth: {gt_6}")
# print(f"Prediction:   {pred_6}")
# print(f"Results: {results_6}\n")
# TP=0, FP=0, FN=2
# P = 0 / (0+0) = 0.0 (Handled by try/except block)
# R = 0 / (0+2) = 0.0