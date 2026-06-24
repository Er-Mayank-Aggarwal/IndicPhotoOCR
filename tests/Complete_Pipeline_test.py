import os
import json
import signal
from tqdm import tqdm
from IndicPhotoOCR.ocr import OCR 

# Configuration - use environment variable or default to ~/storage
DATASET_BASE = os.environ.get("INDICPHOTOOCR_DATA_PATH", os.path.expanduser("~/storage"))
path = os.path.join(DATASET_BASE, "BSTD_Dataset/12C_images")
output_json_file = 'IndicPhotoOCR_sliced.json'
intermediate_json_file = 'IndicPhotoOCR_sliced.json'
exception_file = 'exceptions.log'
MAX_RETRIES = 2  # Number of times to re-attempt failed images
CPU_TIMEOUT = 60 # Seconds before skipping a stuck image on CPU

# Timeout Exception Class
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Processing timed out")

# Initialize OCR Instances
print("Initializing primary GPU OCR instance...")
ocr_gpu = OCR(identifier_lang="auto", device="cuda", verbose=False)

print("Initializing fallback CPU OCR instance for retries...")
ocr_cpu = OCR(identifier_lang="auto", device="cpu", verbose=False)

results = {}
exception_images = []


# Helper function to process an individual image
def process_single_image(image_name, ocr_instance):
    image_path = os.path.join(path, image_name)
    try:
        # Expecting list of (list of texts, bbox)
        detected_words = ocr_instance.ocr(image_path, batch_size=32 if ocr_instance.device == "cuda" else 0)  
        
        polygon_dict = {}
        polygon_idx = 1

        for text_list in detected_words:
            for text in text_list:
                polygon_dict[f"polygon_{polygon_idx}"] = {
                    "text": text
                }
                polygon_idx += 1

        results[image_name] = polygon_dict
        return True  # Success
    except Exception as e:
        return False  # Failed


# Pass 1: Full Dataset Scan (Using GPU)
all_images = os.listdir(path)
print(f"\n--- Starting Pass 1: Processing {len(all_images)} images on GPU ---")

for image in tqdm(all_images):
    success = process_single_image(image, ocr_gpu)
    if not success:
        exception_images.append(image)

# Save Intermediate Results before moving to CPU
print(f"\n--- Saving intermediate GPU results to {intermediate_json_file} ---")
with open(intermediate_json_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
print(f"Intermediate saved. Current successful count: {len(results)}. Exceptions to retry: {len(exception_images)}")


# Pass 2+: Rerun logic for Exceptions Only (Using CPU)
retry_count = 1
while exception_images and retry_count <= MAX_RETRIES:
    print(f"\n--- Starting Retry Pass {retry_count}: Re-processing {len(exception_images)} exceptions on CPU ---")
    
    still_failing = []
    for image in tqdm(exception_images):
        # Set the signal alarm for the timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(CPU_TIMEOUT)
        
        try:
            # Forwarding the image to the CPU OCR instance
            success = process_single_image(image, ocr_cpu)
            if not success:
                still_failing.append(image)
        except TimeoutException:
            print(f"\n[Timeout] Image {image} got stuck on CPU for >{CPU_TIMEOUT}s. Skipping.")
            still_failing.append(image)
        finally:
            # Disable the alarm regardless of success or failure
            signal.alarm(0)
            
    # Update exception tracker with the remaining failures
    exception_images = still_failing
    retry_count += 1


# File IO Saving
print(f"\n--- Processing Completed. Writing output logs ---")

# Save combined successful results to JSON
with open(output_json_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

# Save final persistent exceptions (if any left after CPU passes)
with open(exception_file, 'w') as f:
    for img in exception_images:
        f.write(img + '\n')

print(f"Final saved results count: {len(results)}")
print(f"Final persistent exceptions count: {len(exception_images)}")
