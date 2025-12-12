import os
import numpy as np
import cv2
import csv


# =====================  IMAGE LOADER  ==========================

def load_thermal_image(input_path):
    ext = input_path.split('.')[-1].lower()
    if ext == "csv":
        rows = []
        with open(input_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append([float(val) for val in row])
        data = np.array(rows, dtype=np.float32)
        data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
        gray = np.clip(data, 0, 255).astype(np.uint8)
        color_img = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    else:
        color_img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if color_img is None:
            raise IOError(f"Error reading {input_path}")
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return color_img, gray


# ===================  ROBUST PIPELINE  =========================

def preprocess_image(gray):
    """
    Step 1: Enhance Contrast
    """
    # Bilateral Filter to smooth noise but keep edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # CLAHE to make the warm objects pop against the floor
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    return enhanced


def get_binary_mask(enhanced_gray):
    """
    Step 2: Binary Mask & Hole Filling
    """
    # Otsu Thresholding
    _, binary = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Fill internal holes (donuts)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(binary)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

    return filled_mask


def split_touching_mice(binary_mask):
    """
    Step 3: Watershed Separation
    """
    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

    # Smooth distance map to avoid over-segmenting single mice
    dist_smooth = cv2.GaussianBlur(dist, (5, 5), 0)
    dist_norm = cv2.normalize(dist_smooth, None, 0, 1.0, cv2.NORM_MINMAX)

    # Strict threshold for markers (only the hottest centers)
    _, sure_fg = cv2.threshold(dist_norm, 0.35, 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary_mask, kernel, iterations=3)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    ws_input = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(ws_input, markers)

    final_labels = markers.copy()
    final_labels[final_labels <= 1] = 0

    return final_labels


def filter_false_positives(label_img, intensity_image, min_size=800, min_peak_brightness=130):
    """
    Step 4: Advanced Filtering
    1. Removes blobs that are too small (Noise)
    2. Removes blobs that are 'warm' but not 'hot' (Floor spots)
    """
    final_labels = np.zeros_like(label_img)
    unique_labels = np.unique(label_img)
    new_id = 1

    for lab in unique_labels:
        if lab == 0: continue

        mask = (label_img == lab)

        # 1. Size Check
        area = np.sum(mask)
        if area < min_size:
            continue

        # 2. Intensity (Heat) Check
        # We extract the pixel values of the original/enhanced image for this blob
        # If the brightest pixel in the blob is too dim, it's just a warm floor spot.
        blob_pixels = intensity_image[mask]
        max_brightness = np.max(blob_pixels)

        # Threshold 130 is an estimate: Green is usually ~100-140, Red/Yellow is >180
        if max_brightness < min_peak_brightness:
            print(f"Removed label {lab} (Area: {area}, Max Brightness: {max_brightness}) -> Too Cold")
            continue

        final_labels[mask] = new_id
        new_id += 1

    return final_labels


# =====================  VISUALIZATION  =========================

def label_to_color_and_numbers(label_img):
    H, W = label_img.shape
    colored = np.zeros((H, W, 3), dtype=np.uint8)
    unique_labels = np.unique(label_img)

    rng = np.random.default_rng(42)
    colors = {lab: rng.integers(50, 255, size=3).tolist() for lab in unique_labels if lab != 0}

    for lab in unique_labels:
        if lab == 0: continue
        colored[label_img == lab] = colors[lab]

        ys, xs = np.where(label_img == lab)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        cv2.putText(colored, str(lab), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return colored


def count_regions(label_img):
    unique = np.unique(label_img)
    return len(unique) - (1 if 0 in unique else 0)


# =========================   MAIN    ===========================

def main():

    input_dir = r"/Users/krist/Documents/IP_project/data/thermal_mouse/src/models/model_detectron/output_images_panoptic"
    output_dir = r"/Users/krist/Documents/IP_project/RicardosScript/results"


    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".csv"))])
    print(f"Found {len(image_files)} images. Processing with False Positive removal...")

    for i, fname in enumerate(image_files):
        img_path = os.path.join(input_dir, fname)
        base_name = os.path.splitext(fname)[0]

        try:
            color_img, gray = load_thermal_image(img_path)
        except:
            continue

        # 1. Preprocess (Enhance)
        enhanced = preprocess_image(gray)

        # 2. Binary Mask
        binary_mask = get_binary_mask(enhanced)

        # 3. Watershed Separation
        labels = split_touching_mice(binary_mask)

        # 4. Filter Noise AND Cold Spots
        # We pass 'enhanced' here to check the brightness of the blob
        final_labels = filter_false_positives(labels, enhanced, min_size=800, min_peak_brightness=150)

        # Save outputs
        result_visual = label_to_color_and_numbers(final_labels)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_result.png"), result_visual)

        print(f"[{i + 1}] {fname}: {count_regions(final_labels)} mice detected.")


if __name__ == "__main__":
    main()