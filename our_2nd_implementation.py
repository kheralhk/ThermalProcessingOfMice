import os
import numpy as np
import cv2
import csv


# ===============================================================
# =====================  LOADER IMMAGINI  =======================
# ===============================================================

def read_csv_image(csv_file_path):
    rows = []
    with open(csv_file_path, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            rows.append([float(val) for val in row])
    data = np.array(rows, dtype=np.float32)
    data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    data = np.clip(data, 0, 255).astype(np.uint8)
    return data


def load_thermal_image(input_path):
    ext = input_path.split('.')[-1].lower()

    if ext == "csv":
        print(f"Reading images from CSV: {input_path}")
        gray = read_csv_image(input_path)
        color_img = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    else:
        print(f"Reading images from PNG/JPG: {input_path}")
        color_img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if color_img is None:
            raise IOError("Errore: impossibile leggere l'immagine.")
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    return color_img, gray


# ===============================================================
# =====================  PRE-PROCESSING  ========================
# ===============================================================

def thermal_denoise(gray):
    """Denoising per immagini termiche (median + bilateral)."""
    den = cv2.medianBlur(gray, 3)
    den = cv2.bilateralFilter(den, 5, 50, 5)
    return den


def gradient_aware_morphology(binary, gray):
    bin_uint = (binary.astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    opened = cv2.morphologyEx(bin_uint, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    # gradiente Sobel
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    grad_mag = cv2.magnitude(gx, gy)
    grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, edges = cv2.threshold(grad_norm, 40, 255, cv2.THRESH_BINARY)
    edges_dil = cv2.dilate(edges, kernel)

    refined = cv2.bitwise_or(closed, edges_dil)

    return refined > 0


def split_overlapping_mice(binary):
    bin_uint = (binary.astype(np.uint8)) * 255

    dist = cv2.distanceTransform(bin_uint, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    _, sure_fg = cv2.threshold(dist_norm, 0.4, 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(bin_uint, kernel, 3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0

    ws_img = cv2.cvtColor(bin_uint, cv2.COLOR_GRAY2BGR)
    cv2.watershed(ws_img, markers)

    markers[markers <= 1] = 0
    return markers


# ===============================================================
# =====================   BASELINE TOOLS   ======================
# ===============================================================

def to_binary_from_gray(gray):
    """Otsu threshold classico."""
    _, bin_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bin_otsu > 0


def two_pass(binary_img, connectivity=8):
    """Classical 2-pass CCA."""
    H, W = binary_img.shape
    labels = np.zeros((H, W), dtype=np.int32)
    curr_label = 1
    parent = {curr_label: curr_label}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    offsets = [(-1, 0), (0, -1)] if connectivity == 4 else [(-1, 0), (0, -1), (-1, -1), (-1, 1)]

    for r in range(H):
        for c in range(W):
            if binary_img[r, c]:
                neighbor_labels = []
                for dr, dc in offsets:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and labels[rr, cc] != 0:
                        neighbor_labels.append(labels[rr, cc])
                if not neighbor_labels:
                    parent[curr_label] = curr_label
                    labels[r, c] = curr_label
                    curr_label += 1
                else:
                    smallest = min(neighbor_labels)
                    labels[r, c] = smallest
                    for nl in neighbor_labels:
                        union(smallest, nl)

    label_map = {}
    new_label = 1

    for r in range(H):
        for c in range(W):
            if labels[r, c] != 0:
                root = find(labels[r, c])
                if root not in label_map:
                    label_map[root] = new_label
                    new_label += 1
                labels[r, c] = label_map[root]

    return labels


def contour(binary_img):
    """Contour-based CCA."""
    binary_uint8 = (binary_img.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    label_image = np.zeros(binary_img.shape, dtype=np.int32)
    label_id = 1
    for cnt in contours:
        cv2.drawContours(label_image, [cnt], -1, label_id, -1)
        label_id += 1

    return label_image


def remove_small_regions(label_img, min_size=1000):
    """Filtro regioni piccole."""
    final_labels = np.zeros_like(label_img)
    unique_labels = np.unique(label_img)

    new_id = 1
    for lab in unique_labels:
        if lab == 0:
            continue
        mask = (label_img == lab)
        if np.sum(mask) >= min_size:
            final_labels[mask] = new_id
            new_id += 1

    return final_labels


def count_regions(label_img):
    return len(np.unique(label_img)) - (1 if 0 in np.unique(label_img) else 0)


def apply_mask_on_color(color_img, label_img):
    mask = (label_img > 0).astype(np.uint8)
    masked = cv2.bitwise_and(color_img, color_img, mask=mask)
    return masked


def label_to_color_and_numbers(label_img):
    H, W = label_img.shape
    colored = np.zeros((H, W, 3), dtype=np.uint8)
    unique_labels = np.unique(label_img)

    rng = np.random.default_rng(42)
    colors = {lab: rng.integers(0, 255, size=3) for lab in unique_labels if lab != 0}

    for lab in unique_labels:
        if lab == 0:
            continue
        colored[label_img == lab] = colors[lab]

    for lab in unique_labels:
        if lab == 0:
            continue
        ys, xs = np.where(label_img == lab)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        cv2.putText(colored, str(lab), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return colored


# ===============================================================
# =========================   MAIN    ===========================
# ===============================================================

def main():
    img_path = r"/Users/krist/Documents/IP_project/data/thermal_mouse/src/models/model_detectron/output_images_panoptic/combined_3.png"
    output_dir = r"/Users/krist/Documents/IP_project/RicardosScript/result_riccardos_improvement"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    color_img, gray = load_thermal_image(img_path)

    # ===============================================================
    # BASELINE
    # ===============================================================
    binary_baseline = to_binary_from_gray(gray)

    cv2.imwrite(os.path.join(output_dir, f"{base_name}_binary_baseline.png"),
                (binary_baseline.astype(np.uint8)) * 255)

    # Two-pass baseline
    tp = two_pass(binary_baseline)
    tp = remove_small_regions(tp)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_two_pass_baseline.png"),
                apply_mask_on_color(color_img, tp))
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_two_pass_baseline_colored.png"),
                label_to_color_and_numbers(tp))

    # Contour baseline
    ct = contour(binary_baseline)
    ct = remove_small_regions(ct)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_contour_baseline.png"),
                apply_mask_on_color(color_img, ct))
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_contour_baseline_colored.png"),
                label_to_color_and_numbers(ct))


    # 1) denoising termico
    den = thermal_denoise(gray)

    # 2) Otsu su denoised
    bin_otsu = to_binary_from_gray(den)

    # 3) morphology gradient-aware
    bin_refined = gradient_aware_morphology(bin_otsu, gray)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_binary_improved.png"),
                (bin_refined.astype(np.uint8)) * 255)

    # 4) two-pass migliorato
    tp_imp = two_pass(bin_refined)
    tp_imp = remove_small_regions(tp_imp)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_two_pass_improved.png"),
                apply_mask_on_color(color_img, tp_imp))
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_two_pass_improved_colored.png"),
                label_to_color_and_numbers(tp_imp))

    # 5) watershed per separare topi attaccati
    ws = split_overlapping_mice(bin_refined)
    ws = remove_small_regions(ws)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_watershed.png"),
                apply_mask_on_color(color_img, ws))
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_watershed_colored.png"),
                label_to_color_and_numbers(ws))

    print("\n=== Elaborazione completata! ===")
    print(f"I risultati sono stati salvati in: {output_dir}\n")


if __name__ == "__main__":
    main()
