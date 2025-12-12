import cv2
import numpy as np
import os
import csv


# ----------------- UNION FIND -----------------

class UnionFind:
    def __init__(self):
        self.parent = {}

    def make_set(self, x):
        self.parent[x] = x

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while x != root:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


# ----------------- Specific CSV Reading -----------------

def load_csv_matrix(path):
    """
    Reads a thermal CSV with rows like:
    "23,31"  "23,50"  ...
    seperated by TAB.
    Return a numpy matrix of float32.
    """
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            # skip empty rows:
            if not row or not any(field.strip() for field in row):
                continue

            clean_row = []
            for field in row:
                s = field.strip().strip('"').strip("'")
                if s == "":
                    continue
                s = s.replace(",", ".")
                try:
                    val = float(s)
                except ValueError:
                    continue
                clean_row.append(val)

            if clean_row:
                rows.append(clean_row)

    max_len = max(len(r) for r in rows)
    data = np.full((len(rows), max_len), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        data[i, :len(r)] = r

    return data


# ----------------- Image loading (jpg/png） -----------------

def load_thermal_image(path):
    """
    Returns:
    - color_img: BGR image (thermal color image)
    - gray: grayscale image (uint8)
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        temps = load_csv_matrix(path)

        # Normalizza [min, max] -> [0, 255]
        t_min = np.nanmin(temps)
        t_max = np.nanmax(temps)
        norm = (temps - t_min) / (t_max - t_min + 1e-8)
        gray = np.clip(norm * 255, 0, 255).astype(np.uint8)

        # Colormap termica (BGR)
        color_img = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        return color_img, gray

    # fallback case: image already saved
    color_img = cv2.imread(path)
    if color_img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return color_img, gray


# ----------------- Binarization -----------------

def to_binary_from_gray(gray):
    _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(bool)


# ----------------- CCA: TWO-PASS -----------------

def two_pass(binary, connectivity):
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    uf = UnionFind()
    next_label = 1

    if connectivity == 4:
        neighbors = [(-1, 0), (0, -1)]
    elif connectivity == 8:
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]
    else:
        raise ValueError("Connectivity must be 4 or 8")

    # First pass
    for i in range(h):
        for j in range(w):
            if not binary[i, j]:
                continue
            neighbor_labels = []
            for di, dj in neighbors:
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w and labels[ni, nj] > 0:
                    neighbor_labels.append(labels[ni, nj])
            if not neighbor_labels:
                labels[i, j] = next_label
                uf.make_set(next_label)
                next_label += 1
            else:
                min_label = min(neighbor_labels)
                labels[i, j] = min_label
                for nl in neighbor_labels:
                    if nl != min_label:
                        uf.union(nl, min_label)

    # Second pass: label remapping
    label_map = {}
    new_label = 1
    for i in range(h):
        for j in range(w):
            if labels[i, j] > 0:
                root = uf.find(labels[i, j])
                if root not in label_map:
                    label_map[root] = new_label
                    new_label += 1
                labels[i, j] = label_map[root]

    return labels


# ----------------- CCA: SEED FILLING -----------------

def seed_filling(binary, connectivity):
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    next_label = 1

    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1), (0, 1),
                     (1, -1), (1, 0), (1, 1)]
    else:
        raise ValueError("Connectivity must be 4 or 8")

    for i in range(h):
        for j in range(w):
            if binary[i, j] and labels[i, j] == 0:
                stack = [(i, j)]
                labels[i, j] = next_label

                while stack:
                    x, y = stack.pop()
                    for dx, dy in neighbors:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w:
                            if binary[nx, ny] and labels[nx, ny] == 0:
                                labels[nx, ny] = next_label
                                stack.append((nx, ny))

                next_label += 1

    return labels


# ----------------- CCA: CONTOUR -----------------

def contour(binary):
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    contours, _ = cv2.findContours(binary.astype(np.uint8),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for label, c in enumerate(contours, start=1):
        cv2.drawContours(labels, [c], contourIdx=-1, color=label, thickness=cv2.FILLED)
    return labels


# ----------------- REGION UTILITIES -----------------

def remove_small_regions(labels, min_size):
    unique, counts = np.unique(labels, return_counts=True)
    label_sizes = dict(zip(unique, counts))

    for label, size in label_sizes.items():
        if label == 0:
            continue
        if size < min_size:
            labels[labels == label] = 0

    return labels


def count_regions(labels):
    unique = np.unique(labels)
    num_regions = len(unique) - 1 if 0 in unique else len(unique)
    return num_regions


def apply_mask_on_color(color_img, labels):
    mask = labels > 0
    result = color_img.copy()
    result[~mask] = 0
    return result


# ----------------- MAIN -----------------

def main():
    # ====== Change input path here ======
    input_dir = r"/Users/krist/Documents/IP_project/data/thermal_mouse/src/models/model_detectron/output_images_panoptic"

    # ====== Change output path here ======
    output_dir = r"/Users/krist/Documents/IP_project/RicardosScript/result_baseline"
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".csv"))
    ])

    print(f"Found {len(image_files)} images to process.")

    # Iterate through all images
    for fname in image_files:
        print(f"Processing: {fname}")

        img_path = os.path.join(input_dir, fname)

        # Load single thermal image
        try:
            color_img, gray = load_thermal_image(img_path)
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            continue

        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # Binarization
        binary_img = to_binary_from_gray(gray)

        # B&W image
        binary_visual = (binary_img.astype(np.uint8)) * 255
        binary_output_path = os.path.join(output_dir, f"{base_name}_binary.png")
        cv2.imwrite(binary_output_path, binary_visual)

        # --------- TWO-PASS ----------
        for connectivity in [4, 8]:
            two_pass_label_img = two_pass(binary_img, connectivity)
            two_pass_label_img = remove_small_regions(two_pass_label_img, min_size=1000)
            num_objects = count_regions(two_pass_label_img)
            # print(f"Two-Pass - Connectivity {connectivity}: {num_objects} objects found")

            two_pass_segment = apply_mask_on_color(color_img, two_pass_label_img)
            out_path = os.path.join(output_dir,
                                    f"{base_name}_two-pass_c{connectivity}.png")
            cv2.imwrite(out_path, two_pass_segment)

        # --------- SEED FILLING ----------
        for connectivity in [4, 8]:
            seed_label_img = seed_filling(binary_img, connectivity)
            seed_label_img = remove_small_regions(seed_label_img, min_size=1000)
            num_objects = count_regions(seed_label_img)
            # print(f"Seed Filling - Connectivity {connectivity}: {num_objects} objects found")

            seed_segment = apply_mask_on_color(color_img, seed_label_img)
            out_path = os.path.join(output_dir,
                                    f"{base_name}_seed_c{connectivity}.png")
            cv2.imwrite(out_path, seed_segment)

        # --------- CONTOUR-BASED ----------
        contour_label_img = contour(binary_img)
        contour_label_img = remove_small_regions(contour_label_img, min_size=1000)
        num_objects = count_regions(contour_label_img)
        # print(f"Contour-based CCA: {num_objects} objects found")

        contour_segment = apply_mask_on_color(color_img, contour_label_img)
        out_path = os.path.join(output_dir,
                                f"{base_name}_contour.png")
        cv2.imwrite(out_path, contour_segment)

    print("Done processing all images.")


if __name__ == "__main__":
    main()