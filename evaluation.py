import os
import cv2

# ================= CONFIGURATION =================
# Folder 1: Baseline
BASELINE_DIR = r"/Users/krist/Documents/IP_project/RicardosScript/result_baseline"

# Folder 2: Improved
IMPROVED_DIR = r"/Users/krist/Documents/IP_project/RicardosScript/my_results"


# Minimum size (pixels) to count as a mouse
MIN_MOUSE_SIZE = 600


def count_blobs(image_path):
    """
    Reads an image (Binary or Colored Result) and counts the blobs.
    """
    if not os.path.exists(image_path):
        return -1

        # Read as color first, to handle both types
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_color is None:
        return -1

    # Convert to Grayscale
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # THRESHOLD STRATEGY:
    # If it's a _result.png, the background is Black (0) and mice are Colored (>0).
    # If it's a _binary.png, the background is Black (0) and mice are White (255).
    # So, we just say: "Anything brighter than 0 is a mouse".
    _, thresh = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)

    # Connected Components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    valid_count = 0
    # Iterate labels (0 is background, start from 1)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > MIN_MOUSE_SIZE:
            valid_count += 1

    return valid_count


def get_base_name(filename):
    """
    Strips suffixes to find the 'core' name (e.g., 'combined_0').
    """
    name = filename
    # Remove known suffixes
    for suffix in ["_result.png", "_binary.png", "_solid.png", ".png", ".jpg"]:
        name = name.replace(suffix, "")
    return name


def main():
    print(f"{'IMAGE NAME':<30} | {'BASELINE':<10} | {'IMPROVED':<10} | {'DIFF':<10}")
    print("-" * 70)

    # Get list of files from Improved directory
    # We filter for _result.png because that is what is inside result_improvement2
    imp_files = sorted([f for f in os.listdir(IMPROVED_DIR) if f.endswith("_result.png")])

    # Get list of files from Baseline directory
    base_files = os.listdir(BASELINE_DIR)

    baseline_totals = []
    improved_totals = []

    for fname_imp in imp_files:
        core_name = get_base_name(fname_imp)

        # Find corresponding file in Baseline folder
        fname_base = None
        for f in base_files:
            if core_name in f and f.endswith(".png") and "debug" not in f:
                # Prioritize binary masks if multiple exist
                if "binary" in f:
                    fname_base = f
                    break
                fname_base = f  # Fallback

        if not fname_base:
            continue

        path_imp = os.path.join(IMPROVED_DIR, fname_imp)
        path_base = os.path.join(BASELINE_DIR, fname_base)

        # Count
        count_base = count_blobs(path_base)
        count_imp = count_blobs(path_imp)

        if count_base == -1 or count_imp == -1:
            continue

        baseline_totals.append(count_base)
        improved_totals.append(count_imp)

        diff = count_imp - count_base
        diff_str = f"+{diff}" if diff > 0 else str(diff)

        print(f"{core_name:<30} | {count_base:<10} | {count_imp:<10} | {diff_str:<10}")

    # SUMMARY
    if len(baseline_totals) > 0:
        avg_base = sum(baseline_totals) / len(baseline_totals)
        avg_imp = sum(improved_totals) / len(improved_totals)

        print("-" * 70)
        print(f"SUMMARY ({len(baseline_totals)} images):")
        print(f"Avg Count (Baseline): {avg_base:.2f}")
        print(f"Avg Count (Improved): {avg_imp:.2f}")
        print("-" * 70)


if __name__ == "__main__":
    main()