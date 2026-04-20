import json
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# -----------------------------
# CONFIG
# -----------------------------
COLORS_PER_ROW = 10


# -----------------------------
# LOAD COLOR BOOK
# -----------------------------
def load_color_book_json(path):

    with open(path, "r") as f:
        data = json.load(f)

    colors = []

    for c in data["colors"]:

        hex_color = c["hex"].lstrip("#")

        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        colors.append({
            "color_id": c["color_id"],
            "hex": c["hex"],
            "rgb": (r, g, b)
        })

    return colors


# -----------------------------
# DISPLAY COLOR BOOK
# -----------------------------
def display_color_book(colors):

    total_colors = len(colors)

    rows = math.ceil(total_colors / COLORS_PER_ROW)

    fig, axes = plt.subplots(rows, COLORS_PER_ROW, figsize=(12, rows * 1.5))

    if rows == 1:
        axes = [axes]

    for i in range(rows * COLORS_PER_ROW):

        r = i // COLORS_PER_ROW
        c = i % COLORS_PER_ROW

        ax = axes[r][c] if rows > 1 else axes[c]

        ax.axis("off")

        if i >= total_colors:
            continue

        color = colors[i]

        rgb = np.array(color["rgb"], dtype=np.uint8).reshape(1, 1, 3)

        ax.imshow(rgb)

        ax.text(
            0.5,
            -0.25,
            f"ID: {color['color_id']}\n{color['hex']}\nRGB{color['rgb']}",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
            color="black"
        )
        
    fig.patch.set_facecolor("#F0F0F0")

    plt.suptitle("Color Book", fontsize=14, fontweight='bold', color='black')
    plt.tight_layout()
    plt.show()


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    COLOR_BOOK_PATH = os.path.join(BASE_DIR, "color_book.json")

    colors = load_color_book_json(COLOR_BOOK_PATH)

    display_color_book(colors)