# Card Stunt Generator

Convert any image into a card stunt instruction sheet — a grid of color-coded cards for stadium crowd performances.

Given a source image, the tool quantizes it to a fixed 50-color palette, splits the result into a plate grid (25 rows × 50 columns), and exports per-plate color assignments to Excel so each section of a crowd knows exactly which card to hold.

---

## Features

- **Image preprocessing** — CLAHE contrast enhancement, bilateral denoising, and soft sharpening before quantization
- **Interactive crop** — draw a fixed-aspect-ratio crop window with undo/reset support; auto-crops to center if skipped
- **Color quantization** — nearest-neighbor matching in CIE LAB space using a custom 50-color palette (`color_book.json`)
- **Top-N color reduction** — optionally restrict the final image to your N most-used colors for simpler card management
- **Vertical pair enforcement** — forces every two vertically adjacent pixels to share the same color, matching the physical 1×2 card format
- **Three card layout options** — 1:16 (4×4), 1:20 (4×5), or 1:25 (5×5) cards per plate
- **Excel export** — one sheet with all 1,250 plates labeled and filled with color IDs
- **Interactive plate viewer** — inspect any individual plate (e.g. `B12`) after generation
- **Performance metrics** — MSE, PSNR, SSIM, and ΔE (CIE76) comparing the quantized grid to the resized original
- **Color usage table** — ranked breakdown of every color used, with pixel count and percentage

---

## Project Structure

```
project/
├── card_stunt_generator.py   # Main script
├── metrics.py                # Image quality metrics (MSE, PSNR, SSIM, ΔE)
├── show_color_book.py        # Visualize the 50-color palette
├── color_book.json           # Color palette definition
├── requirements.txt          # Python dependencies
├── setup_Mac.sh              # Mac/Linux environment setup
├── setup_Win.bat             # Windows environment setup
├── images/                   # Place input images here
├── output/                   # Generated grid images (auto-created)
├── plate_sheet/              # Exported Excel files (auto-created)
└── performance/              # Metric reports (auto-created)
```

---

## Setup

### Mac / Linux

```bash
bash setup_Mac.sh
source projEnv/bin/activate
```

### Windows

```bat
setup_Win.bat
call projEnv\Scripts\activate
```

Both scripts create a virtual environment called `projEnv` and install all dependencies from `requirements.txt`.

---

## Usage

### 1. Add your image

Place one or more image files in the `images/` folder. Supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp`, `.gif`, `.jp2`, `.ppm`, `.pgm`, `.pbm`

### 2. Run the generator

```bash
python card_stunt_generator.py
```

The script will prompt you interactively:

| Prompt | Description |
|--------|-------------|
| **Select image** | Choose from images found in the `images/` folder |
| **Card layout** | `1` = 1:16 (4×4), `2` = 1:20 (4×5), `3` = 1:25 (5×5) |
| **Number of colors** | Enter 1–50; limits the palette to the top N most-used colors |
| **Crop window** | Drag to select a region; press `ENTER` to confirm, `U` to undo, `R` to reset. Press `ENTER` without selecting to auto-center-crop |

### 3. Interactive plate viewer

After generation, the full grid is displayed and the viewer starts:

```
Enter plate (A1 etc), 'full', 'nogrid', or 'quit':
```

(Please close the current window before use these command.)

| Command | Action |
|---------|--------|
| `A1`, `B12`, etc. | Show that specific plate with color IDs overlaid |
| `full` | Show the full grid with plate boundary lines |
| `nogrid` | Show the full grid without plate lines |
| `quit` | Exit |

---

## Outputs

| File | Location | Description |
|------|----------|-------------|
| `<image>_grid<timestamp>.png` | `output/` | Full card grid image (10× upscaled) |
| `<image>_plates_<timestamp>.xlsx` | `plate_sheet/` | Excel sheet with all plate color assignments |
| `<image>_report_<timestamp>.txt` | `performance/` | Quality metrics report |

---

## Color Palette

The palette is defined in `color_book.json` under the name `Stunt_Card_50`. It contains 50 colors across 11 hue groups:

| Group | Color IDs |
|-------|-----------|
| Neutral | 1–5 |
| Red | 6–10 |
| Orange | 11–15 |
| Yellow | 16–20 |
| Green | 21–25 |
| Dark Green | 26–30 |
| Cyan | 31–35 |
| Blue | 36–40 |
| Pink | 41–42 |
| Purple | 43–45 |
| Brown | 46–50 |

To visualize the palette:

```bash
python show_color_book.py
```

---

## Grid Layout

The full grid is always **25 rows × 50 columns** of plates, labeled rows A–Y and columns 1–50. The card layout you select controls how many physical cards make up each plate:

| Layout | Cards per plate | Grid resolution (R×C) | Picture resolution |
|--------|----------------|-----------------|-----------------|
| 1:16 (4×4) | 16 | 100 × 200 cards | 200 × 200 cards |
| 1:20 (4×5) | 20 | 100 × 250 cards | 200 × 250 cards
| 1:25 (5×5) | 25 | 125 × 250 cards | 250 × 250 cards

Each plate in the Excel sheet is labeled (e.g. `-A1-`) followed by rows of color IDs that correspond to entries in `color_book.json`.

---

## Performance Metrics

After quantization, four metrics are computed between the quantized grid and the resized original image:

| Metric | Meaning |
|--------|---------|
| **MSE** | Mean Squared Error — lower is better |
| **PSNR** | Peak Signal-to-Noise Ratio (dB) — higher is better |
| **SSIM** | Structural Similarity Index (0–1) — higher is better |
| **ΔE** | Perceptual color difference (CIE76) — lower is better |

Results are printed to the console and saved as a `.txt` report in `performance/`.

---

## Dependencies

```
opencv-python
numpy
matplotlib
openpyxl
scikit-image
```

Install manually if not using the setup scripts:

```bash
pip install -r requirements.txt
```

---

## Requirements

- Python 3.8 or later
- A display environment is required (matplotlib and OpenCV windows are used for cropping and visualization)