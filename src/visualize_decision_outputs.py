import csv
import os
import struct
import zlib


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

WHITE = (255, 255, 255)
BLACK = (35, 35, 35)
GRAY = (225, 225, 225)
BLUE = (31, 119, 180)
GREEN = (44, 160, 44)
RED = (214, 39, 40)


def save_png(path, pixels):
    height = len(pixels)
    width = len(pixels[0])
    raw = b"".join(
        b"\x00" + bytes(channel for pixel in row for channel in pixel)
        for row in pixels
    )

    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        f.write(chunk(b"IEND", b""))


def new_canvas(width=900, height=520):
    return [[WHITE for _ in range(width)] for _ in range(height)]


def set_pixel(pixels, x, y, color):
    if 0 <= y < len(pixels) and 0 <= x < len(pixels[0]):
        pixels[y][x] = color


def draw_line(pixels, x0, y0, x1, y1, color, width=1):
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        for ox in range(-(width // 2), width // 2 + 1):
            for oy in range(-(width // 2), width // 2 + 1):
                set_pixel(pixels, x0 + ox, y0 + oy, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def draw_rect(pixels, x0, y0, x1, y1, color):
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            set_pixel(pixels, x, y, color)


def draw_circle(pixels, cx, cy, radius, color):
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                set_pixel(pixels, x, y, color)


def scale(value, src_min, src_max, dst_min, dst_max):
    if src_max == src_min:
        return (dst_min + dst_max) // 2
    ratio = (value - src_min) / (src_max - src_min)
    return int(dst_min + ratio * (dst_max - dst_min))


def draw_axes_and_grid(pixels, margin_left=80, margin_bottom=70, margin_top=50):
    width = len(pixels[0])
    height = len(pixels)
    x0 = margin_left
    y0 = height - margin_bottom
    x1 = width - 40
    y1 = margin_top

    for i in range(6):
        y = y0 - int(i * (y0 - y1) / 5)
        draw_line(pixels, x0, y, x1, y, GRAY)
    for i in range(6):
        x = x0 + int(i * (x1 - x0) / 5)
        draw_line(pixels, x, y0, x, y1, GRAY)

    draw_line(pixels, x0, y0, x1, y0, BLACK, width=2)
    draw_line(pixels, x0, y0, x0, y1, BLACK, width=2)
    return x0, y0, x1, y1


def read_threshold_profit(path):
    thresholds = []
    profits = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            thresholds.append(float(row["threshold"]))
            profits.append(float(row["profit_proxy"]))
    return thresholds, profits


def read_profit_differences(path):
    differences = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            baseline = float(row["E_profit_baseline_1.0x"])
            optimized = float(row["E_profit_opt"])
            differences.append(optimized - baseline)
    return differences


def plot_threshold_profit_curve():
    input_file = os.path.join(OUTPUT_DIR, "threshold_profit_scan.csv")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Missing {input_file}. Run src/decision_logic.py first.")

    thresholds, profits = read_threshold_profit(input_file)
    pixels = new_canvas()
    x0, y0, x1, y1 = draw_axes_and_grid(pixels)

    min_profit = min(profits)
    max_profit = max(profits)
    points = [
        (
            scale(t, min(thresholds), max(thresholds), x0, x1),
            scale(p, min_profit, max_profit, y0, y1),
        )
        for t, p in zip(thresholds, profits)
    ]
    for (xa, ya), (xb, yb) in zip(points, points[1:]):
        draw_line(pixels, xa, ya, xb, yb, BLUE, width=3)

    best_idx = max(range(len(profits)), key=profits.__getitem__)
    best_x, best_y = points[best_idx]
    draw_line(pixels, best_x, y0, best_x, y1, RED, width=2)
    draw_circle(pixels, best_x, best_y, 7, RED)

    output_file = os.path.join(OUTPUT_DIR, "threshold_profit_curve.png")
    save_png(output_file, pixels)
    print(f"Saved: {output_file}")


def plot_profit_distribution():
    input_file = os.path.join(OUTPUT_DIR, "simulation_per_loan_sample.csv")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Missing {input_file}. Run src/simulation_profit.py first.")

    values = read_profit_differences(input_file)
    pixels = new_canvas()
    x0, y0, x1, y1 = draw_axes_and_grid(pixels)

    bins = 50
    min_value = min(values)
    max_value = max(values)
    counts = [0] * bins
    for value in values:
        idx = int((value - min_value) / (max_value - min_value) * (bins - 1))
        counts[idx] += 1

    max_count = max(counts)
    bar_width = max(1, (x1 - x0) // bins)
    for i, count in enumerate(counts):
        left = x0 + i * bar_width
        right = left + bar_width - 2
        top = scale(count, 0, max_count, y0, y1)
        draw_rect(pixels, left, top, right, y0 - 1, GREEN)

    if min_value <= 0 <= max_value:
        zero_x = scale(0, min_value, max_value, x0, x1)
        draw_line(pixels, zero_x, y0, zero_x, y1, BLACK, width=2)

    output_file = os.path.join(OUTPUT_DIR, "profit_distribution.png")
    save_png(output_file, pixels)
    print(f"Saved: {output_file}")


def main():
    plot_threshold_profit_curve()
    plot_profit_distribution()


if __name__ == "__main__":
    main()
