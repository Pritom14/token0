"""Generate diverse test images for benchmarking Token0 optimizations."""

import os
import random

import numpy as np
from PIL import Image, ImageDraw

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BENCHMARK_DIR, "images")


def ensure_dir():
    os.makedirs(IMAGES_DIR, exist_ok=True)


def generate_large_photo(filename="large_photo.jpg"):
    """4000x3000 photo-like image — tests resize optimization."""
    ensure_dir()
    # Generate a colorful, noisy image that simulates a photo
    arr = np.random.randint(50, 220, (3000, 4000, 3), dtype=np.uint8)
    # Add some structure — gradient overlay
    for y in range(3000):
        for c in range(3):
            arr[y, :, c] = np.clip(arr[y, :, c].astype(int) + int(50 * y / 3000), 0, 255)
    img = Image.fromarray(arr)
    path = os.path.join(IMAGES_DIR, filename)
    img.save(path, "JPEG", quality=90)
    print(
        f"  Created: {filename} ({img.size[0]}x{img.size[1]}, {os.path.getsize(path) / 1024:.0f}KB)"
    )
    return path


def generate_document_screenshot(filename="document_screenshot.png"):
    """800x1100 document-like screenshot — tests OCR routing."""
    ensure_dir()
    img = Image.new("RGB", (800, 1100), color="white")
    draw = ImageDraw.Draw(img)

    # Title
    draw.rectangle([60, 40, 740, 80], fill="black")
    draw.rectangle([65, 45, 400, 75], fill="white")  # "title" text block

    # Body text — simulate paragraphs
    y = 120
    for para in range(6):
        lines_in_para = random.randint(3, 7)
        for line in range(lines_in_para):
            line_width = random.randint(500, 680)
            draw.rectangle([60, y, 60 + line_width, y + 10], fill=(30, 30, 30))
            # Simulate word gaps
            x = 60
            while x < 60 + line_width - 30:
                word_len = random.randint(15, 50)
                draw.rectangle([x, y + 1, x + word_len, y + 9], fill=(20, 20, 20))
                x += word_len + random.randint(4, 10)
            y += 18
        y += 24  # paragraph gap

    # Add some "table" structure
    for row in range(4):
        for col in range(3):
            x1 = 60 + col * 220
            y1 = y + row * 30
            draw.rectangle([x1, y1, x1 + 210, y1 + 25], outline=(100, 100, 100))
            draw.rectangle(
                [x1 + 5, y1 + 5, x1 + 60 + random.randint(0, 100), y1 + 18], fill=(40, 40, 40)
            )

    path = os.path.join(IMAGES_DIR, filename)
    img.save(path, "PNG")
    print(
        f"  Created: {filename} ({img.size[0]}x{img.size[1]}, {os.path.getsize(path) / 1024:.0f}KB)"
    )
    return path


def generate_small_photo(filename="small_photo.jpg"):
    """300x300 small photo — tests detail mode selection."""
    ensure_dir()
    arr = np.random.randint(80, 200, (300, 300, 3), dtype=np.uint8)
    # Add a simple "object" — circle in center
    for y in range(300):
        for x in range(300):
            if (x - 150) ** 2 + (y - 150) ** 2 < 80**2:
                arr[y, x] = [200, 50, 50]
    img = Image.fromarray(arr)
    path = os.path.join(IMAGES_DIR, filename)
    img.save(path, "JPEG", quality=85)
    print(
        f"  Created: {filename} ({img.size[0]}x{img.size[1]}, {os.path.getsize(path) / 1024:.0f}KB)"
    )
    return path


def generate_receipt(filename="receipt.png"):
    """400x900 receipt image — tests OCR routing on narrow text."""
    ensure_dir()
    img = Image.new("RGB", (400, 900), color=(252, 250, 245))
    draw = ImageDraw.Draw(img)

    # Store name
    draw.rectangle([120, 30, 280, 55], fill=(20, 20, 20))

    # Items
    y = 90
    items = [
        ("Coffee Latte", "$4.50"),
        ("Blueberry Muffin", "$3.25"),
        ("Orange Juice", "$2.75"),
        ("Breakfast Sandwich", "$6.50"),
        ("Cappuccino", "$4.00"),
        ("Cookie", "$2.00"),
    ]
    for item_name, price in items:
        # Item name (left aligned)
        name_width = len(item_name) * 7
        draw.rectangle([40, y, 40 + name_width, y + 12], fill=(30, 30, 30))
        # Price (right aligned)
        price_width = len(price) * 7
        draw.rectangle([360 - price_width, y, 360, y + 12], fill=(30, 30, 30))
        # Dotted line between
        for dot_x in range(40 + name_width + 10, 360 - price_width - 10, 8):
            draw.rectangle([dot_x, y + 5, dot_x + 2, y + 7], fill=(100, 100, 100))
        y += 30

    # Separator line
    draw.line([(40, y), (360, y)], fill=(100, 100, 100), width=1)
    y += 15

    # Total
    draw.rectangle([40, y, 100, y + 14], fill=(20, 20, 20))  # "TOTAL"
    draw.rectangle([310, y, 360, y + 14], fill=(20, 20, 20))  # Amount
    y += 40

    # Tax, tip lines
    for label in ["Subtotal", "Tax (8%)", "Tip", "Total"]:
        lw = len(label) * 7
        draw.rectangle([40, y, 40 + lw, y + 10], fill=(60, 60, 60))
        draw.rectangle([320, y, 360, y + 10], fill=(60, 60, 60))
        y += 22

    # Barcode-like area
    y += 30
    for bx in range(80, 320, 3):
        bar_h = random.randint(30, 50)
        draw.rectangle([bx, y, bx + 1, y + bar_h], fill=(20, 20, 20))

    path = os.path.join(IMAGES_DIR, filename)
    img.save(path, "PNG")
    print(
        f"  Created: {filename} ({img.size[0]}x{img.size[1]}, {os.path.getsize(path) / 1024:.0f}KB)"
    )
    return path


def generate_already_optimized(filename="already_optimized.jpg"):
    """512x512 JPEG — should passthrough with minimal/no optimization."""
    ensure_dir()
    arr = np.random.randint(60, 200, (512, 512, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    path = os.path.join(IMAGES_DIR, filename)
    img.save(path, "JPEG", quality=80)
    print(
        f"  Created: {filename} ({img.size[0]}x{img.size[1]}, {os.path.getsize(path) / 1024:.0f}KB)"
    )
    return path


def generate_large_png_screenshot(filename="large_screenshot.png"):
    """1920x1080 desktop screenshot PNG — tests resize + PNG→JPEG conversion."""
    ensure_dir()
    img = Image.new("RGB", (1920, 1080), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Title bar
    draw.rectangle([0, 0, 1920, 40], fill=(60, 60, 70))
    draw.rectangle([10, 10, 200, 30], fill=(100, 100, 110))  # window controls

    # Sidebar
    draw.rectangle([0, 40, 250, 1080], fill=(45, 45, 55))
    for i in range(12):
        draw.rectangle([20, 70 + i * 60, 230, 70 + i * 60 + 35], fill=(65, 65, 75))
        draw.rectangle([50, 80 + i * 60, 180, 92 + i * 60], fill=(120, 120, 130))

    # Main content area — text blocks
    for row in range(8):
        y_start = 80 + row * 110
        for line in range(4):
            line_width = random.randint(400, 900)
            draw.rectangle(
                [280, y_start + line * 22, 280 + line_width, y_start + line * 22 + 14],
                fill=(40, 40, 40),
            )

    # Some colored cards
    colors = [(66, 133, 244), (52, 168, 83), (234, 67, 53), (251, 188, 4)]
    for i, color in enumerate(colors):
        x = 280 + i * 380
        draw.rectangle([x, 60, x + 350, 75], fill=color)

    path = os.path.join(IMAGES_DIR, filename)
    img.save(path, "PNG")
    print(
        f"  Created: {filename} ({img.size[0]}x{img.size[1]}, {os.path.getsize(path) / 1024:.0f}KB)"
    )
    return path


def generate_all():
    """Generate all benchmark test images."""
    print("Generating benchmark images...")
    paths = {
        "large_photo": generate_large_photo(),
        "document_screenshot": generate_document_screenshot(),
        "small_photo": generate_small_photo(),
        "receipt": generate_receipt(),
        "already_optimized": generate_already_optimized(),
        "large_screenshot": generate_large_png_screenshot(),
    }
    print(f"\nGenerated {len(paths)} test images in {IMAGES_DIR}/")
    return paths


if __name__ == "__main__":
    generate_all()
