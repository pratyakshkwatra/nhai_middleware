import cv2
import pytesseract
from PIL import Image
import os
import re
import time
import multiprocessing

CHN_INCREMENT_PER_FRAME = 5
CHN_TARGET_INTERVAL = 100
FRAME_SKIP = CHN_TARGET_INTERVAL // CHN_INCREMENT_PER_FRAME  # 100 / 5 = 20

# Stall recovery thresholds
STALL_FRAME_LIMIT = 5
STALL_PROBE_STEP = 3
MAX_STALL_SCAN = 20


def extract_chn_fast(frame, crop_box):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pil = Image.fromarray(gray)
    cropped = pil.crop(crop_box)
    data = pytesseract.image_to_data(cropped, output_type=pytesseract.Output.DICT,
                                     config="--psm 6 -c tessedit_char_whitelist=0123456789.")
    for word in data["text"]:
        if word and re.match(r"^[\d.]+$", word):
            try:
                return int(float(word))
            except:
                continue
    return None


def process_chunk(start_frame, end_frame, crop_box, video_path, fps, frame_skip):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_index = start_frame
    first_found = False
    last_chn = None
    stall_counter = 0

    while frame_index < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        chn = extract_chn_fast(frame, crop_box)
        if chn is None:
            frame_index += 1
            continue

        if not first_found:
            if chn % 100 == 0:
                print(f"[{frame_index}] Chn: {chn} ✅ (start)")
                first_found = True
                last_chn = chn
                frame_index += frame_skip
                for _ in range(frame_skip - 1):
                    cap.grab()
            else:
                frame_index += 1
            continue

        # Stall Detection
        if chn == last_chn:
            stall_counter += 1

            if stall_counter <= STALL_FRAME_LIMIT:
                print(f"[{frame_index}] Chn: {chn} ❌ (stall mode: frame-by-frame)")
                frame_index += 1

            elif stall_counter <= MAX_STALL_SCAN:
                print(f"[{frame_index}] Chn: {chn} ❌ (stall mode: jump by {STALL_PROBE_STEP})")
                frame_index += STALL_PROBE_STEP
                for _ in range(STALL_PROBE_STEP - 1):
                    cap.grab()

            else:
                print(f"[{frame_index}] ⏭️ Skipping stall block after {stall_counter} frozen frames")
                stall_counter = 0
                frame_index += frame_skip
                for _ in range(frame_skip - 1):
                    cap.grab()

            continue  # Skip Chn % 100 check during stall

        # Chn has changed
        stall_counter = 0
        last_chn = chn

        if chn % 100 == 0:
            print(f"[{frame_index}] Chn: {chn} ✅")
            frame_index += frame_skip
            for _ in range(frame_skip - 1):
                cap.grab()
        else:
            frame_index += 1

    cap.release()


if __name__ == "__main__":
    video_path = input("Enter video filename (with extension): ").strip()
    if not os.path.exists(video_path):
        print("Video file not found.")
        exit(1)

    start = time.time()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps:.2f} | Total Frames: {total_frames}")

    # Try to detect crop box from first few frames
    crop_box = None
    for _ in range(5):
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pil = Image.fromarray(gray)
        data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT)
        for i, word in enumerate(data['text']):
            if word and re.match(r"Chn:\s*[\d.]+", word):
                match = re.search(r"Chn:\s*([\d.]+)", word)
                if match:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    crop_box = (max(x - 5, 0), max(y - 5, 0), x + w + 5, y + h + 5)
                    break
        if crop_box:
            break

    cap.release()

    if not crop_box:
        print("❌ Could not find 'Chn:' in initial frames.")
        exit(1)

    num_cores = multiprocessing.cpu_count()
    chunk_size = total_frames // num_cores
    ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_cores)]
    ranges[-1] = (ranges[-1][0], total_frames)  # last chunk ends at total_frames

    print(f"Spawning {num_cores} processes...")

    with multiprocessing.Pool(num_cores) as pool:
        args = [(start_f, end_f, crop_box, video_path, fps, FRAME_SKIP) for (start_f, end_f) in ranges]
        pool.starmap(process_chunk, args)

    print(f"\n✅ Done in {round(time.time() - start, 2)}s.")
