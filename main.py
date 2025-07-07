import cv2
import pytesseract
from PIL import Image
import os
import re
import time
import multiprocessing

def process_chunk(start_frame, end_frame, crop_box, video_path, fps, frame_skip):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_index = start_frame
    while frame_index < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pil_gray = Image.fromarray(gray)

            cropped = pil_gray.crop(crop_box)
            chn_text = pytesseract.image_to_string(cropped)
            match = re.search(r"Chn:\s*([\d.]+)", chn_text)
            chn_value = round(float(match.group(1))) if match else "?"
            if chn_value % 100 == 0:
                print(chn_value)
        frame_index += 1

    cap.release()


if __name__ == "__main__":
    video_path = input("Enter video filename (with extension): ").strip()
    if not os.path.exists(video_path):
        print("Video file not found.")
        exit(1)

    os.makedirs("cropped_images", exist_ok=True)
    start = time.time()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = 1

    print(f"Video FPS: {fps:.2f} | Total Frames: {total_frames}")

    crop_box = None
    for _ in range(5):
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pil_gray = Image.fromarray(gray)

        data = pytesseract.image_to_data(pil_gray, output_type=pytesseract.Output.DICT)
        for i, word in enumerate(data['text']):
            if word and re.match(r"Chn:\s*[\d.]+", word):
                match = re.search(r"Chn:\s*([\d.]+)", word)
                if match:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    x1 = max(x - 5, 0)
                    y1 = max(y - 5, 0)
                    x2 = x + w + 5
                    y2 = y + h + 5
                    crop_box = (x1, y1, x2, y2)
                    break
        if crop_box:
            break

    cap.release()

    if not crop_box:
        print("CHN not found in initial frames. Exiting.")
        exit(1)

    num_cores = multiprocessing.cpu_count()
    chunk_size = total_frames // num_cores
    ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_cores)]
    ranges[-1] = (ranges[-1][0], total_frames)

    print(f"Spawning {num_cores} processes...")

    with multiprocessing.Pool(num_cores) as pool:
        args = [(start_f, end_f, crop_box, video_path, fps, frame_skip) for (start_f, end_f) in ranges]
        pool.starmap(process_chunk, args)

    print(f"\n✅ Done in {round(time.time() - start, 2)}s.")