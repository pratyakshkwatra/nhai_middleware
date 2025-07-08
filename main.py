import cv2
import pytesseract
from PIL import Image
import os
import re
import time
import math
import multiprocessing
import pandas as pd
from multiprocessing import Manager

CHN_INCREMENT_PER_FRAME = 5
CHN_TARGET_INTERVAL = 100
FRAME_SKIP = CHN_TARGET_INTERVAL // CHN_INCREMENT_PER_FRAME
STALL_FRAME_LIMIT = 5
STALL_PROBE_STEP = 3
MAX_STALL_SCAN = 20
TOLERANCE = 0.0000025


def extract_chn_fast(frame, crop_box):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pil = Image.fromarray(gray)
    cropped = pil.crop(crop_box)
    data = pytesseract.image_to_data(cropped, output_type=pytesseract.Output.DICT,
                                     config="--psm 6 -c tessedit_char_whitelist=0123456789.")
    for word in data["text"]:
        if word and re.match(r"^[\d.]+$", word):
            try:
                return float(word)
            except:
                continue
    return None


def process_chunk(start_frame, end_frame, crop_box, video_path, fps, frame_skip, results, chn_excel_list):
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

        timestamp_sec = frame_index / fps
        chn = extract_chn_fast(frame, crop_box)
        if chn is None:
            frame_index += 1
            continue

        matched_chn = None
        for excel_chn in chn_excel_list:
            if abs(chn - excel_chn) <= TOLERANCE * excel_chn:
                matched_chn = excel_chn
                break

        if matched_chn is None:
            frame_index += 1
            continue

        if not first_found:
            results.append((frame_index, matched_chn, timestamp_sec))
            first_found = True
            last_chn = matched_chn
            frame_index += frame_skip
            for _ in range(frame_skip - 1):
                cap.grab()
            continue

        if matched_chn == last_chn:
            stall_counter += 1
            if stall_counter <= STALL_FRAME_LIMIT:
                frame_index += 1
            elif stall_counter <= MAX_STALL_SCAN:
                frame_index += STALL_PROBE_STEP
                for _ in range(STALL_PROBE_STEP - 1):
                    cap.grab()
            else:
                stall_counter = 0
                frame_index += frame_skip
                for _ in range(frame_skip - 1):
                    cap.grab()
            continue

        stall_counter = 0
        last_chn = matched_chn

        if matched_chn % 100 == 0:
            results.append((frame_index, matched_chn, timestamp_sec))
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

    excel_path = input("Enter Excel filename (with extension): ").strip()
    if not os.path.exists(excel_path):
        print("Excel file not found.")
        exit(1)

    lane = input("Enter lane (e.g., L1, L2, R1, R4): ").strip().upper()
    if lane not in ["L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"]:
        print("Invalid lane.")
        exit(1)

    start = time.time()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"🎞️ Video FPS: {fps:.2f} | Total Frames: {total_frames}")

    cap = cv2.VideoCapture(video_path)
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
        print("❌ Could not detect 'Chn:' in initial frames.")
        exit(1)

    df = pd.read_excel(excel_path, header=[0, 1, 2])
    df.columns = [' '.join([str(i) for i in col if str(i) != 'nan']).strip() for col in df.columns]

    global_cols = [
        'Unnamed: 0_level_0 NH Number Unnamed: 0_level_2',
        'Unnamed: 1_level_0 Start Chainage  Unnamed: 1_level_2',
        'Unnamed: 2_level_0 End Chainage  Unnamed: 2_level_2',
        'Unnamed: 3_level_0 Length Unnamed: 3_level_2',
        'Unnamed: 4_level_0 Structure Details Unnamed: 4_level_2'
    ]

    limitation_cols = [
        'Lane R4 Limitation of BI as per MoRT&H Circular (in mm/km) Unnamed: 38_level_2',
        'Lane R4 Limitation of Rut Depth as per Concession Agreement (in mm) Unnamed: 47_level_2',
        'Lane R4 Limitation of Cracking as per Concession Agreement (in % area) Unnamed: 56_level_2',
        'Lane R4 Limitation of Ravelling as per Concession Agreement (in % area) Unnamed: 65_level_2'
    ]

    idx = ["L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"].index(lane)
    lane_cols = [
        f'Lane R4 {lane} Lane Roughness BI (in mm/km) Unnamed: {39 + idx}_level_2',
        f'Lane R4 {lane} Rut Depth (in mm) Unnamed: {48 + idx}_level_2',
        f'Lane R4 {lane} Crack Area (in % area) Unnamed: {57 + idx}_level_2',
        f'Lane R4 {lane} Area (% area) Unnamed: {66 + idx}_level_2'
    ]

    latlong_cols = [
        f'Lane {lane} Start Latitude',
        f'Lane {lane} Start Longitude'
    ]

    selected_cols = global_cols + limitation_cols + lane_cols + latlong_cols
    filtered_df = df[selected_cols].copy()

    filtered_df.columns = [
        "NH Number", "Start Chainage", "End Chainage", "Length", "Structure Details",
        "BI Limit", "Rut Limit", "Crack Limit", "Ravelling Limit",
        "Roughness", "Rut Depth", "Crack Area", "Ravelling",
        "Start Lat", "Start Long"
    ]

    chn_excel_list = filtered_df["Start Chainage"].dropna().astype(int).tolist()

    manager = Manager()
    shared_results = manager.list()
    num_cores = multiprocessing.cpu_count()
    chunk_size = total_frames // num_cores
    ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_cores)]
    ranges[-1] = (ranges[-1][0], total_frames)

    print(f"⚙️ Spawning {num_cores} processes...")
    with multiprocessing.Pool(num_cores) as pool:
        args = [(start_f, end_f, crop_box, video_path, fps, FRAME_SKIP, shared_results, chn_excel_list) for (start_f, end_f) in ranges]
        pool.starmap(process_chunk, args)

    all_results = list(shared_results)
    matched_rows = []
    for frame, chn, timestamp_sec in all_results:
        ms = int(timestamp_sec * 1000)
        matched_rows.append({"CHN": chn, "Frame": frame, "Timestamp_ms": ms})

    df_matched = pd.DataFrame(matched_rows)
    final_df = pd.merge(filtered_df, df_matched, left_on="Start Chainage", right_on="CHN", how="inner")
    final_df.drop(columns=["CHN"], inplace=True)

    final_df.columns = [
        "nh", "chn_start", "chn_end", "length", "structure",
        "roughness_limit", "rut_limit", "crack_limit", "ravelling_limit",
        "roughness", "rut_depth", "crack_area", "area",
        "start_lat", "start_long",
        "frame", "timestamp"
    ]

    final_df.to_csv("output_matched.csv", index=False)
    print(f"\n✅ Done in {round(time.time() - start, 2)}s. Output saved as 'output_matched.csv'")
