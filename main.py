import os
import time
import multiprocessing
import pandas as pd
import cv2
import pytesseract
from PIL import Image
import re
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

        matched_chn = next((echn for echn in chn_excel_list if abs(chn - echn) <= TOLERANCE * echn), None)
        if matched_chn is None:
            frame_index += 1
            continue

        if not first_found:
            results.append((frame_index, matched_chn, timestamp_sec))
            first_found = True
            last_chn = matched_chn
            frame_index += 1
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

def load_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames

def detect_crop_box(video_path):
    cap = cv2.VideoCapture(video_path)
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
                    return (max(x - 5, 0), max(y - 5, 0), x + w + 5, y + h + 5)
    cap.release()
    return None

def load_excel_data(excel_path, lane):
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
    latlong_cols = [f'Lane {lane} Start Latitude', f'Lane {lane} Start Longitude']
    selected_cols = global_cols + limitation_cols + lane_cols + latlong_cols
    filtered_df = df[selected_cols].copy()
    filtered_df.columns = [
        "NH Number", "Start Chainage", "End Chainage", "Length", "Structure Details",
        "BI Limit", "Rut Limit", "Crack Limit", "Ravelling Limit",
        "Roughness", "Rut Depth", "Crack Area", "Ravelling",
        "Start Lat", "Start Long"
    ]
    chn_excel_list = filtered_df["Start Chainage"].dropna().astype(int).tolist()
    return filtered_df, chn_excel_list

def generate_chunks(total_frames, num_chunks):
    chunk_size = total_frames // num_chunks
    chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_chunks)]
    chunks[-1] = (chunks[-1][0], total_frames)
    return chunks

def run_parallel_processing(chunks, crop_box, video_path, fps, chn_excel_list):
    manager = Manager()
    shared_results = manager.list()
    with multiprocessing.Pool(len(chunks)) as pool:
        args = [
            (start, end, crop_box, video_path, fps, FRAME_SKIP, shared_results, chn_excel_list)
            for (start, end) in chunks
        ]
        pool.starmap(process_chunk, args)
    return list(shared_results)

def merge_results_and_save(results, filtered_df, output_csv):
    matched_rows = [{"CHN": chn, "Frame": frame, "Timestamp_ms": int(timestamp_sec * 1000)}
                    for frame, chn, timestamp_sec in results]
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
    final_df.sort_values(by="timestamp", inplace=True)
    final_df.to_csv(output_csv, index=False)

def main(video_path: str, excel_path: str, lane_name: str, roadway_id: int, lane_id: int):
    start = time.time()

    output_csv = f"media/roadways/{roadway_id}/lanes/{lane_id}/output_matched.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    fps, total_frames = load_video_metadata(video_path)

    crop_box = detect_crop_box(video_path)
    if not crop_box:
        raise ValueError("❌ Could not detect 'Chn:' in video frames.")

    filtered_df, chn_excel_list = load_excel_data(excel_path, lane_name)
    chunks = generate_chunks(total_frames, multiprocessing.cpu_count())

    results = run_parallel_processing(chunks, crop_box, video_path, fps, chn_excel_list)
    merge_results_and_save(results, filtered_df, output_csv)
    return True

