#!/usr/bin/env python3
"""
Compare v8 and v9 drone detection models on the same YouTube test data.
Optimized: pre-generate spectrograms, then batch-predict with YOLO.
"""

import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import librosa
from matplotlib.cm import magma as MAGMA_CMAP
from PIL import Image

# ── Spectrogram params (same for v8 and v9) ──
SR = 16000
N_MELS = 320
N_FFT = 2048
HOP_LENGTH = 50
F_MAX = 8000
IMG_SIZE = (640, 640)
DB_MIN = -80.0
DB_MAX = 0.0

# ── Paths ──
V8_MODEL = Path("/home/dremian/drone-sound-detector/model_v8/best.pt")
V9_MODEL = Path("/home/dremian/drone-sound-detector/model_v9/best.pt")
TEST_DIR = Path("/home/dremian/drone-sound-detector/model_v8/youtube_test")
SPEC_DIR = Path("/tmp/drone_comparison_spectrograms")
OUTPUT_FILE = Path("/home/dremian/article_sound_detector/comparison_results.json")

# ── Test files with ground truth ──
TEST_FILES = [
    ("01_Wg5zVxSceDI", "drone", "DJI Air3/Mini3Pro/Mavic3 noise comparison"),
    ("02_3SJoJUyYpxk", "drone", "DJI Neo 2 vs Neo decibel test"),
    ("03_1vm8hux5MiE", "drone", "FPV raw motor noise"),
    ("04_iwn7JPupAUY", "drone", "DJI Mini 5 Pro extreme wind test"),
    ("06_E3Fa4kUXP2c", "not_drone", "Helicopter low flypast"),
    ("07_l0Q_cKMsa_g", "not_drone", "Helicopter Dolby Atmos flyover"),
    ("08_U59zrzPXT_0", "not_drone", "Chainsaw cutting tree"),
    ("09_3eeFFWgn_Gs", "not_drone", "Chainsaw Stihl tree felling"),
    ("10_n--9v3RxYLY", "not_drone", "Harley Iron 883 exhaust"),
    ("11_uRpuGmt4i48", "not_drone", "Triumph Street Triple exhaust"),
    ("12_BAw342Xqxhs", "not_drone", "NYC street ambience"),
    ("13_FWihlmYmEZE", "not_drone", "Cessna 172 propeller overhead"),
    ("14_JaW6gcNk4tw", "not_drone", "Small propeller plane flyover"),
    ("15_nbQYBGfmuog", "not_drone", "Gas lawn mower"),
    ("16_ke41QysEhuQ", "not_drone", "Leaf blower"),
]

CATEGORIES = {
    "01_Wg5zVxSceDI": "Drone", "02_3SJoJUyYpxk": "Drone",
    "03_1vm8hux5MiE": "Drone", "04_iwn7JPupAUY": "Drone",
    "06_E3Fa4kUXP2c": "Helicopter", "07_l0Q_cKMsa_g": "Helicopter",
    "08_U59zrzPXT_0": "Chainsaw", "09_3eeFFWgn_Gs": "Chainsaw",
    "10_n--9v3RxYLY": "Motorcycle", "11_uRpuGmt4i48": "Motorcycle",
    "12_BAw342Xqxhs": "City traffic",
    "13_FWihlmYmEZE": "Airplane", "14_JaW6gcNk4tw": "Airplane",
    "15_nbQYBGfmuog": "Lawn mower", "16_ke41QysEhuQ": "Leaf blower",
}


def generate_spectrograms_for_file(args):
    """Generate all 1-sec spectrogram PNGs for one audio file. (worker function)"""
    folder, audio_path = args
    out_dir = SPEC_DIR / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    n_segments = len(y) // SR
    generated = 0

    for i in range(n_segments):
        segment = y[i * SR : (i + 1) * SR]
        if len(segment) < SR:
            break

        rms = np.sqrt(np.mean(segment ** 2))
        if rms < 1e-4:
            # Write a marker file for silence
            (out_dir / f"seg_{i:04d}_SILENCE.marker").touch()
            continue

        mel_spec = librosa.feature.melspectrogram(
            y=segment, sr=SR, n_mels=N_MELS, n_fft=N_FFT,
            hop_length=HOP_LENGTH, fmax=F_MAX
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=1.0)
        normalized = np.clip((mel_spec_db - DB_MIN) / (DB_MAX - DB_MIN), 0.0, 1.0)
        normalized = normalized[::-1, :]
        colored = MAGMA_CMAP(normalized)
        rgb = (colored[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(rgb).resize(IMG_SIZE, Image.LANCZOS)
        img.save(out_dir / f"seg_{i:04d}.png")
        generated += 1

    return folder, n_segments, generated


def batch_predict(model_path, spec_dirs):
    """Run YOLO batch prediction on all spectrogram directories."""
    from ultralytics import YOLO
    model = YOLO(str(model_path))

    all_results = {}

    for folder in spec_dirs:
        folder_path = SPEC_DIR / folder
        png_files = sorted(folder_path.glob("seg_*.png"))

        if not png_files:
            all_results[folder] = []
            continue

        # Batch predict all PNGs in this folder
        predictions = model.predict(
            source=[str(f) for f in png_files],
            verbose=False,
            batch=16,
        )

        # Build segment results (including silence markers)
        n_segments_total = max(
            len(list(folder_path.glob("seg_*.png"))) +
            len(list(folder_path.glob("seg_*_SILENCE.marker"))),
            len(predictions)
        )

        # Map predictions by segment index
        pred_map = {}
        for png_f, pred in zip(png_files, predictions):
            seg_idx = int(png_f.stem.split("_")[1])
            probs = pred.probs
            names = pred.names
            drone_idx = next(i for i, n in names.items() if n == "drone")
            drone_conf = float(probs.data[drone_idx])
            is_drone = (drone_idx == int(probs.top1))
            pred_map[seg_idx] = ("drone" if is_drone else "not_drone", drone_conf)

        # Build full list including silence
        silence_markers = sorted(folder_path.glob("seg_*_SILENCE.marker"))
        silence_indices = {int(f.stem.split("_")[1]) for f in silence_markers}

        max_idx = 0
        if pred_map:
            max_idx = max(max_idx, max(pred_map.keys()))
        if silence_indices:
            max_idx = max(max_idx, max(silence_indices))

        segment_results = []
        for i in range(max_idx + 1):
            if i in silence_indices:
                segment_results.append(("silence", 0.0))
            elif i in pred_map:
                segment_results.append(pred_map[i])
            # Skip indices with no data (shouldn't happen)

        all_results[folder] = segment_results

    return all_results


def main():
    t0 = time.time()

    # ── Step 1: Pre-generate all spectrograms (parallel) ──
    print(f"Step 1: Generating spectrograms (using {cpu_count()} cores)...")
    SPEC_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for folder, expected, description in TEST_FILES:
        audio_path = TEST_DIR / folder / "audio.wav"
        if audio_path.exists():
            tasks.append((folder, audio_path))

    with Pool(processes=min(cpu_count(), len(tasks))) as pool:
        results = pool.map(generate_spectrograms_for_file, tasks)

    for folder, n_seg, n_gen in results:
        print(f"  {folder}: {n_seg} segments, {n_gen} spectrograms")

    t1 = time.time()
    print(f"  Spectrograms done in {t1-t0:.1f}s\n")

    # ── Step 2: Batch predict with v8 ──
    folders = [folder for folder, _, _ in TEST_FILES
               if (TEST_DIR / folder / "audio.wav").exists()]

    print("Step 2: Running Model v8 inference...")
    v8_results = batch_predict(V8_MODEL, folders)
    t2 = time.time()
    print(f"  v8 done in {t2-t1:.1f}s\n")

    # ── Step 3: Batch predict with v9 ──
    print("Step 3: Running Model v9 inference...")
    v9_results = batch_predict(V9_MODEL, folders)
    t3 = time.time()
    print(f"  v9 done in {t3-t2:.1f}s\n")

    # ── Step 4: Compile results ──
    print("Step 4: Compiling results...\n")
    all_results = []

    for folder, expected, description in TEST_FILES:
        if folder not in v8_results:
            continue

        res_v8 = v8_results[folder]
        res_v9 = v9_results[folder]
        category = CATEGORIES[folder]
        n_segments = max(len(res_v8), len(res_v9))
        if n_segments == 0:
            continue

        v8_drone = sum(1 for r in res_v8 if r[0] == "drone")
        v9_drone = sum(1 for r in res_v9 if r[0] == "drone")
        v8_drone_pct = v8_drone / n_segments * 100
        v9_drone_pct = v9_drone / n_segments * 100

        if expected == "drone":
            v8_correct_pct = v8_drone_pct
            v9_correct_pct = v9_drone_pct
        else:
            v8_correct_pct = (n_segments - v8_drone) / n_segments * 100
            v9_correct_pct = (n_segments - v9_drone) / n_segments * 100

        entry = {
            "folder": folder,
            "description": description,
            "category": category,
            "expected": expected,
            "total_segments": n_segments,
            "v8_drone_segments": v8_drone,
            "v8_not_drone_segments": sum(1 for r in res_v8 if r[0] == "not_drone"),
            "v8_drone_pct": round(v8_drone_pct, 1),
            "v8_correct_pct": round(v8_correct_pct, 1),
            "v9_drone_segments": v9_drone,
            "v9_not_drone_segments": sum(1 for r in res_v9 if r[0] == "not_drone"),
            "v9_drone_pct": round(v9_drone_pct, 1),
            "v9_correct_pct": round(v9_correct_pct, 1),
            "v8_per_second": [(r[0], round(r[1], 4)) for r in res_v8],
            "v9_per_second": [(r[0], round(r[1], 4)) for r in res_v9],
        }
        all_results.append(entry)

        print(f"  {folder} [{category}] — {description}")
        print(f"    v8: {v8_drone_pct:.1f}% drone ({v8_correct_pct:.1f}% correct)")
        print(f"    v9: {v9_drone_pct:.1f}% drone ({v9_correct_pct:.1f}% correct)")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_v8_correct = 0
    total_v9_correct = 0
    total_segments = 0

    for r in all_results:
        n = r["total_segments"]
        total_segments += n
        if r["expected"] == "drone":
            total_v8_correct += r["v8_drone_segments"]
            total_v9_correct += r["v9_drone_segments"]
        else:
            total_v8_correct += r["v8_not_drone_segments"]
            total_v9_correct += r["v9_not_drone_segments"]

    print(f"Total segments: {total_segments}")
    print(f"v8 overall accuracy: {total_v8_correct/total_segments*100:.1f}%")
    print(f"v9 overall accuracy: {total_v9_correct/total_segments*100:.1f}%")
    print(f"\nTotal time: {time.time()-t0:.1f}s")

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
