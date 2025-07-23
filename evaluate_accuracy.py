# evaluate_accuracy.py
"""
Batch-test extraction accuracy for ImprovedDCTSteganography.

Usage:
    python evaluate_accuracy.py                # default settings
    python evaluate_accuracy.py --trials 10    # more repetitions
"""

import os, glob, random, string, argparse, json, math
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

from crc_utils         import add_crc, check_crc 
from dct_claude_quant import ImprovedDCTSteganography   # <-- your class

###############################################################################
# -----------------------------  CONFIGURATION  ----------------------------- #
###############################################################################

MESSAGE_LENGTHS     = [8, 16]      # characters
JPEG_QUALITIES      = [None, 95, 85]   # None ⇒ keep original format (PNG or JPEG)
TRIALS_PER_SETTING  = 5                # repeat per (image, L, Q) to get variance
STEGO_TMP_PATH      = "tmp_stego.png"  # overwritten each trial

###############################################################################
# ------------------------------  UTILITIES  -------------------------------- #
###############################################################################

def random_msg(n_chars: int) -> str:
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n_chars))

def msg_to_bits(msg: str) -> list[int]:
    return [int(b) for c in msg for b in format(ord(c), "08b")]

def bit_error_rate(bits_a: list[int], bits_b: list[int]) -> float:
    """Pad to equal length, then compute BER."""
    L = max(len(bits_a), len(bits_b))
    a = np.array(bits_a + [0]*(L-len(bits_a)))
    b = np.array(bits_b + [0]*(L-len(bits_b)))
    return float(np.mean(a != b))

###############################################################################
# -----------------------------  CORE EVALUATION  --------------------------- #
###############################################################################

def evaluate_one(image_path: str,
                 msg_len: int,
                 jpeg_quality: int | None = None) -> dict:
    """
    Returns a dict with BER and exact-match flag for one embed→extract round.
    """
    # ------------------------------------------------------------------ load
    orig_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if orig_img is None:
        raise RuntimeError(f"Failed to read {image_path}")

    # ------------------------------------------------------------------ (opt) re-encode JPEG
    work_img = orig_img.copy()
    if jpeg_quality is not None:
        _, enc = cv2.imencode(".jpg", orig_img,
                              [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        work_img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # ------------------------------------------------------------------ embed
    # steg = ImprovedDCTSteganography(alpha=0.05)
    steg = ImprovedDCTSteganography(alpha=0.012, repetition_factor=1)
    gt_msg = random_msg(msg_len)

    # stego_array = steg.embed(work_img, gt_msg)

    # --- (New added code)
    try:
        stego_array = steg.embed(work_img, gt_msg)     # ← may raise ValueError
    except ValueError as e:                            #  (capacity exhausted)
        print(f"[skip] {e} – {os.path.basename(image_path)}, "
            f"L={msg_len}, Q={jpeg_quality or 100}")
        return None  # tell caller to ignore
    # --- (End of the added code)                                     

    # save & reload to follow the same pipeline as FastAPI
    cv2.imwrite(STEGO_TMP_PATH, stego_array)
    decoded_msg = steg.extract_text(STEGO_TMP_PATH)

    # ------------------------------------------------------------------ metrics
    ber  = bit_error_rate(msg_to_bits(gt_msg), msg_to_bits(decoded_msg))
    exact = int(decoded_msg == gt_msg)


    # --- (Newly added code)
    # # -----------------------------------------------------------
    # # CRC-protected embed → extract with ONE automatic retry
    # # -----------------------------------------------------------
    # steg  = ImprovedDCTSteganography(alpha=0.012, repetition_factor=7)

    # gt_msg  = random_msg(msg_len)        # ground-truth message
    # payload = add_crc(gt_msg)            # +2 bytes CRC-16 (latin-1 string)
    # print("✅ using CRC payload:", payload, "len=", len(payload)) 

    # def try_round(img: np.ndarray, engine: ImprovedDCTSteganography) -> str | None:
    #     """
    #     One full embed→extract pass.
    #     Returns the decoded message if CRC passes, else None.
    #     May raise ValueError if capacity is exhausted.
    #     """
    #     stego_arr = engine.embed(img, payload)          # ← can raise
    #     cv2.imwrite(STEGO_TMP_PATH, stego_arr)          # simulate disk I/O
    #     raw        = engine.extract_text(STEGO_TMP_PATH)
    #     return check_crc(raw)                           # None if CRC mismatches

    # # -------- first attempt ------------------------------------
    # try:
    #     decoded_msg = try_round(work_img, steg)
    # except ValueError:                                  # not enough capacity
    #     decoded_msg = None

    # # -------- one retry ----------------------------------------
    # if decoded_msg is None:
    #     try:
    #         steg2       = ImprovedDCTSteganography(alpha=0.012, repetition_factor=7)
    #         decoded_msg = try_round(work_img, steg2)
    #     except ValueError:
    #         decoded_msg = None                          # still no capacity

    # # -------- metrics ------------------------------------------
    # if decoded_msg is None:            # both attempts failed
    #     ber   = 1.0                    # treat as all bits wrong
    #     exact = 0
    # else:
    #     ber   = bit_error_rate(msg_to_bits(gt_msg), msg_to_bits(decoded_msg))
    #     exact = int(decoded_msg == gt_msg)
    
    # # --- (End of newly added line)

    return {
        "file":     os.path.basename(image_path),
        "Q":        jpeg_quality if jpeg_quality is not None else 100,
        "Lchars":   msg_len,
        "BER":      ber,
        "exact":    exact
    }

###############################################################################
# -----------------------------  MAIN ROUTINE  ------------------------------ #
###############################################################################

def run_dataset(image_glob: str,
                trials: int = TRIALS_PER_SETTING) -> pd.DataFrame:

    images = sorted(glob.glob(image_glob))
    if not images:
        raise RuntimeError(f"No images matched “{image_glob}”")

    results = []
    for img in tqdm(images, desc="Images"):
        for L in MESSAGE_LENGTHS:
            for Q in JPEG_QUALITIES:
                for _ in range(trials):
                    # results.append(evaluate_one(img, L, Q))
                    # --- (Newly added code)
                    res = evaluate_one(img, L, Q)   # ← may return None
                    if res is not None:             # ← keep only valid trials
                        results.append(res)
                    # --- (End of newly added code)

    return pd.DataFrame(results)

def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (Q, Lchars) and compute mean ± std."""
    grouped = df.groupby(["Q", "Lchars"]).agg(
        BER_mean  = ("BER",  "mean"),
        BER_std   = ("BER",  "std"),
        Exact_per   = ("exact", "mean")   # 0-1 ; multiply by 100 if you prefer %
    ).reset_index()
    return grouped

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob",
                        default="Images/*.[pjPJ][pnPN]*",
                        help="Glob pattern for test images (default: Images/*.png|jpg)")
    parser.add_argument("--trials", type=int, default=TRIALS_PER_SETTING,
                        help="Repetitions per (image, length, quality)")
    args = parser.parse_args()

    df_raw = run_dataset(args.glob, args.trials)
    summary = summarise(df_raw)

    # Pretty print
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n=== Extraction-accuracy summary (Exact-match ↑  /  BER ↓) ===\n")
    print(summary.to_string(index=False))

    # Optionally save to CSV / JSON
    summary.to_csv("accuracy_summary.csv", index=False)
    with open("accuracy_summary.json", "w") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)

    print("\nDetailed per-trial data written to accuracy_summary.csv/.json")
