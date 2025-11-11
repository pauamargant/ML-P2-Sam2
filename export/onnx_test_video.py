# sam2-onnx-cpp/export/onnx_test_video.py
#!/usr/bin/env python3
# CPU-optimized: encoder fast (BASIC/EXTENDED), decoder/memory safe.
import os
# Limit BLAS thread pools before NumPy loads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import time
import argparse
import cv2
import numpy as np
from PyQt5 import QtWidgets

from onnx_test_utils import (
    print_system_info, set_cv2_threads,
    make_encoder_session, make_safe_session,
    prepare_image, prepare_points, prepare_box_prompt,
    run_encoder, run_decoder,
    prefer_quantized_encoder, green_overlay, compute_display_base,
)

def _stack_memory(frames_feats, frames_pos):
    """
    Concatenate a list of memory frames along the time axis expected by memory_attention.
    """
    if not frames_feats or not frames_pos:
        return None, None
    mem_feats_cat = frames_feats[0]
    mem_pos_cat   = frames_pos[0]
    if len(frames_feats) > 1:
        mem_feats_cat = np.concatenate(frames_feats, axis=0).astype(np.float32, copy=False)
        mem_pos_cat   = np.concatenate(frames_pos,   axis=0).astype(np.float32, copy=False)
    return mem_feats_cat, mem_pos_cat

# Stack a list of 2D arrays along axis 0 (e.g., [(1,256), (1,256), ...] -> (T,256))
def _stack_rows(arrs):
    if not arrs:
        return None
    rows = []
    for a in arrs:
        a = a.astype(np.float32, copy=False)
        if a.ndim == 1:
            a = a[None, :]               # (256,) -> (1,256)
        elif a.ndim > 2:
            a = a.reshape(-1, a.shape[-1])  # squeeze any extra dims to (N,256)
        rows.append(a)
    return np.concatenate(rows, axis=0).astype(np.float32, copy=False)

def interactive_select_points(first_bgr, sess_enc, sess_dec, enc_shape):
    enc_h, enc_w = enc_shape
    tensor, (h_org, w_org) = prepare_image(first_bgr, (enc_h, enc_w))
    enc = run_encoder(sess_enc, tensor)
    embed = enc["image_embeddings"]; f0 = enc["high_res_features1"]; f1 = enc["high_res_features2"]

    base, scale = compute_display_base(first_bgr, max_side=1200)
    points, labels = [], []

    def show(mask=None):
        vis = base.copy()
        if mask is not None:
            m = cv2.resize(mask, (base.shape[1], base.shape[0]), cv2.INTER_NEAREST)
            vis = green_overlay(vis, m, 0.5)
        for i, (px, py) in enumerate(points):
            col = (0, 0, 255) if labels[i] == 1 else (255, 0, 0)
            cv2.circle(vis, (int(px*scale), int(py*scale)), 6, col, -1)
        cv2.imshow("First Frame – SAM-2", vis)

    def run():
        if not points:
            show(); return
        pts, lbls = prepare_points(points, labels, (h_org, w_org), (enc_h, enc_w))
        _, mask_hi, _ = run_decoder(sess_dec, pts, lbls, embed, f0, f1)
        show((mask_hi[0, 0] > 0).astype(np.uint8))

    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x/scale), int(y/scale))); labels.append(1); run()
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append((int(x/scale), int(y/scale))); labels.append(0); run()
        elif event == cv2.EVENT_MBUTTONDOWN:
            points.clear(); labels.clear(); run()

    cv2.namedWindow("First Frame – SAM-2")
    cv2.setMouseCallback("First Frame – SAM-2", cb)
    run()
    print("[INFO] L-click=FG, R-click=BG, M-click=reset. ESC/Enter to continue.")
    while True:
        if cv2.waitKey(20) & 0xFF in (27, 13):
            break
    cv2.destroyAllWindows()
    return points, labels, embed, f0, f1, (h_org, w_org)


def interactive_select_box(first_bgr, sess_enc, sess_dec, enc_shape):
    enc_h, enc_w = enc_shape
    tensor, (h_org, w_org) = prepare_image(first_bgr, (enc_h, enc_w))
    enc = run_encoder(sess_enc, tensor)
    embed = enc["image_embeddings"]; f0 = enc["high_res_features1"]; f1 = enc["high_res_features2"]

    base, scale = compute_display_base(first_bgr, max_side=1200)
    rect_s = rect_e = None
    drawing = False

    def show(mask=None):
        vis = base.copy()
        if mask is not None:
            m = cv2.resize(mask, (base.shape[1], base.shape[0]), cv2.INTER_NEAREST)
            vis = green_overlay(vis, m, 0.5)
        if rect_s and rect_e:
            cv2.rectangle(vis, rect_s, rect_e, (0, 255, 255), 2)
        cv2.imshow("First Frame – SAM-2", vis)

    def run():
        if not(rect_s and rect_e):
            show(); return
        x1d, y1d = rect_s; x2d, y2d = rect_e
        x1, x2 = sorted((int(x1d/scale), int(x2d/scale)))
        y1, y2 = sorted((int(y1d/scale), int(y2d/scale)))
        pts, lbls = prepare_box_prompt((x1, y1, x2, y2), (h_org, w_org), (enc_h, enc_w))
        _, mask_hi, _ = run_decoder(sess_dec, pts, lbls, embed, f0, f1)
        show((mask_hi[0, 0] > 0).astype(np.uint8))

    def cb(event, x, y, flags, param):
        nonlocal rect_s, rect_e, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True; rect_s = rect_e = (x, y); show()
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            rect_e = (x, y); show()
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False; rect_e = (x, y); run()
        elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK):
            rect_s = rect_e = None; show()

    cv2.namedWindow("First Frame – SAM-2")
    cv2.setMouseCallback("First Frame – SAM-2", cb)
    show()
    print("[INFO] Draw rectangle, release → preview. ESC/Enter to continue.")
    while True:
        if cv2.waitKey(20) & 0xFF in (27, 13):
            break
    cv2.destroyAllWindows()

    if rect_s and rect_e:
        x1d, y1d = rect_s; x2d, y2d = rect_e
        x1, x2 = sorted((int(x1d/scale), int(x2d/scale)))
        y1, y2 = sorted((int(y1d/scale), int(y2d/scale)))
        box = (x1, y1, x2, y2)
    else:
        box = None
    return box, embed, f0, f1, (h_org, w_org)


def process_video(args: argparse.Namespace) -> None:
    """
    Segment a user-selected video using SAM-2 ONNX components and write a mask overlay video.

    Pipeline per frame:
        1) ENCODER: Produce multi-scale visual features:
            - image_embeddings (low-res, semantic-rich)
            - high_res_features1 (f0) and high_res_features2 (f1) for spatial detail
            - vision_pos_embed (positional encodings; available on frames > 0 path)
        2) MEMORY ATTENTION (fidx > 0): Fuse current visual embedding with rolling memory
           from previous frames to maintain temporal consistency and object identity.
        3) DECODER: Predict mask logits using (fused_embed or enc_embed) + (f0, f1) and,
           on the first frame, the user prompt (points or bounding box).
        4) MEMORY ENCODER: Convert current mask + pixel features into memory tensors
           for the next frame.
        5) OVERLAY & WRITE: Threshold logits, overlay green mask on the original frame, and write.

    Args:
        args.model_size (str):  One of {"tiny", "small", "base_plus", "large"} selecting the
                               checkpoint subfolder under "checkpoints/".
        args.prompt (str):      "seed_points" or "bounding_box"; controls the interactive prompt
                               on the first frame.
        args.max_frames (int):  If > 0, limit the number of processed frames; else process all.
        args.video (str):       Absolute path of the user-selected input video (injected by main()).

    Side effects:
        - Opens GUI windows for first-frame interactive prompting.
        - Creates an output video next to the input with suffix "_mask_overlay.mkv".
        - Prints per-frame timing diagnostics (ms) for encoder/attention/decoder/memory-encoder.

    Raises:
        SystemExit: If required ONNX files are missing, the video cannot be opened, or
                    the output writer cannot be created.
    """
    # -------------------------------------------------------------------------
    # 0) Resolve ONNX file paths for the selected model size
    # -------------------------------------------------------------------------
    # Directory containing four ONNX files: image_encoder(.quant?), image_decoder,
    # memory_encoder, memory_attention.
    ckpt_dir: str = os.path.join("checkpoints", args.model_size)

    # Prefer quantized encoder if available (faster on CPU); returns full path or None.
    enc_path: str = prefer_quantized_encoder(ckpt_dir)
    if enc_path is None:
        sys.exit(f"ERROR: Encoder ONNX not found in {ckpt_dir}")

    # Helper to compose standard ONNX paths for the remaining components.
    paths = lambda name: os.path.join(ckpt_dir, f"{name}.onnx")
    dec_path: str = paths("image_decoder")
    men_path: str = paths("memory_encoder")
    mat_path: str = paths("memory_attention")

    # Validate existence of required ONNX files.
    for p in (dec_path, men_path, mat_path):
        if not os.path.exists(p):
            sys.exit(f"ERROR: ONNX file missing: {p}")

    # -------------------------------------------------------------------------
    # 1) Initialize ONNX Runtime sessions
    # -------------------------------------------------------------------------
    # Encoder session may use different providers/options (e.g., quantized model).
    sess_enc = make_encoder_session(enc_path)

    # "Safe" sessions configure RT options to minimize dynamic-shape pitfalls and
    # improve stability across platforms (CPU-focused; still works with GPU builds).
    sess_dec = make_safe_session(dec_path, tag="decoder")            # image decoder
    sess_men = make_safe_session(men_path, tag="memory_encoder")     # memory encoder
    sess_mat = make_safe_session(mat_path, tag="memory_attention")   # memory attention

    # The encoder's first input has shape [N, C, H, W]; retrieve H and W.
    # These determine how we resize frames for encoder input.
    enc_h, enc_w = sess_enc.get_inputs()[0].shape[2:]
    print(f"[INFO] Encoder input = {(enc_h, enc_w)}")

    # -------------------------------------------------------------------------
    # 2) Open the input video and prepare the output writer
    # -------------------------------------------------------------------------
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit("ERROR: cannot open video")

    # Original video properties (used for output writer and upscaling masks).
    fps: float = cap.get(cv2.CAP_PROP_FPS)
    w_org: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_org: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output file path: same base name + "_mask_overlay.mkv".
    out_path: str = os.path.splitext(args.video)[0] + "_mask_overlay.mkv"

    # Create VideoWriter. XVID works well for many containers; fallback FPS if missing.
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'XVID'),
        fps if fps and fps > 0 else 25.0,
        (w_org, h_org)
    )
    if not writer.isOpened():
        sys.exit("ERROR: cannot open VideoWriter")

    # -------------------------------------------------------------------------
    # 3) First frame acquisition (used for interactive prompt and seeding)
    # -------------------------------------------------------------------------
    ret, first_bgr = cap.read()
    if not ret:
        sys.exit("ERROR: empty video")

    # Rewind to frame 0 so the main loop also processes it.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # -------------------------------------------------------------------------
    # 4) Interactive prompt on first frame + first-frame encoding
    # -------------------------------------------------------------------------
    # On the first frame, we:
    #   - run the encoder (inside the interactive helper),
    #   - let the user define the prompt,
    #   - preview masks live,
    #   - return the encoder outputs for frame 0: embed0, f0_0, f1_0,
    #   - convert the prompt to decoder inputs (pts0, lbls0).
    if args.prompt == "bounding_box":
        # User draws a rectangle; helper encodes the frame and previews masks.
        box, embed0, f0_0, f1_0, (h_org, w_org) = interactive_select_box(
            first_bgr, sess_enc, sess_dec, (enc_h, enc_w)
        )
        # Convert the rectangle into SAM2's box-prompt tensor form for the decoder.
        if box:
            pts0, lbls0 = prepare_box_prompt(box, (h_org, w_org), (enc_h, enc_w))
        else:
            pts0, lbls0 = (None, None)  # If user cancels, decoder will run without prompt (not ideal).
    else:
        # User clicks FG/BG points; helper encodes the frame and previews masks.
        pts, lbls, embed0, f0_0, f1_0, (h_org, w_org) = interactive_select_points(
            first_bgr, sess_enc, sess_dec, (enc_h, enc_w)
        )
        # Convert point list + labels into tensors in encoder resolution space.
        pts0, lbls0 = prepare_points(pts, lbls, (h_org, w_org), (enc_h, enc_w))

    # Ensure first-frame encoder outputs are float32 (what ONNX models expect).
    embed0 = embed0.astype(np.float32, copy=False)   # (B, C=256, H/16, W/16) approx
    f0_0   = f0_0.astype(np.float32, copy=False)     # (B, C=256, H/8,  W/8)
    f1_0   = f1_0.astype(np.float32, copy=False)     # (B, C=256, H/4,  W/4)

    # -------------------------------------------------------------------------
    # 5) Initialize per-video state and encoder I/O names
    # -------------------------------------------------------------------------
    fidx: int = 0

    # Rolling memory from previous frames (produced by memory_encoder):
    #   mem_feats: spatial memory features (H/16×W/16-ish grid encoded)
    #   mem_pos:   corresponding positional encodings
    mem_window = max(1, int(args.mem_window))
    mem_feats_list = []   # list of per-frame memory features from memory_encoder
    mem_pos_list   = []   # list of per-frame memory positional embeddings
    mem_obj_list = []     # list of per-frame memory object embeddings


    # Cache encoder I/O names for faster sess_enc.run(...) calls.
    enc_input_name: str = sess_enc.get_inputs()[0].name
    enc_out_names: list[str] = [o.name for o in sess_enc.get_outputs()]

    # -------------------------------------------------------------------------
    # 6) Main per-frame loop
    # -------------------------------------------------------------------------
    while True:
        # Stop when video ends or we have reached user-specified frame limit.
        ret, frame = cap.read()
        if not ret or (args.max_frames > 0 and fidx >= args.max_frames):
            break

        # ----------------------- 6a) ENCODER -----------------------
        # Produce visual features for this frame:
        #   enc_embed: low-res semantic embedding (main pixel features)
        #   f0, f1:    higher-res feature maps for decoder precision
        #   vis_pos:   positional encodings (used by memory attention)
        t_enc = time.time()
        if fidx == 0:
            # Reuse first-frame outputs computed during the interactive prompt.
            enc_embed = embed0
            f0        = f0_0
            f1        = f1_0
            vis_pos   = None  # Not produced on the first-frame interactive path
            enc_ms = (time.time() - t_enc) * 1000.0
        else:
            # Preprocess and run encoder ONNX
            tensor, _ = prepare_image(frame, (enc_h, enc_w))  # (1,3,enc_h,enc_w), float32, normalized
            enc_vals = sess_enc.run(None, {enc_input_name: tensor})
            enc = dict(zip(enc_out_names, enc_vals))

            # Cast to float32 and name the outputs:
            enc_embed = enc["image_embeddings"].astype(np.float32, copy=False)
            f0        = enc["high_res_features1"].astype(np.float32, copy=False)
            f1        = enc["high_res_features2"].astype(np.float32, copy=False)
            vis_pos   = enc["vision_pos_embed"].astype(np.float32, copy=False)   # needed by memory attention
            enc_ms = (time.time() - t_enc) * 1000.0

        # ------------------- 6b) MEMORY ATTENTION -------------------

        if fidx > 0 and len(mem_feats_list) > 0:
            t_mat = time.time()
            mem_feats_cat, mem_pos_cat = _stack_memory(mem_feats_list, mem_pos_list)
            # note: This demo passes memory_0 (per-object tokens) as an empty array,
            #       so only spatial ("memory_1") memory is used.
            attn_inputs = {
                "current_vision_feat":      enc_embed.astype(np.float32, copy=False),
                "current_vision_pos_embed": vis_pos.astype(np.float32, copy=False),
                "memory_0":                 np.zeros((0, 256), np.float32),  # unchanged: no per-object tokens yet
                "memory_1":                 mem_feats_cat.astype(np.float32, copy=False),
                "memory_pos_embed":         mem_pos_cat.astype(np.float32, copy=False),
            }
            # fused_embed has the same spatial resolution as enc_embed but enriched with memory.
            fused_embed = sess_mat.run(None, attn_inputs)[0].astype(np.float32, copy=False)
            mat_ms = (time.time() - t_mat) * 1000.0
        else:
            fused_embed = enc_embed
            mat_ms = 0.0

        # ------------------------ 6c) DECODER ------------------------
        # Predict mask logits. On the first frame, include the user prompt tensors.
        t_dec = time.time()
        if fidx == 0:
            # pts0, lbls0 come from the first-frame prompt (points or box).
            # Decoder also uses f0 and f1 to recover fine spatial detail.
            _, mask_for_mem, _ = run_decoder(sess_dec, pts0, lbls0, fused_embed, f0, f1)
        else:
            # On subsequent frames, omit prompts; rely on fused features + memory.
            _, mask_for_mem, _ = run_decoder(sess_dec, None, None, fused_embed, f0, f1)
        dec_ms = (time.time() - t_dec) * 1000.0

        # --------------------- 6d) MEMORY ENCODER ---------------------
        # Convert current logits + pixel features into memory for next frames.
        t_men = time.time()

        men_inputs = {
            "mask_for_mem": mask_for_mem[:, 0:1].astype(np.float32, copy=False),
            "pix_feat":     fused_embed.astype(np.float32, copy=False),
        }
        men_out = sess_men.run(None, men_inputs)

        mem_feats_new, mem_pos_new, mem_obj_new = [x.astype(np.float32, copy=False) for x in men_out]

        # Append new frame’s memory
        mem_feats_list.append(mem_feats_new)
        mem_pos_list.append(mem_pos_new)
        mem_obj_list.append(mem_obj_new)

        # Prune to sliding window length (drop oldest if beyond mem_window)
        if len(mem_feats_list) > mem_window:
            mem_feats_list.pop(0)
            mem_pos_list.pop(0)
            mem_obj_list.pop(0)

        men_ms = (time.time() - t_men) * 1000.0

        # -------------------- 6e) OVERLAY & WRITE --------------------
        # Convert logits to a binary mask in original resolution and write overlay.
        logits: np.ndarray = mask_for_mem[0, 0]  # encoder-resolution logits (Henc×Wenc)
        mask_hi: np.ndarray = cv2.resize(logits, (w_org, h_org), cv2.INTER_LINEAR)  # upsample
        mask: np.ndarray = (mask_hi > 0).astype(np.uint8)  
        
        # Preview on screen too (optional)
        preview = green_overlay(frame, mask, 0.5)
        cv2.imshow("SAM-2 Video – Live Overlay", preview)
        # Press ESC to stop early
        if (cv2.waitKey(1) & 0xFF) == 27:
            break         
        # threshold @ 0
        writer.write(green_overlay(frame, mask, 0.5))                               # blend & write

        # -------------------------- 6f) LOGGING --------------------------
        if fidx == 0:
            print(f"Frame {fidx:03d} - Enc:{enc_ms:.1f} | Dec:{dec_ms:.1f} | MemEnc:{men_ms:.1f}")
        else:
            print(f"Frame {fidx:03d} - Enc:{enc_ms:.1f} | Attn:{mat_ms:.1f} | Dec:{dec_ms:.1f} | MemEnc:{men_ms:.1f}")

        fidx += 1

    # -------------------------------------------------------------------------
    # 7) Cleanup & summary
    # -------------------------------------------------------------------------
    cap.release()
    writer.release()
    print(f"Done! Wrote {fidx} frames with overlays to {out_path}")



def main():
    print_system_info()
    set_cv2_threads(1)

    ap = argparse.ArgumentParser(description="Video segmentation demo for SAM-2 ONNX (CPU-optimized)")
    ap.add_argument("--model_size", default="tiny", choices=["base_plus", "large", "small", "tiny"])
    ap.add_argument("--prompt", default="seed_points", choices=["seed_points", "bounding_box"])
    ap.add_argument("--max_frames", type=int, default=0, help="Max frames to process (0 = all).")
    ap.add_argument("--mem_window", type=int, default=1, help="How many past frames to keep in the memory bank (>=1).")
    args = ap.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    vid, _ = QtWidgets.QFileDialog.getOpenFileName(
        None, "Select Video", "",
        "Video files (*.mp4 *.mkv *.avi *.mov *.m4v);;All files (*.*)")
    if not vid:
        sys.exit("No video selected – exiting.")
    args.video = vid
    print(f"[INFO] Selected video: {vid}")

    process_video(args)

if __name__ == "__main__":
    main()
