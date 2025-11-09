"""
Interactive Video Annotator for SAM-2 ONNX Models.

This script provides an interactive GUI for annotating video frames using SAM-2 (Segment Anything Model 2)
with ONNX runtime for efficient inference. Users can add foreground/background points on frames,
generate masks, and propagate annotations across frames using memory features.

Features:
- Point-based annotation with real-time mask prediction
- Memory-based propagation to adjacent frames
- Thumbnail carousel for navigating annotated frames
- Integrated memory status display on thumbnails

Controls:
- Left-click: Add foreground point
- Right-click: Add background point
- Middle-click: Clear points on current frame
- 'n': Predict next frame using memory
- 'b': Predict previous frame using memory
- Spacebar: Toggle sequential prediction playback
- 'd'/'a': Navigate to next/previous frame
- 'j'/'l': Scroll carousel left/right
- ESC or 'q': Quit
"""

import os
# Limit BLAS thread pools before NumPy loads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import numpy as np
import argparse
from pathlib import Path
import sys
from PyQt5 import QtWidgets

from onnx_test_utils import (
    print_system_info, set_cv2_threads,
    make_encoder_session, make_safe_session,
    prepare_image, prepare_points,
    run_encoder, run_decoder,
    prefer_quantized_encoder, green_overlay,
)

class InteractiveAnnotator:
    """
    Interactive video annotator using SAM-2 ONNX models.

    Provides a GUI for annotating video frames with points, generating masks,
    and propagating annotations across frames using memory features.
    """

    def __init__(self, args):
        """
        Initialize the annotator with models and video.

        Args:
            args: Parsed command-line arguments containing model_size and video path.
        """
        self.args = args
        self.load_models()
        self.load_video()

        self.annotations = {}  # {frame_idx: {"points": [], "labels": [], "mask": None}}
        self.frame_cache = {}  # Cache for embeddings
        self.memory_cache = {} # Cache for memory features

        self.current_frame_idx = 0
        self.max_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.base_image = None
        self.display_scale = 1.0
        self.window_scale = 1.5
        self.display_width = 1280  # Max width for the main display

        self.carousel_image = None
        self.thumbnail_width = 160
        self.thumbnail_height = 90
        self.carousel_scroll_x = 0
        self.is_playing = False
        self.status_message = ""

        self.init_ui()

    def load_models(self):
        """
        Load the ONNX models for encoder, decoder, memory encoder, and memory attention.
        """
        REPO_ROOT = Path(__file__).resolve().parent.parent
        ckpt_dir = REPO_ROOT / "checkpoints" / self.args.model_size
        enc_path = prefer_quantized_encoder(str(ckpt_dir))
        if enc_path is None:
            sys.exit(f"ERROR: Encoder ONNX not found in {ckpt_dir}")

        dec_path = str((ckpt_dir / "image_decoder.onnx").resolve())
        men_path = str((ckpt_dir / "memory_encoder.onnx").resolve())
        mat_path = str((ckpt_dir / "memory_attention.onnx").resolve())
        
        for p in [dec_path, men_path, mat_path]:
            if not os.path.exists(p):
                sys.exit(f"ERROR: ONNX file missing: {p}")

        self.sess_enc = make_encoder_session(enc_path)
        self.sess_dec = make_safe_session(dec_path, tag="decoder")
        self.sess_men = make_safe_session(men_path, tag="memory_encoder")
        self.sess_mat = make_safe_session(mat_path, tag="memory_attention")
        self.enc_h, self.enc_w = self.sess_enc.get_inputs()[0].shape[2:]
        print(f"[INFO] Encoder input = {(self.enc_h, self.enc_w)}")

    def load_video(self):
        """
        Load the video file and get its properties.
        """
        self.cap = cv2.VideoCapture(self.args.video)
        if not self.cap.isOpened():
            sys.exit("ERROR: cannot open video")
        self.w_org = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h_org = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def init_ui(self):
        """
        Initialize the GUI windows and callbacks.
        """
        cv2.namedWindow("Interactive Annotator", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Frame", "Interactive Annotator", 0, self.max_frames - 1, self.on_trackbar_change)
        cv2.setMouseCallback("Interactive Annotator", self.on_mouse_event)

        cv2.namedWindow("Annotation Browser")
        cv2.setMouseCallback("Annotation Browser", self.on_carousel_mouse_event)

        self.update_frame()
        self.update_carousel()

    def on_trackbar_change(self, frame_idx):
        if frame_idx != self.current_frame_idx:
            self.current_frame_idx = frame_idx
            self.update_frame()

    def on_mouse_event(self, event, x, y, flags, param):
        """
        Handle mouse events on the main annotator window.

        Left-click adds foreground point, right-click background, middle-click clears.
        """
        if self.is_playing or self.base_image is None:
            return

        # Co-ordinates on the original image
        px = int(x / self.display_scale)
        py = int(y / self.display_scale)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.add_point(px, py, 1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.add_point(px, py, 0)
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.clear_points()

    def add_point(self, px, py, label):
        """
        Add a point annotation to the current frame.

        Args:
            px, py: Point coordinates
            label: 1 for foreground, 0 for background
        """
        frame_annotations = self.annotations.setdefault(self.current_frame_idx, {"points": [], "labels": [], "mask": None})
        frame_annotations["points"].append((px, py))
        frame_annotations["labels"].append(label)
        self.run_prediction()
        self.update_carousel()

    def clear_points(self):
        if self.current_frame_idx in self.annotations:
            self.annotations[self.current_frame_idx] = {"points": [], "labels": [], "mask": None}
        if self.current_frame_idx in self.memory_cache:
            del self.memory_cache[self.current_frame_idx]
        self.update_frame(show_mask=False)
        self.update_carousel()

    def get_frame_features(self, frame_idx):
        if frame_idx in self.frame_cache:
            return self.frame_cache[frame_idx]

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, bgr_frame = self.cap.read()
        if not ret:
            return None

        tensor, (h_org, w_org) = prepare_image(bgr_frame, (self.enc_h, self.enc_w))
        enc = run_encoder(self.sess_enc, tensor)
        
        features = {
            "image_embeddings": enc["image_embeddings"],
            "high_res_features1": enc["high_res_features1"],
            "high_res_features2": enc["high_res_features2"],
            "vision_pos_embed": enc.get("vision_pos_embed"),
            "original_size": (h_org, w_org),
            "bgr_frame": bgr_frame
        }
        self.frame_cache[frame_idx] = features
        return features

    def update_frame(self, show_mask=True):
        features = self.get_frame_features(self.current_frame_idx)
        if features is None:
            return

        self.base_image = features["bgr_frame"]
        
        frame_annotations = self.annotations.get(self.current_frame_idx, {})
        has_points = "points" in frame_annotations and frame_annotations["points"]
        has_mask = "mask" in frame_annotations and frame_annotations["mask"] is not None

        if show_mask and has_points:
            self.run_prediction()
        elif show_mask and has_mask:
            self.draw_ui(frame_annotations["mask"])
        else:
            self.draw_ui()

    def run_prediction(self):
        features = self.get_frame_features(self.current_frame_idx)
        if features is None:
            return

        frame_annotations = self.annotations.get(self.current_frame_idx)
        if not frame_annotations or not frame_annotations["points"]:
            self.draw_ui()
            return

        points = frame_annotations["points"]
        labels = frame_annotations["labels"]
        h_org, w_org = features["original_size"]

        pts, lbls = prepare_points(points, labels, (h_org, w_org), (self.enc_h, self.enc_w))
        
        self.status_message = "Predicting mask..."
        self.draw_ui()  # Update display with message
        
        _, mask_hi, _ = run_decoder(
            self.sess_dec, pts, lbls, 
            features["image_embeddings"], 
            features["high_res_features1"], 
            features["high_res_features2"]
        )
        
        mask = (mask_hi[0, 0] > 0).astype(np.uint8)
        frame_annotations["mask"] = mask # Store the generated mask

        # Generate and cache memory features
        men_out = self.sess_men.run(None, {
            "mask_for_mem": mask_hi[:, 0:1].astype(np.float32, copy=False),
            "pix_feat":     features["image_embeddings"].astype(np.float32, copy=False),
        })
        self.memory_cache[self.current_frame_idx] = [x.astype(np.float32, copy=False) for x in men_out[:2]]

        self.status_message = ""
        self.draw_ui(mask)
        self.update_carousel()

    def draw_ui(self, mask=None):
        if self.base_image is None:
            return
            
        vis = self.base_image.copy()
        if mask is not None:
            m = cv2.resize(mask, (self.base_image.shape[1], self.base_image.shape[0]), cv2.INTER_NEAREST)
            vis = green_overlay(vis, m, 0.5)
        else:
            # Check if there's a stored mask from prediction
            frame_annotations = self.annotations.get(self.current_frame_idx)
            if frame_annotations and frame_annotations.get("mask") is not None:
                m = cv2.resize(frame_annotations["mask"], (self.base_image.shape[1], self.base_image.shape[0]), cv2.INTER_NEAREST)
                vis = green_overlay(vis, m, 0.5)

        # Calculate display scale
        h, w = vis.shape[:2]
        self.display_scale = self.display_width / w
        new_h, new_w = int(h * self.display_scale), self.display_width
        vis_display = cv2.resize(vis, (new_w, new_h))

        frame_annotations = self.annotations.get(self.current_frame_idx)
        if frame_annotations:
            points = frame_annotations["points"]
            labels = frame_annotations["labels"]
            for i, (px, py) in enumerate(points):
                col = (0, 0, 255) if labels[i] == 1 else (255, 0, 0)
                # Scale points to the displayed image size
                disp_x = int(px * self.display_scale)
                disp_y = int(py * self.display_scale)
                cv2.circle(vis_display, (disp_x, disp_y), 6, col, -1)

        cv2.imshow("Interactive Annotator", vis_display)
        cv2.setTrackbarPos("Frame", "Interactive Annotator", self.current_frame_idx)
        self.update_carousel()

        # Show status message if any
        if self.status_message:
            cv2.putText(vis_display, self.status_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("Interactive Annotator", vis_display)

    def on_carousel_mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            annotated_frames = sorted(self.annotations.keys())
            if not annotated_frames:
                return

            clicked_idx = (x + self.carousel_scroll_x) // self.thumbnail_width
            if 0 <= clicked_idx < len(annotated_frames):
                self.on_trackbar_change(annotated_frames[clicked_idx])

    def update_carousel(self):
        """
        Update the annotation browser carousel with thumbnails and info.
        """
        annotated_frames = sorted([idx for idx, data in self.annotations.items() if data.get("points") or data.get("mask") is not None])
        if not annotated_frames:
            self.carousel_image = np.zeros((self.thumbnail_height, 800, 3), dtype=np.uint8)
            cv2.putText(self.carousel_image, "No annotations yet", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Annotation Browser", self.carousel_image)
            return

        carousel_width = len(annotated_frames) * self.thumbnail_width
        self.carousel_image = np.zeros((self.thumbnail_height, carousel_width, 3), dtype=np.uint8)

        for i, frame_idx in enumerate(annotated_frames):
            features = self.get_frame_features(frame_idx)
            if features:
                thumb = cv2.resize(features["bgr_frame"], (self.thumbnail_width, self.thumbnail_height))
                
                # Draw mask on thumbnail
                frame_annotations = self.annotations.get(frame_idx)
                if frame_annotations and (frame_annotations.get("points") or frame_annotations.get("mask") is not None):
                    thumb_mask = frame_annotations.get("mask")
                    if thumb_mask is None: # Re-generate mask if not present
                        h_org, w_org = features["original_size"]
                        pts, lbls = prepare_points(frame_annotations["points"], frame_annotations["labels"], (h_org, w_org), (self.enc_h, self.enc_w))
                        _, mask_hi, _ = run_decoder(self.sess_dec, pts, lbls, features["image_embeddings"], features["high_res_features1"], features["high_res_features2"])
                        thumb_mask = (mask_hi[0, 0] > 0).astype(np.uint8)

                    m_resized = cv2.resize(thumb_mask, (self.thumbnail_width, self.thumbnail_height), cv2.INTER_NEAREST)
                    thumb = green_overlay(thumb, m_resized, 0.6)

                # Add info text on thumbnail
                info_lines = []
                if frame_annotations:
                    fg_count = frame_annotations["labels"].count(1) if frame_annotations.get("labels") else 0
                    bg_count = frame_annotations["labels"].count(0) if frame_annotations.get("labels") else 0
                    info_lines.append(f"FG:{fg_count} BG:{bg_count}")
                if frame_idx in self.memory_cache:
                    info_lines.append("Mem")
                info_text = " ".join(info_lines)
                cv2.putText(thumb, info_text, (5, self.thumbnail_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                if frame_idx == self.current_frame_idx:
                    cv2.rectangle(thumb, (0, 0), (self.thumbnail_width - 1, self.thumbnail_height - 1), (0, 255, 0), 3)
                
                self.carousel_image[:, i*self.thumbnail_width:(i+1)*self.thumbnail_width] = thumb

        view_width = 800
        if carousel_width > view_width:
            max_scroll = carousel_width - view_width
            self.carousel_scroll_x = min(self.carousel_scroll_x, max_scroll)
            self.carousel_scroll_x = max(0, self.carousel_scroll_x)
            view = self.carousel_image[:, self.carousel_scroll_x:self.carousel_scroll_x + view_width]
        else:
            view = self.carousel_image
        
        cv2.imshow("Annotation Browser", view)

    def predict_from_memory(self, target_frame_idx):
        if self.current_frame_idx not in self.memory_cache:
            print("[WARNING] No memory features for current frame. Annotate it first.")
            return

        mem_feats, mem_pos = self.memory_cache[self.current_frame_idx]
        
        features = self.get_frame_features(target_frame_idx)
        if features is None:
            return

        enc_embed = features["image_embeddings"]
        f0 = features["high_res_features1"]
        f1 = features["high_res_features2"]
        vis_pos = features.get("vision_pos_embed") # May not exist in ONNX-exported features

        if vis_pos is not None:
            attn_inputs = {
                "current_vision_feat": enc_embed.astype(np.float32, copy=False),
                "current_vision_pos_embed": vis_pos.astype(np.float32, copy=False),
                "memory_0": np.zeros((0, 256), np.float32),
                "memory_1": mem_feats.astype(np.float32, copy=False),
                "memory_pos_embed": mem_pos.astype(np.float32, copy=False),
            }
            fused_embed = self.sess_mat.run(None, attn_inputs)[0].astype(np.float32, copy=False)
        else:
            # Fallback if vision_pos_embed is not available
            fused_embed = enc_embed

        self.status_message = "Propagating mask..."
        self.draw_ui()  # Update display with message

        _, mask_hi, _ = run_decoder(self.sess_dec, None, None, fused_embed, f0, f1)
        
        mask = (mask_hi[0, 0] > 0).astype(np.uint8)
        
        # Store the predicted mask
        frame_annotations = self.annotations.setdefault(target_frame_idx, {"points": [], "labels": [], "mask": None})
        frame_annotations["mask"] = mask
        frame_annotations["points"] = [] # Clear points as this is a prediction
        frame_annotations["labels"] = []

        # Generate and cache memory features for the newly predicted frame to allow chaining predictions
        men_out = self.sess_men.run(None, {
            "mask_for_mem": mask_hi[:, 0:1].astype(np.float32, copy=False),
            "pix_feat":     features["image_embeddings"].astype(np.float32, copy=False),
        })
        self.memory_cache[target_frame_idx] = [x.astype(np.float32, copy=False) for x in men_out[:2]]

        self.status_message = ""
        self.on_trackbar_change(target_frame_idx)
        self.update_frame(show_mask=True)

    def run(self):
        print("[INFO] L-click=FG, R-click=BG, M-click=reset. ESC/q to quit.")
        print("[INFO] 'n' to predict next, 'b' to predict previous.")
        print("[INFO] Spacebar to play/pause sequential prediction.")
        while True:
            if self.is_playing:
                if self.current_frame_idx < self.max_frames - 1:
                    self.predict_from_memory(self.current_frame_idx + 1)
                else:
                    self.is_playing = False
                    print("[INFO] Reached end of video. Paused.")

            key = cv2.waitKey(1 if self.is_playing else 20) & 0xFF
            
            if key == ord(' '):
                if self.is_playing:
                    self.is_playing = False
                    print("[INFO] Paused sequential prediction.")
                else:
                    if self.current_frame_idx in self.memory_cache:
                        self.is_playing = True
                        print("[INFO] Started sequential prediction.")
                    else:
                        print("[WARNING] Cannot start playing. Annotate the current frame first.")

            if key in (27, ord('q')):
                break
            elif key == ord('d'): # Next frame
                new_idx = min(self.current_frame_idx + 1, self.max_frames - 1)
                self.on_trackbar_change(new_idx)
            elif key == ord('a'): # Previous frame
                new_idx = max(self.current_frame_idx - 1, 0)
                self.on_trackbar_change(new_idx)
            elif key == ord('j'): # Scroll carousel left
                self.carousel_scroll_x = max(0, self.carousel_scroll_x - 50)
                self.update_carousel()
            elif key == ord('l'): # Scroll carousel right
                self.carousel_scroll_x += 50
                self.update_carousel()
            elif key == ord('n'): # Predict next frame
                if self.current_frame_idx < self.max_frames - 1:
                    self.predict_from_memory(self.current_frame_idx + 1)
            elif key == ord('b'): # Predict previous frame
                if self.current_frame_idx > 0:
                    self.predict_from_memory(self.current_frame_idx - 1)

        cv2.destroyAllWindows()
        self.cap.release()
        print("[INFO] Annotator closed.")


def main():
    print_system_info()
    set_cv2_threads(1)

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    ap = argparse.ArgumentParser(description="Interactive video annotator for SAM-2 ONNX")
    ap.add_argument("--model_size", default="tiny", choices=["base_plus", "large", "small", "tiny"])
    args = ap.parse_args()

    vid, _ = QtWidgets.QFileDialog.getOpenFileName(
        None, "Select Video", "",
        "Video files (*.mp4 *.mkv *.avi *.mov *.m4v);;All files (*.*)")
    if not vid:
        sys.exit("No video selected – exiting.")
    args.video = vid
    print(f"[INFO] Selected video: {vid}")

    annotator = InteractiveAnnotator(args)
    annotator.run()

if __name__ == "__main__":
    main()
