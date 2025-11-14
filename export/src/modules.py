# sam2-onnx-cpp/export/src/modules.py

import torch
from torch import nn
from sam2.modeling.sam2_base import SAM2Base


class ImageEncoder(nn.Module):
    """
    Wrapper around SAM-2's image encoder.

    Goal:
        Take an input RGB image and produce:
          - Pixel features for segmentation
          - Hi-res FPN features for precise boundaries
          - Features + positional encodings for temporal memory attention

    Input:
        image: [B, 3, 1024, 1024]
            - B is batch size (we always use B=1 in ONNX export/demo).

    Output (tuple):
        pix_feat:              [1, 256, 64, 64]
            - "Image embeddings" / pixel features.
            - Semantic feature map at 1/16 resolution (1024 → 64) (4x4 pixel box).

        high_res_features_0:   [1, 32, 256, 256]
            - High-res feature map (1/4 resolution).
            - Helps the decoder recover sharp boundaries.

        high_res_features_1:   [1, 64, 128, 128]
            - Another high-res map (1/8 resolution).

        current_vision_feat2:  [1, 256, 64, 64]
            - Feature map prepared for the MEMORY ATTENTION module.
            - Same spatial size and channels as pix_feat.
            - Has SAM2's `no_mem_embed` added in, as done in the original model.

        current_vision_pos_embeds[-1]: [4096, 1, 256]
            - Positional embeddings for the current frame at 64x64 resolution.
            - 4096 = 64 * 64 flattened pixels.
            - Used as `curr_pos` in memory attention.
    """

    def __init__(self, sam_model) -> None:
        super().__init__()
        self.model = sam_model

        # Learned tensor added to current_vision_feat when there is "no memory".
        # It conditions the model on the "no-memory" case.
        self.no_mem_embed = sam_model.no_mem_embed

        # Low-level encoder backbone (ViT+FPN etc.) from SAM2.
        self.image_encoder = sam_model.image_encoder

        # Helper from SAM2 that prepares multi-scale backbone features, positions, etc.
        self.prepare_backbone_features = sam_model._prepare_backbone_features

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        # Run SAM-2's image encoder.
        # Returns a dict containing:
        #   - "vision_features"  : core feature map (pix_feat)
        #   - "vision_pos_enc"   : positional encodings for different levels
        #   - "backbone_fpn"     : FPN features at several scales
        backbone_out = self.image_encoder(image)

        # Apply the same conv_s0 / conv_s1 used inside SAM2's mask decoder
        # to the first two FPN levels so that their shapes & channels match
        # what the decoder expects.
        backbone_out["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0]
        )
        backbone_out["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )

        # Extract components
        vision_pos_enc = backbone_out["vision_pos_enc"]    # list of [B,C,H,W] or [C,H,W]
        backbone_fpn   = backbone_out["backbone_fpn"]      # list of FPN maps at different scales
        pix_feat       = backbone_out["vision_features"]   # [B, 256, 64, 64]

        # Ensure each FPN and positional encoding has an explicit batch dimension.
        # Sometimes they come as [C,H,W], and we need [B,C,H,W].
        for i in range(len(backbone_fpn)):
            if backbone_fpn[i].dim() == 3:
                backbone_fpn[i] = backbone_fpn[i].unsqueeze(0)
        for i in range(len(vision_pos_enc)):
            if vision_pos_enc[i].dim() == 3:
                vision_pos_enc[i] = vision_pos_enc[i].unsqueeze(0)

        # Let SAM2 compute nicely prepared backbone features and their pos encodings.
        # This returns:
        #   current_vision_feats:      list of feature maps at different resolutions
        #   current_vision_pos_embeds: list of position embeddings matched to them
        _, current_vision_feats, current_vision_pos_embeds, _ = \
            self.prepare_backbone_features({
                "backbone_fpn": backbone_fpn,
                "vision_pos_enc": vision_pos_enc,
            })

        # The last vision feature (highest semantic, coarsest spatial) is used
        # as the base for the memory attention.
        # We add no_mem_embed here (original SAM2 convention for "no memory").
        current_vision_feat = current_vision_feats[-1] + self.no_mem_embed
        # Reshape from [HW, B, C] or similar internal format into [1, 256, 64, 64].
        # Here 64x64 is hard-coded to match the 1024x1024 input.
        current_vision_feat2 = current_vision_feat.reshape(64, 64, 1, 256).permute(2, 3, 0, 1)

        # Prepare high-resolution features for the decoder:
        # - current_vision_feats[0]: finest resolution (256x256)
        # - current_vision_feats[1]: mid resolution (128x128)
        # We reshape them into [1, C, H, W] with the right channel counts.
        high_res_features_0 = current_vision_feats[0].reshape(256, 256, 1, 32).permute(2, 3, 0, 1)
        high_res_features_1 = current_vision_feats[1].reshape(128, 128, 1, 64).permute(2, 3, 0, 1)

        # Return encoder outputs in a fixed order.
        # Note: current_vision_pos_embeds[-1] corresponds to the 64x64 level.
        return (
            pix_feat,                      # [1,256,64,64]
            high_res_features_0,           # [1,32,256,256]
            high_res_features_1,           # [1,64,128,128]
            current_vision_feat2,          # [1,256,64,64]
            current_vision_pos_embeds[-1]  # [4096,1,256]
        )


class ImageDecoder(nn.Module):
    """
    Wrapper around SAM-2's mask decoder.

    Goal:
        Given:
          - an image-level feature map (with or without temporal fusion),
          - two high-resolution FPN feature maps,
          - and a prompt (points),
        predict segmentation masks and produce a mask suitable for the memory encoder.

    Inputs:
        point_coords:    [num_labels, num_points, 2]
            - Prompt point coordinates (in normalized or scaled image space).
        point_labels:    [num_labels, num_points]
            - 1 for foreground, 0 for background (and possibly -1 for "not used").
        image_embed:     [1, 256, 64, 64]
            - The feature map fed as "backbone_features" into SAM2's heads.
            - In a pure single-frame setup, this would be the encoder's pix_feat.
            - In our temporal setup, this is the memory-fused feature.
        high_res_feats_0:[1, 32, 256, 256]
        high_res_feats_1:[1, 64, 128, 128]
            - High-resolution maps that let the decoder recover fine boundaries.

    Outputs:
        obj_ptr:      tensor with object pointer embeddings (used internally by SAM2).
        mask_for_mem: [1, num_masks, 1024, 1024]
            - High-resolution mask logits/scores, passed through a sigmoid+scale+bias.
            - This is what the memory encoder expects to build temporal memory.
        pred_mask:    [1, num_masks, 256, 256]
            - Low-resolution predicted masks for visualization or further processing.
    """

    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model

        # Optional scaling/bias for the mask that goes into memory encoding.
        # These are attributes used by the original SAM2 memory encoder.
        self.sigmoid_scale_for_mem_enc = getattr(sam_model, "sigmoid_scale_for_mem_enc", 1.0)
        self.sigmoid_bias_for_mem_enc  = getattr(sam_model, "sigmoid_bias_for_mem_enc", 0.0)

    @torch.no_grad()
    def forward(
        self,
        point_coords: torch.Tensor,      # [num_labels, num_points, 2]
        point_labels: torch.Tensor,      # [num_labels, num_points]
        image_embed: torch.Tensor,       # [1,256,64,64]
        high_res_feats_0: torch.Tensor,  # [1,32,256,256]
        high_res_feats_1: torch.Tensor,  # [1,64,128,128]
    ):
        # 1) Package point prompts in the format SAM2 expects.
        point_inputs = {
            "point_coords": point_coords,
            "point_labels": point_labels,
        }
        high_res_feats = [high_res_feats_0, high_res_feats_1]

        # 2) Call SAM2's internal heads (mask decoder + upsampling).
        #    This returns multiple resolutions and internal head outputs:
        (
            _,
            _,
            _,
            low_res_masks,   # [1, num_masks, 256, 256]
            high_res_masks,  # [1, num_masks, 1024, 1024]
            obj_ptr,         # object pointer embeddings
            _,
        ) = self.model._forward_sam_heads(
            backbone_features=image_embed,
            point_inputs=point_inputs,
            mask_inputs=None,          # we don't use an existing-mask prompt here
            high_res_features=high_res_feats,
            multimask_output=True,
        )

        # 3) Post-process the high-res masks for the memory encoder:
        #    Apply sigmoid to get mask probabilities, then scale/bias as SAM2 expects.
        mask_for_mem = torch.sigmoid(high_res_masks)
        mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc

        # 4) low_res_masks is directly the "prediction" we care about for segmentation.
        pred_mask = low_res_masks

        return obj_ptr, mask_for_mem, pred_mask


class MemAttention(nn.Module):
    """
    Wraps SAM2's memory_attention module.

    Goal:
        Take:
          - current frame features + position encodings,
          - memory from previous frames (spatial + object tokens),
        and produce a fused feature map that incorporates temporal information.

    Inputs:
        current_vision_feat:      [1, 256, 64, 64]
            - Feature map of the current frame (encoder output).
            - Should include the no_mem_embed that was added in ImageEncoder.

        current_vision_pos_embed: [4096, 1, 256]
            - Positional encodings for the 64x64 grid of the current frame.
            - 4096 = 64 * 64 flattened.

        memory_0:                 [num_obj_ptr, 256]
            - Object / temporal tokens aggregated from past frames.
            - Each row is a 256-D token. In this wrapper, we interpret each
              token as comprising 4 sub-tokens of dimension 64 (4 * 64 = 256).

        memory_1:                 [n, 64, 64, 64]
            - Spatial memory per frame.
            - n = number of frames in the memory bank.
            - For each frame: 64x64 grid with 64-channel features.

        memory_pos_embed:         [N, 1, 64]
            - Positional embeddings for the memory tokens.
            - First part typically encodes spatial positions (for memory_1),
              and extra rows can encode or act as placeholders for object tokens.
            - N must match the total number of memory tokens:
              (n * 4096 spatial tokens) + (num_obj_ptr_tokens).

    Output:
        fused_bcHW: [1, 256, 64, 64]
            - Current frame features enriched with temporal context from memory.
            - Used as input both to the decoder and to the memory encoder.

    Pseudo-Code:

        curr_with_pos   = curr   + curr_pos        # [4096, 1, 256]
        mem_with_pos    = memory + memory_pos      # [M, 1, 64]
        then Q, K, V from these, and cross-attn between curr and mem
    """

    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.memory_attention = sam_model.memory_attention

        # Same no_mem_embed as used in ImageEncoder.
        # Here we subtract it before passing to memory_attention, so:
        #   (added in encoder) + (subtracted here) = original feature baseline.
        self.no_mem_embed = sam_model.no_mem_embed

    @torch.no_grad()
    def forward(
        self,
        current_vision_feat: torch.Tensor,      # [1,256,64,64]
        current_vision_pos_embed: torch.Tensor, # [4096,1,256]
        memory_0: torch.Tensor,                 # [num_obj_ptr,256]
        memory_1: torch.Tensor,                 # [n,64,64,64]
        memory_pos_embed: torch.Tensor          # [N,1,64]
    ) -> torch.Tensor:

        # current_vision_feat: (B=1, C=256, H=64, W=64)
        B, C, H, W = current_vision_feat.shape

        # Flatten spatially: (B,C,H,W) -> (H*W, B, C) = (4096, 1, 256)
        # SAM2 memory_attention expects sequences of shape [HW, B, C].
        feat_hwbc = current_vision_feat.permute(2, 3, 0, 1).reshape(H * W, B, C)

        # Undo the no_mem_embed that was added in ImageEncoder:
        # this restores the expected feature distribution for memory_attention.
        feat_hwbc = feat_hwbc - self.no_mem_embed

        # ---- Handle object tokens (memory_0) ----
        # memory_0: [num_obj_ptr, 256]
        # We interpret each 256-D token as 4 sub-tokens of 64 channels:
        num_obj_ptr = memory_0.shape[0]
        memory_0 = memory_0.reshape(num_obj_ptr, 4, 64)    # [num_obj_ptr, 4, 64]
        memory_0 = memory_0.unsqueeze(1)                   # [num_obj_ptr, 1, 4, 64]
        memory_0 = memory_0.permute(0, 2, 1, 3)            # [num_obj_ptr, 4, 1, 64]
        memory_0 = memory_0.reshape(num_obj_ptr * 4, 1, 64)# [num_obj_ptr*4, 1, 64]
        # Now we have num_obj_ptr_tokens = num_obj_ptr * 4 tokens of size 64.

        # ---- Handle spatial memory (memory_1) ----
        # memory_1: [n, 64, 64, 64]  (n frames)
        mem_1_n = memory_1.shape[0]
        # Reshape to [n, 64, 4096] (H*W) -> [n, 4096, 64] -> [n*4096, 1, 64]
        memory_1 = memory_1.reshape(mem_1_n, 64, 64 * 64)  # [n, 64, 4096]
        memory_1 = memory_1.permute(0, 2, 1)               # [n, 4096, 64]
        memory_1 = memory_1.reshape(-1, 1, 64)             # [n*4096, 1, 64]

        # Concatenate spatial memory tokens + object tokens along the sequence dim:
        # shape: [n*4096 + num_obj_ptr*4, 1, 64]
        memory = torch.cat((memory_1, memory_0), dim=0)

        # Let memory_attention know how many tokens at the end of 'memory'
        # correspond to object pointers (vs spatial memory).
        num_obj_ptr_tokens = num_obj_ptr * 4

        # Run SAM2 memory_attention:
        #   curr:     [HW, B, C]
        #   curr_pos: [HW, B, C]
        #   memory:   [M, 1, 64]
        #   memory_pos: [M, 1, 64]
        # Returns fused sequence [HW, B, C].
        fused_hwbc = self.memory_attention(
            curr=feat_hwbc,
            curr_pos=current_vision_pos_embed,  # [4096,1,256]
            memory=memory,                      # [M,1,64]
            memory_pos=memory_pos_embed,        # [M,1,64]
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )

        # Reshape back to [B, C, H, W]
        fused_bcHW = fused_hwbc.permute(1, 2, 0).reshape(B, C, H, W)
        return fused_bcHW


class MemEncoder(nn.Module):
    """
    Wrapper around SAM-2's memory encoding path.

    Goal:
        Given:
          - a high-res mask for the current frame,
          - a pixel feature map,
        produce:
          - spatial memory features,
          - positional encodings for those features,
          - temporal (object/slot) codes for this memory entry.

    Inputs:
        mask_for_mem: [1, 1, 1024, 1024]
            - High-resolution mask from the decoder (after sigmoid / scaling).
            - Indicates which regions belong to the object we track.

        pix_feat:     [1, 256, 64, 64]
            - Pixel-level feature map at 1/16 resolution.
            - Typically the same type of feature passed to the decoder
              (often already fused with memory).

    Outputs (tuple):
        maskmem_features:  Tensor, typically [1, 64, 64, 64]
            - Spatial memory feature map for this frame.
            - This is what becomes `memory_1` for this frame in the bank.

        maskmem_pos_enc_tensor: [H*W, 1, 64] = [4096, 1, 64]
            - Per-pixel positional encodings for the spatial memory.
            - Used to build `memory_pos_embed` across frames.

        maskmem_tpos_enc:  same as sam_model.maskmem_tpos_enc
            - Temporal/slot encoding tensor from the base model.
            - Represents how this memory entry is positioned in "time" / slot space.
            - Used as the basis for object tokens / temporal codes.

    
    """

    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model

        # Learned temporal position encoding tensor used for memory entries
        # (e.g., per-slot temporal codes).
        self.maskmem_tpos_enc = sam_model.maskmem_tpos_enc

        # Feature map sizes at different scales used by SAM2 internally.
        # (256x256, 128x128, 64x64)
        self.feat_sizes = [(256, 256), (128, 128), (64, 64)]

    @torch.no_grad()
    def forward(
        self,
        mask_for_mem: torch.Tensor,  # [1,1,1024,1024]
        pix_feat: torch.Tensor       # [1,256,64,64]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Flatten pix_feat from [B, C, H, W] to [H*W, B, C] = [4096, 1, 256]
        B, C, H, W = pix_feat.shape
        flattened = pix_feat.view(B, C, H * W).permute(2, 0, 1)

        # Placeholder logits: some API hooks expect per-object logits.
        object_score_logits = torch.zeros(1, 1, device=pix_feat.device)

        # Let SAM2 compute memory features and position encodings from the mask.
        # current_vision_feats is a list; we pass [flattened] so SAM2 treats it
        # as a feature level of size feat_sizes[-1] = (64,64).
        maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
            current_vision_feats=[flattened],
            feat_sizes=self.feat_sizes,
            pred_masks_high_res=mask_for_mem,
            object_score_logits=object_score_logits,
            is_mask_from_pts=True,
        )
        ## This internally uses self.maskmem_tpos_enc

        # We pick the last entry of maskmem_pos_enc (corresponding to the
        # lowest-resolution feature: 64x64).
        maskmem_pos_enc_tensor = maskmem_pos_enc[-1]
        # This is often [1, 64, H*W]; reshape it into [H*W, 1, 64].
        maskmem_pos_enc_tensor = maskmem_pos_enc_tensor.view(1, 64, H * W).permute(2, 0, 1)

        # Outputs:
        #   maskmem_features:      spatial memory per frame
        #   maskmem_pos_enc_tensor:positional encodings for each spatial location
        #   self.maskmem_tpos_enc: temporal codes / slot embeddings from the model
        return maskmem_features, maskmem_pos_enc_tensor, self.maskmem_tpos_enc
