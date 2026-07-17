
#--------------------------
#Safe MASK Custom Node
#--------------------------

import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import imageio
import scipy.ndimage
from PIL import Image
from typing_extensions import override
import folder_paths
import math

import comfy
import comfy.utils
import node_helpers

from comfy_api.latest import ComfyExtension, IO, UI
from comfy.utils import ProgressBar

import nodes
from nodes import MAX_RESOLUTION
import logging
    
#--------------------------
# Header Utility Code(Baseif)
#--------------------------

def normalize_mask_tensor(mask):

    if not isinstance(mask, torch.Tensor):
        mask_tensor = torch.from_numpy(np.array(mask)).float()
    else:
        mask_tensor = mask.float()

    if mask_tensor.ndim == 2:
        return mask_tensor
    elif mask_tensor.ndim == 3:
        if mask_tensor.shape[0] == 1:
            return mask_tensor.squeeze(0)
        elif mask_tensor.shape[-1] == 1:
            return mask_tensor.squeeze(-1)
        else:
            return mask_tensor[0]
    elif mask_tensor.ndim == 4:
        return mask_tensor[0, 0, :, :]
    else:
        raise ValueError(f"Unexpected mask shape: {mask_tensor.shape}")


def ensure_image_tensor(arr):
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(np.array(arr)).float()

    if arr.dim() == 2:
        arr = arr.unsqueeze(0).unsqueeze(0)

    elif arr.dim() == 3:
        if arr.shape[-1] in (1,3,4):
            arr = arr.permute(2,0,1).unsqueeze(0)
        else:
            arr = arr.unsqueeze(0)

    elif arr.dim() == 4:
        if arr.shape[-1] in (1,3,4):
            arr = arr.permute(0,3,1,2)

    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")


    return arr.float()


def ensure_mask_tensor(t: torch.Tensor) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        t = torch.from_numpy(np.array(t)).float()
    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 3:
        t = t.unsqueeze(1)
    elif t.dim() == 4:
        pass
    else:
        raise ValueError(f"Unsupported mask shape: {t.shape}")
    return t.float()
    
    
def composite(destination, source, mask, alpha, blend_mode, resize_source=False):

    source = source.to(destination.device)

    if resize_source:
        source = torch.nn.functional.interpolate(
            source,
            size=(destination.shape[2], destination.shape[3]),  # H, W
            mode="bilinear",
            align_corners=False
        )

    source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])
    mask   = mask.to(destination.device, copy=True)

    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = torch.nn.functional.interpolate(
            mask,
            size=(source.shape[2], source.shape[3]),
            mode="bilinear",
            align_corners=False
        )
        mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])

    elif mask.ndim == 4 and mask.shape[1] == 3:
        mask = mask.mean(dim=1, keepdim=True)
        mask = torch.nn.functional.interpolate(
            mask,
            size=(source.shape[2], source.shape[3]),
            mode="bilinear",
            align_corners=False
        )
        mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])

    else:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")

    inverse_mask = torch.ones_like(mask) - mask
    dest_out = (mask * source * alpha) + inverse_mask * destination

    if blend_mode == "soft":
        return destination * (1 - mask) + dest_out * mask
    elif blend_mode == "darken":
        blended = torch.min(destination, dest_out)
        return destination * (1 - mask) + blended * mask
    elif blend_mode == "lighten":
        blended = torch.max(destination, dest_out)
        return destination * (1 - mask) + blended * mask
    else:  # normal
        return dest_out
        
    output = dest_out

    return output

def compositeflow(destination, source, mask, alpha, resize_source=False):
    # source: NCDHW
    source = source.to(destination.device)

    if resize_source:
        source = torch.nn.functional.interpolate(
            source,
            size=(destination.shape[2], destination.shape[3], destination.shape[4]),  # D, H, W
            mode="trilinear",  # 3D match
            align_corners=False
        )

    source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])

    mask = mask.to(destination.device, copy=True)

    if mask.ndim == 4:
        mask = mask.unsqueeze(2)
    if mask.ndim == 5 and mask.shape[1] == 1:
        mask = torch.nn.functional.interpolate(
            mask,
            size=(source.shape[2], source.shape[3], source.shape[4]),
            mode="trilinear",
            align_corners=False
        )
        mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])

    elif mask.ndim == 5 and mask.shape[1] == 3:
        mask = mask.mean(dim=1, keepdim=True)
        mask = torch.nn.functional.interpolate(
            mask,
            size=(source.shape[2], source.shape[3], source.shape[4]),
            mode="trilinear",
            align_corners=False
        )
        mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])

    else:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")

    inverse_mask = torch.ones_like(mask) - mask
    output = (mask * source * alpha) + inverse_mask * destination

    return output

def sobel_2d(tensor: torch.Tensor) -> torch.Tensor:
    # NCHW
    sobel_x = torch.tensor([[[[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]]]], 
                           dtype=tensor.dtype, device=tensor.device)
    sobel_y = torch.tensor([[[[-1, -2, -1],
                              [ 0,  0,  0],
                              [ 1,  2,  1]]]], 
                           dtype=tensor.dtype, device=tensor.device)

    grad_x = torch.nn.functional.conv2d(tensor, sobel_x, padding=1)
    grad_y = torch.nn.functional.conv2d(tensor, sobel_y, padding=1)
    edge_map = torch.sqrt(grad_x**2 + grad_y**2)
    return edge_map

def sobel_3d(tensor: torch.Tensor) -> torch.Tensor:
    # NCDHW

    sobel_x = torch.tensor(
        [[[[[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]],

           [[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]],

           [[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]]]]],
        dtype=tensor.dtype, device=tensor.device
    )  # shape: (1,1,3,3,3)

    sobel_y = torch.tensor(
        [[[[[-1,-2,-1],
            [ 0, 0, 0],
            [ 1, 2, 1]],

           [[-1,-2,-1],
            [ 0, 0, 0],
            [ 1, 2, 1]],

           [[-1,-2,-1],
            [ 0, 0, 0],
            [ 1, 2, 1]]]]],
        dtype=tensor.dtype, device=tensor.device
    )  # shape: (1,1,3,3,3)

    sobel_z = torch.tensor(
        [[[[[-1,-2,-1],
            [-2, 0, 2],
            [ 1, 2, 1]],

           [[-1,-2,-1],
            [-2, 0, 2],
            [ 1, 2, 1]],

           [[-1,-2,-1],
            [-2, 0, 2],
            [ 1, 2, 1]]]]],
        dtype=tensor.dtype, device=tensor.device
    )  # shape: (1,1,3,3,3)

    grad_x = torch.nn.functional.conv3d(tensor, sobel_x, padding=1)
    grad_y = torch.nn.functional.conv3d(tensor, sobel_y, padding=1)
    grad_z = torch.nn.functional.conv3d(tensor, sobel_z, padding=1)

    edge_map = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    return edge_map

COLOR_MAP = {
    "red":   (255, 0, 0, 128),
    "green": (0, 255, 0, 128),
    "blue":  (0, 0, 255, 128),
    "yellow":(255, 255, 0, 128),
}

def get_color(name: str):
    return COLOR_MAP.get(name, COLOR_MAP["red"])
    
def dilate_tensor(mask_tensor, kernel_size=5, iterations=1):

    stride = 1

    for _ in range(iterations):
        mask_tensor = F.max_pool2d(mask_tensor, kernel_size, stride=1, padding=kernel_size//2)
    return mask_tensor
    
def gaussian_blur(tensor, kernel_size=5, sigma=2):

    k = kernel_size // 2
    x = torch.arange(-k, k+1, dtype=torch.float32)
    gauss = torch.exp(-(x**2)/(2*sigma**2))
    gauss = gauss / gauss.sum()
    kernel1d = gauss.unsqueeze(0)
    kernel2d = gauss.unsqueeze(0) @ gauss.unsqueeze(1)
    kernel2d = kernel2d / kernel2d.sum()
    kernel2d = kernel2d.unsqueeze(0).unsqueeze(0)

    blurred = F.conv2d(tensor, kernel2d, padding=k)
    return blurred


def dilate_mtensor(mask_tensor, kernel_size=5, sigma=2, iterations=1, tapered_corners=True):
    if tapered_corners:
        for _ in range(iterations):
            blurred = gaussian_blur(mask_tensor, kernel_size=kernel_size, sigma=sigma)
            mask_tensor = (blurred > 0.3).float()
    else:
        stride=1
        for _ in range(iterations):
            mask_tensor = F.max_pool2d(mask_tensor, kernel_size, stride=1, padding=kernel_size//2)
    return mask_tensor



def erode_mtensor(mask_tensor, kernel_size=5, sigma=2, iterations=1, tapered_corners=True):
    inv = 1.0 - mask_tensor
    if tapered_corners:
        for _ in range(iterations):
            blurred = gaussian_blur(inv, kernel_size=kernel_size, sigma=sigma)
            inv = (blurred > 0.3).float()
    else:
        stride=1
        for _ in range(iterations):
            inv = F.max_pool2d(inv, kernel_size, stride=1, padding=kernel_size//2)
    return 1.0 - inv



def squeeze_mask_output(mask_tensor: torch.Tensor) -> torch.Tensor:

    if mask_tensor.ndim == 4 and mask_tensor.shape[1] == 1:
        return mask_tensor.squeeze(1)
    return mask_tensor

def apply_feathering(mask_tensor: torch.Tensor, feather_size: int, feather_strength: float) -> torch.Tensor:

    if feather_size <= 0:
        return mask_tensor

    kernel_size = feather_size * 2 + 1
    if kernel_size % 2 == 0:
        kernel_size += 1

    orig_shape = mask_tensor.shape
    H, W = orig_shape[-2], orig_shape[-1]
    
    total_elements = mask_tensor.numel()
    batch_size = total_elements // (H * W)
    mask4d = mask_tensor.view(batch_size, 1, H, W).float()

    blurred_mask = F.avg_pool2d(mask4d, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    if feather_strength != 1.0 and feather_strength > 0:
        blurred_mask = blurred_mask ** (1.0 / feather_strength)

    return blurred_mask.view(orig_shape)
    
def blur_tensor(mask_tensor, k=5):

    return F.avg_pool2d(mask_tensor, kernel_size=k, stride=1, padding=k//2)
    
_counter = 0

def resolve_filename(prefix: str, save_dir: str) -> str:
    base = prefix.replace("%number", "")
    existing = [f for f in os.listdir(save_dir) if f.startswith(base)]
    number = len(existing) + 1
    return prefix.replace("%number", str(number))


def progressbar_update(pbar, step, total_steps):
    pbar.update(step+1)

mask_dir = os.path.join(folder_paths.get_input_directory(), "Mask")
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir, exist_ok=True)

sample_path = os.path.join(mask_dir, "sample.png")
if not os.path.exists(sample_path):
    arr = np.zeros((64,64), dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(sample_path)

def ensure_maskout_tensor(t: torch.Tensor) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        t = torch.from_numpy(np.array(t)).float()
    # (B, 1, H, W) -> (B, H, W) 또는 (1, 1, H, W) -> (1, H, W)
    if t.dim() == 4:
        t = t.squeeze(1)
    if t.dim() == 2:
        # (H, W) -> (1, H, W)
        t = t.unsqueeze(0)
    elif t.dim() == 3:
        if t.shape[0] not in [1, 2, 4]:
            # (H, W, C)=> (C, H, W)
            t = t.permute(2, 0, 1)
    else:
        # (dim=5)
        raise ValueError(f"Unsupported mask shape: {t.shape}")
        
    return t.float()

def ensure_mask_output_shape(mask_tensor: torch.Tensor) -> torch.Tensor:

    if mask_tensor.ndim == 2:   # [H, W] -> [1, 1, H, W]
        return mask_tensor.unsqueeze(0).unsqueeze(0)
    elif mask_tensor.ndim == 3: # [B, H, W],[1, H, W] -> [B, 1, H, W]
        return mask_tensor.unsqueeze(1)
    elif mask_tensor.ndim == 4: # [B, C, H, W] -> [B, 1, H, W]
        return mask_tensor[:, 0:1, :, :]
    return mask_tensor

def generate_auto_mask(arr_bgr, threshold_sat, amp_factor, blur_kernel):
    hsv = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1].astype(np.float32)
    
    # 1. threshold Otsh (auto threshold, threshold_sat, dilate_iter)
    s_channel_u8 = s_channel.astype(np.uint8)
    _, sat_mask = cv2.threshold(s_channel_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    sat_mask = sat_mask.astype(np.float32)

    # 2. saturation >=50
    s_channel_f32 = s_channel.astype(np.float32)
    healthy_val = (threshold_sat / 10.0) * 255
    healthy_mask = cv2.threshold(s_channel, healthy_val, 255, cv2.THRESH_BINARY)[1]

    # 3. combined_mask (threshold Otsh - healthy_mask)
    combined = np.clip(sat_mask - healthy_mask, 0, 255)
    
    combined = cv2.GaussianBlur(combined, (blur_kernel * 2 + 1, blur_kernel * 2 + 1), 0)

    # 4. amp_factor
    combined_mask = np.clip(combined * (1.0 + amp_factor), 0, 255)

    return combined_mask

def generate_auto_noise_mask(arr_bgr, threshold_sat, amp_factor, blur_kernel):
    # --- [Tier 1] sd 1.5 pixel collapse ---
    h, w = arr_bgr.shape[:2]
    hsv = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # sat >=thresh_val : low saturation collapse
    thresh_val = (threshold_sat / 10.0)
    sat_low_mask = (hsv[:, :, 1] < thresh_val).astype(np.float32)
    

    # grayscale collapse
    blurred = cv2.GaussianBlur(gray, (blur_kernel | 1, blur_kernel | 1), 0)
    diff = np.abs(gray - blurred)
    texture_broken_mask = (diff > 10.0).astype(np.float32)


    # --- [Tier 2] overtue or noise burn ---
    s_burn_mask = (hsv[:, :, 1] > 230.0).astype(np.float32)
    v_burn_mask = (hsv[:, :, 2] > 240.0).astype(np.float32)
    
    # --- [composite] ---
    combined_mask = np.maximum(sat_low_mask, texture_broken_mask)
    combined_mask = np.maximum(combined_mask, s_burn_mask)
    combined_mask = np.maximum(combined_mask, v_burn_mask)
    
    final_mask = np.clip(combined_mask * (1.0 + amp_factor), 0, 1) * 255

    return final_mask

def generate_auto_mask_pixel(arr_bgr, threshold_sat, amp_factor, blur_kernel):
    # --- [Tier 1] sd 1.5 pixel collapse ---
    gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


    # 2. find collapse pixel to Laplacian
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    
    lap_threshold = threshold_sat * 5.0
    
    
    # 3. find edge
    mask = cv2.threshold(laplacian, max(1, lap_threshold), 255, cv2.THRESH_BINARY_INV)[1]
    
    # 4. amp_factor
    automask = np.clip(mask * (1.0 + amp_factor), 0, 255)

    return automask

def generate_auto_mask_min(arr_bgr, threshold_sat, amp_factor, k_size):
    adj_threshold = threshold_sat * 2.0
    # --- [Tier 1] sd 1.5 pixel collapse ---
    gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # 2. mean check
    mean = cv2.blur(gray, (k_size, k_size))
    
    # 3. mean/sq variance
    mean_sq = cv2.blur(gray**2, (k_size, k_size))
    var = np.clip(mean_sq - mean**2, 0, None)
    mask = cv2.threshold(var, threshold_sat * 2.0, 255, cv2.THRESH_BINARY_INV)[1]

    # 4. amp_factor
    automask = np.clip(mask * (1.0 + amp_factor), 0, 255)
    
    return automask

def generate_advanced_mask(arr_bgr, threshold_sat, amp_factor, blur_kernel):
    # 1. multy channel parser
    hsv = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # 2. multyscale var
    def get_var(img, k):
        mean = cv2.blur(img, (k, k))
        return np.clip(cv2.blur(img**2, (k, k)) - mean**2, 0, None)
    # 3. sat/collaps score (low saturation + collaps lightbalance)
    collapse_score = (1.0 / (get_var(gray, 3) + 1e-5)) * 0.4 + (1.0 / (get_var(gray, 11) + 1e-5)) * 0.6
    sat_inv = np.clip(1.0 - (hsv[:,:,1].astype(np.float32) / (threshold_sat * 2 + 1)), 0, 1)
    
    final_mask = cv2.normalize(collapse_score * sat_inv, None, 0, 255, cv2.NORM_MINMAX)# 4. normalize

    automask = np.clip(final_mask * (1.0 + amp_factor), 0, 255)

    return automask

def generate_tiled_mask(arr_bgr, threshold_sat, amp_factor, blur_kernel, tile_size=16):
    gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    kernel_size = blur_kernel
    mean = cv2.blur(gray, (kernel_size, kernel_size))
    mean_sq = cv2.blur(gray**2, (kernel_size, kernel_size))
    variance = np.clip(mean_sq - mean**2, 0, None)
    
    # 1. texture_strength
    # 0(collapse)< variance< 1(normal pixel)
    texture_strength = cv2.normalize(variance, None, 0, 1, cv2.NORM_MINMAX)
    
    # 2. tile scorer
    score_map = np.zeros((h, w), dtype=np.float32)
    global_mean_var = np.mean(texture_strength) + 1e-5
    
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile_data = texture_strength[y:y_end, x:x_end]

            # percentile
            val = np.percentile(tile_data, 25)

            # threshold_sat
            ratio = (global_mean_var / (val + 1e-5))
            score = np.clip(ratio / (threshold_sat + 1.0) - 1.0, 0, 1)
            score_map[y:y_end, x:x_end] = score
    
    # 3. amp factor settings
    score_map = cv2.GaussianBlur(score_map, (tile_size + 1, tile_size + 1), 0)
    automask = np.clip(score_map * 255 * (1.0 + amp_factor), 0, 255)

    return automask.astype(np.float32)

def generate_tiled_edge_mask(arr_bgr, threshold_sat, amp_factor, blur_kernel, tile_size=16):
    gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. auto Canny
    # auto shreshold
    v = np.median(gray)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(gray, lower, upper)
    
    # 2. tile edge min check
    h, w = edges.shape
    score_map = np.zeros((h, w), dtype=np.float32)
    
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            
            # edge_density
            edge_density = np.sum(edges[y:y_end, x:x_end] > 0) / (tile_size * tile_size)
            
            # (score>=score 1, error pixel)
            # threshold_sat
            score = np.clip(1.0 - (edge_density * (10.0 / (threshold_sat + 1e-5))), 0, 1)
            score_map[y:y_end, x:x_end] = score
            
    # 3. amp factor settings
    score_map = cv2.GaussianBlur(score_map, (tile_size + 1, tile_size + 1), 0)
    automask = np.clip(score_map * 255 * (1.0 + amp_factor), 0, 255)
    
    return automask.astype(np.float32)

#----------------------------------------------------
# Mask Preview - original implement from
# https://github.com/cubiq/ComfyUI_essentials/blob/9d9f4bedfc9f0321c19faf71855e228c93bd0dc9/mask.py#L81
# upstream requested in https://github.com/Kosinkadink/rfcs/blob/main/rfcs/0000-corenodes.md#preview-nodes

class AutoMaskGenerator(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="AutoMaskGenerator",
            display_name="자동 마스크 생성기",
            category="커스텀마스크/체커",
            description="이미지에서 색감 붕괴 및 텍스처 파괴 영역을 자동으로 감지하여 마스크를 생성합니다. 라인 붕괴점은 잘 탐지하지만, 색상 영역 탐지는 아직 미흡합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="대상 이미지"),
                IO.Combo.Input("mode", options=["saturation", "noiseburn", "collapse_pixel", "minstd", "advanced", "tiled", "tiled_edge"], default="saturation", tooltip="붕괴점 탐지용 마스크 준비 로직"),
                IO.Float.Input("threshold_sat", default=2.0, min=0.0, max=10.0, step=0.1, tooltip="채도 붕괴 임계값"),
                IO.Float.Input("amp_factor", default=0.000, min=0.000, max=1.999, step=0.001, tooltip="마스크 증폭 횟수"),
                IO.Combo.Input("blur_kernel", options=["1", "2", "3", "4"], default="1", tooltip="블러 커널 크기"),
                IO.Combo.Input("tile_size", options=["skip", "1", "2"], default="skip", tooltip="타일 매칭 파악시 참조 범위. 1=16, 2=32픽셀 기준이 됩니다."),
                IO.Boolean.Input("show_preview", default=False, tooltip="프리뷰 표시 여부"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[IO.Mask.Output("mask")],
        )
    @classmethod
    def execute(cls, image, mode, threshold_sat, amp_factor, blur_kernel, tile_size, show_preview) -> IO.NodeOutput:
        arr_orig = ensure_image_tensor(image)
        H, W = arr_orig.shape[2:]
        arr = arr_orig[0].permute(1,2,0).cpu().numpy()
        arr = (arr * 255).clip(0,255).astype(np.uint8)
        arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        
        kernel=int(blur_kernel)
        kernel_map = {1:1, 2:3, 3:5, 4:7} 
        k_size = kernel_map.get(kernel, 1.0)
        if tile_size !="skip":
            tile=int(tile_size)
            tile_map = {1:16, 2:32} 
            t_size = tile_map.get(tile, 1.0)
        amp_factor = min(max(amp_factor,0.000),1.999)
        if mode == "collapse_pixel":
            mask_arr = generate_auto_mask_pixel(arr_bgr, threshold_sat, amp_factor, k_size)
        elif mode == "saturation":
            mask_arr = generate_auto_mask(arr_bgr, threshold_sat, amp_factor, k_size)
        elif mode == "advanced":
            mask_arr = generate_advanced_mask(arr_bgr, threshold_sat, amp_factor, k_size)
        elif mode == "tiled" and tile_size != "skip":
            mask_arr = generate_tiled_mask(arr_bgr, threshold_sat, amp_factor, k_size, t_size)
        elif mode == "tiled_edge" and tile_size != "skip":
            mask_arr = generate_tiled_edge_mask(arr_bgr, threshold_sat, amp_factor, k_size, t_size)
        elif mode == "noiseburn":
            mask_arr = generate_auto_noise_mask(arr_bgr, threshold_sat, amp_factor, k_size)
        else:
            mask_arr = generate_auto_mask_min(arr_bgr, threshold_sat, amp_factor, k_size)

        mask_tensor = torch.from_numpy(mask_arr).float() / 255.0
        mask_tensor = ensure_mask_output_shape(mask_tensor)
        
        if show_preview:
            preview_mask = mask_tensor # [B, 1, H, W]
            return IO.NodeOutput(mask_tensor, ui=UI.PreviewMask(preview_mask))
        else:
            return IO.NodeOutput(mask_tensor, )

#----------------------------------------------------
#Registration
#----------------------------------------------------
    
    
TEST_NODE_CLASS_MAPPINGS = {
    "AutoMaskGenerator": AutoMaskGenerator
}

TEST_NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoMaskGenerator": "자동 마스크 생성기"
}
