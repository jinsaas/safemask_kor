
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
import random

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

#--------------------------
#  Utillity Header
#--------------------------

def ensure_mask_2d_tensor(t: torch.Tensor, to_rgb: bool=False) -> torch.Tensor:
    """
    Exactly (B,1,H,W) Setting
    to_rgb=True, (B,3,H,W) Expansion Setting.
    """
    if not isinstance(t, torch.Tensor):
        t = torch.from_numpy(np.array(t)).float()

    if t.dim() == 2:          # (H,W)
        t = t.unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
    elif t.dim() == 3:        # (B,H,W)
        t = t.unsqueeze(1)               # (B,1,H,W)
    elif t.dim() == 4:        # (B,C,H,W)
        pass
    else:
        raise ValueError(f"Unsupported mask shape: {t.shape}")

    t = t.float()

    if to_rgb:
        if t.shape[1] == 1:   # (B,1,H,W)
            t = t.repeat(1,3,1,1)   # (B,3,H,W)

    return t

def normalize_mask(mask_tensor: torch.Tensor) -> torch.Tensor:
    return (mask_tensor - mask_tensor.min()) / (mask_tensor.max() - mask_tensor.min() + 1e-8)

 
 
def make_feather_mask(h, w, c, border=16):
    mask = torch.ones((h,w,c), dtype=torch.float32)
    for k in range(border):
        alpha = k / border
        mask[k,:,:] *= alpha
        mask[-k-1,:,:] *= alpha
        mask[:,k,:] *= alpha
        mask[:,-k-1,:] *= alpha
    return mask


def subtract_mask(base, *removes):
    # Multi Remover mask Settings
    combined_remove = np.maximum.reduce(removes)
    return np.clip(base - combined_remove, 0, 1)

def difference_mask(mask_a, mask_b):
    return np.abs(mask_a - mask_b)


#--------------------------
#  Mask Utillity Node
#--------------------------

class SafeTileSoftFillng(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SafeTileSoftFillng",
            display_name="타일링 보정채우기(pytorch)",
            category="커스텀마스크/유틸",
            description="pytorch기반 타일링 채우기. 부분이미지 보정용으로 쓸 수 있습니다.",
            inputs=[
                IO.Image.Input("tile_image", tooltip="크롭된 타일 이미지"),
                IO.Int.Input("canvas_width", default=512, min=64, max=8192, step=8, tooltip="캔버스 X축 크기"),
                IO.Int.Input("canvas_height", default=512, min=64, max=8192, step=8, tooltip="캔버스 Y축 크기"),
                IO.Int.Input("rematch", default=0, min=0, max=64, step=1, tooltip="경계 보정 횟수"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="보정된 캔버스 이미지")
            ]
        )

    @classmethod
    def execute(cls, tile_image, canvas_width, canvas_height, rematch) -> IO.NodeOutput:
        tile_image = tile_image.float()
        B, H, W, C = tile_image.shape

        # stride logic
        rematch = min(rematch, min(H, W) - 1)
        stride_x = max(W - rematch, 1)
        stride_y = max(H - rematch, 1)

        repeat_x = canvas_width // stride_x
        repeat_y = canvas_height // stride_y

        # Canvas Settings
        canvas = torch.zeros((B, canvas_height, canvas_width, C), dtype=torch.float32)
        weight = torch.zeros_like(canvas)

        # Make feather mask
        mask = make_feather_mask(H, W, C, border=min(rematch, 16))

        # Tile layout + Boundary correction
        for i in range(repeat_y + 1):
            for j in range(repeat_x + 1):
                y0 = i * stride_y
                x0 = j * stride_x
                y1 = min(y0 + H, canvas_height)
                x1 = min(x0 + W, canvas_width)

                h_slice = y1 - y0
                w_slice = x1 - x0

                if h_slice > 0 and w_slice > 0:
                    # Tile layout
                    canvas[:, y0:y1, x0:x1, :] += tile_image[:, :h_slice, :w_slice, :]
                    weight[:, y0:y1, x0:x1, :] += 1.0

                    # Boundary correction
                    canvas[:, y0:y1, x0:x1, :] += tile_image[:, :h_slice, :w_slice, :] * mask[:h_slice, :w_slice, :]
                    weight[:, y0:y1, x0:x1, :] += mask[:h_slice, :w_slice, :]

        # Averaging and blank handling
        canvas = canvas / torch.clamp(weight, min=1.0)
        canvas[weight == 0] = 0

        return IO.NodeOutput(canvas)



#--------------------------

class SafeMask_Subeditor(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SafeMask_Subeditor",
            display_name="마스크 서브에디터",
            category="커스텀마스크/유틸",
            description="마스크 추가 작업용 에디터. 과처리된 마스크를 손질하거나 덧붙일 수 있습니다.",
            inputs=[
                IO.Mask.Input("base_mask", tooltip="작업할 마스크"),
                IO.Mask.Input("user_mask", tooltip="지울 영역 선택"),
                IO.Combo.Input("subtract_mode", options=["delete","subtract","difference"], default="delete",
                               tooltip="제거 방식.\n 완전제거-부분제거-차이값만 남김"),
                IO.Combo.Input("blur_mode", options=["none","soft","hard"], default="none", tooltip="블러 셋팅"),
                IO.Combo.Input("blur_str", options=["0","1","2","3","4","5","6","7","8","9","10"], default="0", tooltip="블러 커널 크기"),
                IO.Mask.Input("add_mask", tooltip="덧붙일 추가 마스크", optional=True),
            ],
            outputs=[IO.Mask.Output("mask", tooltip="편집된 마스크")]
        )

    @classmethod
    def execute(cls, base_mask, user_mask, subtract_mode, blur_mode, blur_str="0", add_mask=None) -> IO.NodeOutput:

        result = ensure_mask_tensor(base_mask)
        user_mask = ensure_mask_tensor(user_mask)

        # Removal mode
        if subtract_mode == "delete":
            result = torch.where(user_mask > 0.5, torch.zeros_like(result), result)
        elif subtract_mode == "subtract":
            result = torch.clamp(result - user_mask, 0.0, 1.0)
        elif subtract_mode == "difference":
            result = torch.abs(result - user_mask)

        # Additional mask
        if add_mask is not None:
            add_mask = ensure_mask_tensor(add_mask)
            result = torch.clamp(result + add_mask, 0.0, 1.0)

        # Blur
        blur_strength = int(blur_str)
        if blur_strength > 0 and blur_mode != "none":
            if blur_mode == "soft":
                k = min(2 * blur_strength - 1, 11)
                kernel = torch.ones((1,1,k,k), device=result.device) / (k*k)
                result = F.conv2d(ensure_mask_tensor(result), kernel, padding=k//2)
            elif blur_mode == "hard":
                k = min(2 * blur_strength + 1, 13)
                kernel = torch.ones((1,1,k,k), device=result.device) / (k*k)
                result = F.conv2d(ensure_mask_tensor(result), kernel, padding=k//2)

        result = ensure_mask_tensor(result)
        return IO.NodeOutput(result)

#--------------------------

class SafeMask_CompositeAdv(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SafeMask_CompositeAdv",
            display_name="콤포짓 마스크 어드밴스(멀티)",
            category="커스텀마스크/유틸",
            description="다중 마스크 합성 + 경계보정 + 팽창/축소(소프트) + 블러 처리.",
            inputs=[
                IO.Mask.Input("base_mask", tooltip="원본 마스크"),
                IO.Mask.Input("user_mask", tooltip="합성 대상 마스크"),
                IO.Combo.Input("blend_mode", options=["max","average","overwrite","or","and"], default="max", tooltip="합성 방식 지정"),
                IO.Float.Input("feather_strength", default=0, min=0, max=5, step=1, tooltip="페더링 강도 설정"),
                IO.Int.Input("mask_adjust_unit", default=0, min=-5, max=5, step=1, tooltip="양수=팽창, 음수=축소 (소프트 처리)"),
                IO.Combo.Input("blur_mode", options=["none","soft","hard"], default="none", tooltip="블러 셋팅"),
                IO.Combo.Input("blur_str", options=["0","1","2","3","4","5","6","7","8","9","10"], default="0", tooltip="블러 강도 선택"),
                IO.Combo.Input("invert", options=["off","on"], default="off", tooltip="반전"),
                IO.Mask.Input("add_mask1", tooltip="추가 마스크", optional=True),
                IO.Mask.Input("add_mask2", tooltip="추가 마스크", optional=True),
                IO.Mask.Input("add_mask3", tooltip="추가 마스크", optional=True),
                IO.Mask.Input("add_mask4", tooltip="추가 마스크", optional=True),
                IO.Mask.Input("add_mask5", tooltip="추가 마스크", optional=True),
            ],
            outputs=[IO.Mask.Output("mask", tooltip="합성된 마스크")]
        )

    @classmethod
    def execute(cls, base_mask, user_mask, blend_mode="max", feather_strength=0, mask_adjust_unit=0,
                blur_mode="none", blur_str="0", invert="off", add_mask1=None, add_mask2=None, add_mask3=None,
                add_mask4=None, add_mask5=None) -> IO.NodeOutput:

        result = ensure_mask_tensor(base_mask)
        user_mask = ensure_mask_tensor(user_mask)

        # Basic Maskcomposite
        if blend_mode == "max":
            result = torch.max(result, user_mask)
        elif blend_mode == "average":
            result = (result + user_mask) / 2
        elif blend_mode == "overwrite":
            result = torch.where(user_mask > 0.5, torch.ones_like(result), result)
        elif blend_mode == "or":
            result = torch.clamp(result + user_mask, 0, 1)
        elif blend_mode == "and":
            result = result * user_mask

        # Additional Maskcomposite
        for add_mask in [add_mask1, add_mask2, add_mask3, add_mask4, add_mask5]:
            if add_mask is not None:
                add_mask = ensure_mask_tensor(add_mask)
                if blend_mode == "max":
                    result = torch.max(result, add_mask)
                elif blend_mode == "average":
                    result = (result + add_mask) / 2
                elif blend_mode == "overwrite":
                    result = torch.where(add_mask > 0.5, torch.ones_like(result), result)
                elif blend_mode == "or":
                    result = torch.clamp(result + add_mask, 0, 1)
                elif blend_mode == "and":
                    result = result * add_mask

        # Feathering
        if feather_strength > 0:
            k = max(1, 2 * int(feather_strength) - 1)
            kernel = torch.ones((1,1,k,k), device=result.device) / (k*k)
            result = F.conv2d(result, kernel, padding=k//2)

        if mask_adjust_unit != 0:
            k = min(2*abs(mask_adjust_unit)+1, 9)
            kernel = torch.ones((1,1,k,k), device=result.device) / (k*k)
            result = F.conv2d(result, kernel, padding=k//2)
            if mask_adjust_unit < 0:
                result = torch.sigmoid(5*(result-0.5))
            else:
                result = torch.clamp(result, 0, 1)

        blur_strength = int(blur_str)
        if blur_strength > 0 and blur_mode != "none":
            k = max(1, 2 * blur_strength - 1)
            kernel = torch.ones((1,1,k,k), device=result.device) / (k*k)
            if blur_mode == "soft":
                result = F.conv2d(result, kernel, padding=k//2)
            elif blur_mode == "hard":
                result = F.conv2d(result, kernel, padding=k//2)
                result = F.conv2d(result, kernel, padding=k//2)

        # Invert
        if invert == "on":
            result = 1 - result

        result = torch.clamp(result, 0, 1)
        result = ensure_mask_tensor(result)
        return IO.NodeOutput(result)


#--------------------------

class SafeMaskAmplifier(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SafeMaskAmplifier",
            display_name="마스크 증폭기",
            category="커스텀마스크/유틸",
            description="입력 마스크를 기반으로 증폭/보정 후 유저마스크와 합성 + 부분강조",
            inputs=[
                IO.Mask.Input("mask", tooltip="입력 마스크"),
                IO.Float.Input("amp_factor", default=1.0, min=1.0, max=1.999, step=0.001, tooltip="출력 증폭 계수"),
                IO.Combo.Input("invert", options=["off","on"], default="off", tooltip="반전"),
                IO.Float.Input("gamma", default=0.6, min=0.001, max=1.999, step=0.001, tooltip="감마 보정 (pow 값)"),
                IO.Combo.Input("clahe", options=["off","on"], default="off", tooltip="클라헤 보정"),
                IO.Mask.Input("user_mask", tooltip="합성 대상 마스크", optional=True),
                IO.Combo.Input("blend_mode", options=["max","average","overwrite"], default="max", tooltip="합성 방식 지정"),
                IO.Combo.Input("highlight_region", options=["none","left","right","top","bottom"], default="none", tooltip="부분 강조 방향"),
                IO.Float.Input("highlight_strength", default=0.0, min=0.0, max=2.0, step=0.1, tooltip="부분 강조 강도"),
            ],
            outputs=[IO.Mask.Output("mask", tooltip="증폭된 마스크")]
        )

    @classmethod
    def execute(cls, mask, amp_factor=1.0, invert="off", gamma=0.6, clahe="off", user_mask=None, blend_mode="max",
                highlight_region="none", highlight_strength=0.0) -> IO.NodeOutput:

        amplication = min(max(amp_factor,0.000),1.999)
        mask_tensor = ensure_mask_tensor(mask)
        mask_tensor = torch.clamp(mask_tensor * amplication, 0, 1)

        # CLAHE
        if clahe == "on":
            mask_img = (mask_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
            clahe_obj = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(32,32))
            mask_img = clahe_obj.apply(mask_img)
            mask_tensor = ensure_mask_tensor(torch.from_numpy(mask_img).float() / 255.0)

        # Blending Usermask
        if user_mask is not None:
            user_mask = ensure_mask_tensor(user_mask)
            if blend_mode == "max":
                mask_tensor = torch.max(mask_tensor, user_mask)
            elif blend_mode == "average":
                mask_tensor = (mask_tensor + user_mask) / 2
            elif blend_mode == "overwrite":
                mask_tensor = torch.where(user_mask > 0.5, torch.ones_like(mask_tensor), mask_tensor)

        # Emphasis on part
        if highlight_region != "none" and highlight_strength > 0.0:
            B, C, H, W = mask_tensor.shape
            if highlight_region == "left":
                grad = torch.linspace(highlight_strength, 1.0, W, device=mask_tensor.device)
                grad = grad.unsqueeze(0).unsqueeze(0).unsqueeze(2).repeat(B,1,H,1)
            elif highlight_region == "right":
                grad = torch.linspace(1.0, highlight_strength, W, device=mask_tensor.device)
                grad = grad.unsqueeze(0).unsqueeze(0).unsqueeze(2).repeat(B,1,H,1)
            elif highlight_region == "top":
                grad = torch.linspace(highlight_strength, 1.0, H, device=mask_tensor.device)
                grad = grad.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(B,1,1,W)
            elif highlight_region == "bottom":
                grad = torch.linspace(1.0, highlight_strength, H, device=mask_tensor.device)
                grad = grad.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(B,1,1,W)
            mask_tensor = torch.clamp(mask_tensor * grad, 0, 1)

        # Gamma Setting
        gamma_factor = min(max(gamma,0.000),1.999)
        mask_tensor = mask_tensor.pow(gamma_factor)

        # Invert
        if invert == "on":
            mask_tensor = 1 - mask_tensor

        mask_tensor = ensure_mask_tensor(mask_tensor)
        return IO.NodeOutput(mask_tensor)



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
    "SafeTileSoftFillng": SafeTileSoftFillng,
    "SafeMask_Subeditor": SafeMask_Subeditor,
    "SafeMask_CompositeAdv": SafeMask_CompositeAdv,
    "SafeMaskAmplifier": SafeMaskAmplifier,
    "AutoMaskGenerator": AutoMaskGenerator,
}

TEST_NODE_DISPLAY_NAME_MAPPINGS = {
    "SafeTileSoftFillng": "타일링 보정채우기(pytorch)",
    "SafeMask_Subeditor": "마스크 서브에디터",
    "SafeMask_CompositeAdv": "콤포짓 마스크 어드밴스(멀티)",
    "SafeMaskAmplifier": "마스크 증폭기",
    "AutoMaskGenerator": "자동 마스크 생성기"
}
