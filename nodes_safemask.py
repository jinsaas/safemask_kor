
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

def ensure_mask_output_shape(mask_tensor: torch.Tensor) -> torch.Tensor:

    if mask_tensor.ndim == 2:   # [H, W] -> [1, 1, H, W]
        return mask_tensor.unsqueeze(0).unsqueeze(0)
    elif mask_tensor.ndim == 3: # [B, H, W],[1, H, W] -> [B, 1, H, W]
        return mask_tensor.unsqueeze(1)
    elif mask_tensor.ndim == 4: # [B, C, H, W] -> [B, 1, H, W]
        return mask_tensor[:, 0:1, :, :]
    return mask_tensor

def apply_feathering(mask_tensor: torch.Tensor, feather_size: int, feather_strength: float) -> torch.Tensor:
    if feather_size <= 0:
        return mask_tensor

    if mask_tensor.ndim == 3:
        mask_tensor = mask_tensor.unsqueeze(1)

    kernel_size = feather_size * 2 + 1
    sigma = feather_size / 2.0
    
    k = kernel_size // 2
    x = torch.arange(-k, k + 1, dtype=mask_tensor.dtype, device=mask_tensor.device)
    gauss = torch.exp(-(x**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    kernel2d = (gauss.unsqueeze(1) @ gauss.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    channels = mask_tensor.shape[1]
    kernel2d = kernel2d.repeat(channels, 1, 1, 1)

    # 4. apply Conv2d
    blurred = F.conv2d(mask_tensor, kernel2d, padding=k, groups=channels)
    
    # 5. Strength and Clamp
    if feather_strength != 1.0:
        blurred = torch.clamp(blurred * feather_strength, 0.0, 1.0)
        
    return blurred
    
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
#----------------------------------------------------
# Mask Node Code(Base)
#----------------------------------------------------
# Mask Preview - original implement from
# https://github.com/cubiq/ComfyUI_essentials/blob/9d9f4bedfc9f0321c19faf71855e228c93bd0dc9/mask.py#L81
# upstream requested in https://github.com/Kosinkadink/rfcs/blob/main/rfcs/0000-corenodes.md#preview-nodes
class SafeMaskLoader(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        files = [f for f in os.listdir(mask_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not files:
            files = ["sample.png"]

        return IO.Schema(
            node_id="SafeMaskLoader",
            display_name="마스크 로더",
            category="커스텀마스크/기본형",
            description="인풋 폴더 내 mask 디렉토리에서 마스크 이미지를 불러옵니다.\n"
            "마스크를 미리보는 기능도 있지만 프리뷰의 이미지 에디터 기능으로 수정된 내용은 실제 마스크 데이터에 반영되지 않으며,\n"
            "데이터 손상/왜곡의 원인이 될 수 있습니다.",
            inputs=[
                IO.Combo.Input("mask_file", options=files, default=files[0], tooltip="불러올 마스크 파일 이름. 위젯 선택시 반드시 [Mask]폴더 안에서 가져오는걸 권장합니다.", upload=IO.UploadType.image, image_folder=IO.FolderType.input),
                IO.Boolean.Input("invert", default=False, tooltip="마스크 반전 여부"),
                IO.Boolean.Input("show_preview", default=False, tooltip="프리뷰 표시 여부"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[IO.Mask.Output("mask_out", tooltip="불러온 마스크 텐서")],
        )

    @classmethod
    def execute(cls, mask_file, invert=False, show_preview=False) -> IO.NodeOutput:
        mask_dir = os.path.join(folder_paths.get_input_directory(), "Mask")
        filepath = os.path.join(mask_dir, mask_file)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Mask file not found: {filepath}")

        # 1. image load & grayscale convert
        with Image.open(filepath) as img:
            if img.mode in ('I;16', 'I;16L', 'I;16B'):
                raise ValueError(f"지원하지 않는 이미지 형식입니다: 16비트 이상의 이미지는 마스크로 사용할 수 없습니다. (파일: {mask_file})")
            arr = np.array(img, dtype=np.float32) / 255.0
            image_tensor = ensure_image_tensor(arr)
            gray_mask = image_tensor[:,0:1,:,:]

        # 3. normalize mask
        mask = normalize_mask_tensor(gray_mask)

        # 4. mask dim check
        mask = ensure_mask_tensor(mask)

        # 5. invert
        if invert:
            mask = 1.0 - mask
            mask = torch.clamp(mask, 0.0, 1.0)

        mask = ensure_mask_output_shape(mask)
        
        if show_preview:
            preview_mask = mask # [B, 1, H, W]
            return IO.NodeOutput(mask, ui=UI.PreviewMask(preview_mask))
        else:
            return IO.NodeOutput(mask, )


#----------------------------------------------------

class SafeMaskToImage(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SafeMaskToImage",
            display_name="마스크 이미지 변환",
            category="커스텀마스크/기본형",
            description="마스크 텐서를 그레이스케일 이미지로 변환합니다.",
            inputs=[
                IO.Mask.Input("mask", tooltip="입력된 마스크 텐서를 그레이스케일 이미지로 변환합니다.")
            ],
            outputs=[
                IO.Image.Output("image", tooltip="그레이스케일 이미지")
            ],
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "입력된 마스크 텐서를 그레이스케일 이미지로 변환합니다."}),
            }
        }

    @classmethod
    def execute(cls, mask) -> IO.NodeOutput:
        mask_tensor = normalize_mask_tensor(mask)

        if mask_tensor.max() > 1.0:
            mask_tensor = mask_tensor / 255.0
        else:
            mask_tensor = mask_tensor.clamp(0, 1)

        result = mask_tensor.unsqueeze(-1).expand(-1, -1, 3)

        result = result.unsqueeze(0)

        return IO.NodeOutput(result,)

#----------------------------------------------------

class SafeImageToMask(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SafeImageToMask",
            display_name="이미지 마스크 변환",
            category="커스텀마스크/기본형",
            description="이미지에서 특정 채널을 추출하여 마스크로 변환합니다.",
            inputs=[
                IO.Image.Input("base_Image", tooltip="마스크를 추출할 대상 이미지"),
                IO.Combo.Input("channel", options=["red","green","blue","alpha"], default="red", tooltip="마스크 추출에 사용할 채널\n"
                                    "알파 채널 선택시 알파채널이 없으면 흰색(불투명)으로 나옵니다."),
            ],
            outputs=[
                IO.Mask.Output("mask", tooltip="변환된 마스크 이미지"),
            ],
        )

    @classmethod
    def execute(cls, base_Image, channel="red") -> IO.NodeOutput:
        image = ensure_image_tensor(base_Image)
        channels = ["red", "green", "blue", "alpha"]
        idx = channels.index(channel)
        print(">>> type(image):", type(image))
        if isinstance(image, torch.Tensor):
            print(">>> image.shape:", image.shape)
        elif isinstance(image, np.ndarray):
            print(">>> image.shape:", image.shape)
        else:
            print(">>> image:", image)


        image = ensure_image_tensor(image)

        if image.max() > 1.0:
            image = image / 255.0
        else:
            image = image.clamp(0, 1)

        if idx >= image.shape[1]:
            if channel == "alpha":
                mask = torch.ones((image.shape[2], image.shape[3]), dtype=torch.float32, device=image.device)
            else:
                raise ValueError(f"Channel '{channel}' not available in image with {image.shape[1]} channels")
        else:
            mask = image[0, idx, :, :] 
            
        mask = squeeze_mask_output(mask) 
        
        return IO.NodeOutput(mask,)
#----------------------------------------------------
# Mask Preview - original implement from
# https://github.com/cubiq/ComfyUI_essentials/blob/9d9f4bedfc9f0321c19faf71855e228c93bd0dc9/mask.py#L81
# upstream requested in https://github.com/Kosinkadink/rfcs/blob/main/rfcs/0000-corenodes.md#preview-nodes

class SafeImageColorToMask(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SafeImageColorToMask",
            display_name="색상 마스크 변환",
            category="커스텀마스크/기본형",
            description="이미지에서 지정된 색상을 on/off로 선택하여 마스크 생성.",
            inputs=[
                IO.Image.Input("base_Image", tooltip="마스크를 생성할 대상 이미지"),
                IO.Boolean.Input("red", default=False, tooltip="적색 기반 마스크"),
                IO.Boolean.Input("green", default=False, tooltip="녹색 기반 마스크"),
                IO.Boolean.Input("blue", default=False, tooltip="청색 기반 마스크"),
                IO.Boolean.Input("yellow", default=False, tooltip="노란색 기반 마스크"),
                IO.Boolean.Input("magenta", default=False, tooltip="자홍색 기반 마스크"),
                IO.Boolean.Input("cyan", default=False, tooltip="청록색 기반 마스크"),
                IO.Boolean.Input("black", default=False, tooltip="흑색 기반 마스크"),
                IO.Combo.Input("mode", options=["switch","hex"], default="switch", tooltip="색상 처리 방식"),
                IO.String.Input("hex_color", default="#FF0000", tooltip="헥스값으로 지정된 색상"),
                IO.Boolean.Input("show_preview", default=False, tooltip="프리뷰 표시 여부"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Mask.Output("mask_out", tooltip="변환된 마스크 이미지"),
            ],
        )

    @classmethod
    def execute(cls, base_Image, red=False, green=False, blue=False, yellow=False, magenta=False,
                cyan=False, black=False, mode="switch", hex_color="#FF0000", show_preview=False) -> IO.NodeOutput:

        image = ensure_image_tensor(base_Image)

        temp = (torch.clamp(image, 0, 1.0) * 255.0).round().to(torch.int)
        R, G, B = temp[:, 0, :, :].float(), temp[:, 1, :, :].float(), temp[:, 2, :, :].float()

        mask = torch.zeros_like(R, dtype=torch.float)


        if mode == "switch":
            if red:
                mask = torch.max(mask, ((R > G + 10) & (R > B + 10)).float())
            if green:
                mask = torch.max(mask, ((G > R + 10) & (G > B + 10)).float())
            if blue:
                mask = torch.max(mask, ((B > R + 10) & (B > G + 10)).float())
            if yellow:
                mask = torch.max(mask, ((R > B + 10) & (G > B + 10)).float())
            if magenta:
                mask = torch.max(mask, ((R > G + 10) & (B > G + 10)).float())
            if cyan:
                mask = torch.max(mask, ((G > R + 10) & (B > R + 10)).float())
            if black:
                mask = torch.max(mask, ((R < 20) & (G < 20) & (B < 20)).float())

        elif mode == "hex":
            hex_color = hex_color.lstrip("#")
            r_target = int(hex_color[0:2], 16)
            g_target = int(hex_color[2:4], 16)
            b_target = int(hex_color[4:6], 16)

            tolerance = 10
            cond_hex = ((R - r_target).abs() < tolerance) & \
                       ((G - g_target).abs() < tolerance) & \
                       ((B - b_target).abs() < tolerance)
            mask = cond_hex.float()

        mask = ensure_mask_output_shape(mask)
        
        if show_preview:
            preview_mask = mask # [B, 1, H, W]
            return IO.NodeOutput(mask, ui=UI.PreviewMask(preview_mask))
        else:
            return IO.NodeOutput(mask, )

#----------------------------------------------------
# Mask Node Code(Edit)
#----------------------------------------------------

class SafeImageComposite(IO.ComfyNode):

    @classmethod
    def sanitize_input_mask(cls, mask):
        if mask is None:
            return None

        if isinstance(mask, dict):
            if "latent_mask" in mask:
                mask = mask["latent_mask"]
            elif "noise_mask" in mask:
                mask = mask["noise_mask"]
            elif "mask" in mask:
                mask = mask["mask"]
            elif "samples" in mask:
                mask = mask["samples"]

        if hasattr(mask, "dtype") and mask.dtype == torch.bool:
            mask = mask.float()
        return ensure_mask_tensor(mask)

    @classmethod
    def image_alpha_fix_NCHW(cls, destination, source): 

        # destination
        if destination.shape[1] == 1:  # grayscale → RGB
            destination = destination.repeat(1, 3, 1, 1)
        elif destination.shape[1] == 4:  # RGBA → RGB
            destination = destination[:, :3]

        # source
        if source.shape[1] == 1:  # grayscale → RGB
            source = source.repeat(1, 3, 1, 1)
        elif source.shape[1] == 4:  # RGBA → RGB
            source = source[:, :3]

        return destination, source

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SafeImageComposite",
            display_name="안정화 이미지 합성",
            category="커스텀마스크/에디터",
            description="두 이미지를 마스크를 사용해 합성합니다. 부분 블렌딩에 적합합니다.\n"
                        "소스와 마스크는 리사이징이 가능하지만, 품질 하락이 발생하기 때문에\n"
                        "사이즈는 일치시키는게 좋습니다.",
            inputs=[
                IO.Image.Input("destination", tooltip="합성에 사용할 원본 이미지"),
                IO.Image.Input("source", tooltip="합성에 사용할 추가 이미지"),
                IO.Mask.Input("mask", tooltip="합성에 사용할 사용할 마스크"),
                IO.Boolean.Input("resize_source", tooltip="소스 이미지를 대상 크기에 맞게 자동 조정", default=False),
                IO.Float.Input("alpha", min=0.0, max=1.0, step=0.1, default=0.0, tooltip="추가 이미지 주입 비율"),
                IO.Combo.Input("blend_mode", options=["normal", "soft", "darken","lighten"], default="normal", tooltip="블렌딩 타입 선택"),
                IO.Int.Input("spread_mode", min=0, max=10, step=1, default=0, tooltip="스프레딩 마스크 세팅 선택"),
            ],
            outputs=[
                IO.Image.Output("image_out", tooltip="합성된 이미지")
                ],
        )

    @classmethod
    def execute(cls, destination, source, mask, resize_source=False, alpha=0.0, blend_mode="normal", spread_mode=0) -> IO.NodeOutput:
        destination = ensure_image_tensor(destination)
        source      = ensure_image_tensor(source)

        mask = cls.sanitize_input_mask(mask)

        alpha = min(max(alpha, 0.0), 1.0)
        spread_mode = min(max(spread_mode, 0), 10)
        kernel_map = {i: (2*i-1) for i in range(1,11)}
        kernel_size = kernel_map.get(spread_mode, 0)
        if kernel_size > 0:
            kernel = torch.ones((1,1,kernel_size,kernel_size), dtype=mask.dtype, device=mask.device)
            pad = kernel_size // 2
            mask = torch.nn.functional.conv2d(mask, kernel, padding=pad)
            mask = mask / (kernel_size * kernel_size)

        mask = mask

        mask_rgb = mask.repeat(1,3,1,1)

        H, W = destination.shape[2], destination.shape[3]
        source = source[:, :, :H, :W]
        mask   = mask_rgb[:, :, :H, :W]

        destination, source = cls.image_alpha_fix_NCHW(destination, source)

        dest_out = composite(destination, source, mask, alpha, blend_mode, resize_source)

        if dest_out.shape[1] == 1:
            output = dest_out.repeat(1,3,1,1)
        else :
            output = dest_out

        output = output.permute(0, 2, 3, 1)

        return IO.NodeOutput(output,)

#----------------------------------------------------

class SafeMaskComposite(IO.ComfyNode):

    @classmethod
    def sanitize_input_mask(cls, mask):
        if mask is None:
            return None

        if isinstance(mask, dict):
            if "latent_mask" in mask:
                mask = mask["latent_mask"]
            elif "noise_mask" in mask:
                mask = mask["noise_mask"]
            elif "mask" in mask:
                mask = mask["mask"]
            elif "samples" in mask:
                mask = mask["samples"]

        if hasattr(mask, "dtype") and mask.dtype == torch.bool:
            mask = mask.float()
        return ensure_mask_tensor(mask)

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SafeMaskComposite",
            display_name="마스크 합성",
            category="커스텀마스크/에디터",
            description="두 마스크를 결합합니다. add, multiply, subtract, 논리 연산(and, or, xor) 지원.\n"
                        "spread 옵션으로 마스크 확장 가능.",
            inputs=[
                IO.Mask.Input("destination", tooltip="결합에 사용할 첫 번째 마스크"),
                IO.Mask.Input("source", tooltip="결합에 사용할 두 번째 마스크"),
                IO.Combo.Input("operation", options=["multiply","add","subtract","and","or","xor"], default="add", tooltip="마스크 결합 방식"),
                IO.Int.Input("spread", min=0, max=10, step=1, default=0, tooltip="마스크 확장 정도"),
                IO.Combo.Input("blend_mode", options=["normal", "soft", "screen","overlay","darken","lighten"], default="normal", tooltip="블렌딩 타입 선택")
            ],
            outputs=[
                IO.Mask.Output("mask_out", tooltip="합성된 마스크")
            ],
        )

    @classmethod
    def execute(cls, destination, source, operation="add", blend_mode="normal", spread=0) -> IO.NodeOutput:
        destination = cls.sanitize_input_mask(destination)
        source      = cls.sanitize_input_mask(source)

        H = min(destination.shape[-2], source.shape[-2])
        W = min(destination.shape[-1], source.shape[-1])
        destination = destination[:, :, :H, :W]
        source      = source[:, :, :H, :W]

        spread = min(max(spread, 0), 10)
        kernel_map = {i: (2*i-1) for i in range(1,11)}
        kernel_size = kernel_map.get(spread, 0)
        
        if kernel_size > 0:
            kernel = torch.ones((1,1,kernel_size,kernel_size), dtype=destination.dtype, device=destination.device)
            pad = kernel_size // 2
            destination = torch.nn.functional.conv2d(destination, kernel, padding=pad)
            destination = destination / (kernel_size * kernel_size)

        if operation == "multiply":
            combined = destination * source
        elif operation == "add":
            combined = destination + source
        elif operation == "subtract":
            combined = destination - source
        elif operation == "and":
            combined = torch.bitwise_and(destination.round().bool(), source.round().bool()).float()
        elif operation == "or":
            combined = torch.bitwise_or(destination.round().bool(), source.round().bool()).float()
        elif operation == "xor":
            combined = torch.bitwise_xor(destination.round().bool(), source.round().bool()).float()
        else:
            combined = source

        if blend_mode == "soft":
            output = (destination * 0.5 + source * 0.5)
        elif blend_mode == "screen":
            output = 1 - ((1 - destination) * (1 - source))
        elif blend_mode == "overlay":
            output = torch.where(
                destination < 0.5,
                2 * destination * source,
                1 - 2 * (1 - destination) * (1 - source)
            )
        elif blend_mode == "darken":
            output = torch.min(destination, source)
        elif blend_mode == "lighten":
            output = torch.max(destination, source)
        else:
            output = combined

        output = torch.clamp(output, 0.0, 1.0)
        return IO.NodeOutput(output,)

#----------------------------------------------------

class SafeLatentComposite(IO.ComfyNode):

    @classmethod
    def sanitize_input_mask(cls, mask):
        if mask is None:
            return None

        if isinstance(mask, dict):
            if "latent_mask" in mask:
                mask = mask["latent_mask"]
            elif "noise_mask" in mask:
                mask = mask["noise_mask"]
            elif "mask" in mask:
                mask = mask["mask"]
            elif "samples" in mask:
                mask = mask["samples"]

        if hasattr(mask, "dtype") and mask.dtype == torch.bool:
            mask = mask.float()
        return ensure_mask_tensor(mask)

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SafeLatentComposite",
            display_name="안정화 라텐트 합성",
            category="커스텀마스크/에디터",
            description="원본 라텐트에 소스 라텐트를 영역에 맞게 합성합니다.\n"
                        "마스크를 사용해 합성영역을 조절할 수 있습니다.\n"
                        "엣지 어텐션을 사용 시 라텐트 영역을 추가 조절할 수 있는 기능이 열립니다.\n"
                        "소스와 마스크는 리사이징이 가능하지만, 품질 하락이 발생하기 때문에\n"
                        "사이즈는 일치시키는게 좋습니다.",
            inputs=[
                IO.Latent.Input("destination", tooltip="합성에 사용할 원본 라텐트"),
                IO.Latent.Input("source", tooltip="합성에 사용할 추가 라텐트"),
                IO.Boolean.Input("resize_source", default=False, tooltip="추가 라텐트 자동 리사이징"),
                IO.Mask.Input("mask", tooltip="합성에 사용할 참고 마스크", optional=True),
                IO.Float.Input("alpha", min=0.0, max=1.0, default=0.0, step=0.1, tooltip="추가 라텐트 주입 비율"),
                IO.Boolean.Input("edge_attention", tooltip="라텐트 경계마스크 추가여부", default=False),
                IO.Float.Input("brightness_strength", min=0.0, max=2.0, default=1.0, step=0.1, tooltip="추가 밝기 조정 지시"),
                IO.Float.Input("contrast_strength", min=0.0, max=2.0, default=1.0, step=0.1, tooltip="추가 대비 조정 지시"),
                IO.Float.Input("sharpen_strength", min=0.0, max=2.0, default=1.0, step=0.1, tooltip="추가 샤픈 조정 지시")
            ],
            outputs=[
                IO.Latent.Output("samples", tooltip="합성된 라텐트")
            ],
        )

    @classmethod
    def execute(cls, destination, source, mask=None, alpha=0.0, resize_source=False, edge_attention=False, brightness_strength=1.0, contrast_strength=1.0, sharpen_strength=1.0) -> IO.NodeOutput:
        steps=7
        pbar = ProgressBar(int(steps))
        alpha = min(max(alpha, 0.0), 1.0)
        destination = destination["samples"].clone()
        progressbar_update(pbar, 1, steps)
        source = source["samples"]
        if destination.ndim == 4:
            if mask is None:
                mask = torch.ones((destination.shape[0], 1, *destination.shape[2:]), device=destination.device)
                progressbar_update(pbar, 1, steps)
            else:
                mask = cls.sanitize_input_mask(mask)
                progressbar_update(pbar, 1, steps)
            combined = composite(destination, source, mask, alpha, blend_mode="normal", resize_source=resize_source)
            progressbar_update(pbar, 1, steps)

        elif destination.ndim == 5:
            if mask is None:
                mask = torch.ones((destination.shape[0], 1, destination.shape[2], destination.shape[3], destination.shape[4]),
                          device=destination.device)
                progressbar_update(pbar, 1, steps)
            else:
                pass
                progressbar_update(pbar, 1, steps)

            combined = compositeflow(destination, source, mask, alpha, resize_source)
            progressbar_update(pbar, 1, steps)

        else:
            raise ValueError(f"Unexpected latent shape: {destination.shape}")



        if edge_attention:
            # create edge mask
            if combined.ndim == 4:  # NCHW
                edge_map = sobel_2d(combined.mean(dim=1, keepdim=True))
            elif combined.ndim == 5:  # NCDHW
                edge_map = sobel_3d(combined.mean(dim=1, keepdim=True))
            attention = torch.clamp(edge_map.mean(dim=1, keepdim=True), 0.0, 1.0)
            progressbar_update(pbar, 1, steps)

            brightness_strength = min(max(brightness_strength, 0.0), 2.0)
            contrast_strength = min(max(contrast_strength, 0.0), 2.0)
            sharpen_strength = min(max(sharpen_strength, 0.0), 2.0)
            if brightness_strength != 1.0:
                combined = combined * (1.0 + attention * (brightness_strength - 1.0))
            progressbar_update(pbar, 1, steps)

            if contrast_strength != 1.0:
                mean_val = combined.mean()
                combined = (combined - mean_val) * (1.0 + attention * (contrast_strength - 1.0)) + mean_val
            progressbar_update(pbar, 1, steps)

            if sharpen_strength > 0.0:
                if combined.ndim == 4:  # NCHW
                    blur = torch.nn.functional.avg_pool2d(combined, kernel_size=3, stride=1, padding=1)
                elif combined.ndim == 5:  # NCDHW
                    if combined.shape[2] == 1:  # D=1 → 2D
                        blur = torch.nn.functional.avg_pool2d(combined.squeeze(2), kernel_size=3, stride=1, padding=1)
                        blur = blur.unsqueeze(2)  # NCDHW
                    else:
                        blur = torch.nn.functional.avg_pool3d(combined, kernel_size=3, stride=1, padding=1)
                combined = combined + attention * sharpen_strength * (combined - blur)
            progressbar_update(pbar, 1, steps)
        else:
            progressbar_update(pbar, 4, steps)

        output = {}
        output["samples"] = combined
        return IO.NodeOutput(output)

    composite = execute  # TODO: remove
#----------------------------------------------------
# Mask Node Code(Transform)
#----------------------------------------------------

class SafeInvertMask(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SafeInvertMask",
            display_name="마스크 반전",
            category="커스텀마스크/변형",
            description="마스크 값을 반전합니다.",
            inputs=[IO.Mask.Input("mask", tooltip="원본 마스크")],
            outputs=[IO.Mask.Output("mask_out", tooltip="반전한 마스크 텐서")],
        )

    @classmethod
    def execute(cls, mask) -> IO.NodeOutput:
        mask = ensure_mask_tensor(mask)

        out = 1.0 - mask
        out = torch.clamp(out, 0.0, 1.0)

        return IO.NodeOutput(out,)

#----------------------------------------------------
# Mask Preview - original implement from
# https://github.com/cubiq/ComfyUI_essentials/blob/9d9f4bedfc9f0321c19faf71855e228c93bd0dc9/mask.py#L81
# upstream requested in https://github.com/Kosinkadink/rfcs/blob/main/rfcs/0000-corenodes.md#preview-nodes

class SafeGrowMask(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SafeGrowMask",
            display_name="마스크 확장",
            category="커스텀마스크/변형",
            description="마스크 영역을 지정한 픽셀 수만큼 확장합니다.",
            inputs=[
                IO.Mask.Input("mask", tooltip="확장할 대상 마스크"),
                IO.Int.Input("expand", default=0, min=0, max=25, step=1, tooltip="확장할 픽셀 수"),
                IO.Boolean.Input("tapered_corners", default=False, tooltip="모서리를 부드럽게 처리할지 여부"),
                IO.Boolean.Input("show_preview", default=False, tooltip="프리뷰 표시 여부"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Mask.Output("mask_out", tooltip="확장된 마스크"),
            ],
        )

    @classmethod
    def execute(cls, mask, expand, tapered_corners, show_preview=False) -> IO.NodeOutput:
        
        mask = ensure_mask_tensor(mask)

        expand = min(max(expand, 0), 25)
        if expand > 0:
            mask = dilate_mtensor(mask, kernel_size=3, iterations=expand, tapered_corners=tapered_corners)

        mask = ensure_mask_output_shape(mask)
        
        if show_preview:
            preview_mask = mask # [B, 1, H, W]
            return IO.NodeOutput(mask, ui=UI.PreviewMask(preview_mask))
        else:
            return IO.NodeOutput(mask, )

#----------------------------------------------------
# Mask Preview - original implement from
# https://github.com/cubiq/ComfyUI_essentials/blob/9d9f4bedfc9f0321c19faf71855e228c93bd0dc9/mask.py#L81
# upstream requested in https://github.com/Kosinkadink/rfcs/blob/main/rfcs/0000-corenodes.md#preview-nodes

class SafeShrinkMask(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SafeShrinkMask",
            display_name="마스크 수축",
            category="커스텀마스크/변형",
            description="마스크 영역을 지정한 픽셀 수만큼 수축합니다.",
            inputs=[
                IO.Mask.Input("mask", tooltip="축소할 원본 마스크"),
                IO.Int.Input("shrink", default=0, min=0, max=25, step=1, tooltip="축소할 픽셀 수"),
                IO.Boolean.Input("tapered_corners", default=False, tooltip="모서리를 부드럽게 처리할지 여부"),
                IO.Boolean.Input("show_preview", default=False, tooltip="프리뷰 표시 여부"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Mask.Output("mask_out", tooltip="수축된 마스크"),
            ],
        )

    @classmethod
    def execute(cls, mask, shrink, tapered_corners, show_preview=False) -> IO.NodeOutput:

        mask = ensure_mask_tensor(mask)

        shrink = min(max(shrink, 0), 25)
        if shrink > 0:
            mask = erode_mtensor(mask, kernel_size=3, iterations=shrink, tapered_corners=tapered_corners)

        mask = ensure_mask_output_shape(mask)
        
        if show_preview:
            preview_mask = mask # [B, 1, H, W]
            return IO.NodeOutput(mask, ui=UI.PreviewMask(preview_mask))
        else:
            return IO.NodeOutput(mask, )

#----------------------------------------------------
# Mask Preview - original implement from
# https://github.com/cubiq/ComfyUI_essentials/blob/9d9f4bedfc9f0321c19faf71855e228c93bd0dc9/mask.py#L81
# upstream requested in https://github.com/Kosinkadink/rfcs/blob/main/rfcs/0000-corenodes.md#preview-nodes

class SafeTransformMask(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SafeTransformMask",
            display_name="마스크 변형",
            category="커스텀마스크/변형",
            description="마스크 영역의 크기를 Lanczos 보간법으로 변형합니다.",
            inputs=[
                IO.Mask.Input("mask", tooltip="변형할 원본 마스크"),
                IO.Int.Input("width", default=512, min=16, max=2048, step=1, tooltip="출력 마스크 너비"),
                IO.Int.Input("height", default=512, min=16, max=2048, step=1, tooltip="출력 마스크 높이"),
                IO.Boolean.Input("tapered_corners", default=False, tooltip="모서리를 부드럽게 처리할지 여부"),
                IO.Boolean.Input("show_preview", default=False, tooltip="프리뷰 표시 여부"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Mask.Output("mask_out", tooltip="변형된 마스크"),
            ],
        )

    @classmethod
    def execute(cls, mask, width, height, tapered_corners, show_preview=False) -> IO.NodeOutput:
        
        mask = ensure_mask_tensor(mask)

        width = min(max(width, 16), 2048)
        height = min(max(height, 16), 2048)
        out = []
        for m in mask:
            arr = m[0].cpu().numpy().astype(np.float32)
            resized = cv2.resize(arr, (width, height), interpolation=cv2.INTER_LANCZOS4)

            if tapered_corners:
                kernel = np.array([[0,1,0],
                                   [1,1,1],
                                   [0,1,0]])
                resized = scipy.ndimage.grey_dilation(resized, footprint=kernel)

            out.append(torch.from_numpy(resized))

        mask_out = torch.stack(out, dim=0)
        mask = ensure_mask_output_shape(mask_out)
        
        if show_preview:
            preview_mask = mask # [B, 1, H, W]
            return IO.NodeOutput(mask, ui=UI.PreviewMask(preview_mask))
        else:
            return IO.NodeOutput(mask, )

#----------------------------------------------------
# Mask Preview - original implement from
# https://github.com/cubiq/ComfyUI_essentials/blob/9d9f4bedfc9f0321c19faf71855e228c93bd0dc9/mask.py#L81
# upstream requested in https://github.com/Kosinkadink/rfcs/blob/main/rfcs/0000-corenodes.md#preview-nodes

class SafeThresholdMask(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SafeThresholdMask",
            display_name="마스크 임계값 처리",
            category="커스텀마스크/변형",
            description="마스크를 임계값 기준으로 이진화합니다.",
            inputs=[
                IO.Mask.Input("mask", tooltip="이진화할 원본 마스크"),
                IO.Float.Input("value", default=0.5, min=0.0, max=1.0, step=0.01, tooltip="마스크를 이진화할 기준 값"),
                IO.Boolean.Input("show_preview", default=False, tooltip="프리뷰 표시 여부"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Mask.Output("mask_out", tooltip="임계값 처리된 마스크"),
            ],
        )

    @classmethod
    def execute(cls, mask, value, show_preview=False) -> IO.NodeOutput:
        mask = ensure_mask_tensor(mask)

        value = min(max(value, 0.0), 1.0)
        mask = (mask > value).float()
        
        mask = ensure_mask_output_shape(mask)
        
        if show_preview:
            preview_mask = mask # [B, 1, H, W]
            return IO.NodeOutput(mask, ui=UI.PreviewMask(preview_mask))
        else:
            return IO.NodeOutput(mask, )

#----------------------------------------------------

class SafeSolidMask(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SafeSolidMask",
            display_name="단색 마스크 생성",
            category="커스텀마스크/변형",
            description="지정된 크기와 값으로 단색 마스크를 생성합니다.",
            inputs=[
                IO.Float.Input("value", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="마스크 픽셀 값 (0=검정, 1=흰색)"),
                IO.Int.Input("width", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1, tooltip="생성할 마스크의 너비"),
                IO.Int.Input("height", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1, tooltip="생성할 마스크의 높이"),
            ],
            outputs=[
                IO.Mask.Output("mask_out", tooltip="생성된 단색 마스크"),
            ],
        )

    @classmethod
    def execute(cls, value, width, height) -> IO.NodeOutput:
        value = min(max(value, 0.0), 1.0)
        width = min(max(width, 1), nodes.MAX_RESOLUTION)
        height = min(max(height, 1), nodes.MAX_RESOLUTION)
        mask = torch.full((1, 1, height, width), value, dtype=torch.float32, device="cpu")
        mask = ensure_mask_output_shape(mask)

        return IO.NodeOutput(out,)

#----------------------------------------------------

class SafeImagePadding(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SafeImagePadding",
            display_name="이미지 패딩",
            category="커스텀마스크/변형",
            description="이미지를 상하좌우로 확장하고 지정된 색상으로 채웁니다.",
            inputs=[
                IO.Image.Input("image", tooltip="패딩을 적용할 원본 이미지"),
                IO.Int.Input("pad_top", default=0, min=0, max=2048, step=1, tooltip="위쪽으로 확장할 픽셀 수"),
                IO.Int.Input("pad_bottom", default=0, min=0, max=2048, step=1, tooltip="아래쪽으로 확장할 픽셀 수"),
                IO.Int.Input("pad_left", default=0, min=0, max=2048, step=1, tooltip="왼쪽으로 확장할 픽셀 수"),
                IO.Int.Input("pad_right", default=0, min=0, max=2048, step=1, tooltip="오른쪽으로 확장할 픽셀 수"),
                IO.Combo.Input("padding_color", options=["black", "white"], default="black", tooltip="패딩 영역에 채울 색상"),
            ],
            outputs=[
                IO.Image.Output("image_out", tooltip="패딩이 적용된 이미지"),
            ],
        )

    @classmethod
    def execute(cls, image, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, padding_color="black") -> IO.NodeOutput:
        
        image_tensor = ensure_image_tensor(image)

        pad_top = min(max(pad_top, 0), 2048)
        pad_bottom = min(max(pad_bottom, 0), 2048)
        pad_left = min(max(pad_left, 0), 2048)
        pad_right = min(max(pad_right, 0), 2048)
        b, c, h, w = image_tensor.shape
        fill_val = 0.0 if padding_color == "black" else 1.0

        canvas = torch.full(
            (b, c, h + pad_top + pad_bottom, w + pad_left + pad_right),
            fill_val,
            dtype=image_tensor.dtype,
            device=image_tensor.device
        )

        canvas[:, :, pad_top:pad_top + h, pad_left:pad_left + w] = image_tensor
        canvas = canvas.permute(0, 2, 3, 1)

        return IO.NodeOutput(canvas,)

#----------------------------------------------------

class SafeMaskPadding(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SafeMaskPadding",
            display_name="마스크 패딩",
            category="커스텀마스크/변형",
            description="마스크를 상하좌우로 확장하고 지정된 값으로 채웁니다.",
            inputs=[
                IO.Mask.Input("mask", tooltip="패딩을 적용할 원본 마스크"),
                IO.Int.Input("pad_top", default=0, min=0, max=2048, step=1, tooltip="위쪽으로 확장할 픽셀 수"),
                IO.Int.Input("pad_bottom", default=0, min=0, max=2048, step=1, tooltip="아래쪽으로 확장할 픽셀 수"),
                IO.Int.Input("pad_left", default=0, min=0, max=2048, step=1, tooltip="왼쪽으로 확장할 픽셀 수"),
                IO.Int.Input("pad_right", default=0, min=0, max=2048, step=1, tooltip="오른쪽으로 확장할 픽셀 수"),
                IO.Combo.Input("padding_color", options=["black", "white"], default="black", tooltip="패딩 영역에 채울 값"),
                IO.Int.Input("feather_size", default=0, min=0, max=50, step=1, tooltip="페더링 커널 크기"),
                IO.Float.Input("feather_strength", default=0.0, min=0.0, max=5.0, step=0.1, tooltip="페더링 강도"),
            ],
            outputs=[
                IO.Mask.Output("mask_out", tooltip="패딩이 적용된 마스크"),
            ],
        )

    @classmethod
    def execute(cls, mask, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, padding_color="black",
                feather_size=0, feather_strength=0.0) -> IO.NodeOutput:
        
        mask = ensure_mask_tensor(mask)

        b, c, h, w = mask.shape

        pad_top = min(max(pad_top, 0), 2048)
        pad_bottom = min(max(pad_bottom, 0), 2048)
        pad_left = min(max(pad_left, 0), 2048)
        pad_right = min(max(pad_right, 0), 2048)
        feather_size = min(max(feather_size, 0), 50)
        feather_strength = min(max(feather_strength, 0.0), 5.0)
        
        fill_val = 0.0 if padding_color == "black" else 1.0

        canvas = torch.full(
            (b, c, h + pad_top + pad_bottom, w + pad_left + pad_right),
            fill_val,
            dtype=mask.dtype,
            device=mask.device
        )
        canvas[:, :, pad_top:pad_top + h, pad_left:pad_left + w] = mask
    
        canvas = apply_feathering(canvas, feather_size, feather_strength)
        canvas = ensure_mask_output_shape(mask)

        return IO.NodeOutput(canvas,)

#----------------------------------------------------
# Mask Node Code(Cuttings)
#----------------------------------------------------
# Mask Preview - original implement from
# https://github.com/cubiq/ComfyUI_essentials/blob/9d9f4bedfc9f0321c19faf71855e228c93bd0dc9/mask.py#L81
# upstream requested in https://github.com/Kosinkadink/rfcs/blob/main/rfcs/0000-corenodes.md#preview-nodes
class SafeCropMask(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SafeCropMask",
            display_name="마스크 크롭",
            category="커스텀마스크/컷팅",
            description="마스크를 지정된 위치와 크기로 잘라냅니다.",
            inputs=[
                IO.Mask.Input("mask", tooltip="잘라낼 원본 마스크 텐서"),
                IO.Int.Input("x", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="잘라낼 영역의 X 좌표"),
                IO.Int.Input("y", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="잘라낼 영역의 Y 좌표"),
                IO.Int.Input("width", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1, tooltip="잘라낼 영역의 너비"),
                IO.Int.Input("height", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1, tooltip="잘라낼 영역의 높이"),
                IO.Boolean.Input("show_preview", default=False, tooltip="프리뷰 표시 여부"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Mask.Output("mask_out", tooltip="크롭된 마스크"),
            ],
        )

    @classmethod
    def execute(cls, mask, x, y, width, height, show_preview=False) -> IO.NodeOutput:

        mask = ensure_mask_tensor(mask)

        b, c, H, W = mask.shape

        x = min(max(x, 0), W - 1)
        y = min(max(y, 0), H - 1)
        width = min(max(width, 1), W - x)
        height = min(max(height, 1), H - y)

        cropped = mask[:, :, y:y + height, x:x + width]


        mask = ensure_mask_output_shape(cropped)
        
        if show_preview:
            preview_mask = mask # [B, 1, H, W]
            return IO.NodeOutput(mask, ui=UI.PreviewMask(preview_mask))
        else:
            return IO.NodeOutput(mask, )

#----------------------------------------------------
# Mask Preview - original implement from
# https://github.com/cubiq/ComfyUI_essentials/blob/9d9f4bedfc9f0321c19faf71855e228c93bd0dc9/mask.py#L81
# upstream requested in https://github.com/Kosinkadink/rfcs/blob/main/rfcs/0000-corenodes.md#preview-nodes


class SafeCenterCropMask(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SafeCenterCropMask",
            display_name="마스크 중앙 크롭",
            category="커스텀마스크/컷팅",
            description="마스크를 중앙 기준으로 잘라냅니다.",
            inputs=[
                IO.Mask.Input("mask", tooltip="잘라낼 원본 마스크 텐서"),
                IO.Int.Input("left", default=256, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="중앙 기준 왼쪽으로 잘라낼 픽셀 수"),
                IO.Int.Input("right", default=256, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="중앙 기준 오른쪽으로 잘라낼 픽셀 수"),
                IO.Int.Input("top", default=256, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="중앙 기준 위쪽으로 잘라낼 픽셀 수"),
                IO.Int.Input("bottom", default=256, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="중앙 기준 아래쪽으로 잘라낼 픽셀 수"),
                IO.Boolean.Input("show_preview", default=False, tooltip="프리뷰 표시 여부"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Mask.Output("mask_out", tooltip="중앙 기준으로 크롭된 마스크"),
            ],
        )

    @classmethod
    def execute(cls, mask, left, right, top, bottom, show_preview=False) -> IO.NodeOutput:
        
        mask = ensure_mask_tensor(mask)

        b, c, H, W = mask.shape
        cx, cy = W // 2, H // 2

        left = min(max(left, 0), cx-1)
        right = min(max(right, 0), cx-1)
        top = min(max(top, 0), cy-1)
        bottom = min(max(bottom, 0), cy-1)

        cropped = mask[:, :, cy - top: cy + bottom, cx - left: cx + right]

        mask = ensure_mask_output_shape(cropped)
        
        if show_preview:
            preview_mask = mask # [B, 1, H, W]
            return IO.NodeOutput(mask, ui=UI.PreviewMask(preview_mask))
        else:
            return IO.NodeOutput(mask, )

#----------------------------------------------------
# Mask Preview - original implement from
# https://github.com/cubiq/ComfyUI_essentials/blob/9d9f4bedfc9f0321c19faf71855e228c93bd0dc9/mask.py#L81
# upstream requested in https://github.com/Kosinkadink/rfcs/blob/main/rfcs/0000-corenodes.md#preview-nodes

class SafeFeatherMask(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SafeFeatherMask",
            display_name="마스크 페더링",
            category="커스텀마스크/컷팅",
            description="마스크 가장자리를 부드럽게 처리합니다.",
            inputs=[
                IO.Mask.Input("mask", tooltip="페더링을 적용할 원본 마스크"),
                IO.Int.Input("feather_size", default=0, min=0, max=50, step=1, tooltip="페더링 커널 크기 지정 (0=블러 없음)"),
                IO.Float.Input("feather_strength", default=0.0, min=0.0, max=5.0, step=0.1, tooltip="페더링 강도"),
                IO.Boolean.Input("show_preview", default=False, tooltip="프리뷰 표시 여부"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Mask.Output("mask_out", tooltip="페더링이 적용된 마스크"),
            ],
        )

    @classmethod
    def execute(cls, mask, feather_size, feather_strength, show_preview=False) -> IO.NodeOutput:

        mask = ensure_mask_tensor(mask)
       
        b, c, H, W = mask.shape
        feather_size    = min(max(feather_size, 0), 50)
        feather_strength = min(max(feather_strength, 0.0), 5.0)
        
        output = apply_feathering(mask, feather_size, feather_strength)

        mask = ensure_mask_output_shape(output)
        
        if show_preview:
            preview_mask = mask # [B, 1, H, W]
            return IO.NodeOutput(mask, ui=UI.PreviewMask(preview_mask))
        else:
            return IO.NodeOutput(mask, )


#----------------------------------------------------
# Mask Preview - original implement from
# https://github.com/cubiq/ComfyUI_essentials/blob/9d9f4bedfc9f0321c19faf71855e228c93bd0dc9/mask.py#L81
# upstream requested in https://github.com/Kosinkadink/rfcs/blob/main/rfcs/0000-corenodes.md#preview-nodes
class SafeMaskPreview(IO.ComfyNode):
    classname    = "SafeMaskPreview"
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SafeMaskPreview",
            display_name="마스크 미리보기",
            category="커스텀마스크/체커",
            description="GUI에서 마스크 출력을 직접 미리보기 합니다.",
            inputs=[IO.Mask.Input("mask")],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, mask, filename_prefix="ComfyUI") -> IO.NodeOutput:

        if mask is not None:
            if isinstance(mask, dict):
                if "latent_mask" in mask:
                    mask = mask["latent_mask"]
                elif "noise_mask" in mask:
                    mask = mask["noise_mask"]
                elif "mask" in mask:
                    mask = mask["mask"]
            if mask.dtype == torch.bool:
                mask = mask.float()
            denoise_mask = mask
        mask_tensor = normalize_mask_tensor(mask)

        if mask_tensor.max() > 1.0:
            mask_tensor = mask_tensor / 255.0
        else:
            mask_tensor = mask_tensor.clamp(0, 1)
     

        return IO.NodeOutput(ui=UI.PreviewMask(mask))
        
#----------------------------------------------------        

class SafeMaskSaveOnly(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SafeMaskSaveOnly",
            display_name="마스크 저장 (출력 없음)",
            category="커스텀마스크/체커",
            description="마스크를 'ComfyUI/output/mask' 폴더에 저장합니다. 출력은 없습니다.",
            inputs=[
                IO.Mask.Input("mask", tooltip="저장할 마스크 텐서"),
                IO.String.Input("filename_prefix", default="mask_%number", tooltip="저장할 파일 이름 접두사"),
            ],
            hidden=[IO.Hidden.prompt],
            is_output_node=True, 
            outputs=[],
        )

    @classmethod
    def execute(cls, mask, filename_prefix="mask_%number") -> IO.NodeOutput:

        mask = ensure_mask_tensor(mask) 
        mask_tensor = normalize_mask_tensor(mask)

        mask_img = mask_tensor.cpu().numpy().squeeze()
        
        output_dir = os.path.join(folder_paths.get_output_directory(), "mask")
        os.makedirs(output_dir, exist_ok=True)

        filename = resolve_filename(filename_prefix, output_dir) + ".png"
        filepath = os.path.join(output_dir, filename)

        imageio.imwrite(filepath, (mask_img * 255).astype("uint8"))
        return IO.NodeOutput()

#----------------------------------------------------
# Mask Preview - original implement from
# https://github.com/cubiq/ComfyUI_essentials/blob/9d9f4bedfc9f0321c19faf71855e228c93bd0dc9/mask.py#L81
# upstream requested in https://github.com/Kosinkadink/rfcs/blob/main/rfcs/0000-corenodes.md#preview-nodes

class SafeMaskSaveLink(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SafeMaskSaveLink",
            display_name="마스크 저장 (링크 출력)",
            category="커스텀마스크/체커",
            description="마스크를 'ComfyUI/output/mask' 폴더에 저장하고, 동시에 출력으로 연결합니다.",
            inputs=[
                IO.Mask.Input("mask", tooltip="저장할 마스크 텐서"),
                IO.String.Input("filename_prefix", default="mask_%number", tooltip="저장할 파일 이름 접두사"),
                IO.Boolean.Input("save_switch", default=False, tooltip="저장 여부 (off/on)"),
                IO.Boolean.Input("show_preview", default=False, tooltip="프리뷰 표시 여부"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Mask.Output("mask_out", tooltip="출력으로 연결된 마스크"),
            ],
        )

    @classmethod
    def execute(cls, mask, filename_prefix="mask_%number", save_switch=False, show_preview=False) -> IO.NodeOutput:
        
        mask_tensor = ensure_mask_tensor(mask)

        if save_switch:

            mask_img = normalize_mask_tensor(mask_tensor).cpu().numpy().squeeze()

            output_dir = os.path.join(folder_paths.get_output_directory(), "mask")
            os.makedirs(output_dir, exist_ok=True)

            filename = resolve_filename(filename_prefix, output_dir) + ".png"
            filepath = os.path.join(output_dir, filename)

            imageio.imwrite(filepath, (mask_img * 255).astype("uint8"))
            
        mask = ensure_mask_output_shape(mask_tensor)

        if show_preview:
            preview_mask = mask # [B, 1, H, W]
            return IO.NodeOutput(mask, ui=UI.PreviewMask(preview_mask))
        else:
            return IO.NodeOutput(mask, )

#----------------------------------------------------

class SafeMaskChecker(IO.ComfyNode):

    @staticmethod
    def mask_to_image(mask):
        arr = mask.cpu().numpy().astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]

        rgb = np.stack([arr]*3, axis=-1).astype(np.float32)
        return torch.from_numpy(rgb).unsqueeze(0)

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SafeMaskChecker",
            display_name="마스크 체커",
            category="커스텀마스크/체커",
            description="편집된 마스크를 원본 마스크 위에 오버레이하여 시각적으로 비교합니다.",
            inputs=[
                IO.Mask.Input("base_mask", tooltip="비교 기준이 되는 원본 마스크"),
                IO.Mask.Input("edit_mask", tooltip="편집된 마스크"),
                IO.Combo.Input("color", options=["red","green","blue", "yellow"], default="red", tooltip="시각화에 사용할 색상"),
                IO.Boolean.Input("preview_mode", default=False, tooltip="이미지 미리보기")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Image.Output("diff_image", tooltip="적용 영역이 강조된 결과 이미지"),
            ],
        )

    @classmethod
    def execute(cls, base_mask, edit_mask, color="red", preview_mode=False) -> IO.NodeOutput:
        
        base_mask = ensure_mask_tensor(base_mask)

        edit_mask = ensure_mask_tensor(edit_mask)

        if base_mask.shape[-2:] != edit_mask.shape[-2:]:
            base_mask_resized = F.interpolate(base_mask, size=edit_mask.shape[-2:], mode="nearest")
        else:
            base_mask_resized = base_mask

        # mask to grayscale image (1,3,H,W)
        base_gray = cls.mask_to_image(base_mask_resized.squeeze(0))
        base_gray = base_gray.clamp(0, 1)  #(1,H,W,3)
        base_gray = base_gray.permute(0, 3, 1, 2)  #(1,3,H,W)

        # edit mask to float(0/1)
        edit_mask_bin = (edit_mask.squeeze(1) > 0.3).float()

        # COLOR_MAP
        COLOR_MAP = {"red": (1.0, 0.0, 0.0),
                     "green": (0.0, 1.0, 0.0),
                     "blue": (0.0, 0.0, 1.0),
                     "yellow": (1.0, 1.0, 0.0),}
        color_rgb = torch.tensor(COLOR_MAP.get(color, COLOR_MAP["red"]),
                             dtype=torch.float32, device=base_gray.device).view(1, 3, 1, 1)

        # composite : background 0.5, overlay 0.5
        background = base_gray * 0.5
        overlay = (color_rgb * edit_mask_bin.unsqueeze(1)) * 0.5

        checked_tensor = (background + overlay).clamp(0,1)

        
        checked_tensor = checked_tensor.permute(0, 2, 3, 1)
        

        if preview_mode:
            preview_tensor = checked_tensor  # [1,H,W,3]

            return IO.NodeOutput(checked_tensor,ui=UI.PreviewImage(preview_tensor))
        else:
            return IO.NodeOutput(checked_tensor,)

#----------------------------------------------------

class SafeMaskCheckerDiff(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SafeMaskCheckerDiff",
            display_name="마스크 차이 체커",
            category="커스텀마스크/체커",
            description="원본 이미지를 입력하고, 편집된 마스크를 비교하여 원본 이미지 위에 차이 영역을 강조 표시합니다.",
            inputs=[
                IO.Image.Input("base_Image", tooltip="덮어씌워질 대상 이미지"),
                IO.Mask.Input("edit_mask", tooltip="편집된 마스크"),
                IO.Combo.Input("color", options=["red","green","blue", "yellow"], default="red", tooltip="차이 영역 강조 색상"),
                IO.Boolean.Input("preview_mode", default=False, tooltip="이미지 미리보기")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Image.Output("diff_image", tooltip="차이 영역이 강조된 결과 이미지"),
            ],
        )

    @classmethod
    def execute(cls, base_Image, edit_mask, color="red", preview_mode=False) -> IO.NodeOutput:
        
        base_img = ensure_image_tensor(base_Image)
        edit_mask = ensure_mask_tensor(edit_mask)
        base_mask = torch.zeros_like(edit_mask)

        base_img = base_img.float()
        if base_img.max() > 1.0:
            base_img = base_img / 255.0
        else:
            base_img = base_img.clamp(0, 1)

        base_mask = normalize_mask_tensor(base_mask)
        edit_mask = normalize_mask_tensor(edit_mask) 
        
        if base_mask.shape[-2:] != edit_mask.shape[-2:]:
            base_mask = base_mask.unsqueeze(1)
            base_mask = F.interpolate(base_mask, size=edit_mask.shape[-2:], mode="nearest")
            base_mask = base_mask.squeeze(1)

        alpha = 0.3
        COLOR_MAP = {
            "red":   (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue":  (0.0, 0.0, 1.0),
            "yellow": (1.0, 1.0, 0.0),
        }

        base_arr = np.squeeze(base_mask.cpu().numpy()).astype(np.float32)
        edit_arr = np.squeeze(edit_mask.cpu().numpy()).astype(np.float32)

        threshold = 0.3
        diff_mask_np = (np.abs(edit_arr - base_arr) > threshold).astype(np.float32)

        if diff_mask_np.sum() == 0:
            result_rgb = base_img

        else:
            diff_mask = torch.from_numpy(diff_mask_np)
            mask2d = (diff_mask > 0).numpy() 

            result_rgb = base_img.clone()

            color_rgb = torch.tensor(COLOR_MAP.get(color, COLOR_MAP["red"]), dtype=torch.float32)
            color_rgb = color_rgb.view(1, 3, 1, 1).expand_as(base_img)
            
            mask_idx = torch.from_numpy(mask2d).to(base_img.device).unsqueeze(0).unsqueeze(0)
            mask_idx = mask_idx.expand_as(base_img)

            result_rgb = base_img * (1 - mask_idx * alpha) + color_rgb * (mask_idx * alpha)
            checked_tensor = result_rgb.clamp(0, 1)
            
        if preview_mode:
            result_rgb = checked_tensor.permute(0, 2, 3, 1)
            return IO.NodeOutput(result_rgb,ui=UI.PreviewImage(result_rgb))
        else:
            result_rgb = checked_tensor.permute(0, 2, 3, 1)
            return IO.NodeOutput(result_rgb,)




#----------------------------------------------------
#Registration
#----------------------------------------------------
    
    
Safemask_NODE_CLASS_MAPPINGS = {
    "SafeMaskLoader": SafeMaskLoader,
    "SafeMaskToImage": SafeMaskToImage,
    "SafeImageToMask": SafeImageToMask,
    "SafeImageColorToMask": SafeImageColorToMask,
    "SafeSolidMask": SafeSolidMask,
    "SafeImageComposite": SafeImageComposite,
    "SafeMaskComposite": SafeMaskComposite,
    "SafeLatentComposite": SafeLatentComposite,
    "SafeGrowMask": SafeGrowMask,
    "SafeShrinkMask": SafeShrinkMask,
    "SafeTransformMask": SafeTransformMask,
    "SafeThresholdMask": SafeThresholdMask,
    "SafeInvertMask": SafeInvertMask,
    "SafeImagePadding": SafeImagePadding,
    "SafeMaskPadding": SafeMaskPadding,
    "SafeCropMask": SafeCropMask,
    "SafeCenterCropMask": SafeCenterCropMask,
    "SafeFeatherMask": SafeFeatherMask,
    "SafeMaskPreview": SafeMaskPreview,
    "SafeMaskSaveOnly": SafeMaskSaveOnly,
    "SafeMaskSaveLink": SafeMaskSaveLink,
    "SafeMaskChecker": SafeMaskChecker,
    "SafeMaskCheckerDiff": SafeMaskCheckerDiff,
}

Safemask_NODE_DISPLAY_NAME_MAPPINGS = {
    "SafeMaskLoader": "마스크 로더",
    "SafeMaskToImage": "안정화 마스크 이미지 변환",
    "SafeImageToMask": "안정화 이미지 마스크변환",
    "SafeImageColorToMask": "안정화 이미지색상마스크",
    "SafeSolidMask": "안정화 단색마스크",
    "SafeImageComposite": "안정화 이미지 합성",
    "SafeMaskComposite": "안정화 마스크합성",
    "SafeLatentComposite": "안정화 라텐트 합성",
    "SafeGrowMask": "안정화 마스크확장",
    "SafeShrinkMask": "안정화 마스크수축",
    "SafeTransformMask": "안정화 마스크변형",
    "SafeThresholdMask": "안정화 임계값 마스크",
    "SafeInvertMask": "안정화 마스크반전",
    "SafeImagePadding": "이미지 패딩",
    "SafeMaskPadding": "마스크 패딩",
    "SafeCropMask": "안정화 마스크자르기",
    "SafeCenterCropMask": "안정화 선택마스크자르기",
    "SafeFeatherMask": "안정화 마스크페더링",
    "SafeMaskPreview": "안정화 마스크미리보기",
    "SafeMaskSaveOnly": "안정화 마스크 저장",
    "SafeMaskSaveLink": "안정화 마스크 저장&링크",
    "SafeMaskChecker": "안정화 마스크 체커",
    "SafeMaskCheckerDiff": "안정화 마스크 강조 체커",
}
