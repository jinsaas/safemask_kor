
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

import comfy
import comfy.utils
import node_helpers

from comfy_api.latest import ComfyExtension, IO, UI

import nodes
from nodes import MAX_RESOLUTION
import logging
    
#--------------------------
# Header Utility Code(Baseif)
#--------------------------

def ensure_image_tensor(arr):
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(np.array(arr)).float()
    if arr.dim() == 2:  # (H,W) → 흑백
        arr = arr.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    elif arr.dim() == 3:
        if arr.shape[0] in (1,3):  # (H,W,C)
            arr = arr.permute(2,0,1).unsqueeze(0)  # (1,C,H,W)
        else:  # (B,H,W)
            arr = arr.unsqueeze(1)  # (B,1,H,W)
    elif arr.dim() == 4:  # (B,C,H,W)
        pass
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    return arr




def ensure_mask_tensor(t: torch.Tensor) -> torch.Tensor:
    """항상 (B,1,H,W) 형태로 보정"""
    if not isinstance(t, torch.Tensor):
        t = torch.from_numpy(np.array(t)).float()
    if t.dim() == 2:          # (H,W)
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 3:        # (B,H,W)
        t = t.unsqueeze(1)
    elif t.dim() == 4:        # (B,C,H,W)
        pass
    else:
        raise ValueError(f"Unsupported mask shape: {t.shape}")
    return t.float()
    
    
def composite(destination, source, mask, resize_source=False):
    # destination, source: (B,H,W,C)
    # mask: (B,1,H,W) / (B,H,W,1) / (B,H,W,3)

    source = source.to(destination.device)

    # 필요 시 소스를 대상 크기에 맞게 리사이즈
    if resize_source:
        source = torch.nn.functional.interpolate(
            source.permute(0,3,1,2),  # (B,H,W,C) → (B,C,H,W)
            size=(destination.shape[1], destination.shape[2]),
            mode="bilinear"
        ).permute(0,2,3,1)          # 다시 (B,H,W,C)

    source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])

    # --- 마스크 처리 ---
    mask = mask.to(destination.device, copy=True)

    if mask.ndim == 4 and mask.shape[1] == 1:
        # (B,1,H,W) → (B,H,W,1)
        mask = torch.nn.functional.interpolate(
            mask,
            size=(source.shape[1], source.shape[2]),
            mode="bilinear"
        )
        mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])
        mask = mask.permute(0,2,3,1)

    elif mask.ndim == 4 and mask.shape[-1] == 1:
        # (B,H,W,1)
        mask = torch.nn.functional.interpolate(
            mask.permute(0,3,1,2),
            size=(source.shape[1], source.shape[2]),
            mode="bilinear"
        ).permute(0,2,3,1)
        mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])

    elif mask.ndim == 4 and mask.shape[-1] == 3:
        # (B,H,W,3) → 그레이스케일 변환
        mask = mask.mean(dim=-1, keepdim=True)  # (B,H,W,1)
        mask = torch.nn.functional.interpolate(
            mask.permute(0,3,1,2),
            size=(source.shape[1], source.shape[2]),
            mode="bilinear"
        ).permute(0,2,3,1)
        mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])

    else:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")

    # 합성
    inverse_mask = torch.ones_like(mask) - mask
    output = mask * source + inverse_mask * destination  # (B,H,W,C)

    return output




# 색상 지정 + 반투명 alpha
COLOR_MAP = {
    "red":   (255, 0, 0, 128),
    "green": (0, 255, 0, 128),
    "blue":  (0, 0, 255, 128),
    "yellow":(255, 255, 0, 128),
}

def get_color(name: str):
    """문자열 키로 색상 RGBA 튜플을 반환"""
    return COLOR_MAP.get(name, COLOR_MAP["red"])
    
def dilate_tensor(mask_tensor, kernel_size=5, iterations=1):
    """
    PyTorch 기반 딜레이션 (max pooling 활용)
    mask_tensor: (B,1,H,W) float tensor
    """
    for _ in range(iterations):
        mask_tensor = F.max_pool2d(mask_tensor, kernel_size, stride=1, padding=kernel_size//2)
    return mask_tensor

def blur_tensor(mask_tensor, k=5):
    """
    PyTorch 기반 블러 (avg pooling으로 근사)
    mask_tensor: (B,1,H,W) float tensor
    """
    return F.avg_pool2d(mask_tensor, kernel_size=k, stride=1, padding=k//2)
    


#--------------------------
# Mask Node Code(Base)
#--------------------------

class SafeMaskToImage:
    classname = "SafeMaskToImage"
    node_id = "SafeMaskToImage"
    DISPLAY_NAME = "마스크 이미지 변환"
    DESCRIPTION = "마스크 텐서를 그레이스케일 이미지로 변환합니다."
    CATEGORY = "커스텀마스크/기본형"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "입력된 마스크 텐서를 그레이스케일 이미지로 변환합니다."}),
            }
        }

    @classmethod
    def execute(cls, mask):
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(np.array(mask)).float()

        if mask.ndim == 2:          # (H,W)
            result = mask.unsqueeze(-1)   # (H,W,1)
        elif mask.ndim == 3:        # (B,H,W)
            if mask.shape[0] == 1:
                result = mask.squeeze(0).unsqueeze(-1) # (H,W,1)
            else:
                raise ValueError("Batch dimension >1 not supported for SafeMaskToImage")
        elif mask.ndim == 4:        # (B,H,W,C)
            if mask.shape[0] == 1:
                result = mask.squeeze(0)            # (H,W,C)
            else:
                raise ValueError("Batch dimension >1 not supported for SafeMaskToImage")
        else:
            raise ValueError(f"Unsupported mask ndim={mask.ndim}")
                    
        # 값 범위 정규화
        if result.max() > 1.0:
            result = result / 255.0
        else:
            result = result.clamp(0, 1)
            
        if result.shape[-1] == 1:

            # (H,W,1) → (H,W,3)
            result = result.expand(-1, -1, 3)            # (H,W,3)
            
        result = result.unsqueeze(0)           # (1,H,W,3)


        return (result,)



class SafeImageToMask:
    classname = "SafeImageToMask"
    node_id = "SafeImageToMask"
    DISPLAY_NAME = "이미지 마스크 변환"
    DESCRIPTION = "이미지에서 특정 채널을 추출하여 마스크로 변환합니다."
    CATEGORY = "커스텀마스크/기본형"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "마스크를 추출할 대상 이미지"}),
                "channel": (["red","green","blue","alpha"], {"default":"red", "tooltip":"마스크 추출에 사용할 채널\n"
                                    "알파 채널 선택시 알파채널이 없으면 흰색(불투명)으로 나옵니다."}),
            }
        }

    @classmethod
    def execute(cls, image, channel):
        channels = ["red", "green", "blue", "alpha"]
        idx = channels.index(channel)
        # numpy → torch 변환
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()

        # ndim 보정
        if image.ndim == 2:
            # H, W → (1, H, W, 1)
            image = image.unsqueeze(0).unsqueeze(-1)
        elif image.ndim == 3:
            # H, W, C → (1, H, W, C)
            image = image.unsqueeze(0)
        elif image.ndim == 4:
            # B, H, W, C → 그대로 사용
            pass
        else:
            raise ValueError(f"Unsupported image ndim={image.ndim}")
                    
        # 값 범위 정규화
        if image.max() > 1.0:
            image = image / 255.0
        else:
            image = image.clamp(0, 1)
            
        if idx >= image.shape[-1]:
            if channel == "alpha":
                mask = torch.ones(
                    (image.shape[0], image.shape[1], image.shape[2]),
                    dtype=torch.float32,
                    device=image.device,
                )
            else:
                raise ValueError(f"Channel '{channel}' not available in image with {image.shape[-1]} channels")
        else:
            mask = image[:, :, :, idx]
        
        # 결과를 항상 ndim=3 (H, W, C)로 반환
        result = mask.squeeze(0)  # 배치 제거 → (H, W, 1)

        return (mask,)


class SafeImageColorToMask:
    classname = "SafeImageColorToMask"
    node_id = "SafeImageColorToMask"
    DISPLAY_NAME = "색상 마스크 변환"
    DESCRIPTION = "이미지에서 지정된 색상을 on/off로 선택하여 마스크 생성"
    CATEGORY = "커스텀마스크/기본형"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "마스크를 생성할 대상 이미지"}),
                "red": (["off","on"], {"default":"off", "tooltip": "적색 기반 마스크"}),
                "green": (["off","on"], {"default":"off", "tooltip": "녹색 기반 마스크"}),
                "blue": (["off","on"], {"default":"off", "tooltip": "청색 기반 마스크"}),
                "yellow": (["off","on"], {"default":"off", "tooltip": "노란색 기반 마스크"}),
                "magenta": (["off","on"], {"default":"off", "tooltip": "자홍색 기반 마스크"}),
                "cyan": (["off","on"], {"default":"off", "tooltip": "청록색 기반 마스크"}),
                "black": (["off","on"], {"default":"off", "tooltip": "흑색 기반 마스크"}),
            }
        }


    @classmethod
    def execute(cls, image,
                red="off", green="off", blue="off",
                yellow="off", magenta="off", cyan="off", black="off"):

        # numpy → torch 변환
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()

        # ndim 보정
        if image.ndim == 2:
            image = image.unsqueeze(0).unsqueeze(-1)
        elif image.ndim == 3:
            image = image.unsqueeze(0)
        elif image.ndim == 4:
            pass
        else:
            raise ValueError(f"Unsupported image ndim={image.ndim}")

        # (B,H,W,C) → 0~255 int
        temp = (torch.clamp(image, 0, 1.0) * 255.0).round().to(torch.int)
        R, G, B = temp[:, :, :, 0].float(), temp[:, :, :, 1].float(), temp[:, :, :, 2].float()

        mask = torch.zeros_like(R, dtype=torch.float)

        # 비율 기반 색상 판별
        if red == "on":
            cond_red = (R > G + 10) & (R > B + 10)
            mask = torch.max(mask, cond_red.float())
            
        if green == "on":
            cond_green = (G > R + 10) & (G > B + 10)
            mask = torch.max(mask, cond_green.float())
            
        if blue == "on":
            cond_blue = (B > R + 10) & (B > G + 10)
            mask = torch.max(mask, cond_blue.float())
            
        if yellow == "on":
            cond_yellow = (R > B + 10) & (G > B + 10)
            mask = torch.max(mask, cond_yellow.float())
            
        if magenta == "on":
            cond_magenta = (R > G + 10) & (B > G + 10)
            mask = torch.max(mask, cond_magenta.float())
            
        if cyan == "on":
            cond_cyan = (G > R + 10) & (B > R + 10)
            mask = torch.max(mask, cond_cyan.float())

        if black == "on":
            cond_black = (R < 20) & (G < 20) & (B < 20)
            mask = torch.max(mask, cond_black.float())

        # 최종 (B,H,W,)
        mask = mask.squeeze(0)

        return (mask,)






#--------------------------
# Mask Node Code(Edit)
#--------------------------

class SafeImageComposite:
    classname = "SafeImageComposite"
    node_id = "SafeImageComposite"
    DISPLAY_NAME = "이미지 합성"
    DESCRIPTION = "두 이미지를 마스크를 사용해 합성합니다. 부분 블렌딩에 적합합니다."
    CATEGORY = "커스텀마스크/에디터"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "destination": ("IMAGE", {"tooltip": "덮어씌워질 대상 이미지"}),
                "source": ("IMAGE", {"tooltip": "합성에 사용할 원본 이미지"}),
                "resize_source": (["off","on"], {"default":"off", "tooltip":"소스 이미지를 대상 크기에 맞게 자동 조정"}),
                "mask": ("MASK", {"tooltip": "합성 시 사용할 마스크"})
            },
            "optional": {
                "blend_mode": (
                    ["normal", "soft", "darken","lighten"],
                    {"default":"normal", "tooltip":"블렌딩 타입 선택"}),
                "spread_mode": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1, "tooltip":"스프레딩 마스크 세팅 선택"})
            }
        }

    @classmethod
    def execute(cls, destination, source, mask, resize_source="off", blend_mode="normal", spread_mode=0):
        # destination/source → (B,C,H,W)
        destination = ensure_image_tensor(destination)
        source      = ensure_image_tensor(source)

        # mask 보정 → (B,1,H,W)
        mask = ensure_mask_tensor(mask)

        # spread_mode 적용
        kernel_map = {i: (2*i-1) for i in range(1,11)}
        kernel_size = kernel_map.get(spread_mode, 0)
        if kernel_size > 0:
            kernel = torch.ones((1,1,kernel_size,kernel_size), dtype=mask.dtype, device=mask.device)
            pad = kernel_size // 2
            mask = torch.nn.functional.conv2d(mask, kernel, padding=pad)
            mask = mask / (kernel_size * kernel_size)

        mask = mask.permute(0,2,3,1)

        # RGB 확장 → (B,H,W,3)
        mask_rgb = mask.repeat(1,1,1,3)

        # 좌표 범위 맞추기
        H, W = destination.shape[2], destination.shape[3]  # BCHW
        source = source[:, :, :H, :W]
        mask   = mask_rgb[:, :, :H, :W]

        # 알파 채널 보정
        destination, source = node_helpers.image_alpha_fix(destination, source)

        # 합성
        dest_out = composite(destination, source, mask, resize_source == "on")

        # 블렌드 모드
        if blend_mode == "soft":
            output = destination * (1 - mask) + dest_out * mask
        elif blend_mode == "darken":
            blended = torch.min(destination, dest_out)
            output = destination * (1 - mask) + blended * mask
        elif blend_mode == "lighten":
            blended = torch.max(destination, dest_out)
            output = destination * (1 - mask) + blended * mask
        else:
            output = dest_out

        # 채널이 1개라면 RGB로 확장
        if output.shape[-1] == 1:
            output = output.repeat(1,1,1,3)

        return (output,)



    
class SafeMaskComposite:
    classname = "SafeMaskComposite"
    node_id = "SafeMaskComposite"
    DISPLAY_NAME = "마스크 합성"
    DESCRIPTION = "두 마스크를 결합합니다. add, multiply, subtract, 논리 연산(and, or, xor) 지원. spread 옵션으로 마스크 확장 가능."
    CATEGORY = "커스텀마스크/에디터"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "destination": ("MASK", {"tooltip": "결합에 사용할 첫 번째 마스크"}),
                "source": ("MASK", {"tooltip": "결합에 사용할 두 번째 마스크"}),
                "operation": (["multiply","add","subtract","and","or","xor"], {"default":"add", "tooltip":"마스크 결합 방식"}),
                "spread": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1, "tooltip":"마스크 확장 정도"})
            },
            "optional": {
                "blend_mode": (
                    ["normal", "soft", "screen","overlay","darken","lighten"],
                    {"default":"normal", "tooltip":"블렌딩 타입 선택"})
            }
        }

    @classmethod
    def execute(cls, destination, source, operation="add", blend_mode="normal", spread=0):
        destination = ensure_mask_tensor(destination)
        source      = ensure_mask_tensor(source)

        # 크기 맞추기
        H = min(destination.shape[-2], source.shape[-2])
        W = min(destination.shape[-1], source.shape[-1])
        destination = destination[:, :, :H, :W]
        source      = source[:, :, :H, :W]

        # spread 적용
        kernel_map = {i: (2*i-1) for i in range(1,11)}
        kernel_size = kernel_map.get(spread, 0)
        if kernel_size > 0:
            kernel = torch.ones((1,1,kernel_size,kernel_size), dtype=destination.dtype, device=destination.device)
            pad = kernel_size // 2
            destination = torch.nn.functional.conv2d(destination, kernel, padding=pad)
            destination = destination / (kernel_size * kernel_size)

        # 결합 방식
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

        # 블렌드 모드
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
        return (output,)



    

#--------------------------
# Mask Node Code(Transform)
#--------------------------

class SafeInvertMask:
    classname = "SafeInvertMask"
    DISPLAY_NAME = "마스크 반전"
    DESCRIPTION = "마스크 값을 1.0에서 빼서 전경/배경을 반전합니다."
    CATEGORY = "커스텀마스크/변형"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "반전할 원본 마스크 텐서"}),
            }
        }

    @classmethod
    def execute(cls, mask):
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)

        if mask.ndim == 2:          # (H,W) → (1,H,W)
            mask = mask.unsqueeze(0)
        elif mask.ndim == 3:        # (B,H,W)
            pass
        elif mask.ndim == 4:        # (B,C,H,W)
            mask = mask[:,0,:,:]

        elif mask.ndim != 3:        # (B,H,W)만 허용
            raise ValueError(f"Unsupported mask shape: {mask.shape}")

        out = 1.0 - mask
        out = torch.clamp(out, 0.0, 1.0)

        return (out,)

class SafeGrowMask:
    classname    = "SafeGrowMask"
    node_id      = "SafeGrowMask"
    DISPLAY_NAME = "마스크 확장"
    DESCRIPTION  = "마스크 영역 내부의 픽셀을 확장합니다."
    CATEGORY     = "커스텀마스크/변형"
    RETURN_TYPES = ("MASK",)
    FUNCTION     = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "확장할 대상 마스크"}),
                "expand": ("INT", {"default": 0, "min": 0, "max": 25, "step": 1, "tooltip": "확장할 픽셀 수"}),
                "tapered_corners": ("BOOLEAN", {"default": True, "tooltip": "모서리를 부드럽게 처리할지 여부"}),
            }
        }

    @classmethod
    def execute(cls, mask, expand, tapered_corners):
        # 입력 타입 정리
        if not isinstance(mask, torch.Tensor):
            mask_tensor = torch.from_numpy(np.array(mask))
        else:
            mask_tensor = mask

        # 차원 정규화
        if mask_tensor.ndim == 2:  # (H,W)
            mask = mask_tensor
        elif mask_tensor.ndim == 3:
            if mask_tensor.shape[0] == 1:  # (1,H,W)
                mask = mask_tensor.squeeze(0)
            elif mask_tensor.shape[-1] == 1:  # (H,W,1)
                mask = mask_tensor.squeeze(-1)
            else:  # (B,H,W) → 첫 배치만
                mask = mask_tensor[0]
        elif mask_tensor.ndim == 4:  # (B,C,H,W)
            mask = mask_tensor[0, 0, :, :]
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c],
                           [1, 1, 1],
                           [c, 1, c]])

        mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        out = []
        for m in mask:
            arr = m.cpu().numpy()
            for _ in range(abs(expand)):
                if expand < 0:
                    arr = scipy.ndimage.grey_erosion(arr, footprint=kernel, iterations=expand)
                else:
                    arr = scipy.ndimage.grey_dilation(arr, footprint=kernel, iterations=expand)
            out.append(torch.from_numpy(arr))

        return (torch.stack(out, dim=0),)
    
    
    
class SafeShrinkMask:
    classname    = "SafeShrinkMask"
    node_id      = "SafeShrinkMask"
    DISPLAY_NAME = "마스크 수축"
    DESCRIPTION  = "마스크 내부의 픽셀을 수축합니다."
    CATEGORY     = "커스텀마스크/변형"
    RETURN_TYPES = ("MASK",)
    FUNCTION     = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "축소할 원본 마스크"}),
                "shrink": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1, "tooltip": "축소할 픽셀 수"}),
                "tapered_corners": ("BOOLEAN", {"default": True, "tooltip": "모서리를 부드럽게 처리할지 여부"}),
            }
        }

    @classmethod
    def execute(cls, mask, shrink, tapered_corners):
        # 입력 타입 정리
        if not isinstance(mask, torch.Tensor):
            mask_tensor = torch.from_numpy(np.array(mask))
        else:
            mask_tensor = mask

        # 차원 정규화
        if mask_tensor.ndim == 2:  # (H,W)
            mask = mask_tensor
        elif mask_tensor.ndim == 3:
            if mask_tensor.shape[0] == 1:  # (1,H,W)
                mask = mask_tensor.squeeze(0)
            elif mask_tensor.shape[-1] == 1:  # (H,W,1)
                mask = mask_tensor.squeeze(-1)
            else:  # (B,H,W) → 첫 배치만
                mask = mask_tensor[0]
        elif mask_tensor.ndim == 4:  # (B,C,H,W)
            mask = mask_tensor[0, 0, :, :]
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        if shrink <= 0:
            return (mask,)

        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c],
                           [1, 1, 1],
                           [c, 1, c]])

        mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        out = []
        for m in mask:
            arr = m.cpu().numpy()
            shrink_val = min(shrink, min(m.shape[-2], m.shape[-1]))
            for _ in range(shrink_val):
                arr = scipy.ndimage.grey_erosion(arr, footprint=kernel, iterations=shrink_val)
            out.append(torch.from_numpy(arr))

        return (torch.stack(out, dim=0),)
        

class SafeTransformMask:
    classname    = "SafeTransformMask"
    node_id      = "SafeTransformMask"
    DISPLAY_NAME = "마스크 변형"
    DESCRIPTION  = "마스크 영역의 크기를 Lanczos 보간법으로 변형합니다."
    CATEGORY     = "커스텀마스크/변형"
    RETURN_TYPES = ("MASK",)
    FUNCTION     = "execute"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "변형할 원본 마스크"}),
                "width": ("INT", {"default": 512, "min": 16, "max": 2048, "step": 1}),
                "height": ("INT", {"default": 512, "min": 16, "max": 2048, "step": 1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
            }
        }

    @classmethod
    def execute(cls, mask, width, height, tapered_corners):
        if not isinstance(mask, torch.Tensor):
            mask_tensor = torch.from_numpy(np.array(mask))
        else:
            mask_tensor = mask

        # 차원 정규화
        if mask_tensor.ndim == 2:
            mask = mask_tensor
        elif mask_tensor.ndim == 3:
            if mask_tensor.shape[0] == 1:
                mask = mask_tensor.squeeze(0)
            elif mask_tensor.shape[-1] == 1:
                mask = mask_tensor.squeeze(-1)
            else:
                mask = mask_tensor[0]
        elif mask_tensor.ndim == 4:
            mask = mask_tensor[0, 0, :, :]
        else:
            raise ValueError(f"Unexpected mask shape: {mask_tensor.shape}")

        mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        out = []
        for m in mask:
            arr = m.cpu().numpy().astype(np.float32)
            resized = cv2.resize(arr, (width, height), interpolation=cv2.INTER_LANCZOS4)

            if tapered_corners:
                kernel = np.array([[0,1,0],
                                   [1,1,1],
                                   [0,1,0]])
                resized = scipy.ndimage.grey_dilation(resized, footprint=kernel)

            out.append(torch.from_numpy(resized))

        return (torch.stack(out, dim=0),)






class SafeThresholdMask:
    classname    = "SafeThresholdMask"
    node_id      = "SafeThresholdMask"
    DISPLAY_NAME = "마스크 임계값 처리"
    DESCRIPTION  = "마스크를 임계값 기준으로 이진화합니다."
    CATEGORY     = "커스텀마스크/변형"
    RETURN_TYPES = ("MASK",)
    FUNCTION     = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "이진화할 원본 마스크"}),
                "value": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "마스크를 이진화할 기준 값"}),
            }
        }

    @classmethod
    def execute(cls, mask, value):
        # 입력 타입 정리
        if not isinstance(mask, torch.Tensor):
            mask_tensor = torch.from_numpy(np.array(mask))
        else:
            mask_tensor = mask

        # 차원 정규화
        if mask_tensor.ndim == 2:  # (H,W)
            mask = mask_tensor
        elif mask_tensor.ndim == 3:
            if mask_tensor.shape[0] == 1:  # (1,H,W)
                mask = mask_tensor.squeeze(0)
            elif mask_tensor.shape[-1] == 1:  # (H,W,1)
                mask = mask_tensor.squeeze(-1)
            else:  # (B,H,W) → 첫 배치만
                mask = mask_tensor[0]
        elif mask_tensor.ndim == 4:  # (B,C,H,W)
            mask = mask_tensor[0, 0, :, :]
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        mask = (mask > value).float()
        
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return (mask,)


class SafeSolidMask:
    classname    = "SafeSolidMask"
    node_id      = "SafeSolidMask"
    DISPLAY_NAME = "단색 마스크 생성"
    DESCRIPTION  = "지정된 크기와 값으로 단색 마스크를 생성합니다."
    CATEGORY     = "커스텀마스크/변형"
    RETURN_TYPES = ("MASK",)
    FUNCTION     = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "마스크 픽셀 값 (0=검정, 1=흰색)"}),
                "width": ("INT", {"default": 512, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "생성할 마스크의 너비"}),
                "height": ("INT", {"default": 512, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "생성할 마스크의 높이"}),
            }
        }

    @classmethod
    def execute(cls, value, width, height):

        out = torch.full((1, height, width), value, dtype=torch.float32, device="cpu")
        return (out,)
    

class safeImagePadding:
    classname    = "safeImagePadding"
    node_id      = "safeImagePadding"
    DISPLAY_NAME = "이미지 패딩"
    DESCRIPTION  = "이미지를 상하좌우로 확장하고 지정된 색상으로 채웁니다."
    CATEGORY     = "커스텀마스크/변형"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION     = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "패딩을 적용할 원본 이미지"}),
                "pad_top": ("INT", {"default": 0, "min": 0, "max": 2048, "tooltip": "위쪽으로 확장할 픽셀 수"}),
                "pad_bottom": ("INT", {"default": 0, "min": 0, "max": 2048, "tooltip": "아래쪽으로 확장할 픽셀 수"}),
                "pad_left": ("INT", {"default": 0, "min": 0, "max": 2048, "tooltip": "왼쪽으로 확장할 픽셀 수"}),
                "pad_right": ("INT", {"default": 0, "min": 0, "max": 2048, "tooltip": "오른쪽으로 확장할 픽셀 수"}),
                "padding_color": ("COMBO", {"options": ["black", "white"], "default": "black", "tooltip": "패딩 영역에 채울 색상"}),
                "feather_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "패딩 경계의 부드러움 정도"}),
            }
        }

    @classmethod
    def execute(cls, image, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, padding_color="black", feather_strength=0.0):
        # numpy → torch 변환
        if not isinstance(image, torch.Tensor):
            image_tensor = torch.from_numpy(np.array(image))
        else:
            image_tensor = image
            
         # 차원 정규화
        if image_tensor.ndim == 3:  # (H,W,C)
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.ndim == 4:  # (B,H,W,C)
            pass
        else:
            raise ValueError(f"Unexpected image shape: {image_tensor.shape}")

            
        # 값 범위 정규화
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
        else:
            image_tensor = image_tensor.clamp(0, 1)

        b, h, w, c = image.shape
        fill_val = 0.0 if padding_color == "black" else 1.0
        canvas = torch.full(
            (b, h + pad_top + pad_bottom, w + pad_left + pad_right, c),
            fill_val,
            dtype=image.dtype,
            device=image.device
        )
        canvas[:, pad_top:pad_top + h, pad_left:pad_left + w, :] = image

        if feather_strength > 0.0:
            # TODO: feathering logic
            pass

        return (canvas,)


class safeMaskPadding:
    classname    = "safeMaskPadding"
    node_id      = "safeMaskPadding"
    DISPLAY_NAME = "마스크 패딩"
    DESCRIPTION  = "마스크를 상하좌우로 확장하고 지정된 값으로 채웁니다."
    CATEGORY     = "커스텀마스크/변형"
    RETURN_TYPES = ("MASK",)
    FUNCTION     = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "패딩을 적용할 원본 마스크"}),
                "pad_top": ("INT", {"default": 0, "min": 0, "max": 2048, "tooltip": "위쪽으로 확장할 픽셀 수"}),
                "pad_bottom": ("INT", {"default": 0, "min": 0, "max": 2048, "tooltip": "아래쪽으로 확장할 픽셀 수"}),
                "pad_left": ("INT", {"default": 0, "min": 0, "max": 2048, "tooltip": "왼쪽으로 확장할 픽셀 수"}),
                "pad_right": ("INT", {"default": 0, "min": 0, "max": 2048, "tooltip": "오른쪽으로 확장할 픽셀 수"}),
                "padding_color": ("COMBO", {"options": ["black", "white"], "default": "black", "tooltip": "패딩 영역에 채울 색상"}),
                "feather_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "패딩 경계의 부드러움 정도"}),
            }
        }

    @classmethod
    def execute(cls, mask, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, padding_color="black", feather_strength=0.0):
        # 입력 타입 정리
        if not isinstance(mask, torch.Tensor):
            mask_tensor = torch.from_numpy(np.array(mask))
        else:
            mask_tensor = mask

        # 차원 정규화
        if mask_tensor.ndim == 2:  # (H,W)
            mask = mask_tensor.unsqueeze(0)  # (1,H,W)
        elif mask_tensor.ndim == 3:
            if mask_tensor.shape[0] == 1:  # (1,H,W)
                mask = mask_tensor
            elif mask_tensor.shape[-1] == 1:  # (H,W,1)
                mask = mask_tensor.squeeze(-1).unsqueeze(0)  # (1,H,W)
            else:  # (B,H,W) → 첫 배치만
                mask = mask_tensor

        elif mask_tensor.ndim == 4:  # (B,C,H,W)
            mask = mask_tensor[:, 0, :, :]  # (1,H,W)
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")
            
        if mask is None:
            raise ValueError("MaskPadding node requires a mask input.")
            
        # 값 범위 정규화
        if mask.max() > 1.0:
            mask = mask / 255.0
        else:
            mask = mask.clamp(0, 1)
                   
        b, h, w = mask.shape
        fill_val = 0.0 if padding_color == "black" else 1.0
        canvas = torch.full(
            (b, h + pad_top + pad_bottom, w + pad_left + pad_right),
            fill_val,
            dtype=mask.dtype,
            device=mask.device
        )
        canvas[:, pad_top:pad_top + h, pad_left:pad_left + w] = mask

        if feather_strength > 0.0:
            # TODO: feathering logic
            pass

        return (canvas,)



#--------------------------
# Mask Node Code(Cuttings)
#--------------------------

class SafeCropMask:
    classname    = "SafeCropMask"
    node_id      = "SafeCropMask"
    DISPLAY_NAME = "마스크 크롭"
    DESCRIPTION  = "마스크를 지정된 위치와 크기로 잘라냅니다."
    CATEGORY     = "커스텀마스크/컷팅"
    RETURN_TYPES = ("MASK",)
    FUNCTION     = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "잘라낼 원본 마스크 텐서"}),
                "x": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "잘라낼 영역의 X 좌표"}),
                "y": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "잘라낼 영역의 Y 좌표"}),
                "width": ("INT", {"default": 512, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "잘라낼 영역의 너비"}),
                "height": ("INT", {"default": 512, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "잘라낼 영역의 높이"}),
            }
        }

    @classmethod
    def execute(cls, mask, x, y, width, height):
        # 입력 타입 정리
        if not isinstance(mask, torch.Tensor):
            mask_tensor = torch.from_numpy(np.array(mask))
        else:
            mask_tensor = mask

        # 차원 정규화
        if mask_tensor.ndim == 2:  # (H,W)
            mask = mask_tensor
        elif mask_tensor.ndim == 3:
            if mask_tensor.shape[0] == 1:  # (1,H,W)
                mask = mask_tensor.squeeze(0)
            elif mask_tensor.shape[-1] == 1:  # (H,W,1)
                mask = mask_tensor.squeeze(-1)
            else:  # (B,H,W) → 첫 배치만
                mask = mask_tensor[0]
        elif mask_tensor.ndim == 4:  # (B,C,H,W)
            mask = mask_tensor[0, 0, :, :]
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")
            
        # 값 범위 정규화
        if mask.max() > 1.0:
            mask = mask / 255.0
        else:
            mask = mask.clamp(0, 1)
                   
        mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        H, W = mask.shape[-2:]

        # 안전 제한 적용
        x = min(x, W - 1)
        y = min(y, H - 1)
        width = min(width, W - x)
        height = min(height, H - y)

        out = mask[:, y:y + height, x:x + width]
        return (out,)
        

class SafeCenterCropMask:
    classname    = "SafeCenterCropMask"
    node_id      = "SafeCenterCropMask"
    DISPLAY_NAME = "마스크 중앙 크롭"
    DESCRIPTION  = "마스크를 중앙 기준으로 잘라냅니다."
    CATEGORY     = "커스텀마스크/컷팅"
    RETURN_TYPES = ("MASK",)
    FUNCTION     = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "잘라낼 원본 마스크 텐서"}),
                "left": ("INT", {"default": 256, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "중앙 기준 왼쪽으로 잘라낼 픽셀 수"}),
                "right": ("INT", {"default": 256, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "중앙 기준 오른쪽으로 잘라낼 픽셀 수"}),
                "top": ("INT", {"default": 256, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "중앙 기준 위쪽으로 잘라낼 픽셀 수"}),
                "bottom": ("INT", {"default": 256, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "중앙 기준 아래쪽으로 잘라낼 픽셀 수"}),
            }
        }

    @classmethod
    def execute(cls, mask, left, right, top, bottom):
        # 입력 타입 정리
        if not isinstance(mask, torch.Tensor):
            mask_tensor = torch.from_numpy(np.array(mask))
        else:
            mask_tensor = mask

        # 차원 정규화
        if mask_tensor.ndim == 2:  # (H,W)
            mask = mask_tensor
        elif mask_tensor.ndim == 3:
            if mask_tensor.shape[0] == 1:  # (1,H,W)
                mask = mask_tensor.squeeze(0)
            elif mask_tensor.shape[-1] == 1:  # (H,W,1)
                mask = mask_tensor.squeeze(-1)
            else:  # (B,H,W) → 첫 배치만
                mask = mask_tensor[0]
        elif mask_tensor.ndim == 4:  # (B,C,H,W)
            mask = mask_tensor[0, 0, :, :]
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")
            
        # 값 범위 정규화
        if mask.max() > 1.0:
            mask = mask / 255.0
        else:
            mask = mask.clamp(0, 1)
                   
        mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        H, W = mask.shape[-2:]
        cx, cy = W // 2, H // 2

        # 입력값을 절반 기준으로 제한
        left = min(left, cx)
        right = min(right, cx)
        top = min(top, cy)
        bottom = min(bottom, cy)

        out = mask[:, cy - top: cy + bottom, cx - left: cx + right]
        return (out,)


class SafeFeatherMask:
    classname    = "SafeFeatherMask"
    node_id      = "SafeFeatherMask"
    DISPLAY_NAME = "마스크 페더링"
    DESCRIPTION  = "마스크 가장자리를 부드럽게 처리합니다."
    CATEGORY     = "커스텀마스크/컷팅"
    RETURN_TYPES = ("MASK",)
    FUNCTION     = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "페더링을 적용할 원본 마스크"}),
                "left": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "좌측 가장자리를 흐리게 할 픽셀 수"}),
                "top": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "상단 가장자리를 흐리게 할 픽셀 수"}),
                "right": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "우측 가장자리를 흐리게 할 픽셀 수"}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "하단 가장자리를 흐리게 할 픽셀 수"}),
            }
        }

    @classmethod
    def execute(cls, mask, left, top, right, bottom):
        # 입력 타입 정리
        if not isinstance(mask, torch.Tensor):
            mask_tensor = torch.from_numpy(np.array(mask))
        else:
            mask_tensor = mask

        # 차원 정규화
        if mask_tensor.ndim == 2:  # (H,W)
            pass
        elif mask_tensor.ndim == 3:
            if mask_tensor.shape[0] == 1:  # (1,H,W)
                mask_tensor = mask_tensor.squeeze(0)
            elif mask_tensor.shape[-1] == 1:  # (H,W,1)
                mask_tensor = mask_tensor.squeeze(-1)
            else:  # (B,H,W) → 첫 배치만
                mask_tensor = mask_tensor[0]
        elif mask_tensor.ndim == 4:  # (B,C,H,W)
            mask_tensor = mask_tensor[0, 0, :, :]
        else:
            raise ValueError(f"Unexpected mask shape: {mask_tensor.shape}")

            
        # 값 범위 정규화
        if mask_tensor.max() > 1.0:
            mask_tensor = mask_tensor / 255.0
        else:
            mask_tensor = mask_tensor.clamp(0, 1)
       

        # 출력 준비
        output = mask_tensor.clone().unsqueeze(0)  # (1,H,W) 형태로 맞춤

        H, W = output.shape[-2:]
        left   = min(left, W)
        right  = min(right, W)
        top    = min(top, H)
        bottom = min(bottom, H)

        # 좌우 페더링
        for x in range(left):
            feather_rate = (x + 1.0) / left
            output[:, :, x] *= feather_rate
        for x in range(right):
            feather_rate = (x + 1.0) / right
            output[:, :, -(x+1)] *= feather_rate

        # 상하 페더링
        for y in range(top):
            feather_rate = (y + 1.0) / top
            output[:, y, :] *= feather_rate
        for y in range(bottom):
            feather_rate = (y + 1.0) / bottom
            output[:, -(y+1), :] *= feather_rate

        return (output,)



#--------------------------
# Mask Node Code(Check)
#--------------------------
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
        # 입력 타입 정리
        if not isinstance(mask, torch.Tensor):
            mask_tensor = torch.from_numpy(np.array(mask))
        else:
            mask_tensor = mask

        # 차원 정규화
        if mask_tensor.ndim == 2:  # (H,W)
            pass
        elif mask_tensor.ndim == 3:
            if mask_tensor.shape[0] == 1:  # (1,H,W)
                mask_tensor = mask_tensor.squeeze(0)
            elif mask_tensor.shape[-1] == 1:  # (H,W,1)
                mask_tensor = mask_tensor.squeeze(-1)
            else:  # (B,H,W) → 첫 배치만
                mask_tensor = mask_tensor[0]
        elif mask_tensor.ndim == 4:  # (B,C,H,W)
            mask_tensor = mask_tensor[0, 0, :, :]
        else:
            raise ValueError(f"Unexpected mask shape: {mask_tensor.shape}")
            
        # 값 범위 정규화
        if mask_tensor.max() > 1.0:
            mask_tensor = mask_tensor / 255.0
        else:
            mask_tensor = mask_tensor.clamp(0, 1)
       


        # 변환된 텐서를 UI에 넘김

        return IO.NodeOutput(ui=UI.PreviewMask(mask))





        

class SafeMaskSaveOnly:
    classname    = "SafeMaskSaveOnly"
    node_id      = "SafeMaskSaveOnly"
    DISPLAY_NAME = "마스크 저장 (출력 없음)"
    DESCRIPTION  = "마스크를 'ComfyUI output/mask' 폴더에 저장합니다. 출력은 없습니다."
    CATEGORY     = "커스텀마스크/체커"
    RETURN_TYPES = ()
    FUNCTION     = "execute"
    OUTPUT_NODE  = True  # 출력 노드임을 표시

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "저장할 마스크 텐서"}),
                "filename_prefix": ("STRING", {"default": "mask", "tooltip": "저장할 파일 이름 접두사"}),
            }
        }

    @classmethod
    def execute(cls, mask, filename_prefix="mask"):
        # 입력 타입 정리
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(np.array(mask))

        # 차원 보정
        if mask.ndim == 2:          # (H,W)
            mask_img = mask.cpu().numpy()
        elif mask.ndim == 3:        # (B,H,W)
            mask_img = mask[0].cpu().numpy()
        elif mask.ndim == 4:        # (B,C,H,W) → 첫 배치/첫 채널
            mask_img = mask[0,0].cpu().numpy()
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        mask_img = mask_img.squeeze()

        # 저장 경로 준비
        output_dir = os.path.join(folder_paths.get_output_directory(), "mask")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{filename_prefix}.png")

        # 저장 (0~1 범위 → 0~255 변환)
        imageio.imwrite(filepath, (mask_img * 255).astype("uint8"))

        return {}




class SafeMaskSaveLink:
    classname    = "SafeMaskSaveLink"
    node_id      = "SafeMaskSaveLink"
    DISPLAY_NAME = "마스크 저장 (링크 출력)"
    DESCRIPTION  = "마스크를 'ComfyUI output/mask' 폴더에 저장하고, 동시에 출력으로 연결합니다."
    CATEGORY     = "커스텀마스크/체커"
    RETURN_TYPES = ("MASK",)
    FUNCTION     = "execute"
    OUTPUT_NODE  = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "저장할 마스크 텐서"}),
                "filename_prefix": ("STRING", {"default": "mask", "tooltip": "저장할 파일 이름 접두사"}),
            }
        }

    @classmethod
    def execute(cls, mask, filename_prefix="mask"):
        # 입력 타입 정리
        if isinstance(mask, torch.Tensor):
            mask_tensor = mask
        else:
            mask_tensor = torch.from_numpy(np.array(mask))

        # 차원 보정
        if mask.ndim == 2:          # (H,W)
            mask_img = mask.cpu().numpy()
        elif mask.ndim == 3:        # (B,H,W)
            mask_img = mask[0].cpu().numpy()
        elif mask.ndim == 4:        # (B,C,H,W) → 첫 배치/첫 채널
            mask_img = mask[0,0].cpu().numpy()
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        mask_img = mask[0].cpu().numpy().squeeze()
        output_dir = os.path.join(folder_paths.get_output_directory(), "mask")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{filename_prefix}.png")
        imageio.imwrite(filepath, (mask_img * 255).astype("uint8"))
        return (mask,)


class SafeMaskChecker:
    classname    = "SafeMaskChecker"
    node_id      = "SafeMaskChecker"
    DISPLAY_NAME = "마스크 체커"
    DESCRIPTION  = "편집된 마스크를 원본 마스크 위에 오버레이하여 시각적으로 비교합니다."
    CATEGORY     = "커스텀마스크/체커"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION     = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_mask": ("MASK", {"tooltip": "비교 기준이 되는 원본 마스크"}),
                "edit_mask": ("MASK", {"tooltip": "편집된 마스크"}),
                "color": ("COMBO", {"options": ["red", "green", "blue"], "default": "red", "tooltip": "시각화에 사용할 색상"}),
            }
        }


    # 마스크 → RGB 이미지 (H,W,3)
    @staticmethod
    def mask_to_image(mask):
        arr = mask.cpu().numpy().astype(np.float32)
        if arr.ndim == 3:   # (B,H,W) → 첫 배치만 사용
            arr = arr[0]
        rgb = np.stack([arr]*3, axis=-1).astype(np.float32)  # (H,W,3)
        return torch.from_numpy(rgb)

    @classmethod
    def execute(cls, base_mask, edit_mask, color="red"):
        if not isinstance(base_mask, torch.Tensor):
            base_mask = torch.from_numpy(np.array(base_mask))
        if not isinstance(edit_mask, torch.Tensor):
            edit_mask = torch.from_numpy(np.array(edit_mask))

        # 차원 맞추기
        if base_mask.ndim == 2:
            base_mask = base_mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        elif base_mask.ndim == 3:
            base_mask = base_mask.unsqueeze(1)               # (B,1,H,W)

        if edit_mask.ndim == 2:
            edit_mask = edit_mask.unsqueeze(0).unsqueeze(0)
        elif edit_mask.ndim == 3:
            edit_mask = edit_mask.unsqueeze(1)

        # 값 범위 정규화
        base_mask = base_mask.float()
        edit_mask = edit_mask.float()

        if base_mask.max() > 1.0:
            base_mask = base_mask / 255.0
        else:
            base_mask = base_mask.clamp(0, 1)

        if edit_mask.max() > 1.0:
            edit_mask = edit_mask / 255.0
        else:
            edit_mask = edit_mask.clamp(0, 1)

        # 크기 비교 후 리사이즈
        if base_mask.shape[-2:] != edit_mask.shape[-2:]:
            base_mask = F.interpolate(base_mask, size=edit_mask.shape[-2:], mode="nearest")
            edit_mask = F.interpolate(edit_mask, size=base_mask.shape[-2:], mode="nearest")

        base_mask = base_mask.squeeze(1)  # (B,H,W)
        edit_mask = edit_mask.squeeze(1)

        # 배경: 원본 마스크 → RGB
        base_img = cls.mask_to_image(base_mask[0]).to(edit_mask.device)  # (H,W,3)
        
        # diff 마스크 계산
        base_arr = np.squeeze(base_mask.cpu().numpy()).astype(np.float32)
        edit_arr = np.squeeze(edit_mask.cpu().numpy()).astype(np.float32)
        diff_mask = (np.abs(edit_arr - base_arr) > 0.5).astype("float32")  # (H,W)

        # torch 기반 overlay 생성 (곱셈 방식)
        COLOR_MAP = {
            "red":   (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue":  (0.0, 0.0, 1.0),
        }
        color_rgb = torch.tensor(COLOR_MAP.get(color, COLOR_MAP["red"]), dtype=torch.float32, device=base_img.device)  # (3,)

        mask_t = torch.from_numpy(diff_mask).to(base_img.device).unsqueeze(-1).float()  # (H,W,1)
        overlay_tensor = mask_t * color_rgb  # (H,W,3)

        # 합성 (알파 블렌딩)
        alpha = 1
        checked_tensor = base_img * (1 - mask_t * alpha) + overlay_tensor * (mask_t * alpha)
        checked_tensor = checked_tensor.clamp(0,1)  # (H,W,3)

        
        checked_tensor = checked_tensor.unsqueeze(0)        # (1,H,W,3)

        # 최종 반환
        return (checked_tensor,)










class SafeMaskCheckerDiff:
    classname    = "SafeMaskCheckerDiff"
    node_id      = "SafeMaskCheckerDiff"
    DISPLAY_NAME = "마스크 차이 체커"
    DESCRIPTION  = "원본 이미지를 입력하고, 원본 마스크와 편집된 마스크를 비교하여 원본 이미지 위에 차이 영역을 강조 표시합니다."
    CATEGORY     = "커스텀마스크/체커"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION     = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_Image": ("IMAGE", {"tooltip": "덮어씌워질 대상 이미지"}),
                "base_mask": ("MASK", {"tooltip": "비교 기준이 되는 원본 마스크"}),
                "edit_mask": ("MASK", {"tooltip": "편집된 마스크"}),
                "color": ("COMBO", {"options": ["red", "green", "blue"], "default": "red", "tooltip": "차이 영역 강조 색상"}),
            }
        }

    @classmethod
    def execute(cls, base_Image, base_mask, edit_mask, color="red"):
        # 입력 타입 정리
        if not isinstance(base_Image, torch.Tensor):
            base_Image = torch.from_numpy(np.array(base_Image))

        if not isinstance(base_mask, torch.Tensor):
            base_mask = torch.from_numpy(np.array(base_mask))
        else:
            base_mask = base_mask

        if not isinstance(edit_mask, torch.Tensor):
            edit_mask = torch.from_numpy(np.array(edit_mask))
        else:
            edit_mask = edit_mask
            
        if base_Image.ndim == 3:   # (H,W,C)
            base_img = base_Image.unsqueeze(0).float()   # (1,H,W,C)
        elif base_Image.ndim == 4: # (B,H,W,C)
            base_img = base_Image.float()
        else:
            raise ValueError(f"Unexpected image shape: {base_Image.shape}")

        if base_mask.ndim == 2:
            base_mask = base_mask.unsqueeze(0)  # (1,H,W)
            
        if edit_mask.ndim == 2:
            edit_mask = edit_mask.unsqueeze(0)  # (1,H,W)
            
        # 값 범위 정규화
        if base_mask.max() > 1.0:
            base_mask = base_mask / 255.0
        else:
            base_mask = base_mask.clamp(0, 1)
        
        if edit_mask.max() > 1.0:
            edit_mask = edit_mask / 255.0
        else:
            edit_mask = edit_mask.clamp(0, 1)
        
        # 크기 비교 후 리사이즈 (edit_mask 기준)
        if base_mask.shape[-2:] != edit_mask.shape[-2:]:
            # (B,H,W) → (B,1,H,W) 로 바꿔서 interpolate
            base_mask = base_mask.unsqueeze(1)
            base_mask = F.interpolate(base_mask, size=edit_mask.shape[-2:], mode="nearest")
            base_mask = base_mask.squeeze(1)
            
        # 항상 0~1 범위로 normalize
        if base_img.max() > 1.0:
            base_img = base_img / 255.0
        else:
            base_img = base_img.clamp(0, 1)

        alpha = 0.3
        COLOR_MAP = {
            "red":   (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue":  (0.0, 0.0, 1.0),
        }

        # 마스크 배열 준비
        base_arr = np.squeeze(base_mask.cpu().numpy()).astype(np.float32)
        edit_arr = np.squeeze(edit_mask.cpu().numpy()).astype(np.float32)

        # 차이 마스크 계산
        threshold = 0.3
        diff_mask_np = (np.abs(edit_arr - base_arr) > threshold).astype(np.float32)

        if diff_mask_np.sum() == 0:
            result_rgb = base_img

        else:
            diff_mask = torch.from_numpy(diff_mask_np)
            mask2d = (diff_mask > 0).numpy()   # (H,W)

            # 결과 이미지 복사
            result_rgb = base_img.clone()

            # 오버레이 색상 벡터
            color_rgb = torch.tensor(COLOR_MAP.get(color, COLOR_MAP["red"]), dtype=torch.float32)
            color_rgb = color_rgb.view(1, 1, 3).expand_as(base_img)  # (H,W,3)
            mask_idx = torch.from_numpy(mask2d).unsqueeze(-1).expand_as(base_img)  # (H,W,3)

            # 마스크 영역만 알파 블렌딩 적용
            result_rgb = base_img * (1 - mask_idx * alpha) + color_rgb * (mask_idx * alpha)
            result_rgb = result_rgb.clamp(0, 1)

        return (result_rgb,)




#--------------------------
#Registration
#--------------------------
    
    
NODE_CLASS_MAPPINGS = {
    "SafeMaskToImage": SafeMaskToImage,
    "SafeImageToMask": SafeImageToMask,
    "SafeImageColorToMask": SafeImageColorToMask,
    "SafeSolidMask": SafeSolidMask,
    "SafeImageComposite": SafeImageComposite,
    "SafeMaskComposite": SafeMaskComposite,
    "SafeGrowMask": SafeGrowMask,
    "SafeShrinkMask": SafeShrinkMask,
    "SafeTransformMask": SafeTransformMask,
    "SafeThresholdMask": SafeThresholdMask,
    "SafeInvertMask": SafeInvertMask,
    "safeImagePadding": safeImagePadding,
    "safeMaskPadding": safeMaskPadding,
    "SafeCropMask": SafeCropMask,
    "SafeCenterCropMask": SafeCenterCropMask,
    "SafeFeatherMask": SafeFeatherMask,
    "SafeMaskPreview": SafeMaskPreview,
    "SafeMaskSaveOnly": SafeMaskSaveOnly,
    "SafeMaskSaveLink": SafeMaskSaveLink,
    "SafeMaskChecker": SafeMaskChecker,
    "SafeMaskCheckerDiff": SafeMaskCheckerDiff,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SafeMaskToImage": "안정화 마스크 이미지 변환",
    "SafeImageToMask": "안정화 이미지 마스크변환",
    "SafeImageColorToMask": "안정화 이미지색상마스크",
    "SafeSolidMask": "안정화 단색마스크",
    "SafeImageComposite": "안정화 이미지 합성",
    "SafeMaskComposite": "안정화 마스크합성",
    "SafeGrowMask": "안정화 마스크확장",
    "SafeShrinkMask": "안정화 마스크수축",
    "SafeTransformMask": "안정화 마스크변형",
    "SafeThresholdMask": "안정화 임계값 마스크",
    "SafeInvertMask": "안정화 마스크반전",
    "safeImagePadding": "이미지 패딩",
    "safeMaskPadding": "마스크 패딩",
    "SafeCropMask": "안정화 마스크자르기",
    "SafeCenterCropMask": "안정화 선택마스크자르기",
    "SafeFeatherMask": "안정화 마스크페더링",
    "SafeMaskPreview": "안정화 마스크미리보기",
    "SafeMaskSaveOnly": "안정화 마스크 저장",
    "SafeMaskSaveLink": "안정화 마스크 저장&링크",
    "SafeMaskChecker": "안정화 마스크 체커",
    "SafeMaskCheckerDiff": "안정화 마스크 강조 체커",
}