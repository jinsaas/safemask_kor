import os, logging, nodes

version = "3.0.2"
logging.info(f"### Loading: ComfyUI_SafeMask_Pack (v{version})")

# 노드 정의 임포트
from .nodes_safemask import (
    SafeMaskToImage,
    SafeImageToMask,
    SafeImageColorToMask,
    SafeSolidMask,
    SafeInvertMask,
    SafeImageComposite,
    SafeMaskComposite,
    SafeGrowMask,
    SafeShrinkMask,
    SafeTransformMask,
    SafeThresholdMask,
    safeImagePadding,
    safeMaskPadding,
    SafeCropMask,
    SafeCenterCropMask,
    SafeFeatherMask,
    SafeMaskPreview,
    SafeMaskSaveOnly,
    SafeMaskSaveLink,
    SafeMaskChecker,
    SafeMaskCheckerDiff,
)

# 노드 등록 매핑
NODE_CLASS_MAPPINGS = {
    "SafeMaskToImage": SafeMaskToImage,
    "SafeImageToMask": SafeImageToMask,
    "SafeImageColorToMask": SafeImageColorToMask,
    "SafeSolidMask": SafeSolidMask,
    "SafeInvertMask": SafeInvertMask,

    "SafeImageComposite": SafeImageComposite,
    "SafeMaskComposite": SafeMaskComposite,

    "SafeGrowMask": SafeGrowMask,
    "SafeShrinkMask": SafeShrinkMask,
    "SafeTransformMask": SafeTransformMask,
    "SafeThresholdMask": SafeThresholdMask,
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
    "SafeInvertMask": "안정화 마스크반전",

    "SafeImageComposite": "안정화 이미지 합성",
    "SafeMaskComposite": "안정화 마스크합성",

    "SafeGrowMask": "안정화 마스크확장",
    "SafeShrinkMask": "안정화 마스크수축",
    "SafeTransformMask": "안정화 마스크변형",
    "SafeThresholdMask": "안정화 임계값 마스크",
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
