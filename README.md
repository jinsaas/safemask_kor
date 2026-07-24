# Modified from ComfyUI (AGPL-3.0)
# Original project: https://github.com/comfyanonymous/ComfyUI

# comfy-Custom-mask-nodes

This project provides enhanced mask-processing utilities for ComfyUI, including
conversion, preview, and compositing functions.

Based on ComfyUI’s original mask node implementation, this version improves
stability by reducing alpha-channel inconsistencies and offering additional
mask manipulation features.

## Attention
This project runs correctly on ComfyUI version 0.092 or higher.  
It also works with the latest ComfyUI versions.

## Version History

### V3.0.0
- Fixed bugs where GrowMask, CropMask, and SelectCrop would break when exceeding maximum values.
- Crop/SelectCrop no longer cut off when values exceed the original image size.
- GrowMask no longer processes pixel shrink; capped at 25× expansion.  
  (If you want larger expansion, chain multiple GrowMask nodes.)
- For shrink behavior, use ShrinkMask instead. This was split for stability.
- Added TransformMask node to resize masks in pixel units.
- Masks saved with numbered suffixes.

### V4.0.0
- Unified schema logic across all nodes.
- Fixed all feathering mask logic errors.
- Re‑coded cyclic logic in checker nodes.
- Removed original mask input from difference checker node.
- Added internal output functionality to nodes.


### V4.0.1
-Updates in this version

-Further improvements to the Mask Load node

-You can now open files using the upload widget as well. (Loading masks from the 'Mask' folder is recommended to minimize the risk of conflicts.)

### V4.0.2
-Updates in this version

-composite node error Fixed

-Add 4 node

## Extended Features

- **Safe MaskToImage Node**: Converts a mask into an image (e.g., grayscale visualization).
- **Safe ImageToMask Node**: Extracts a mask from an image based on a specific channel.
- **Safe ImageColorToMask Node**: Generates a mask from a specified color range in the image.
- **Safe SolidMask Node**: Creates a mask filled with uniform values.
- **Safe InvertMask Node**: Inverts an existing mask.

- **Safe ImageCompositeMasked Node**: Composites images using a mask.
- **Safe MaskComposite Node**: Composites multiple masks together.

- **Safe CropMask Node**: Crops a mask to a specified region.
- **Safe Select CropMask Node**: Stabilized node. Allows direct pixel cropping from left, right, top, and bottom.
- **ImagePadding Node**: Expands the input image by adding padding of specified pixels on all sides.
- **MaskPadding Node**: Expands the input mask by adding padding of specified pixels on all sides.
- **Safe FeatherMask Node**: Applies feathering (soft edges) to a mask.

- **Safe GrowMask Node**: Expands the mask pixel region outward (maximum 25 pixels per operation; chain nodes for further expansion).
- **Safe ShrinkMask Node**: Shrinks the mask pixel region inward.
- **Safe TransformMask Node**: Resizes the mask region by arbitrary pixel values.
- **Safe ThresholdMask Node**: Converts grayscale masks into binary masks using a threshold.

- **Safe MaskPreview Node**: Previews mask output directly in the GUI.
- **Safe MaskSaveOnly Node**: Saves mask data as a grayscale image.
- **Safe MaskSaveLink Node**: Saves mask data as a grayscale image and outputs the mask connection line.
- **SafeMaskChecker Node**: Visualizes the edit mask with a chosen color for safe inspection.
- **SafeMaskDiffChecker Node**: Highlights differences between the original and edited masks, showing only the changed regions.


## Getting Started

### Installation

1. **Download ZIP**:  
   Download this repository as a `.zip` file.

2. **Place in custom_nodes**:  
   Extract the zip and put the folder inside the `custom_nodes` directory of your ComfyUI installation.

3. **Restart ComfyUI**:  
   Restart ComfyUI to load the new custom nodes.


### Usage

### basics

### Safe Maskloader Node
Includes built‑in preview functionality. Does not resize. Mask editor is not supported (risk of errors).
- **Inputs**:
  - `invert`: Invert mask
  - `show_preview`: Preview option (if enabled, preview mode applies)
- **Outputs**:
  - `mask`: Mask tensor

#### Safe MaskToImage Node
Stabilization node. If an error occurs, returns and re‑outputs to induce stabilization.
- **Inputs**:
  - `mask`: Mask tensor to convert
- **Outputs**:
  - `image`: Grayscale image representing the mask

#### Safe ImageToMask Node
Stabilization node. If an error occurs, returns and re‑outputs to induce stabilization.
- **Inputs**:
  - `image`: Source image
  - `channel`: Channel to extract (`red`, `green`, `blue`, `alpha`)
- **Outputs**:
  - `mask`: Generated mask

#### Safe ImageColorToMask Node
Stabilization node. Generates a mask from a specified color range, useful for color‑based selection.
- **Inputs**:
  - `image`: Source image
  - `color`: Target color or range
  - `mode`: Processing mode (color switch combination / hex code)
  - `hex_color`: Output based on hex code
  - `show_preview`: Preview option
- **Outputs**:
  - `mask`: Mask generated from color selection

### Editor
#!!Note: The `resize_source` option may cause pixel fragmentation and boundary distortion when compositing latent/mask. 
#It is recommended to use inputs of the same size as the target.

#### Safe ImageCompositeMasked Node
Stabilization node. Composites two images using a mask, suitable for partial blending.
- **Inputs**:
  - `destination`: Target image
  - `source`: Overlay image
  - `resize source`: Resize source and mask
  - `alpha`: Blend priority
  - `mask`: Optional mask
- **Outputs**:
  - `image`: Composite image

#### Safe MaskComposite Node
Stabilization node. If an error occurs, returns and re‑outputs to induce stabilization.
- **Inputs**:
  - `mask_a`: First mask
  - `mask_b`: Second mask
  - `blend_mode`: Combination mode (add, multiply, etc.)
- **Outputs**:
  - `mask`: Composite mask

#### Safe LatentCompositeMasked Node
Stabilization node. Composites two latents using a mask, suitable for partial blending.
- **Inputs**:
  - `destination`: Target latent
  - `source`: Overlay latent
  - `resize source`: Resize source and mask
  - `alpha`: Blend priority
  - `mask`: Optional mask
  - `edge_attention`: Whether to generate edge mask
  - `brightness_strength`: Brightness adjustment
  - `contrast_strength`: Contrast adjustment
  - `sharpen_strength`: Sharpen adjustment
- **Outputs**:
  - `image`: Composite image

### Transform

#### Safe SolidMask Node
Stabilization node. If an error occurs, returns and re‑outputs to induce stabilization.
- **Inputs**:
  - `width`: Mask width
  - `height`: Mask height
  - `value`: Fill value (0–1)
  - `show_preview`: Preview option
- **Outputs**:
  - `mask`: Solid mask

#### Safe InvertMask Node
Stabilization node. Includes built‑in preview.
- **Inputs**:
  - `mask`: Mask tensor to invert
- **Outputs**:
  - `mask`: Inverted mask

#### Safe GrowMask Node
Expands mask pixels outward. Maximum expansion capped at 25 pixels. Chain nodes for further expansion. Includes built‑in preview.
- **Inputs**:
  - `mask`: Mask tensor
  - `amount`: Pixels to expand
  - `show_preview`: Preview option
- **Outputs**:
  - `mask`: Expanded mask

#### Safe ShrinkMask Node
Shrinks mask pixels inward proportionally. Includes built‑in preview.
- **Inputs**:
  - `mask`: Mask tensor
  - `amount`: Pixels to shrink
  - `show_preview`: Preview option
- **Outputs**:
  - `mask`: Shrunk mask

#### Safe TransformMask Node
Resizes mask dimensions in pixel units. Includes built‑in preview.
- **Inputs**:
  - `mask`: Mask tensor
  - `amount`: Horizontal pixels
  - `amount`: Vertical pixels
  - `show_preview`: Preview option
- **Outputs**:
  - `mask`: Transformed mask

#### Safe ThresholdMask Node
Stabilization node. Converts grayscale mask into binary mask using threshold. Includes built‑in preview.
- **Inputs**:
  - `mask`: Mask tensor
  - `threshold`: Threshold (0–1)
  - `show_preview`: Preview option
- **Outputs**:
  - `mask`: Binary mask

#### ImagePadding Node
Expands image canvas by specified pixels. Supports black/white fill and feathering. Stabilization included.
- **Inputs**:
  - `image`: Input image (batch dimension corrected if needed)
  - `pad_top`, `pad_bottom`, `pad_left`, `pad_right`: Padding size
  - `padding_color`: Fill color (black/white)
  - `feather_strength`: Feathering strength
- **Outputs**:
  - `image`: Padded image

#### MaskPadding Node
Expands mask canvas by specified pixels. Supports black/white fill and feathering. Stabilization included.
- **Inputs**:
  - `mask`: Input mask (batch dimension corrected if needed)
  - `pad_top`, `pad_bottom`, `pad_left`, `pad_right`: Padding size
  - `padding_color`: Fill color (black/white)
  - `feather_strength`: Feathering strength
- **Outputs**:
  - `mask`: Padded mask

### Cutting

#### Safe CropMask Node
Stabilization node. Default crop origin is top‑left. Includes built‑in preview.
- **Inputs**:
  - `mask`: Mask tensor
  - `x`, `y`: Crop position
  - `width`, `height`: Crop size
  - `show_preview`: Preview option
- **Outputs**:
  - `mask`: Cropped mask

#### Safe Select CropMask Node
Stabilization node. Crops mask relative to center by specifying pixels for left, right, top, bottom. Includes built‑in preview.
- **Inputs**:
  - `mask`: Mask tensor
  - `L`, `R`, `T`, `B`: Crop values
  - `show_preview`: Preview option
- **Outputs**:
  - `mask`: Cropped mask

#### Safe FeatherMask Node
Mask feathering node. Reduces conflicts. Includes built‑in preview.
- **Inputs**:
  - `mask`: Mask tensor
  - `radius`: Feather radius
  - `show_preview`: Preview option
- **Outputs**:
  - `mask`: Feathered mask

### Checker

#### Safe MaskPreview Node
Stabilization node. If values remain, returns and re‑outputs to induce stabilization. Bug fixes applied.
- **Inputs**:
  - `mask`: Mask tensor
- **Outputs**:
  - `image`: Mask preview image

#### Safe MaskSaveOnly Node
Stabilization node. Saves mask to ComfyUI output folder. If error occurs, returns and re‑outputs.
- **Inputs**:
  - `mask`: Mask tensor

#### Safe MaskSaveLink Node
Stabilization node. Saves mask to ComfyUI output folder and outputs mask connection. Includes built‑in preview.
- **Inputs**:
  - `mask`: Mask tensor
  - `show_preview`: Preview option
- **Outputs**:
  - `mask`: Mask tensor

#### SafeMaskChecker Node
Stabilization node. Overlays edit mask on base mask for pre‑inspection. Includes built‑in preview.
- **Inputs**:
  - `base_mask`: Base mask (converted from original image)
  - `edit_mask`: Edit mask (drawn via mask editor)
  - `color`: Display color (red/green/blue)
  - `show_preview`: Preview option
- **Outputs**:
  - `preview`: Edit mask visualized with color overlay
Processing: Converts edit mask to grayscale, overlays with chosen color, outputs preview image.

#### SafeMaskDiffChecker Node
Stabilization node. Compares original image and edit mask, highlights differences. Includes built‑in preview.
- **Inputs**:
  - `base_image`: Original image
  - `edit_mask`: Edit mask
  - `color`: Highlight color (red/green/blue/yellow)
  - `show_preview`: Preview option
- **Outputs**:
  - `preview`: Image highlighting changed regions
Processing: Compares original and edited masks, extracts difference regions, renders full mask semi‑transparent, emphasizes differences with chosen color.

