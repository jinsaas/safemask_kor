# 이 프로젝트는 ComfyUI(AGPL-3.0)를 기반으로 수정된 버전입니다.
# 원본 프로젝트: https://github.com/comfyanonymous/ComfyUI

# comfy-Custom-mask-nodes

이 프로젝트는 ComfyUI용 확장 마스크 처리 노드들을 제공합니다.  
마스크 변환, 미리보기, 합성 등 다양한 마스크 유틸리티 기능을 포함하고 있습니다.

ComfyUI 기본 마스크 노드를 기반으로 하여,  
마스크 연산 과정에서 발생하던 알파 채널 오류를 줄이고  
보다 안정적인 마스크 처리 기능을 제공하도록 개선되었습니다.


-Attention-
# 이 프로젝트는 ComfyUI 0.092이상에서 정상작동하는 버전입니다. ComfyUI 최신버전에서도 기동합니다.
# V 3.0.0 정보
그로우마스크와 크롭마스크, 셀렉트 크롭의 경우 최대값 오버하면 깨지던 버그가 있었습니다.
최대값을 받은 이미지에 비례해 처리하는 식으로 바꿨습니다. 
-크롭/셀렉트 크롭은 원본보다 높은 값 넣으면 안 잘립니다.
-그로우마스크는 픽셀축소부분을 처리 안하게 바꿨고, 원본의 25배 이상은 처리하지 않습니다. 굳이 더 크게 하시려면 추가로 붙이세요.
-그로우마스크의 픽셀축소부분 동작이 필요하시면 쉬링크마스크 쓰시면 됩니다. 안정성을 위해 쪼갰습니다.
-마스크를 크기를 조정해 맞춰서 쓰고싶은 분들 있을거같아서 축을 픽셀단위로 늘리는 변형마스크 놔뒀습니다.
-마스크 저장시 번호가 붙습니다

# V4.0.0 정보
이번 버전업 내용
모든 스키마 로직 통일
페더링 마스크 로직 오류 전체 수정
체커 노드 순환로직 오류 리코딩
차이 체크 노드 원본마스크 투입 삭제
노드 내부 출력기능 추가

## 주요 기능 (Features)

- **Safe MaskToImage Node**: 마스크를 이미지로 변환 (예: 그레이스케일 시각화).
- **Safe ImageToMask Node**: 이미지에서 특정 채널을 기반으로 마스크 추출.
- **Safe ImageColorToMask Node**: 이미지의 특정 색상 범위로부터 마스크 생성.
- **Safe SolidMask Node**: 균일한 값으로 채워진 마스크 생성.
- **Safe InvertMask Node**: 기존 마스크 반전.

- **Safe ImageCompositeMasked Node**: 마스크를 사용하여 이미지 합성
- **Safe MaskComposite Node**: 여러 마스크를 합성.

- **Safe CropMask Node**: 지정된 영역으로 마스크 잘라내기.
- **Safe Select CropMask Node**:안정화노드. 상하좌우로 직접 자를 픽셀을 지정.
- **ImagePadding : 입력 이미지를 상하좌우로 지정한 픽셀만큼 패딩하여 확장
- **MaskPadding : 입력 마스크를 상하좌우로 지정한 픽셀만큼 패딩하여 확장
- **Safe FeatherMask Node**: 마스크에 페더링(부드러운 가장자리) 적용.

- **Safe GrowMask Node**: 마스크 영역의 픽셀크기를 바깥쪽으로 확장(확장픽셀기준25[연속으로 쓰면 추가확장가능]).
- **Safe ShrinkMask Node**: 마스크 영역의 픽셀 크기를 안쪽으로 축소.
- **Safe TransformMask Node**: 마스크 영역자체의 크기를 임의 픽셀로 재조정.
- **Safe ThresholdMask Node**: 그레이스케일 마스크를 임계값으로 이진 마스크로 변환.

- **Safe MaskPreview Node**: GUI에서 마스크 출력 미리보기.
- **Safe MaskSaveOnly**: 마스크 데이터를 흑백이미지로 저장.
- **Safe MaskSaveLink**: 마스크 데이터를 흑백이미지로 저장 후 마스크 연결선 출력.
- **SafeMaskChecker : 편집 마스크를 지정 색상으로 시각화하여 안전하게 확인
- **SafeMaskDiffChecker : 원본과 편집 마스크의 차이를 강조하여 변경된 영역만 시각화


## 설치 방법 (Getting Started)

### Installation

1. **ZIP 다운로드**  
   이 저장소를 `.zip` 파일로 다운로드합니다.

2. **custom_nodes 폴더에 배치**  
   압축을 해제한 후, ComfyUI 설치 경로의 `custom_nodes` 디렉토리에 폴더를 넣습니다.

3. **ComfyUI 재시작**  
   ComfyUI를 재시작하면 새로운 커스텀 노드가 로드됩니다.

## 사용법 (Usage)
 
### 기본형

##Safe Maskloader Node : 자체미리보기 기능 포함, 리사이즈는 하지 않음, 마스크에디터는 지원하지 않음(에러 유발 위험)
- **Inputs**:
  - `invert`: 마스크 반전 기능
  - `show_preview`: 미리보기 기능(켰을 경우 프리뷰 모드도 적용)
- **Outputs**:  
  - `mask`: 마스크 이미지 텐서


#### Safe MaskToImage Node : 안정화 노드. 오류발생시 리턴하여 안정화를 유도해 재출력함.
- **Inputs**:  
  - `mask`: 변환할 마스크 텐서  
- **Outputs**:  
  - `image`: 마스크를 표현하는 그레이스케일 이미지  

#### Safe ImageToMask Node : 안정화 노드. 오류발생시 리턴하여 안정화를 유도해 재출력함.
- **Inputs**:  
  - `image`: 원본 이미지  
  - `channel`: 추출할 채널 (`red`, `green`, `blue`, `alpha`)  
- **Outputs**:  
  - `mask`: 생성된 마스크  

#### Safe ImageColorToMask Node : 안정화 노드. 특정 색상 범위를 지정해 마스크를 생성할 수 있어, 색상 기반 선택 작업에 유용.
- **Inputs**:  
  - `image`: 원본 이미지  
  - `color`: 지정된 색상 또는 범위  
  - `mode`: 처리 방식. 스위치에 따라 컬러스위치 조합/헥스코드 처리로 진행  
  - `hex_color`: 지정된 헥스코드 기반 출력 
  - `show_preview`: 미리보기 기능(켰을 경우 프리뷰 모드도 적용)
- **Outputs**:  
  - `mask`: 색상 선택으로 생성된 마스크  

### 에디터
주의: resize_source 옵션은 라텐트/마스크 합성 시 픽셀 파편화 및 경계 왜곡을 유발할 수 있습니다.
가능하다면 대상과 동일한 크기의 입력을 사용하는 것을 권장합니다.

#### Safe ImageCompositeMasked Node : 안정화 노드. 마스크를 이용해 두 이미지를 합성할 수 있어, 부분 합성이나 블렌딩 작업에 적합.
- **Inputs**:  
  - `destination`: 합성 대상 이미지  
  - `source`: 덮어씌울 이미지
  - `resize source`: 소스 및 마스크의 크기 조정
  - `alpha`: 블렌딩 시 영역의 이미지 우선도 조정
  - `mask`: 블렌딩을 제어하는 선택적 마스크  
- **Outputs**:  
  - `image`: 합성된 이미지  
  
#### Safe MaskComposite Node : 안정화 노드. 오류발생시 리턴하여 안정화를 유도해 재출력함.
 - **Inputs**:  
  - `mask_a`: 첫 번째 마스크  
  - `mask_b`: 두 번째 마스크  
  - `blend_mode`: 결합 방식 (add, multiply 등)  
- **Outputs**:  
  - `mask`: 합성된 마스크  

#### Safe LatentCompositeMasked Node : 안정화 노드. 마스크를 이용해 두 라텐트를 합성할 수 있어, 부분 합성이나 블렌딩 작업에 적합.
- **Inputs**:  
  - `destination`: 합성 대상 라텐트
  - `source`: 덮어씌울 라텐트
  - `resize source`: 소스 및 마스크의 크기 조정
  - `alpha`: 블렌딩 시 영역의 라텐트 우선도 조정
  - `mask`: 블렌딩을 제어하는 선택적 마스크 
  - `edge_attention`: 라텐트의 엣지 마스크를 만들지의 확인
  - `brightness_strength`: 추가 밝기 조정 지시  
  - `contrast_strength`: 추가 대비 조정 지시  
  - `sharpen_strength`: 추가 샤픈 조정 지시  

- **Outputs**:  
  - `image`: 합성된 이미지  

### 변형

#### Safe SolidMask Node : 안정화 노드. 오류발생시 리턴하여 안정화를 유도해 재출력함.

- **Inputs**:  
  - `width`: 마스크 너비  
  - `height`: 마스크 높이  
  - `value`: 채울 값 (0–1) 
  - `show_preview`: 미리보기 기능(켰을 경우 프리뷰 모드도 적용) 
- **Outputs**:  
  - `mask`: 단색 마스크  


#### Safe InvertMask Node : 안정화 노드. 오류발생시 리턴하여 안정화를 유도해 재출력함.
##오류수정 자체미리보기 추가

- **Inputs**:  
  - `mask`: 반전할 마스크 텐서  
- **Outputs**:  
  - `mask`: 반전된 마스크  
  
#### Safe GrowMask Node : 마스크의 픽셀 크기를 늘림. 최대제한 픽셀을 25로 변경. 재확장시 추가연결하면 됩니다.
##오류수정 자체미리보기 추가

- **Inputs**:  
  - `mask`: 마스크 텐서  
  - `amount`: 확장할 픽셀 수
  - `show_preview`: 미리보기 기능(켰을 경우 프리뷰 모드도 적용)
- **Outputs**:  
  - `mask`: 확장된 마스크  

#### Safe ShrinkMask Node : 마스크의 픽셀을 입력치에 비례해서 픽셀을 축소합니다.
##오류수정 자체미리보기 추가

- **Inputs**:  
  - `mask`: 마스크 텐서  
  - `amount`: 축소할 픽셀 수
  - `show_preview`: 미리보기 기능(켰을 경우 프리뷰 모드도 적용)
- **Outputs**:  
  - `mask`: 확장된 마스크  

#### Safe TransformMask Node : 마스크를 입력치에 맞춰 크기를 조정. 픽셀단위로 조정합니다.
##오류수정 자체미리보기 추가

- **Inputs**:  
  - `mask`: 마스크 텐서  
  - `amount`: 조정할 가로 픽셀 수  
  - `amount`: 조정할 세로 픽셀 수
  - `show_preview`: 미리보기 기능(켰을 경우 프리뷰 모드도 적용)
- **Outputs**:  
  - `mask`: 변형된 마스크  

#### Safe ThresholdMask Node : 안정화노드. 임계값에 맞춰 마스크를 이진화합니다.
##오류수정, 자체미리보기 추가


- **Inputs**:  
  - `mask`: 마스크 텐서  
  - `threshold`: 임계값 (0–1)
  - `show_preview`: 미리보기 기능(켰을 경우 프리뷰 모드도 적용)
- **Outputs**:  
  - `mask`: 이진화된 마스크   
  
  
#### ImagePadding Node : 이미지의 캔버스 크기를 지정한 방향으로 늘려서 칠함. 흑백 지원, 경계 페더링 지원. 이미지가 원본이어서 에러위험이 있어어도 리턴 후 보정처리하여 적용함.
- **Inputs**: 
  - `image` : 입력 이미지(배치데이터 체킹 후 아닌경우 배치차원 추가하여 리턴 후 재연결)
  - `pad_top`, `pad_bottom`, `pad_left`, `pad_right` : 각 방향별 패딩 크기
  - `padding_color` : 패딩 영역 색상(black/white)
  - `feather_strength` : 패딩 경계 페더링 강도
- **Outputs**: 
  - `image : 패딩이 적용된 이미지
처리방식
입력 이미지를 배치 차원 보정 후 지정한 픽셀만큼 상하좌우에 패딩을 추가하고 선택한 색상으로 채움

#### MaskPadding Node : 마스크의 캔버스 크기를 지정한 방향으로 늘려서 칠함. 흑백 지원, 경계 페더링 지원. 마스크데이터가 원본이어서 에러위험이 있어도 리턴 후 보정처리하여 적용함.
- **Inputs**: 
  - `mask` : 입력 마스크(배치데이터 체킹 후 아닌경우 배치차원 추가하여 리턴 후 재연결)
  - `pad_top, pad_bottom, pad_left, pad_right : 각 방향별 패딩 크기
  - `padding_color` : 패딩 영역 색상(black/white)
  - `feather_strength` : 패딩 경계 페더링 강도
- **Outputs**: 
  - `mask` : 패딩이 적용된 마스크
처리방식
입력 마스크를 배치 차원 보정 후 지정한 픽셀만큼 상하좌우에 패딩을 추가하고 선택한 색상으로 채움


  
### 컷팅
  

#### Safe CropMask Node : 안정화는 했지만 잘라내는 위치는 추가조정 필요. 기본값은 왼쪽 위입니다.
##오류수정, 자체미리보기 추가

- **Inputs**:  
  - `mask`: 마스크 텐서  
  - `x`, `y`: 잘라낼 위치  
  - `width`, `height`: 잘라낼 크기
  - `show_preview`: 미리보기 기능(켰을 경우 프리뷰 모드도 적용)
- **Outputs**:  
  - `mask`: 잘라낸 마스크  

#### Safe Select CropMask Node : 안정화노드. 중앙을 기준으로 상하좌우로 직접 자를 픽셀을 지정.
##오류수정, 자체미리보기 추가

- **Inputs**:  
  - `mask`: 마스크 텐서  
  - `L`, `R`, `T`, `B`: 잘라낼 위치 및 크기
  - `show_preview`: 미리보기 기능(켰을 경우 프리뷰 모드도 적용)
- **Outputs**:  
  - `mask`: 잘라낸 마스크  
  
#### Safe FeatherMask Node : 마스크 페더링 노드. 기본노드와 크게 차이는 없지만, 충돌 자체는 줄임.
##오류수정, 자체미리보기 추가

- **Inputs**:  
  - `mask`: 마스크 텐서  
  - `radius`: 페더링 반경
  - `show_preview`: 미리보기 기능(켰을 경우 프리뷰 모드도 적용)
- **Outputs**:  
  - `mask`: 페더링된 마스크  



### 체커

#### Safe MaskPreview Node : 안정화 노드. '값이 남아있긴 하다면', 오류발생시 리턴하여 안정화를 유도해 재출력함.(깨지던 버그 수정.)
- **Inputs**:  
  - `mask`: 마스크 텐서  
- **Outputs**:  
  - `image`: 마스크 미리보기 이미지  

#### Safe MaskSaveOnly Node : 안정화 노드. 마스크를 읽고 '콤피UI 출력 폴더의 마스크폴더에 저장'. 오류발생시 리턴하여 안정화를 유도해 재출력함.
- **Inputs**:  
  - `mask`: 마스크 텐서 
  
#### Safe MaskSaveLink Node : 안정화 노드. 마스크를 읽고 '콤피UI 출력 폴더의 마스크폴더에 저장 후' 마스크 출력 노드로 연결. 오류발생시 리턴하여 안정화를 유도해 재출력함.
##오류수정, 자체미리보기 추가

- **Inputs**:  
  - `mask`: 마스크 텐서
  - `show_preview`: 미리보기 기능(켰을 경우 프리뷰 모드도 적용)
- **Outputs**: 
   - `mask`: 마스크 텐서 
   
#### SafeMaskChecker Node : 안정화 노드. 원본이미지의 마스크 위에 수정마스크를 덮어서 이미지 미리보기에 연결하여 사전확인 가능하게 함.
##오류수정, 강조 색상 노랑 추가, 자체미리보기 추가

- **Inputs**: 
  - `base_mask` : 기준 마스크(원본이미지를 마스크 컨버트로 출력후 연결)
  - `edit_mask` : 편집 마스크(마스크 에디터로 수정영역 그린 후 연결)
  - `color` : 표시 색상(red/green/blue)
  - `show_preview`: 미리보기 기능(켰을 경우 프리뷰 모드도 적용)
- **Outputs**: 
  - `preview` : 편집 마스크를 색상 오버레이로 표현한 이미지
처리방식
편집 마스크를 흑백 이미지로 변환 후 선택한 색상으로 오버레이하여 시각적으로 확인 가능하게 출력

#### SafeMaskDiffChecker Node : 원본이미지의 마스크와 수정마스크의 차이점을 대조 후 변경예상점을 이미지 미리보기에 연결하여 사전확인 가능하게 함.(깨지던 버그 수정.)
##버전업. 이제 원본마스크 연결 안합니다. 준비한 마스크와 원본이미지 연결하면 확인 가능합니다. 강조 색상 노랑 추가, 자체미리보기 추가

- **Inputs**: 
  - `base_image` : 원본 이미지
  - `edit_mask` : 수정 마스크(마스크 에디터로 수정영역 그린 후 연결)
  - `color` : 강조 색상(red/green/blue/yellow)
  - `show_preview`: 미리보기 기능(켰을 경우 프리뷰 모드도 적용)
- **Outputs**: 
  - `preview` : 변경된 영역을 색상으로 강조한 이미지
처리방식
원본과 편집 마스크를 비교하여 차이 영역만 추출, 전체 영역은 반투명 처리하고 차이 영역은 지정 색상으로 강조


