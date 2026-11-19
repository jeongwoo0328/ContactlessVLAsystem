# VLAsystem

`VLAsystem`은 비전 인식 모델과 소형 언어 모델(SLM), 로봇 시뮬레이션 및 실제 하드웨어 제어를 하나의 파이썬 파이프라인으로 결합한 비접촉형 VLA(Vision-Language-Action) 실험 시스템입니다.

이 프로젝트는 사전 학습된 비전 및 언어 모델과 규칙 기반 로봇 스킬을 조합하며, PyBullet 시뮬레이션과 Arduino 서보모터 하드웨어를 지원합니다.

> ⚠️ 이 저장소의 로봇 제어 코드는 실제 하드웨어에 영향을 줄 수 있습니다. 실제 장비를 조작하기 전 관절 제한과 작업 공간을 확인하고, 먼저 시뮬레이션 모드로 충분히 검증하세요.

## 주요 기능

- 단일 이미지로 물체의 위치와 크기 추정
- FastSAM 기반 물체 분할
- TinyCLIP 기반 물체 의미 분류
- Depth Anything V2 기반 깊이 추정
- 3D 포인트클라우드 및 선택적 메쉬 생성
- Qwen2.5 기반 대화와 명령 라우팅
- PyBullet 기반 5축 로봇팔 시뮬레이션
- Arduino 및 서보모터 기반 실제 하드웨어 제어
- 최근 대화와 장기 대화 기억 관리
- `look`, `point`, `raise_low`, `motion`, `grasp`, `idle` 로봇 스킬

## 시스템 구성

```text
사용자 명령 / 카메라 이미지
          │
          ▼
┌──────────────────────────┐
│ Vision3DModule           │
│ Depth + FastSAM + CLIP   │
└─────────────┬────────────┘
              │ scene.json / 3D 객체 정보
              ▼
┌──────────────────────────┐
│ AttentionRobot           │
│ 명령 분류 및 스킬 계획     │
└─────────────┬────────────┘
              │ 관절 또는 시퀀스 명령
       ┌──────┴──────┐
       ▼             ▼
   PyBullet       Arduino
   시뮬레이션      서보모터
```

## 사용 모델

| 구성 요소 | 모델 또는 라이브러리 | 역할 |
|---|---|---|
| 깊이 추정 | Depth-Anything-V2-Small | 이미지의 상대 깊이 추정 |
| 물체 분할 | FastSAM-s | 물체 마스크 추출 |
| 시각 의미 분류 | TinyCLIP-ViT-39M | 물체 종류와 색상 관련 의미 추정 |
| 언어 및 라우팅 | Qwen2.5-0.5B-Instruct | 대화, 명령 라우팅, 스킬 계획 |
| 장기 대화 기억 | SentenceTransformer | 과거 대화의 벡터 검색 |
| 물리 시뮬레이션 | PyBullet | 로봇팔 및 객체 시뮬레이션 |

모델 가중치는 실행 시 Hugging Face 또는 Ultralytics에서 다운로드될 수 있습니다.

## 주요 파일

| 파일 또는 디렉터리 | 설명 |
|---|---|
| `main.py` | 전체 파이프라인 실행과 사용자 명령 루프 |
| `perception_core.py` | 깊이 추정, 분할, 물체 분류, 포인트클라우드 생성 |
| `phi_reasoning_module.py` | 대화 라우팅, 기억 관리 연동, 로봇 스킬 선택 |
| `memory_manager.py` | 단기 및 장기 대화 기억 관리 |
| `semantic_analyzer.py` | 명령의 의미 원소 분석 |
| `pybullet_module.py` | 5축 로봄팔 PyBullet 시뮬레이션 |
| `robot_geometry.py` | 카메라 좌표와 PyBullet 좌표 변환 상수 |
| `skills/` | `look`, `point`, `grasp` 등 로봇 스킬 구현 |
| `words_data.json` | 의미 원소와 단어 매핑 데이터 |
| `sketch_jul12a.ino` | Arduino 서보모터 시리얼 제어 코드 |

## 요구 사항

- Python 3.10 이상 권장
- Windows 권장: 기본 설정은 COM 포트와 PySerial을 사용합니다
- Linux 및 macOS에서도 시뮬레이션은 실행할 수 있습니다
- CUDA GPU는 선택 사항이며, CPU에서도 실행할 수 있지만 추론 속도가 느릴 수 있습니다

### 하드웨어

- 선택 사항: Arduino 또는 호환 보드
- 서보모터 또는 유사한 로봇팔 하드웨어
- USB 시리얼 연결
- 카메라 또는 분석할 정지 이미지

실제 하드웨어가 없어도 시뮬레이션을 실행할 수 있습니다.

## 설치

```bash
git clone https://github.com/jeongwoo0328/VLAsystem.git
cd VLAsystem

python -m venv .venv
```

Windows 활성화:

```bash
.venv\Scripts\activate
```

Linux 또는 macOS 활성화:

```bash
source .venv/bin/activate
```

의존성 설치:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

GPU를 사용하는 경우 PyTorch의 CUDA 버전을 먼저 설치하세요.

```text
https://pytorch.org/get-started/locally/
```

## 설정

`main.py`의 `IMAGE_PATH`를 분석할 이미지 경로로 변경하세요.

```python
IMAGE_PATH = r"sample.jpg"
OUTPUT_DIR = "output"
```

실제 하드웨어를 사용하는 경우 `main.py`의 시리얼 설정을 수정하세요.

```python
ARDUINO_PORT = "COM6"   # Windows 예시
BAUD_RATE = 9600
```

Linux에서는 예를 들어 `/dev/ttyUSB0` 또는 `/dev/ttyACM0`을 사용할 수 있습니다.

### 주요 환경변수

| 환경변수 | 기본값 | 설명 |
|---|---:|---|
| `VLA_RESIZE_W` | CPU 640 / GPU 800 | 입력 이미지 가로 크기 |
| `VLA_POINT_STEP` | CPU 3 / GPU 2 | 포인트클라우드 샘플링 간격 |
| `VLA_MAX_OBJECTS` | CPU 15 / GPU 24 | 처리할 최대 물체 수 |
| `VLA_CLIP_BATCH` | 16 | TinyCLIP 배치 크기 |
| `VLA_ENABLE_MESH` | 0 | 메쉬 생성 여부 |
| `VLA_MESH_DEPTH` | 7 | 메쉬 생성 깊이 |
| `VLA_AI_ROUTING` | 1 | 언어 기반 라우팅 활성화 여부 |
| `VLA_SIM_STEP_SLEEP` | 0.002 | PyBullet 시뮬레이션 단계 간 지연 |

예시:

```bash
VLA_ENABLE_MESH=1 python main.py
```

Windows PowerShell:

```powershell
$env:VLA_ENABLE_MESH="1"
python main.py
```

## 실행

```bash
python main.py
```

실행 중 다음 명령을 사용할 수 있습니다.

| 명령 | 설명 |
|---|---|
| `objects` | 감지된 객체 목록 표시 |
| `history` 또는 `memory` | 최근 대화 기록 표시 |
| `clear` | 대화 기억 삭제 |
| `exit` | 프로그램 종료 |

예시 명령:

```text
Look at the red object
Point to id 2
Raise your arm
Lower your arm
Wave
Grasp object
What can you see?
```

## 생성 결과

실행 후 `output/` 디렉터리에 다음 파일이 생성될 수 있습니다.

- `scene.ply`: 전체 포인트클라우드
- `scene_mesh.ply`: 선택적으로 생성되는 메쉬
- `object_*.ply`: 감지된 개별 객체 포인트클라우드
- `scene.json`: 객체 위치, 크기, 색상 및 의미 라벨

장기 대화 기억은 기본적으로 `memory/conversations/memory_state.json`에 저장됩니다.

## 안전 주의사항

- 실제 로봇 작동 전 작업 공간에 사람이 없는지 확인하세요
- 서보모터 전원과 관절 각도 제한을 확인하세요
- 긴급 정지 스위치와 물리적 고정 장치를 준비하세요
- 자동 집기 명령은 예측 불가능한 하드웨어 동작을 만들 수 있습니다
- 이 프로젝트는 연구 및 시연 목적이며 사람 주변에서의 무인 작업을 보장하지 않습니다

## 현재 제한 사항

- 물체 깊이와 카메라 보정이 환경에 따라 달라질 수 있습니다
- 일부 좌표 변환과 로봇 관절 값은 특정 하드웨어에 맞춰져 있습니다
- 집기 동작은 감지 데이터가 아닌 고정 관절 시퀀스를 사용합니다
- 실험 환경 외부에서는 안정적인 실제 로봇 제어를 보장하지 않습니다
- `semantic_analyzer.py`에는 현재 환경에 종속된 경로가 있으므로 다른 컴퓨터에서는 상대 경로 설정이 필요합니다

## 라이선스

아직 라이선스 파일이 없습니다. 공개 저장소의 사용 조건을 명확히 하려면 라이선스를 추가해야 합니다.

## 기여

버그나 개선 아이디어는 GitHub Issues를 통해 등록할 수 있습니다.
