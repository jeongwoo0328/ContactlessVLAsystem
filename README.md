# NoncontactVLAsystem
비접촉 VLA시스템

# Vision-Language-Action Robot System

사진 한 장으로부터 3D 환경을 추정하고, 객체를 인식하며, 자연어 명령을 이해하여 로봇팔이 행동할 수 있도록 구성된 Vision-Language-Action(VLA) 시스템입니다.

## 개요

이 프로젝트는 다음 4개의 핵심 모듈로 구성됩니다.

1. **Perception (Vision3DModule)**

   * 단일 RGB 이미지 입력
   * Depth Anything 기반 깊이 추정
   * MobileSAM 기반 객체 분할
   * SIGLIP 기반 객체 의미 인식
   * 3D Point Cloud 생성
   * 객체 중심 좌표 및 크기 추정

2. **Reasoning (AttentionRobot)**

   * Phi-3 Mini 기반 언어 추론
   * 대화 관리
   * 명령 분류
   * 객체 참조 해석
   * 로봇 행동 생성

3. **Simulation (PyBulletModule)**

   * 5-DOF 로봇팔 시뮬레이션
   * 객체 시각화
   * 물리 시뮬레이션
   * 행동 실행

4. **Memory (MemoryManager)**

   * 최근 대화 저장
   * 장기 기억 압축
   * 태그 기반 검색
   * 대화 컨텍스트 복원

---

# 시스템 구조

```text
RGB Image
    │
    ▼
Vision3DModule
    │
    ├─ Depth Estimation
    ├─ Object Segmentation
    ├─ Semantic Recognition
    └─ 3D Reconstruction
    │
    ▼
Scene Data
    │
    ▼
AttentionRobot
    │
    ├─ Chat
    ├─ Vision QA
    ├─ Skill Selection
    └─ Action Planning
    │
    ▼
Robot Action
    │
    ▼
PyBulletModule
    │
    ▼
Robot Motion
```

---

# 주요 기능

## 객체 인식

* 책
* 노트북
* 모니터
* 키보드
* 마우스
* 스마트폰
* 컵
* 병
* 박스
* 각종 사무용품

## 공간 이해

* 객체 위치 추정
* 객체 크기 추정
* 상대 거리 계산
* 3D 포인트 클라우드 생성
* 장면 재구성

## 자연어 이해

예시:

```text
저 책을 봐
컵을 가리켜
가장 가까운 물체는 뭐야?
오른쪽 물체를 보여줘
팔을 더 높이 올려
```

## 로봇 스킬

### Look Skill

특정 객체를 바라봄

### Point Skill

특정 객체를 가리킴

### Raise/Lower Skill

팔 높이 조절

### Motion Skill

* Wave
* Scan
* Home
* Extend
* Fold

### Idle Skill

대기 자세

---

# 프로젝트 구조

```text
project/

├── main.py
├── perception_core.py
├── phi_reasoning_module.py
├── pybullet_module.py
├── memory_manager.py

├── skills/
│   ├── look.py
│   ├── point.py
│   ├── motion.py
│   ├── raise_low.py
│   └── idle.py

├── output/
├── memory/
└── models/
```

---

# 사용 모델

## Vision

### Depth Anything V2

깊이 추정

```text
depth-anything/Depth-Anything-V2
```

### MobileSAM

객체 분할

```text
mobile_sam.pt
```

### SIGLIP

객체 의미 인식

```text
google/siglip-base-patch16-224
```

## Language

### Phi-3 Mini

```text
microsoft/Phi-3-mini-4k-instruct
```

역할:

* 명령 분류
* 대화
* 객체 참조 해석
* 행동 계획

---

# 설치

## Python

```bash
Python 3.10+
```

## 필수 라이브러리

```bash
pip install torch
pip install transformers
pip install ultralytics
pip install opencv-python
pip install scipy
pip install numpy
pip install pillow
pip install open3d
pip install pybullet
```

---

# 실행

`main.py`에서 이미지 경로를 설정합니다.

```python
IMAGE_PATH = r"your_image.jpg"
```

실행:

```bash
python main.py
```

---

# 명령어

## 시스템 명령

```text
objects
```

인식된 객체 목록 출력

```text
history
```

대화 기록 출력

```text
memory
```

대화 기억 출력

```text
clear
```

대화 기록 초기화

```text
exit
```

종료

---

# 예시 출력

```text
OBJECT 3

CENTER:
[0.42, 0.15, 2.81]

SIZE:
[0.21, 0.05, 0.17]

COLOR:
white

LABELS:
['book', 'notebook']
```

---

# 현재 구현 상태

### 구현 완료

* 단일 이미지 3D 추정
* 객체 분할
* 의미 인식
* 3D 좌표 계산
* Phi-3 기반 추론
* 로봇 스킬 시스템
* PyBullet 시뮬레이션
* 대화 메모리

### 향후 계획

* 실물 로봇 연동
* 다중 카메라 지원
* 실시간 영상 입력
* Grasping
* Pick & Place
* 경로 계획(Path Planning)
* 장기 메모리 강화
* 멀티모달 추론 고도화

---

# 라이선스

본 프로젝트는 연구 및 개발 목적으로 제작되었습니다.

사용된 오픈소스 모델과 라이브러리의 라이선스를 각각 준수해야 합니다.

* Microsoft Phi-3
* Depth Anything
* MobileSAM
* SIGLIP
* PyBullet
* Open3D

---

# 한 줄 소개

**"사진을 이해하고, 공간을 추론하며, 자연어 명령을 행동으로 변환하는 Vision-Language-Action 로봇 시스템"**


이는 chat gpt가 작성한 글이며 모든 코드는 chat gpt와 codex cli가 작성한 것임을 참고해주세요.
