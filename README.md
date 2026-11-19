# VLAsystem
이 VLA 시스템은 각각 분류된 SLM과 비전모델을 결합하여 파이튜닝 없이 코드로 결합한 시스템 입니다.
해당 코드는 크게 3가지 부분으로 분류 할 수 있습니다.
1. main.py
2. memory_manager.py
3. perception_core.py
4. phi_reasoning_module.py(코드에는 phi-3 mini가 아닌 Qwen2.5-1.5B 모델이 있음)
5. pybullet_module.py & robot_geometry.py
6. semantic_analyzer.py

코드에서 사용된 AI는 총 4개로 Depth Anything V2 Small, FastSAM-s, TinyCLIP ViT-39M, Qwen2.5-1.5B, Julia-1이 사용되었습니다.

이 시스템의 목적은 사용자의 보조 역할을 하거나 로봇팔 형태의 하드웨어에 연결하여 테스트를 할 수 있도록 하기 위해 개발되었습니다.

해당 시스템의 코드는 AI의 도움을 받아 작성된 코드 입니다.
