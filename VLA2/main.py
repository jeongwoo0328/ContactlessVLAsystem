# =========================================================
# main.py
# =========================================================& .\.venv\Scripts\python.exe .\VLA2\main.py  

import json
import threading
import math
import time
import serial
import pybullet as p

from perception_core import Vision3DModule
from phi_reasoning_module import AttentionRobot
from pybullet_module import PyBulletModule

# =========================================================
# SETTINGS
# =========================================================

IMAGE_PATH = (
    r"C:\Users\0328j\OneDrive\Desktop\IMG_2560.jpg"
)
OUTPUT_DIR = "output"

# 아두이노 시리얼 설정
ARDUINO_PORT = 'COM6' 
BAUD_RATE = 9600

# 서보모터 매핑 핀 정의
PIN_JOINT_0 = 2  # joint_0 제어용 핀
PIN_JOINT_1 = 7  # joint_1 제어용 핀

# =========================================================
# PYBULLET THREAD
# =========================================================

def pybullet_loop(sim):

    while True:

        try:

            sim.update()

        except:
            break

# =========================================================
# ARDUINO SYNC THREAD (아두이노 동기화 스레드 루프)
# =========================================================

def arduino_sync_loop(sim):
    """
    PyBullet 시뮬레이션의 joint_0과 joint_1 값을 실시간으로 감시하여
    각각 2번 핀과 7번 핀 서보모터로 "핀번호,각도\n" 형태로 전송합니다.
    """
    print("\n[아두이노] 시리얼 포트 연결 시도 중...")
    try:
        py_serial = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
        # 아두이노 리셋 대기 및 안정화
        time.sleep(2)
        print(f"[아두이노] {ARDUINO_PORT} 포트연결")
        print(f"제어 핀: 2번 핀(joint_0), 7번 핀(joint_1)")
    except Exception as e:
        print(f"[아두이노] 연결X 시뮬레이션만 진행(에러: {e})")
        return

    # 파이썬 내부에서 기억할 각 서보모터의 가상 현재 각도 (초기값 90도 중심 설정)
    current_angle_2 = 90  # joint_0 (Pin 2)
    current_angle_7 = 90  # joint_1 (Pin 7)

    while True:
        try:
            # PyBullet 로봇 모델 및 최소 2개 이상의 관절 존재 여부 확인
            if sim.robot_id is not None and len(sim.joint_indices) >= 2:
                
                # -------------------------------------------------
                # 1. joint_0 (Pin 2) 목표 각도 계산
                # -------------------------------------------------
                joint_0_index = sim.joint_indices[0]
                state_0 = p.getJointState(sim.robot_id, joint_0_index)
                joint_0_radian = state_0[0]
                
                target_angle_2 = int(math.degrees(joint_0_radian)) + 90
                target_angle_2 = max(0, min(180, target_angle_2)) # 구동 범위 제한
                
                # -------------------------------------------------
                # 2. joint_1 (Pin 7) 목표 각도 계산 (반대 방향 작동 해결)
                # -------------------------------------------------
                joint_1_index = sim.joint_indices[1]
                state_1 = p.getJointState(sim.robot_id, joint_1_index)
                joint_1_radian = state_1[0]
                
                # 기존의 "각도 + 90"에서 역방향인 "90 - 각도"로 공식 변경
                target_angle_7 = 90 - int(math.degrees(joint_1_radian)) +20
                target_angle_7 = max(0, min(180, target_angle_7)) # 구동 범위 제한

                # -------------------------------------------------
                # 3. joint_0 (Pin 2) 이동 및 전송
                # -------------------------------------------------
                if current_angle_2 != target_angle_2:
                    if current_angle_2 < target_angle_2:
                        current_angle_2 += 5
                        if current_angle_2 > target_angle_2:
                            current_angle_2 = target_angle_2
                    else:
                        current_angle_2 -= 5
                        if current_angle_2 < target_angle_2:
                            current_angle_2 = target_angle_2
                    
                    # 포맷: "핀번호,각도\n" (예: "2,090\n")
                    data_to_send = f"{PIN_JOINT_0},{current_angle_2:03d}\n"
                    py_serial.write(data_to_send.encode())
                    py_serial.flush()
                    print(f"{PIN_JOINT_0}번 핀(joint_0) -> {current_angle_2:03d}도")
                    time.sleep(0.01) # 시리얼 전송 간의 초단기 간격 확보 (충돌 방지)

                # -------------------------------------------------
                # 4. joint_1 (Pin 7) 이동 및 전송
                # -------------------------------------------------
                if current_angle_7 != target_angle_7:
                    if current_angle_7 < target_angle_7:
                        current_angle_7 -= 5
                        if current_angle_7 < target_angle_7:
                            current_angle_7 = target_angle_7
                    else:
                        current_angle_7 += 5
                        if current_angle_7 > target_angle_7:
                            current_angle_7 = target_angle_7
                    
                    # 포맷: "핀번호,각도\n" (예: "7,090\n")
                    data_to_send = f"{PIN_JOINT_1},{current_angle_7:03d}\n"
                    py_serial.write(data_to_send.encode())
                    py_serial.flush()
                    print(f"{PIN_JOINT_1}번 핀(joint_1) -> {current_angle_7:03d}도")
            
            # 과도한 전송으로 인한 버퍼링 부하를 막기 위해 50ms마다 제어 주기 설정
            time.sleep(0.05)
            
        except Exception as e:
            print(f"\n[아두이노] 동기화 중 오류 발생: {e}")
            break

    # 루프 탈출 시 안전하게 포트 닫기
    try:
        py_serial.close()
        print("[아두이노] 시리얼 연결 종료")
    except:
        pass

# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("VISION LANGUAGE ACTION SYSTEM")
    print("=" * 60)

    # =====================================================
    # VISION
    # =====================================================

    vision = Vision3DModule()

    # =====================================================
    # PROCESS IMAGE
    # =====================================================

    scene_data = vision.process_image(

        image_path=IMAGE_PATH,

        output_dir=OUTPUT_DIR
    )

    # =====================================================
    # PYBULLET
    # =====================================================

    print("=" * 60)
    print("INITIALIZING PYBULLET")

    sim = PyBulletModule()

    # =====================================================
    # LOAD SCENE INTO PYBULLET
    # =====================================================

    sim.load_scene(
        scene_data
    )

    # =====================================================
    # START PHYSICS THREAD
    # =====================================================

    physics_thread = threading.Thread(

        target=pybullet_loop,

        args=(sim,),

        daemon=True
    )

    physics_thread.start()

    # =====================================================
    # START ARDUINO SYNC THREAD (아두이노 동기화 스레드 시작)
    # =====================================================

    arduino_thread = threading.Thread(
        target=arduino_sync_loop,
        args=(sim,),
        daemon=True
    )
    arduino_thread.start()

    # =====================================================
    # PHI-3
    # =====================================================

    print("=" * 60)
    print("INITIALIZING PHI-3")

    phi = AttentionRobot()

    # =====================================================
    # READY
    # =====================================================

    print("=" * 60)
    print("SYSTEM READY")
    print("=" * 60)

    print("COMMANDS:")
    print("  exit")
    print("  objects")
    print("  history")
    print("  clear")

    print("=" * 60)

    # =====================================================
    # LOOP
    # =====================================================

    while True:

        try:

            print()

            user_input = input(
                "USER > "
            ).strip()

            # =================================================
            # EMPTY
            # =================================================

            if len(user_input) == 0:
                continue

            # =================================================
            # EXIT
            # =================================================

            if user_input.lower() == "exit":

                print("=" * 60)
                print("SHUTDOWN")
                print("=" * 60)

                sim.close()

                break

            # =================================================
            # OBJECTS
            # =================================================

            if user_input.lower() == "objects":

                print("=" * 60)
                print("OBJECT LIST")
                print("=" * 60)

                for obj in scene_data["objects"]:

                    print()

                    print(
                        f"OBJECT {obj['id']}"
                    )

                    print(
                        "CENTER:",
                        obj["center"]
                    )

                    print(
                        "SIZE:",
                        obj["size"]
                    )

                    print(
                        "DEPTH:",
                        obj["depth"]
                    )

                    print(
                        "COLOR:",
                        obj[
                            "dominant_color"
                        ]
                    )

                    print(
                        "LABELS:",
                        obj[
                            "semantic_labels"
                        ]
                    )

                continue

            # =================================================
            # LOCAL PHI-3 HISTORY
            # =================================================

            if user_input.lower() in ["history", "memory"]:

                print("=" * 60)
                print("PHI-3 LOCAL HISTORY")
                print("=" * 60)

                memories = phi.get_conversation_history(limit=10)

                if len(memories) == 0:

                    print("NO LOCAL HISTORY")

                else:

                    for m in memories:

                        print()

                        print(
                            json.dumps(
                                m,
                                indent=4,
                                ensure_ascii=False
                            )
                        )

                continue

            # =================================================
            # CLEAR
            # =================================================

            if user_input.lower() == "clear":

                phi.clear_conversation_history()
                print("PHI-3 LOCAL HISTORY CLEARED")

                continue

            # =================================================
            # ASK PHI
            # =================================================

            print("=" * 60)
            print("PHI-3 THINKING")

            route = phi.classify_request(

                user_message=user_input,

                scene_data=scene_data
            )

            response = phi.ask_question(

                user_message=user_input,

                scene_data=scene_data,

                memory_context="",

                route=route
            )

            # =================================================
            # RESPONSE
            # =================================================

            print()

            print("PHI-3 >")
            print(response)

            # =================================================
            # ACTION GENERATION
            # =================================================

            print()
            print("=" * 60)
            print("GENERATING ROBOT ACTION")

            action = phi.generate_robot_action(

                user_message=user_input,

                scene_data=scene_data,

                route=route
            )

            print(
                "ACTION:",
                action
            )

            # =================================================
            # EXECUTE ACTION
            # =================================================

            if action.get("type") != "conversation":
                sim.execute_action(action)

        except KeyboardInterrupt:

            print()
            print("=" * 60)
            print("INTERRUPTED")
            print("=" * 60)

            sim.close()

            break

        except Exception as e:

            print()
            print("=" * 60)
            print("ERROR")
            print(e)
            print("=" * 60)

# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    main()