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
    r"C:\Users\0328j\OneDrive\Desktop\WIN_20260613_13_59_39_Pro.jpg"
)
OUTPUT_DIR = "output"

# 아두이노 시리얼 설정
ARDUINO_PORT = 'COM6' 
BAUD_RATE = 9600

# 서보모터 매핑 핀 정의 (4개 모터 구동)
PIN_JOINT_0 = 2  # joint_0 제어용 핀
PIN_JOINT_1 = 7  # joint_1 제어용 핀
PIN_JOINT_2 = 9  # joint_2 제어용 핀 (기존 추가됨)
PIN_JOINT_3 = 11 # joint_3 제어용 핀 (새로 추가된 그리퍼/집게용 서보 핀 - 11번으로 수정)

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
    PyBullet 시뮬레이션의 joint_0, joint_1, joint_2, joint_3 값을 실시간으로 감시하여
    각각 2번, 7번, 9번, 11번 핀 서보모터로 "핀번호,각도\n" 형태로 전송합니다.
    """
    print("\n[아두이노] 시리얼 포트 연결 시도 중...")
    try:
        py_serial = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
        # 아두이노 리셋 대기 및 안정화
        time.sleep(2)
        print(f"[아두이노] {ARDUINO_PORT} 포트연결 성공!")
        print(f"제어 핀: 2번(joint_0), 7번(joint_1), 9번(joint_2), {PIN_JOINT_3}번(그리퍼)")
    except Exception as e:
        print(f"[아두이노] 연결 실패! 시뮬레이션만 진행합니다. (에러: {e})")
        return

    # 파이썬 내부에서 기억할 각 서보모터의 가상 현재 각도
    current_angle_2 = 90   # joint_0 (Pin 2)
    current_angle_7 = 90   # joint_1 (Pin 7)
    current_angle_9 = 90   # joint_2 (Pin 9)
    current_angle_11 = 0   # joint_3 (Pin 11) - 그리퍼 초기화 (절반 열림 상태)

    while True:
        try:
            # PyBullet 로봇 모델 및 최소 4개 이상의 관절 존재 여부 확인
            if sim.robot_id is not None and len(sim.joint_indices) >= 4:
                
                # -------------------------------------------------
                # 1. joint_0 (Pin 2) 목표 각도 계산
                # -------------------------------------------------
                joint_0_index = sim.joint_indices[0]
                state_0 = p.getJointState(sim.robot_id, joint_0_index)
                joint_0_radian = state_0[0]
                
                target_angle_2 = int(math.degrees(joint_0_radian)) + 90
                target_angle_2 = max(0, min(180, target_angle_2)) # 구동 범위 제한
                
                # -------------------------------------------------
                # 2. joint_1 (Pin 7) 목표 각도 계산
                # -------------------------------------------------
                joint_1_index = sim.joint_indices[1]
                state_1 = p.getJointState(sim.robot_id, joint_1_index)
                joint_1_radian = state_1[0]
                
                target_angle_7 = 90 - int(math.degrees(joint_1_radian)) + 20
                target_angle_7 = max(0, min(180, target_angle_7)) # 구동 범위 제한

                # -------------------------------------------------
                # 3. joint_2 (Pin 9) 목표 각도 계산
                # -------------------------------------------------
                joint_2_index = sim.joint_indices[2]
                state_2 = p.getJointState(sim.robot_id, joint_2_index)
                joint_2_radian = state_2[0]
                
                target_angle_9 = int(math.degrees(joint_2_radian)) + 90
                target_angle_9 = max(0, min(180, target_angle_9)) # 구동 범위 제한

                # -------------------------------------------------
                # 4. joint_3 (Pin 11) 목표 각도 계산 (새롭게 추가된 그리퍼)
                # -------------------------------------------------
                joint_3_index = sim.joint_indices[3]
                state_3 = p.getJointState(sim.robot_id, joint_3_index)
                joint_3_radian = state_3[0]
                
                # 시뮬레이션 상의 물리값 기준 변환 (라디안 단위에서 0~180 범위 매핑)
                target_angle_11 = int(math.degrees(joint_3_radian)) + 90
                target_angle_11 = max(0, min(180, target_angle_11))

                # -------------------------------------------------
                # 5. joint_0 (Pin 2) 이동 및 전송
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
                    
                    data_to_send = f"{PIN_JOINT_0},{current_angle_2:03d}\n"
                    py_serial.write(data_to_send.encode())
                    py_serial.flush()
                    print(f"{PIN_JOINT_0}번 핀(joint_0) -> {current_angle_2:03d}도")
                    time.sleep(0.01)

                # -------------------------------------------------
                # 6. joint_1 (Pin 7) 이동 및 전송
                # -------------------------------------------------
                if current_angle_7 != target_angle_7:
                    if current_angle_7 < target_angle_7:
                        current_angle_7 += 5
                        if current_angle_7 > target_angle_7:
                            current_angle_7 = target_angle_7
                    else:
                        current_angle_7 -= 5
                        if current_angle_7 < target_angle_7:
                            current_angle_7 = target_angle_7
                    
                    data_to_send = f"{PIN_JOINT_1},{current_angle_7:03d}\n"
                    py_serial.write(data_to_send.encode())
                    py_serial.flush()
                    print(f"{PIN_JOINT_1}번 핀(joint_1) -> {current_angle_7:03d}도")
                    time.sleep(0.01)

                # -------------------------------------------------
                # 7. joint_2 (Pin 9) 이동 및 전송
                # -------------------------------------------------
                if current_angle_9 != target_angle_9:
                    if current_angle_9 < target_angle_9:
                        current_angle_9 += 5
                        if current_angle_9 > target_angle_9:
                            current_angle_9 = target_angle_9
                    else:
                        current_angle_9 -= 5
                        if current_angle_9 < target_angle_9:
                            current_angle_9 = target_angle_9
                    
                    data_to_send = f"{PIN_JOINT_2},{current_angle_9:03d}\n"
                    py_serial.write(data_to_send.encode())
                    py_serial.flush()
                    print(f"{PIN_JOINT_2}번 핀(joint_2) -> {current_angle_9:03d}도")
                    time.sleep(0.01)

                # -------------------------------------------------
                # 8. joint_3 (Pin 11) 그리퍼 이동 및 전송
                # -------------------------------------------------
                if current_angle_11 != target_angle_11:
                    # 집게 서보는 보간 속도를 원활하게 하기 위해 보조 매핑
                    if current_angle_11 < target_angle_11:
                        current_angle_11 += 10
                        if current_angle_11 > target_angle_11:
                            current_angle_11 = target_angle_11
                    else:
                        current_angle_11 -= 10
                        if current_angle_11 < target_angle_11:
                            current_angle_11 = target_angle_11
                    
                    data_to_send = f"{PIN_JOINT_3},{current_angle_11:03d}\n"
                    py_serial.write(data_to_send.encode())
                    py_serial.flush()
                    print(f"{PIN_JOINT_3}번 핀(그리퍼) -> {current_angle_11:03d}도")
                    time.sleep(0.01)
            
            time.sleep(0.05)
            
        except Exception as e:
            print(f"\n[아두이노] 동기화 중 오류 발생: {e}")
            break

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