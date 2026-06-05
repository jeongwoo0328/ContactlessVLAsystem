# =========================================================
# main.py
# =========================================================

import json
import threading

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
