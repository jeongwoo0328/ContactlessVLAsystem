# =========================================================
# pybullet_module.py
# 5-DOF ARM VERSION
# shoulder yaw/pitch + elbow pitch + wrist pitch/yaw
# shoulder-to-wrist-end length is approximately 1.0 m
# FRONT/BACK FIXED
# OBJECTS SHIFTED RIGHT
# ROBOT ROTATED 90 DEG TO FACE FRONT
# =========================================================

import time
import math
import threading
import os
import numpy as np
import pybullet as p

from robot_geometry import (
    ARM_FOREARM_LEN,
    ARM_TOOL_LEN,
    ARM_TOTAL_LEN,
    ARM_UPPER_LEN,
    ARM_WRIST_LEN,
    BASE_JOINT_HEIGHT,
    MIN_OBJECT_SIZE,
    OBJECT_SCENE_OFFSET,
    ROBOT_BASE_POSITION,
    SCENE_ORIGIN,
    SHOULDER_LINK_HEIGHT,
    vision_to_pybullet_world
)


class PyBulletModule:

    # =====================================================
    # INIT
    # =====================================================

    def __init__(self, gui=True):

        print("=" * 60)
        print("INITIALIZING PYBULLET")

        self.client = p.connect(
            p.GUI if gui else p.DIRECT
        )

        p.setGravity(0, 0, -9.8)

        p.setRealTimeSimulation(0)

        p.configureDebugVisualizer(
            p.COV_ENABLE_GUI,
            0
        )

        p.configureDebugVisualizer(
            p.COV_ENABLE_SHADOWS,
            0
        )

        p.resetDebugVisualizerCamera(
            cameraDistance=3.0,
            cameraYaw=90,
            cameraPitch=-20,
            cameraTargetPosition=[
                0.4,
                0.0,
                0.7
            ]
        )

        # =================================================
        # joint sign calibration
        # =================================================

        self.JOINT_SIGNS = [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0
        ]

        self.arm_upper_len = ARM_UPPER_LEN
        self.arm_forearm_len = ARM_FOREARM_LEN
        self.arm_wrist_len = ARM_WRIST_LEN
        self.arm_tool_len = ARM_TOOL_LEN
        self.arm_total_len = ARM_TOTAL_LEN

        self.joint_lower_limits = [
            -2.5,
            -1.8,
            -3.1,
            -2.8,
            -1.8
        ]

        self.joint_upper_limits = [
            2.5,
            2.6,
            3.1,
            2.8,
            1.8
        ]

        self.robot_id = None

        self.joint_indices = []

        self.object_text_ids = []

        self.object_body_ids = []

        self.step_lock = threading.Lock()

        self.scene_origin = list(SCENE_ORIGIN)

        self.min_object_size = MIN_OBJECT_SIZE
        self.step_sleep = float(os.getenv("VLA_SIM_STEP_SLEEP", "0.002"))

        # =================================================
        # OBJECTS SHIFTED RIGHT
        # x = right
        # y = forward
        # =================================================

        self.object_scene_offset = list(OBJECT_SCENE_OFFSET)

        self.robot_base_position = list(ROBOT_BASE_POSITION)

        # =================================================
        # ROBOT FACES FRONT (+Y) BY 90 DEG ROTATION
        # =================================================

        self.robot_base_orientation = (
            p.getQuaternionFromEuler([
                0.0,
                0.0,
                math.radians(90.0)
            ])
        )

        self.create_ground()

        self.draw_axes()

        self.create_robot()

        self.apply_joint_control({

            "joint_0": 0.0,

            "joint_1": 0.20,

            "joint_2": -0.55,

            "joint_3": -0.10,

            "joint_4": 0.0,

            "steps": 240
        })

        print("=" * 60)
        print("PYBULLET READY")

    # =====================================================
    # GROUND
    # =====================================================

    def create_ground(self):

        plane = p.createCollisionShape(
            p.GEOM_PLANE
        )

        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=plane
        )

    # =====================================================
    # AXES
    # =====================================================

    def draw_axes(self):

        s = 1.0

        p.addUserDebugLine(
            [0, 0, 0],
            [s, 0, 0],
            [1, 0, 0],
            4
        )

        p.addUserDebugLine(
            [0, 0, 0],
            [0, s, 0],
            [0, 1, 0],
            4
        )

        p.addUserDebugLine(
            [0, 0, 0],
            [0, 0, s],
            [0, 0, 1],
            4
        )

    # =====================================================
    # LOAD SCENE
    # =====================================================

    def load_scene(
            self,
            scene_data
    ):

        if "objects" in scene_data:

            self.load_objects(
                scene_data["objects"]
            )

    # =====================================================
    # LOAD OBJECTS
    # =====================================================

    def load_objects(
            self,
            objects
    ):

        for t in self.object_text_ids:

            try:
                p.removeUserDebugItem(t)
            except:
                pass

        self.object_text_ids = []

        for body_id in self.object_body_ids:

            try:
                p.removeBody(body_id)
            except:
                pass

        self.object_body_ids = []

        for obj in objects:

            world_pos = self.vision_to_pybullet(
                obj
            )

            half_extents = self.object_half_extents(
                obj
            )

            rgba = self.color_to_rgba(
                obj.get(
                    "dominant_color",
                    "mixed"
                )
            )

            labels = obj.get(
                "semantic_labels",
                []
            )

            label = (
                labels[0]
                if labels else
                "object"
            )

            shape_type = self.guess_shape(
                label
            )

            if shape_type == "cylinder":

                collision = p.createCollisionShape(
                    p.GEOM_CYLINDER,
                    radius=max(
                        half_extents[0],
                        half_extents[1]
                    ),
                    height=max(
                        self.min_object_size,
                        half_extents[2] * 2.0
                    )
                )

                visual = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=max(
                        half_extents[0],
                        half_extents[1]
                    ),
                    length=max(
                        self.min_object_size,
                        half_extents[2] * 2.0
                    ),
                    rgbaColor=rgba
                )

            else:

                collision = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=half_extents
                )

                visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=half_extents,
                    rgbaColor=rgba
                )

            object_orientation = (
                p.getQuaternionFromEuler([
                    0.0,
                    0.0,
                    0.0
                ])
            )

            shifted_pos = [

                world_pos[0]
                + self.object_scene_offset[0],

                world_pos[1]
                + self.object_scene_offset[1],

                world_pos[2]
                + self.object_scene_offset[2]
            ]

            body_id = p.createMultiBody(

                baseMass=0.0,

                baseCollisionShapeIndex=collision,

                baseVisualShapeIndex=visual,

                basePosition=shifted_pos,

                baseOrientation=object_orientation
            )

            self.object_body_ids.append(
                body_id
            )

            text_id = p.addUserDebugText(

                f"id={obj['id']} {label}",

                [
                    shifted_pos[0],
                    shifted_pos[1],
                    shifted_pos[2]
                    + half_extents[2]
                    + 0.04
                ],

                [1, 1, 1],

                1.2
            )

            self.object_text_ids.append(
                text_id
            )

        self.validate_scene_coordinates(
            objects
        )

    # =====================================================
    # COORDINATE TRANSFORM
    # =====================================================

    def vision_to_pybullet(self, obj):

        center = obj.get(
            "center",
            [0.0, 0.0, 0.5]
        )

        size = obj.get(
            "size",
            [0.08, 0.08, 0.08]
        )

        return vision_to_pybullet_world(
            center,
            size,
            shifted=False
        )

    # =====================================================
    # OBJECT SIZE
    # =====================================================

    def object_half_extents(self, obj):

        size = obj.get(
            "size",
            [0.08, 0.08, 0.08]
        )

        sx = max(
            self.min_object_size,
            min(abs(float(size[0])), 0.35)
        )

        sy = max(
            self.min_object_size,
            min(abs(float(size[2])), 0.35)
        )

        sz = max(
            self.min_object_size,
            min(abs(float(size[1])), 0.18)
        )

        return [
            sx * 0.5,
            sy * 0.5,
            sz * 0.5
        ]

    # =====================================================
    # COLOR
    # =====================================================

    def color_to_rgba(self, color_name):

        colors = {

            "white": [0.92, 0.92, 0.88, 1.0],

            "black": [0.02, 0.02, 0.025, 1.0],

            "gray": [0.45, 0.45, 0.45, 1.0],

            "red": [0.85, 0.08, 0.06, 1.0],

            "green": [0.10, 0.55, 0.16, 1.0],

            "blue": [0.08, 0.20, 0.85, 1.0],

            "mixed": [0.72, 0.55, 0.32, 1.0],
        }

        return colors.get(
            str(color_name).lower(),
            colors["mixed"]
        )

    # =====================================================
    # SHAPE
    # =====================================================

    def guess_shape(self, label):

        text = str(label).lower()

        if any(
                word in text
                for word in [
                    "bottle",
                    "cup",
                    "can",
                    "container"
                ]
        ):
            return "cylinder"

        return "box"

    # =====================================================
    # VALIDATE
    # =====================================================

    def validate_scene_coordinates(
            self,
            objects
    ):

        if not objects:

            print("COORDINATE CHECK: NO OBJECTS")

            return True

        ok = True

        for obj in objects:

            pos = self.vision_to_pybullet(
                obj
            )

            if not all(
                    math.isfinite(v)
                    for v in pos
            ):
                ok = False

            if pos[2] < -1e-4:
                ok = False

        print("=" * 60)
        print("COORDINATE CHECK")
        print("VISION: x=right y=up z=forward")
        print("PYBULLET: x=right y=forward z=up")
        print("STATUS:", "OK" if ok else "INVALID")

        return ok

    # =====================================================
    # CREATE ROBOT
    # =====================================================

    def create_robot(self):

        print("=" * 60)
        print("CREATING 5-DOF ARM ROBOT")

        base_collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=0.22,
            height=0.16
        )

        base_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.22,
            length=0.16,
            rgbaColor=[
                0.15,
                0.15,
                0.15,
                1
            ]
        )

        yaw_len = SHOULDER_LINK_HEIGHT

        yaw_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[
                0.075,
                0.075,
                yaw_len / 2
            ],
            collisionFramePosition=[
                0,
                0,
                yaw_len / 2
            ]
        )

        yaw_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[
                0.075,
                0.075,
                yaw_len / 2
            ],
            visualFramePosition=[
                0,
                0,
                yaw_len / 2
            ],
            rgbaColor=[
                0.9,
                0.4,
                0.2,
                1
            ]
        )

        upper_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[
                self.arm_upper_len / 2,
                0.04,
                0.04
            ],
            collisionFramePosition=[
                self.arm_upper_len / 2,
                0,
                0
            ]
        )

        upper_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[
                self.arm_upper_len / 2,
                0.04,
                0.04
            ],
            visualFramePosition=[
                self.arm_upper_len / 2,
                0,
                0
            ],
            rgbaColor=[
                1,
                0.25,
                0.15,
                1
            ]
        )

        forearm_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[
                self.arm_forearm_len / 2,
                0.035,
                0.035
            ],
            collisionFramePosition=[
                self.arm_forearm_len / 2,
                0,
                0
            ]
        )

        forearm_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[
                self.arm_forearm_len / 2,
                0.035,
                0.035
            ],
            visualFramePosition=[
                self.arm_forearm_len / 2,
                0,
                0
            ],
            rgbaColor=[
                0.95,
                0.55,
                0.12,
                1
            ]
        )

        wrist_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[
                self.arm_wrist_len / 2,
                0.03,
                0.03
            ],
            collisionFramePosition=[
                self.arm_wrist_len / 2,
                0,
                0
            ]
        )

        wrist_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[
                self.arm_wrist_len / 2,
                0.03,
                0.03
            ],
            visualFramePosition=[
                self.arm_wrist_len / 2,
                0,
                0
            ],
            rgbaColor=[
                0.25,
                0.55,
                0.95,
                1
            ]
        )

        tool_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[
                self.arm_tool_len / 2,
                0.022,
                0.022
            ],
            collisionFramePosition=[
                self.arm_tool_len / 2,
                0,
                0
            ]
        )

        tool_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[
                self.arm_tool_len / 2,
                0.022,
                0.022
            ],
            visualFramePosition=[
                self.arm_tool_len / 2,
                0,
                0
            ],
            rgbaColor=[
                1.0,
                0.05,
                0.05,
                1
            ]
        )

        self.robot_id = p.createMultiBody(

            baseMass=0,

            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=base_visual,

            basePosition=self.robot_base_position,

            baseOrientation=self.robot_base_orientation,

            linkMasses=[
                1.0,
                0.8,
                0.55,
                0.25,
                0.1
            ],

            linkCollisionShapeIndices=[
                yaw_collision,
                upper_collision,
                forearm_collision,
                wrist_collision,
                tool_collision
            ],

            linkVisualShapeIndices=[
                yaw_visual,
                upper_visual,
                forearm_visual,
                wrist_visual,
                tool_visual
            ],

            linkPositions=[

                [0, 0, BASE_JOINT_HEIGHT],

                [0, 0, yaw_len],

                [self.arm_upper_len, 0, 0],

                [self.arm_forearm_len, 0, 0],

                [self.arm_wrist_len, 0, 0]
            ],

            linkOrientations=[

                [0, 0, 0, 1],

                [0, 0, 0, 1],

                [0, 0, 0, 1],

                [0, 0, 0, 1],

                [0, 0, 0, 1]
            ],

            linkInertialFramePositions=[

                [0, 0, 0],

                [self.arm_upper_len / 2, 0, 0],

                [self.arm_forearm_len / 2, 0, 0],

                [self.arm_wrist_len / 2, 0, 0],

                [self.arm_tool_len / 2, 0, 0]
            ],

            linkInertialFrameOrientations=[

                [0, 0, 0, 1],

                [0, 0, 0, 1],

                [0, 0, 0, 1],

                [0, 0, 0, 1],

                [0, 0, 0, 1]
            ],

            linkParentIndices=[
                0,
                1,
                2,
                3,
                4
            ],

            linkJointTypes=[
                p.JOINT_REVOLUTE,
                p.JOINT_REVOLUTE,
                p.JOINT_REVOLUTE,
                p.JOINT_REVOLUTE,
                p.JOINT_REVOLUTE
            ],

            linkJointAxis=[

                [0, 0, 1],

                [0, 1, 0],

                [0, 1, 0],

                [0, 1, 0],

                [0, 0, 1]
            ]
        )

        self.joint_indices = []

        count = p.getNumJoints(
            self.robot_id
        )

        for i in range(count):

            info = p.getJointInfo(
                self.robot_id,
                i
            )

            if info[2] == p.JOINT_REVOLUTE:

                self.joint_indices.append(i)

        for j in self.joint_indices:

            p.resetJointState(
                self.robot_id,
                j,
                0.0
            )

    # =====================================================
    # APPLY CONTROL
    # =====================================================

    def apply_joint_control(
            self,
            action
    ):

        steps = int(
            action.get(
                "steps",
                240
            )
        )

        targets = []

        action = self.refine_action_for_target(
            action
        )

        for i in range(len(self.joint_indices)):
            value = float(
                action.get(
                    f"joint_{i}",
                    0.0
                )
            )
            targets.append(
                self.JOINT_SIGNS[i] * value
            )

        for _ in range(steps):

            for i, joint_index in enumerate(
                    self.joint_indices
            ):

                if i >= len(targets):
                    break

                p.setJointMotorControl2(

                    bodyUniqueId=self.robot_id,

                    jointIndex=joint_index,

                    controlMode=p.POSITION_CONTROL,

                    targetPosition=targets[i],

                    force=12000,

                    positionGain=0.95,

                    velocityGain=1.2
                )

            self.step()

        for i, joint_index in enumerate(self.joint_indices):
            if i >= len(targets):
                break

            p.resetJointState(
                self.robot_id,
                joint_index,
                targets[i]
            )

        self.step()

        self.report_target_error(action)

    # =====================================================
    # IK REFINE
    # =====================================================

    def refine_action_for_target(
            self,
            action
    ):

        if not isinstance(action, dict) or "target_world" not in action:
            return action

        if self.robot_id is None or len(self.joint_indices) == 0:
            return action

        try:
            target = np.array(
                [
                    float(v)
                    for v in action["target_world"]
                ],
                dtype=float
            )
        except Exception:
            return action

        saved_joint_states = [
            p.getJointState(
                self.robot_id,
                joint_index
            )[0]
            for joint_index in self.joint_indices
        ]

        initial_q = np.array(
            [
                float(action.get(f"joint_{i}", 0.0))
                for i in range(len(self.joint_indices))
            ],
            dtype=float
        )

        low = np.array(
            self.joint_lower_limits[:len(self.joint_indices)],
            dtype=float
        )
        high = np.array(
            self.joint_upper_limits[:len(self.joint_indices)],
            dtype=float
        )

        def set_joint_positions(values):
            for joint_i, joint_index in enumerate(self.joint_indices):
                p.resetJointState(
                    self.robot_id,
                    joint_index,
                    float(values[joint_i])
                )

        def tip_for(values):
            set_joint_positions(values)
            p.performCollisionDetection()
            tip = self.get_tool_tip_world()
            if tip is None:
                return None
            return np.array(tip, dtype=float)

        def optimize_from(seed):
            q = np.clip(
                np.array(seed, dtype=float),
                low,
                high
            )

            for _ in range(80):
                current_tip = tip_for(q)
                if current_tip is None:
                    break

                error = target - current_tip
                if float(np.linalg.norm(error)) < 0.004:
                    break

                jacobian = np.zeros(
                    (3, len(self.joint_indices)),
                    dtype=float
                )
                eps = 1e-4

                for joint_i in range(len(self.joint_indices)):
                    q_plus = q.copy()
                    q_minus = q.copy()
                    q_plus[joint_i] += eps
                    q_minus[joint_i] -= eps

                    tip_plus = tip_for(q_plus)
                    tip_minus = tip_for(q_minus)
                    if tip_plus is None or tip_minus is None:
                        raise RuntimeError("unable to compute IK jacobian")

                    jacobian[:, joint_i] = (
                        tip_plus
                        - tip_minus
                    ) / (2.0 * eps)

                damping = 0.03
                lhs = (
                    jacobian
                    @ jacobian.T
                    + (damping * damping) * np.eye(3)
                )
                delta = (
                    jacobian.T
                    @ np.linalg.solve(lhs, error)
                )

                max_delta = float(np.max(np.abs(delta)))
                if max_delta > 0.12:
                    delta = delta / max_delta * 0.12

                q = np.clip(
                    q + delta,
                    low,
                    high
                )

            final_tip = tip_for(q)
            if final_tip is None:
                return q, float("inf")

            return q, float(np.linalg.norm(target - final_tip))

        try:
            seed_count = len(self.joint_indices)
            seeds = [
                initial_q,
                np.zeros(seed_count, dtype=float),
                np.array([0.0, 0.8, -2.0, 0.0, 0.0][:seed_count]),
                np.array([0.0, 1.4, -2.6, -0.4, 0.0][:seed_count]),
                np.array([0.0, 2.0, -3.0, -1.0, 0.0][:seed_count]),
            ]

            yaw = float(initial_q[0]) if seed_count > 0 else 0.0
            for shoulder in [0.4, 0.9, 1.4, 1.9, 2.4]:
                for elbow in [-0.8, -1.4, -2.0, -2.6, -3.0]:
                    for wrist in [-2.2, -1.4, -0.6, 0.0, 0.6]:
                        seeds.append(
                            np.array(
                                [yaw, shoulder, elbow, wrist, 0.0][
                                    :seed_count
                                ],
                                dtype=float
                            )
                        )

            best_q = initial_q
            best_error = float("inf")

            for seed in seeds:
                q, error = optimize_from(seed)
                if error < best_error:
                    best_q = q
                    best_error = error

            refined = dict(action)
            for joint_i in range(len(self.joint_indices)):
                refined[f"joint_{joint_i}"] = float(best_q[joint_i])

            return refined

        except Exception as e:
            print("IK REFINE ERROR:", e)
            return action

        finally:
            set_joint_positions(
                saved_joint_states
            )

    # =====================================================
    # TOOL TIP / TARGET ERROR
    # =====================================================

    def get_tool_tip_world(self):

        if self.robot_id is None or len(self.joint_indices) == 0:
            return None

        link_index = self.joint_indices[-1]

        state = p.getLinkState(
            self.robot_id,
            link_index,
            computeForwardKinematics=True
        )

        link_pos = state[4]
        link_orn = state[5]
        rot = p.getMatrixFromQuaternion(link_orn)

        local_x_world = [
            rot[0],
            rot[3],
            rot[6]
        ]

        return [
            link_pos[0] + (local_x_world[0] * self.arm_tool_len),
            link_pos[1] + (local_x_world[1] * self.arm_tool_len),
            link_pos[2] + (local_x_world[2] * self.arm_tool_len),
        ]

    def report_target_error(
            self,
            action
    ):

        if not isinstance(action, dict) or "target_world" not in action:
            return

        tip = self.get_tool_tip_world()
        if tip is None:
            return

        try:
            target = [
                float(v)
                for v in action["target_world"]
            ]
        except Exception:
            return

        error = math.sqrt(
            ((tip[0] - target[0]) ** 2)
            + ((tip[1] - target[1]) ** 2)
            + ((tip[2] - target[2]) ** 2)
        )

        print("=" * 60)
        print("TARGET TRACKING")
        print("TIP:", [round(v, 4) for v in tip])
        print("TARGET:", [round(v, 4) for v in target])
        print("ERROR_M:", round(error, 4))

    # =====================================================
    # EXECUTE
    # =====================================================

    def execute_action(
            self,
            action
    ):

        if (
                not isinstance(action, dict)
                or action.get("type") == "conversation"
        ):
            return

        if action.get("type") == "sequence":
            for item in action.get("actions", []):
                self.apply_joint_control(item)
            return

        self.apply_joint_control(
            action
        )

    # =====================================================
    # STEP
    # =====================================================

    def step(self):

        with self.step_lock:

            p.stepSimulation()

        if self.step_sleep > 0:
            time.sleep(
                self.step_sleep
            )

    # =====================================================
    # UPDATE
    # =====================================================

    def update(self):

        self.step()

    # =====================================================
    # CLOSE
    # =====================================================

    def close(self):

        print("=" * 60)
        print("CLOSING PYBULLET")

        p.disconnect()
