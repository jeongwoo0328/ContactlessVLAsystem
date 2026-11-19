import math

from robot_geometry import (
    ARM_FOREARM_LEN,
    ARM_TOOL_LEN,
    ARM_TOTAL_LEN,
    ARM_UPPER_LEN,
    ARM_WRIST_LEN,
    OBJECT_SCENE_OFFSET,
    ROBOT_BASE_POSITION,
    SCENE_ORIGIN,
    SHOULDER_HEIGHT,
    shoulder_world,
    vision_to_pybullet_world
)


class TargetSkill:
    """Shared target resolution and simple 5-DOF arm aiming math."""

    label = "TARGET"
    steps = 110

    def __init__(self, robot):
        self.robot = robot
        self.YAW_SIGN = 1.0
        self.PITCH_SIGN = -1.0
        self.YAW_MIN = -1.4
        self.YAW_MAX = 1.4
        self.PITCH_MIN = -0.9
        self.PITCH_MAX = 0.9
        self.PITCH_BIAS = 0.0
        self.upper_len = ARM_UPPER_LEN
        self.forearm_len = ARM_FOREARM_LEN
        self.wrist_len = ARM_WRIST_LEN
        self.tool_len = ARM_TOOL_LEN
        self.total_len = ARM_TOTAL_LEN
        self.elbow_rest_bend = -0.35
        self.scene_origin = list(SCENE_ORIGIN)
        self.object_scene_offset = list(OBJECT_SCENE_OFFSET)
        self.robot_base_position = list(ROBOT_BASE_POSITION)
        self.shoulder_height = SHOULDER_HEIGHT
        self.reach_margin = 0.015
        self.min_reach = 0.08

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def target_to_world(self, target):
        center = target.get("center", [0.0, 0.0, 0.5])
        size = target.get("size", [0.08, 0.08, 0.08])
        return vision_to_pybullet_world(
            center,
            size,
            shifted=True
        )

    def world_to_shoulder_direction(self, target):
        world_x, world_y, world_z = self.target_to_world(target)

        shoulder_x, shoulder_y, shoulder_z = shoulder_world()

        dx = world_x - shoulder_x
        dy = world_y - shoulder_y
        dz = world_z - shoulder_z

        return dx, dy, dz, (world_x, world_y, world_z)

    def solve_planar_ik(self, horizontal, dz):
        upper = self.upper_len
        lower = self.forearm_len + self.wrist_len + self.tool_len

        distance = math.sqrt((horizontal * horizontal) + (dz * dz))
        max_reach = max(self.min_reach, upper + lower - self.reach_margin)
        min_reach = max(self.min_reach, abs(upper - lower) + self.reach_margin)

        if distance > max_reach:
            scale = max_reach / max(distance, 1e-6)
            horizontal *= scale
            dz *= scale
            distance = max_reach
        elif distance < min_reach:
            scale = min_reach / max(distance, 1e-6)
            horizontal *= scale
            dz *= scale
            distance = min_reach

        cos_elbow = (
            (distance * distance)
            - (upper * upper)
            - (lower * lower)
        ) / (2.0 * upper * lower)
        cos_elbow = self.clamp(cos_elbow, -1.0, 1.0)

        elbow_math = math.acos(cos_elbow)
        shoulder_math = (
            math.atan2(dz, max(horizontal, 1e-6))
            - math.atan2(
                lower * math.sin(elbow_math),
                upper + lower * math.cos(elbow_math)
            )
        )

        joint_1 = -shoulder_math
        joint_2 = -elbow_math

        joint_1 = self.clamp(
            joint_1,
            getattr(self.robot, "JOINT_1_MIN", -1.25),
            getattr(self.robot, "JOINT_1_MAX", 1.25)
        )

        joint_2 = self.clamp(
            joint_2,
            getattr(self.robot, "JOINT_2_MIN", -2.2),
            getattr(self.robot, "JOINT_2_MAX", 2.2)
        )

        joint_3 = self.clamp(
            0.0,
            getattr(self.robot, "JOINT_3_MIN", -1.4),
            getattr(self.robot, "JOINT_3_MAX", 1.4)
        )

        return joint_1, joint_2, joint_3

    def find_target(self, intent, scene_data):
        target = intent.get("target", None)

        if target is not None and "center" in target:
            return target

        target_id = intent.get("target_id", None)
        if target_id is not None:
            for obj in scene_data.get("objects", []):
                if obj.get("id") == target_id:
                    return obj

        if hasattr(self.robot, "find_best_object"):
            user_text = str(intent.get("user_message", ""))
            best = self.robot.find_best_object(user_text, scene_data)
            if best is not None:
                return best

        objects = scene_data.get("objects", [])
        if len(objects) > 0:
            return objects[0]

        return None

    def build_action(self, target):
        dx, dy, dz, world_target = self.world_to_shoulder_direction(target)

        # Robot base is rotated 90 degrees: local +X points toward world +Y.
        joint_0 = math.atan2(-dx, dy)
        joint_0 = self.clamp(
            joint_0,
            getattr(self.robot, "JOINT_0_MIN", self.YAW_MIN),
            getattr(self.robot, "JOINT_0_MAX", self.YAW_MAX)
        )

        horizontal = math.sqrt((dx * dx) + (dy * dy))
        joint_1, joint_2, joint_3 = self.solve_planar_ik(
            horizontal,
            dz
        )

        joint_4 = 0.0

        return {
            "joint_0": joint_0,
            "joint_1": joint_1,
            "joint_2": joint_2,
            "joint_3": joint_3,
            "joint_4": joint_4,
            "steps": self.steps,
            "target_world": world_target,
            "target_distance": math.sqrt(
                (dx * dx) + (dy * dy) + (dz * dz)
            ),
        }

    def execute(self, intent, scene_data):
        print("=" * 60)
        print(self.label)
        print("=" * 60)

        target = self.find_target(intent, scene_data)

        if target is None:
            print("NO TARGET")
            action = {
                "joint_0": getattr(self.robot, "current_joint_0", 0.0),
                "joint_1": getattr(self.robot, "current_joint_1", 0.0),
                "joint_2": getattr(self.robot, "current_joint_2", 0.0),
                "joint_3": getattr(self.robot, "current_joint_3", 0.0),
                "joint_4": getattr(self.robot, "current_joint_4", 0.0),
                "steps": 60,
            }
            print("ACTION:", action)
            return self.dispatch(action)

        action = self.build_action(target)

        print("TARGET:", target)
        print("ACTION:", action)

        self.robot.current_joint_0 = action["joint_0"]
        self.robot.current_joint_1 = action["joint_1"]
        self.robot.current_joint_2 = action["joint_2"]
        self.robot.current_joint_3 = action["joint_3"]
        self.robot.current_joint_4 = action["joint_4"]

        if hasattr(self.robot, "current_state"):
            self.robot.current_state = {
                "joint_0": action["joint_0"],
                "joint_1": action["joint_1"],
                "joint_2": action["joint_2"],
                "joint_3": action["joint_3"],
                "joint_4": action["joint_4"],
            }

        if hasattr(self.robot, "target_state"):
            self.robot.target_state = dict(self.robot.current_state)

        return self.dispatch(action)

    def dispatch(self, action):
        if hasattr(self.robot, "execute_action") and callable(
                getattr(self.robot, "execute_action")
        ):
            try:
                result = self.robot.execute_action(action)
                return result if result is not None else action
            except TypeError:
                pass

        pybullet = getattr(self.robot, "pybullet", None)
        if pybullet is not None and hasattr(pybullet, "execute_action"):
            pybullet.execute_action(action)
            return action

        return action
