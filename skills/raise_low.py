# =========================================================
# skills/raise_low.py
# 5-DOF raise/lower skill.
# =========================================================


class RaiseLowSkill:

    def __init__(self, robot):
        self.robot = robot
        self.pitch_delta = 0.35

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def execute(self, intent, scene_data):
        print("=" * 60)
        print("RAISE/LOW SKILL")
        print("=" * 60)

        direction = str(intent.get("direction", "raise")).lower()
        current_pitch = float(getattr(self.robot, "current_joint_1", 0.0))
        current_elbow = float(getattr(self.robot, "current_joint_2", 0.0))
        current_wrist = float(getattr(self.robot, "current_joint_3", 0.0))

        if direction in ["low", "lower", "down"]:
            target_pitch = current_pitch + self.pitch_delta
            target_elbow = current_elbow + 0.20
            target_wrist = current_wrist + 0.10
        else:
            target_pitch = current_pitch - self.pitch_delta
            target_elbow = current_elbow - 0.25
            target_wrist = current_wrist - 0.10

        action = {
            "joint_0": float(getattr(self.robot, "current_joint_0", 0.0)),
            "joint_1": self.clamp(
                target_pitch,
                getattr(self.robot, "JOINT_1_MIN", -1.5),
                getattr(self.robot, "JOINT_1_MAX", 1.5)
            ),
            "joint_2": self.clamp(
                target_elbow,
                getattr(self.robot, "JOINT_2_MIN", -2.2),
                getattr(self.robot, "JOINT_2_MAX", 2.2)
            ),
            "joint_3": self.clamp(
                target_wrist,
                getattr(self.robot, "JOINT_3_MIN", -1.4),
                getattr(self.robot, "JOINT_3_MAX", 1.4)
            ),
            "joint_4": float(getattr(self.robot, "current_joint_4", 0.0)),
            "steps": 100,
        }

        return action
