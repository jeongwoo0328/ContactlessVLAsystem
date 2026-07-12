class MotionSkill:
    """Preset motions that do not require a target object."""

    def __init__(self, robot):
        self.robot = robot

    def current(self):
        return {
            "joint_0": float(getattr(self.robot, "current_joint_0", 0.0)),
            "joint_1": float(getattr(self.robot, "current_joint_1", 0.0)),
            "joint_2": float(getattr(self.robot, "current_joint_2", 0.0)),
            "joint_3": float(getattr(self.robot, "current_joint_3", 0.0)),
            "joint_4": float(getattr(self.robot, "current_joint_4", 0.0)),
        }

    def pose(self, j0, j1, j2, j3, j4, steps=80):
        return {
            "joint_0": float(j0),
            "joint_1": float(j1),
            "joint_2": float(j2),
            "joint_3": float(j3),
            "joint_4": float(j4),
            "steps": int(steps),
        }

    def sequence(self, actions):
        return {
            "type": "sequence",
            "actions": actions,
        }

    def execute(self, intent, scene_data):
        print("=" * 60)
        print("MOTION SKILL")
        print("=" * 60)

        motion = str(intent.get("motion", "home")).lower()
        current = self.current()

        if motion == "wave":
            j0 = current["joint_0"]
            return self.sequence([
                self.pose(j0, -0.75, -1.15, -0.20, -0.75, 90),
                self.pose(j0, -0.75, -1.15, -0.20, 0.75, 45),
                self.pose(j0, -0.75, -1.15, -0.20, -0.75, 45),
                self.pose(j0, -0.75, -1.15, -0.20, 0.75, 45),
                self.pose(j0, -0.45, -0.80, -0.10, 0.0, 70),
            ])

        if motion == "scan":
            return self.sequence([
                self.pose(-0.85, -0.25, -0.70, -0.10, 0.0, 90),
                self.pose(0.85, -0.25, -0.70, -0.10, 0.0, 120),
                self.pose(0.0, -0.20, -0.55, -0.10, 0.0, 80),
            ])

        if motion == "extend":
            return self.pose(
                current["joint_0"],
                -0.05,
                -0.20,
                0.0,
                0.0,
                100
            )

        if motion == "fold":
            return self.pose(
                current["joint_0"],
                -0.70,
                -1.55,
                0.60,
                0.0,
                100
            )

        return self.pose(
            0.0,
            0.20,
            -0.55,
            -0.10,
            0.0,
            100
        )
