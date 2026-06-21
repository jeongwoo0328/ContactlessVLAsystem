# =========================================================
# skills/idle.py
# =========================================================


class IdleSkill:

    def __init__(self, robot):
        self.robot = robot

    def execute(
        self,
        intent,
        scene_data
    ):

        print("=" * 60)
        print("IDLE SKILL")
        print("=" * 60)

        action = {
            "joint_0": 0.0,
            "joint_1": 0.20,
            "joint_2": -0.55,
            "joint_3": -0.10,
            "joint_4": 0.0,
            "steps": 80,
        }

        self.robot.current_joint_0 = 0.0
        self.robot.current_joint_1 = 0.20
        self.robot.current_joint_2 = -0.55
        self.robot.current_joint_3 = -0.10
        self.robot.current_joint_4 = 0.0

        if hasattr(self.robot, "current_state"):
            self.robot.current_state["joint_0"] = 0.0
            self.robot.current_state["joint_1"] = 0.20
            self.robot.current_state["joint_2"] = -0.55
            self.robot.current_state["joint_3"] = -0.10
            self.robot.current_state["joint_4"] = 0.0

        if hasattr(self.robot, "target_state"):
            self.robot.target_state = dict(self.robot.current_state)

        return action
