ARM_UPPER_LEN = 0.72
ARM_FOREARM_LEN = 0.56
ARM_WRIST_LEN = 0.19
ARM_TOOL_LEN = 0.13
ARM_TOTAL_LEN = (
    ARM_UPPER_LEN
    + ARM_FOREARM_LEN
    + ARM_WRIST_LEN
    + ARM_TOOL_LEN
)

BASE_JOINT_HEIGHT = 0.25
SHOULDER_LINK_HEIGHT = 0.18

SCENE_ORIGIN = [0.0, 0.0, 0.0]
OBJECT_SCENE_OFFSET = [0.9, 12.5, 0.0]
ROBOT_BASE_POSITION = [0.0, -0.75, 0.0]
SHOULDER_HEIGHT = BASE_JOINT_HEIGHT + SHOULDER_LINK_HEIGHT
MIN_OBJECT_SIZE = 0.04


def vision_to_pybullet_world(center, size=None, shifted=False):
    size = size or [0.08, 0.08, 0.08]

    vision_x = float(center[0])
    vision_y_up = float(center[1])
    vision_z_forward = float(center[2])

    world_x = SCENE_ORIGIN[0] + vision_x
    world_y = SCENE_ORIGIN[1] - vision_z_forward
    world_z = max(
        MIN_OBJECT_SIZE,
        SCENE_ORIGIN[2]
        + vision_y_up
        + max(float(size[1]) * 0.5, MIN_OBJECT_SIZE)
    )

    if shifted:
        world_x += OBJECT_SCENE_OFFSET[0]
        world_y += OBJECT_SCENE_OFFSET[1]
        world_z += OBJECT_SCENE_OFFSET[2]

    return [world_x, world_y, world_z]


def shoulder_world():
    return [
        ROBOT_BASE_POSITION[0],
        ROBOT_BASE_POSITION[1],
        ROBOT_BASE_POSITION[2] + SHOULDER_HEIGHT,
    ]
