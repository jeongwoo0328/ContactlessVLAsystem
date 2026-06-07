import os
import cv2
import torch
import open3d as o3d
import numpy as np

from PIL import Image
from scipy.ndimage import gaussian_filter

from transformers import pipeline
from ultralytics import SAM

# =========================================================
# SETTINGS
# =========================================================

IMAGE_PATH = r"C:\Users\0328j\OneDrive\Desktop\IMG_2560.jpg"

OUTPUT_DIR = r"C:\Users\0328j\OneDrive\Desktop\지각모델\output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =========================================================
# QUALITY SETTINGS
# =========================================================

RESIZE_W = 768

POINT_STEP = 2

MAX_OBJECTS = 20

MIN_MASK_AREA = 3000

DEPTH_SCALE = 16.0

# perspective 강도
PERSPECTIVE_BLEND = 0.18

# 바닥 기울기 보정
GROUND_TILT = 0.32

# 근거리 억제
NEAR_ATTENUATION = 0.55

# =========================================================
# LOAD IMAGE
# =========================================================

print("=" * 60)
print("DEVICE:", DEVICE)

image_bgr = cv2.imread(IMAGE_PATH)

if image_bgr is None:
    raise FileNotFoundError(
        f"IMAGE NOT FOUND:\n{IMAGE_PATH}"
    )

# =========================================================
# RESIZE
# =========================================================

h0, w0 = image_bgr.shape[:2]

scale = RESIZE_W / w0

new_w = RESIZE_W
new_h = int(h0 * scale)

image_bgr = cv2.resize(
    image_bgr,
    (new_w, new_h),
    interpolation=cv2.INTER_AREA
)

image_rgb = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB
)

H, W = image_rgb.shape[:2]

print("IMAGE:", W, "x", H)

# =========================================================
# LOAD DEPTH MODEL
# =========================================================

print("=" * 60)
print("LOADING DEPTH MODEL")

depth_pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Large-hf",
    device=0 if torch.cuda.is_available() else -1
)

# =========================================================
# DEPTH ESTIMATION
# =========================================================

print("=" * 60)
print("PREDICTING DEPTH")

pil_image = Image.fromarray(image_rgb)

depth_result = depth_pipe(pil_image)

depth = np.array(
    depth_result["depth"]
).astype(np.float32)

depth = cv2.resize(
    depth,
    (W, H),
    interpolation=cv2.INTER_CUBIC
)

# =========================================================
# NORMALIZE
# =========================================================

depth -= depth.min()

depth /= (
    depth.max() + 1e-8
)

# =========================================================
# DEPTH REFINEMENT
# =========================================================

print("=" * 60)
print("REFINING DEPTH")

global_depth = gaussian_filter(
    depth,
    sigma=7
)

detail = depth - gaussian_filter(
    depth,
    sigma=1.2
)

depth = (
    global_depth * 0.90
    +
    detail * 0.10
)

depth = gaussian_filter(
    depth,
    sigma=0.8
)

# =========================================================
# DEPTH CURVE STABILIZATION
# =========================================================

depth = np.power(
    depth,
    1.6
)

depth -= depth.min()

depth /= (
    depth.max() + 1e-8
)

# =========================================================
# SAVE DEPTH
# =========================================================

depth_img = (
    depth * 255
).astype(np.uint8)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "depth.png"
    ),
    depth_img
)

print("DEPTH SAVED")

# =========================================================
# LOAD SAM
# =========================================================

print("=" * 60)
print("LOADING SAM")

sam_model = SAM(
    "mobile_sam.pt"
)

# =========================================================
# SEGMENT
# =========================================================

print("=" * 60)
print("SEGMENTING")

sam_results = sam_model(
    image_rgb,
    verbose=False
)

# =========================================================
# STORAGE
# =========================================================

scene_points = []
scene_colors = []

objects = []

# =========================================================
# CAMERA
# =========================================================

fx = W * 1.45
fy = W * 1.45

cx = W / 2
cy = H / 2

# =========================================================
# PROCESS OBJECTS
# =========================================================

print("=" * 60)
print("BUILDING 3D")

object_index = 0

for result in sam_results:

    if result.masks is None:
        continue

    masks = result.masks.data.cpu().numpy()

    for mask in masks:

        if object_index >= MAX_OBJECTS:
            break

        binary = (
            mask > 0.5
        ).astype(np.uint8)

        area = binary.sum()

        if area < MIN_MASK_AREA:
            continue

        ys, xs = np.where(binary > 0)

        if len(xs) < 100:
            continue

        # =================================================
        # FAST SAMPLING
        # =================================================

        sample_idx = np.arange(
            0,
            len(xs),
            POINT_STEP
        )

        xs = xs[sample_idx]
        ys = ys[sample_idx]

        # =================================================
        # DEPTH
        # =================================================

        z = depth[ys, xs]

        valid = z > 0.03

        xs = xs[valid]
        ys = ys[valid]
        z = z[valid]

        if len(z) < 50:
            continue

        # =================================================
        # STABLE DEPTH
        # =================================================

        zz = np.sqrt(z)

        zz *= DEPTH_SCALE

        # =================================================
        # NEAR ATTENUATION
        # =================================================

        near_factor = np.clip(
            zz / 6.0,
            0.0,
            1.0
        )

        near_factor = (
            near_factor * (1.0 - NEAR_ATTENUATION)
            + NEAR_ATTENUATION
        )

        zz *= near_factor

        # =================================================
        # PERSPECTIVE
        # =================================================

        persp_x = (
            (xs - cx)
            * zz
            / fx
        )

        persp_y = (
            -(ys - cy)
            * zz
            / fy
        )

        # =================================================
        # ORTHOGRAPHIC
        # =================================================

        ortho_x = (
            (xs - cx)
            / W
        ) * 18.0

        ortho_y = (
            -(ys - cy)
            / H
        ) * 18.0

        # =================================================
        # DEPTH-BASED BLEND
        # =================================================

        dynamic_blend = (
            PERSPECTIVE_BLEND
            * np.clip(zz / 8.0, 0.2, 1.0)
        )

        xx = (
            persp_x * dynamic_blend
            +
            ortho_x * (1.0 - dynamic_blend)
        )

        yy = (
            persp_y * dynamic_blend
            +
            ortho_y * (1.0 - dynamic_blend)
        )

        # =================================================
        # GROUND TILT FIX
        # =================================================

        yy -= zz * GROUND_TILT

        # =================================================
        # POINTS
        # =================================================

        object_points = np.stack(
            [xx, yy, zz],
            axis=1
        )

        # =================================================
        # COLORS
        # =================================================

        object_colors = (
            image_rgb[ys, xs] / 255.0
        )

        scene_points.append(object_points)
        scene_colors.append(object_colors)

        # =================================================
        # BBOX
        # =================================================

        min_xyz = object_points.min(axis=0)
        max_xyz = object_points.max(axis=0)

        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_xyz,
            max_xyz
        )

        bbox.color = (1, 0, 0)

        objects.append(bbox)

        object_index += 1

        print(
            "OBJECT:",
            object_index,
            "| POINTS:",
            len(object_points)
        )

# =========================================================
# CONCAT
# =========================================================

if len(scene_points) == 0:
    raise RuntimeError(
        "NO POINTS GENERATED"
    )

scene_points = np.concatenate(
    scene_points,
    axis=0
)

scene_colors = np.concatenate(
    scene_colors,
    axis=0
)

print("=" * 60)
print("TOTAL POINTS:", len(scene_points))

# =========================================================
# POINT CLOUD
# =========================================================

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(
    scene_points
)

pcd.colors = o3d.utility.Vector3dVector(
    scene_colors
)

# =========================================================
# DOWNSAMPLE
# =========================================================

print("=" * 60)
print("OPTIMIZING CLOUD")

pcd = pcd.voxel_down_sample(
    voxel_size=0.03
)

# =========================================================
# OUTLIER REMOVE
# =========================================================

pcd, _ = pcd.remove_statistical_outlier(
    nb_neighbors=20,
    std_ratio=2.0
)

# =========================================================
# SAVE
# =========================================================

scene_ply = os.path.join(
    OUTPUT_DIR,
    "scene_fixed_ground.ply"
)

o3d.io.write_point_cloud(
    scene_ply,
    pcd
)

print("=" * 60)
print("PLY SAVED")

# =========================================================
# VISUALIZER
# =========================================================

print("=" * 60)
print("OPENING VIEWER")

vis = o3d.visualization.Visualizer()

vis.create_window(
    window_name="FIXED GROUND 3D",
    width=1600,
    height=1000
)

vis.add_geometry(pcd)

for bbox in objects:
    vis.add_geometry(bbox)

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=5
)

vis.add_geometry(axis)

opt = vis.get_render_option()

opt.point_size = 2.0
opt.light_on = True

ctr = vis.get_view_control()

ctr.set_zoom(0.72)

print("=" * 60)
print("LEFT DRAG  : ROTATE")
print("RIGHT DRAG : MOVE")
print("WHEEL      : ZOOM")

vis.run()

vis.destroy_window()

print("=" * 60)
print("FINISHED")
print("OUTPUT:", OUTPUT_DIR)
print("=" * 60)