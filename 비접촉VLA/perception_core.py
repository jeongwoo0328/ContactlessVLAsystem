# =========================================================
# perception_core.py
# =========================================================

import os
import cv2
import json
import torch
import numpy as np
import open3d as o3d

from PIL import Image

from scipy.ndimage import (
    gaussian_filter
)

from transformers import (
    pipeline,
    AutoProcessor,
    AutoModel
)

from ultralytics import SAM

# =========================================================
# CLASS
# =========================================================

class Vision3DModule:

    # =====================================================
    # INIT
    # =====================================================

    def __init__(self):

        print("=" * 60)
        print("INITIALIZING VISION ENGINE")

        # =================================================
        # DEVICE
        # =================================================

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print("DEVICE:", self.device)

        if self.device == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        else:
            torch.set_num_threads(
                max(1, min(8, os.cpu_count() or 1))
            )
            torch.set_grad_enabled(False)

        cv2.setNumThreads(
            max(1, min(8, os.cpu_count() or 1))
        )

        # =================================================
        # SETTINGS
        # =================================================

        self.cpu_only = self.device == "cpu"

        self.resize_w = (
            int(os.getenv("VLA_RESIZE_W", "768"))
            if self.cpu_only
            else int(os.getenv("VLA_RESIZE_W", "960"))
        )

        self.point_step = (
            int(os.getenv("VLA_POINT_STEP", "2"))
            if self.cpu_only
            else int(os.getenv("VLA_POINT_STEP", "1"))
        )

        self.depth_scale = 12.0

        self.max_objects = (
            int(os.getenv("VLA_MAX_OBJECTS", "18"))
            if self.cpu_only
            else int(os.getenv("VLA_MAX_OBJECTS", "32"))
        )

        self.min_mask_area = (
            int(os.getenv("VLA_MIN_MASK_AREA", "1200"))
            if self.cpu_only
            else int(os.getenv("VLA_MIN_MASK_AREA", "1500"))
        )

        self.voxel_size = 0.025

        self.normal_radius = 0.12

        self.mesh_depth = (
            int(os.getenv("VLA_MESH_DEPTH", "7"))
            if self.cpu_only
            else int(os.getenv("VLA_MESH_DEPTH", "8"))
        )

        self.outlier_neighbors = 30

        self.outlier_std = 1.8

        self.floor_y = 0.0

        self.enable_mesh = os.getenv(
            "VLA_ENABLE_MESH",
            "0" if self.cpu_only else "1"
        ).strip().lower() in ("1", "true", "yes", "on")

        self.parallel_workers = max(
            1,
            min(8, os.cpu_count() or 1)
        )

        self.siglip_batch_size = int(os.getenv("VLA_SIGLIP_BATCH", "8"))
        self.depth_model_name = os.getenv(
            "VLA_DEPTH_MODEL",
            (
                "depth-anything/Depth-Anything-V2-Base-hf"
                if self.cpu_only
                else "depth-anything/Depth-Anything-V2-Large-hf"
            )
        )

        print("CPU PROFILE:", "ON" if self.cpu_only else "OFF")
        print("RESIZE WIDTH:", self.resize_w)
        print("POINT STEP:", self.point_step)
        print("MAX OBJECTS:", self.max_objects)
        print("MESH:", "ON" if self.enable_mesh else "OFF")
        print("DEPTH MODEL:", self.depth_model_name)

        # =================================================
        # DEPTH
        # =================================================

        print("=" * 60)
        print("LOADING DEPTH MODEL")

        self.depth_pipe = pipeline(

            task="depth-estimation",

            model=self.depth_model_name,

            device=0
            if torch.cuda.is_available()
            else -1
        )

        # =================================================
        # SAM
        # =================================================

        print("=" * 60)
        print("LOADING MOBILE SAM")

        self.sam_model = SAM(
            "mobile_sam.pt"
        )

        # =================================================
        # SIGLIP
        # =================================================

        print("=" * 60)
        print("LOADING SIGLIP")

        self.siglip_processor = (
            AutoProcessor.from_pretrained(
                "google/siglip-base-patch16-224"
            )
        )

        self.siglip_model = (
            AutoModel.from_pretrained(
                "google/siglip-base-patch16-224"
            )
        )

        self.siglip_model.to(
            self.device
        )

        self.siglip_model.eval()

        # =================================================
        # LABELS
        # =================================================

        self.semantic_labels = [

            "paper",
            "book",
            "notebook",
            "keyboard",
            "mouse",
            "monitor",
            "laptop",
            "phone",
            "cable",
            "box",
            "container",
            "bottle",
            "cup",
            "electronic device",
            "office item"
        ]

        self.siglip_text_features = self.encode_siglip_texts(
            self.semantic_labels
        )

        print("=" * 60)
        print("VISION ENGINE READY")

    # =====================================================
    # SIGLIP TEXT CACHE
    # =====================================================

    @torch.inference_mode()
    def encode_siglip_texts(
            self,
            labels
    ):

        text_inputs = (
            self.siglip_processor(
                text=labels,
                padding=True,
                return_tensors="pt"
            )
        )

        text_inputs = {
            k: v.to(self.device)
            for k, v in text_inputs.items()
        }

        text_features = (
            self.siglip_model
            .get_text_features(
                **text_inputs
            )
        )

        text_features = (
            text_features
            / text_features.norm(
                dim=-1,
                keepdim=True
            )
        )

        return text_features

    # =====================================================
    # DOMINANT COLOR
    # =====================================================

    def get_dominant_color(
            self,
            crop
    ):

        if crop.size == 0:
            return "unknown"

        mean = crop.mean(axis=(0, 1))

        r, g, b = mean

        if r > 180 and g > 180 and b > 180:
            return "white"

        if r < 60 and g < 60 and b < 60:
            return "black"

        if abs(r - g) < 15 and abs(g - b) < 15:
            return "gray"

        if r > g and r > b:
            return "red"

        if g > r and g > b:
            return "green"

        if b > r and b > g:
            return "blue"

        return "mixed"

    # =====================================================
    # VISUAL SEMANTICS
    # =====================================================

    def get_visual_semantics(
            self,
            crop
    ):

        try:

            pil = Image.fromarray(
                crop
            )

            image_inputs = (
                self.siglip_processor(
                    images=pil,
                    return_tensors="pt"
                )
            )

            image_inputs = {

                k: v.to(self.device)
                for k, v in image_inputs.items()
            }

            with torch.no_grad():

                image_features = (
                    self.siglip_model
                    .get_image_features(
                        **image_inputs
                    )
                )

            image_features = (
                image_features
                / image_features.norm(
                    dim=-1,
                    keepdim=True
                )
            )

            similarity = (
                image_features
                @ self.siglip_text_features.T
            )

            similarity = (
                similarity[0]
                .cpu()
                .numpy()
            )

            top_indices = np.argsort(
                similarity
            )[-3:][::-1]

            labels = []

            for idx in top_indices:

                labels.append(
                    self.semantic_labels[idx]
                )

            return labels

        except Exception as e:

            print("SIGLIP ERROR:", e)

            return ["unknown"]

    # =====================================================
    # VISUAL SEMANTICS BATCH
    # =====================================================

    @torch.inference_mode()
    def get_visual_semantics_batch(
            self,
            crops
    ):

        if len(crops) == 0:
            return []

        labels_by_crop = []

        for start in range(0, len(crops), self.siglip_batch_size):
            batch = crops[start:start + self.siglip_batch_size]

            try:
                images = [
                    Image.fromarray(crop)
                    for crop in batch
                    if crop.size != 0
                ]

                if len(images) != len(batch):
                    labels_by_crop.extend([["unknown"] for _ in batch])
                    continue

                image_inputs = self.siglip_processor(
                    images=images,
                    return_tensors="pt"
                )

                image_inputs = {
                    k: v.to(self.device)
                    for k, v in image_inputs.items()
                }

                image_features = self.siglip_model.get_image_features(
                    **image_inputs
                )
                image_features = image_features / image_features.norm(
                    dim=-1,
                    keepdim=True
                )

                similarity = (
                    image_features
                    @ self.siglip_text_features.T
                ).cpu().numpy()

                top_indices = np.argsort(similarity, axis=1)[:, -3:][:, ::-1]

                for row in top_indices:
                    labels_by_crop.append([
                        self.semantic_labels[int(idx)]
                        for idx in row
                    ])

            except Exception as e:
                print("SIGLIP BATCH ERROR:", e)
                labels_by_crop.extend([["unknown"] for _ in batch])

        return labels_by_crop

    # =====================================================
    # REFINE DEPTH
    # =====================================================

    def refine_depth(
            self,
            depth
    ):

        depth = gaussian_filter(
            depth,
            sigma=2.0
        )

        global_depth = gaussian_filter(
            depth,
            sigma=8
        )

        detail = (
            depth
            - gaussian_filter(
                depth,
                sigma=1.5
            )
        )

        depth = (
            global_depth * 0.92
            + detail * 0.08
        )

        depth = np.power(
            depth,
            1.4
        )

        depth -= depth.min()

        depth /= (
            depth.max()
            + 1e-8
        )

        return depth

    # =====================================================
    # BUILD POINT CLOUD
    # =====================================================

    def build_point_cloud(
            self,
            rgb,
            depth
    ):

        H, W = depth.shape

        fx = W * 1.25
        fy = W * 1.25

        cx = W / 2
        cy = H / 2

        step = max(1, int(self.point_step))

        depth_view = depth[::step, ::step]
        rgb_view = rgb[::step, ::step]

        Hs, Ws = depth_view.shape

        xs, ys = np.meshgrid(

            np.arange(0, W, step)[:Ws],
            np.arange(0, H, step)[:Hs]
        )

        xs = xs.flatten()
        ys = ys.flatten()

        z = depth_view.flatten()

        valid = z > 0.02

        xs = xs[valid]
        ys = ys[valid]
        z = z[valid]

        z = np.sqrt(z)

        z *= self.depth_scale

        x = (
            (xs - cx)
            * z
            / fx
        )

        y = (
            -(ys - cy)
            * z
            / fy
        )

        points = np.stack(

            [x, y, z],

            axis=1
        )

        colors = (
            rgb_view.reshape(-1, 3)[valid]
            / 255.0
        )

        return points, colors

    # =====================================================
    # BUILD MESH
    # =====================================================

    def build_mesh(
            self,
            pcd
    ):

        print("=" * 60)
        print("RECONSTRUCTING SURFACE")

        pcd.estimate_normals(

            search_param=
            o3d.geometry.KDTreeSearchParamHybrid(

                radius=self.normal_radius,

                max_nn=40
            )
        )

        mesh, densities = (
            o3d.geometry.TriangleMesh
            .create_from_point_cloud_poisson(

                pcd,

                depth=self.mesh_depth
            )
        )

        densities = np.asarray(
            densities
        )

        vertices_to_remove = (
            densities
            < np.quantile(
                densities,
                0.03
            )
        )

        mesh.remove_vertices_by_mask(
            vertices_to_remove
        )

        mesh.compute_vertex_normals()

        return mesh

    # =====================================================
    # PROCESS IMAGE
    # =====================================================

    def process_image(
            self,
            image_path,
            output_dir="output"
    ):

        print("=" * 60)
        print("PROCESSING IMAGE")

        os.makedirs(

            output_dir,

            exist_ok=True
        )

        # =================================================
        # LOAD
        # =================================================

        image_bgr = cv2.imread(
            image_path
        )

        if image_bgr is None:

            raise FileNotFoundError(
                image_path
            )

        h0, w0 = image_bgr.shape[:2]

        scale = (
            self.resize_w / w0
        )

        new_w = self.resize_w

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

        # =================================================
        # DEPTH
        # =================================================

        print("=" * 60)
        print("PREDICTING DEPTH")

        pil = Image.fromarray(
            image_rgb
        )

        with torch.inference_mode():
            depth_result = (
                self.depth_pipe(
                    pil
                )
            )

        depth = np.array(

            depth_result["depth"]

        ).astype(np.float32)

        depth = cv2.resize(

            depth,

            (W, H),

            interpolation=cv2.INTER_CUBIC
        )

        depth -= depth.min()

        depth /= (
            depth.max()
            + 1e-8
        )

        depth = self.refine_depth(
            depth
        )

        # =================================================
        # SAVE DEPTH
        # =================================================

        depth_img = (
            depth * 255
        ).astype(np.uint8)

        depth_path = os.path.join(

            output_dir,

            "depth.png"
        )

        cv2.imwrite(
            depth_path,
            depth_img
        )

        # =================================================
        # POINT CLOUD
        # =================================================

        print("=" * 60)
        print("BUILDING POINT CLOUD")

        points, colors = (
            self.build_point_cloud(
                image_rgb,
                depth
            )
        )

        pcd = o3d.geometry.PointCloud()

        pcd.points = (
            o3d.utility.Vector3dVector(
                points
            )
        )

        pcd.colors = (
            o3d.utility.Vector3dVector(
                colors
            )
        )

        # =================================================
        # FILTER
        # =================================================

        pcd = pcd.voxel_down_sample(

            voxel_size=self.voxel_size
        )

        pcd, _ = (
            pcd.remove_statistical_outlier(

                nb_neighbors=
                self.outlier_neighbors,

                std_ratio=
                self.outlier_std
            )
        )

        # =================================================
        # NORMALS
        # =================================================

        if self.enable_mesh:
            pcd.estimate_normals(

                search_param=
                o3d.geometry.KDTreeSearchParamHybrid(

                    radius=0.12,

                    max_nn=40
                )
            )

        # =================================================
        # SAVE PCD
        # =================================================

        pcd_path = os.path.join(

            output_dir,

            "scene.ply"
        )

        o3d.io.write_point_cloud(

            pcd_path,

            pcd
        )

        # =================================================
        # MESH
        # =================================================

        mesh_path = None

        if self.enable_mesh:

            mesh = self.build_mesh(
                pcd
            )

            mesh_path = os.path.join(

                output_dir,

                "scene_mesh.ply"
            )

            o3d.io.write_triangle_mesh(

                mesh_path,

                mesh
            )

        # =================================================
        # SEGMENTATION
        # =================================================

        print("=" * 60)
        print("SEGMENTING OBJECTS")

        with torch.inference_mode():
            sam_results = (
                self.sam_model(

                    image_rgb,

                    verbose=False
                )
            )

        objects = []
        semantic_crops = []

        object_index = 0

        for result in sam_results:

            if result.masks is None:
                continue

            masks = (
                result.masks.data
                .cpu()
                .numpy()
            )

            for mask in masks:

                if (
                    object_index
                    >= self.max_objects
                ):
                    break

                binary = (
                    mask > 0.5
                ).astype(np.uint8)

                area = binary.sum()

                if (
                    area
                    < self.min_mask_area
                ):
                    continue

                ys, xs = np.where(
                    binary > 0
                )

                if len(xs) < 100:
                    continue

                z = depth[ys, xs]

                valid = z > 0.02

                xs = xs[valid]
                ys = ys[valid]
                z = z[valid]

                if len(z) < 50:
                    continue

                zz = np.sqrt(z)

                zz *= self.depth_scale

                fx = W * 1.25
                fy = W * 1.25

                cx = W / 2
                cy = H / 2

                xx = (
                    (xs - cx)
                    * zz
                    / fx
                )

                yy = (
                    -(ys - cy)
                    * zz
                    / fy
                )

                object_points = np.stack(

                    [xx, yy, zz],

                    axis=1
                )

                min_xyz = (
                    object_points.min(axis=0)
                )

                max_xyz = (
                    object_points.max(axis=0)
                )

                center = (
                    (
                        min_xyz
                        + max_xyz
                    ) / 2.0
                )

                size = (
                    max_xyz
                    - min_xyz
                )

                x1 = max(0, xs.min())
                y1 = max(0, ys.min())

                x2 = min(W, xs.max())
                y2 = min(H, ys.max())

                crop = image_rgb[
                    y1:y2,
                    x1:x2
                ]

                dominant_color = (
                    self.get_dominant_color(
                        crop
                    )
                )

                obj = {

                    "id":
                        object_index,

                    "center":
                        center.tolist(),

                    "size":
                        size.tolist(),

                    "depth":
                        float(center[2]),

                    "dominant_color":
                        dominant_color,

                    "semantic_labels":
                        [],

                    "crop_box":
                        [
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2)
                        ]
                }

                objects.append(obj)
                semantic_crops.append(crop)

                print(
                    "OBJECT:",
                    object_index,
                    dominant_color
                )

                object_index += 1

        semantic_results = self.get_visual_semantics_batch(
            semantic_crops
        )

        for obj, labels in zip(objects, semantic_results):
            obj["semantic_labels"] = labels
            print("LABEL:", obj["id"], labels)

        # =================================================
        # SPATIAL GRAPH
        # =================================================

        def build_relation(pair):
            a, b = pair

            if a["id"] == b["id"]:
                return None

            ax, ay, az = a["center"]
            bx, by, bz = b["center"]

            dx = bx - ax
            dy = by - ay
            dz = bz - az

            distance = float(
                np.linalg.norm([
                    dx,
                    dy,
                    dz
                ])
            )

            return {
                "from": a["id"],
                "to": b["id"],
                "distance": distance,
                "left": bool(dx < 0),
                "right": bool(dx > 0),
                "above": bool(dy > 0),
                "below": bool(dy < 0),
                "front": bool(dz < 0),
                "behind": bool(dz > 0)
            }

        spatial_graph = []

        for a in objects:
            for b in objects:
                relation = build_relation((a, b))
                if relation is not None:
                    spatial_graph.append(relation)

        # =================================================
        # CAMERA
        # =================================================

        camera_pose = {

            "fx":
                W * 1.25,

            "fy":
                W * 1.25,

            "cx":
                W / 2,

            "cy":
                H / 2
        }

        # =================================================
        # FINAL
        # =================================================

        scene_data = {

            "point_cloud_path":
                pcd_path,

            "mesh_path":
                mesh_path,

            "depth_path":
                depth_path,

            "objects":
                objects,

            "spatial_graph":
                spatial_graph,

            "camera_pose":
                camera_pose,

            "image_width":
                W,

            "image_height":
                H,

            "coordinate_system":
                {
                    "vision":
                        "x=right, y=up, z=forward",
                    "pybullet":
                        "x=right, y=forward, z=up",
                    "camera_mount":
                        "overhead robot-excluded camera"
                }
        }

        # =================================================
        # SAVE JSON
        # =================================================

        with open(

                os.path.join(
                    output_dir,
                    "scene.json"
                ),

                "w",

                encoding="utf-8"

        ) as f:

            json.dump(

                scene_data,

                f,

                indent=4,

                ensure_ascii=False
            )

        print("=" * 60)
        print("VISION PROCESS COMPLETE")

        return scene_data
