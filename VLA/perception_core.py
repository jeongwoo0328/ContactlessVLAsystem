# =========================================================
# perception_core.py (극단적 초고속 최적화 버전)
# =========================================================

import os
import cv2
import json
import torch
import numpy as np
import open3d as o3d

from PIL import Image
from scipy.ndimage import gaussian_filter
from transformers import pipeline, AutoProcessor, AutoModel
from ultralytics import SAM

class Vision3DModule:
    # =====================================================
    # INIT
    # =====================================================
    def __init__(self):
        print("=" * 60)
        print("INITIALIZING EXTRA FAST VISION ENGINE")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("DEVICE:", self.device)

        if self.device == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        else:
            torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
            torch.set_grad_enabled(False)

        cv2.setNumThreads(max(1, min(8, os.cpu_count() or 1)))

        self.cpu_only = self.device == "cpu"
        # 초고속 처리를 위해 기본 해상도를 한 단계 최적화 (필요시 환경변수로 조절 가능)
        self.resize_w = int(os.getenv("VLA_RESIZE_W", "640")) if self.cpu_only else int(os.getenv("VLA_RESIZE_W", "800"))
        self.point_step = int(os.getenv("VLA_POINT_STEP", "3")) if self.cpu_only else int(os.getenv("VLA_POINT_STEP", "2"))
        self.depth_scale = 12.0
        self.max_objects = int(os.getenv("VLA_MAX_OBJECTS", "15")) if self.cpu_only else int(os.getenv("VLA_MAX_OBJECTS", "24"))
        self.min_mask_area = int(os.getenv("VLA_MIN_MASK_AREA", "1000"))
        
        # 하이브리드 병합 임계값
        self.merge_iou_thresh = 0.4
        self.merge_dist_thresh = 0.5
        self.merge_depth_thresh = 0.3

        self.voxel_size = 0.03   # Voxel 크기를 살짝 키워 Scene PCD 속도 향상
        self.normal_radius = 0.12
        self.mesh_depth = int(os.getenv("VLA_MESH_DEPTH", "7"))
        self.outlier_neighbors = 20
        self.outlier_std = 2.0
        self.floor_y = 0.0
        
        # 메쉬는 속도를 위해 완전히 비활성화 (환경변수로만 켬)
        self.enable_mesh = os.getenv("VLA_ENABLE_MESH", "0").strip().lower() in ("1", "true", "yes", "on")
        self.siglip_batch_size = int(os.getenv("VLA_SIGLIP_BATCH", "16")) # 배치 사이즈 확장
        
        # 경량화 모델 로드 유도 (Depth Anything V2 Small로 변경하여 속도 3배 이상 향상 가능)
        self.depth_model_name = os.getenv(
            "VLA_DEPTH_MODEL",
            "depth-anything/Depth-Anything-V2-Small-hf"
        )

        print("CPU PROFILE:", "ON" if self.cpu_only else "OFF")
        print("RESIZE WIDTH:", self.resize_w)
        print("DEPTH MODEL:", self.depth_model_name)

        print("=" * 60)
        print("LOADING MODELS")
        self.depth_pipe = pipeline(task="depth-estimation", model=self.depth_model_name, device=0 if torch.cuda.is_available() else -1)
        self.sam_model = SAM("mobile_sam.pt")
        self.siglip_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.siglip_model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(self.device).eval()

        self.semantic_labels = ["paper", "book", "notebook", "keyboard", "mouse", "monitor", "laptop", "phone", "cable", "box", "container", "bottle", "cup", "electronic device", "office item"]
        self.siglip_text_features = self.encode_siglip_texts(self.semantic_labels)
        print("VISION ENGINE READY")

    @torch.inference_mode()
    def encode_siglip_texts(self, labels):
        text_inputs = self.siglip_processor(text=labels, padding=True, return_tensors="pt")
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        text_features = self.siglip_model.get_text_features(**text_inputs)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def get_dominant_color(self, crop):
        if crop.size == 0: return "unknown"
        mean = crop.mean(axis=(0, 1))
        r, g, b = mean
        if r > 180 and g > 180 and b > 180: return "white"
        if r < 60 and g < 60 and b < 60: return "black"
        if abs(r - g) < 15 and abs(g - b) < 15: return "gray"
        if r > g and r > b: return "red"
        if g > r and g > b: return "green"
        if b > r and b > g: return "blue"
        return "mixed"

    @torch.inference_mode()
    def get_visual_semantics_batch(self, crops):
        if len(crops) == 0: return []
        labels_by_crop = []
        for start in range(0, len(crops), self.siglip_batch_size):
            batch = crops[start:start + self.siglip_batch_size]
            try:
                images = [Image.fromarray(crop) for crop in batch if crop.size != 0]
                if len(images) != len(batch):
                    labels_by_crop.extend([["unknown"] for _ in batch])
                    continue
                image_inputs = self.siglip_processor(images=images, return_tensors="pt").to(self.device)
                image_features = self.siglip_model.get_image_features(**image_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ self.siglip_text_features.T).cpu().numpy()
                top_indices = np.argsort(similarity, axis=1)[:, -3:][:, ::-1]
                for row in top_indices:
                    labels_by_crop.append([self.semantic_labels[int(idx)] for idx in row])
            except Exception as e:
                print("SIGLIP BATCH ERROR:", e)
                labels_by_crop.extend([["unknown"] for _ in batch])
        return labels_by_crop

    def refine_depth(self, depth):
        # 조금 더 빠른 연산을 위해 시그마 축소 및 다운스케일 정제
        depth = gaussian_filter(depth, sigma=1.5)
        depth = np.power(depth, 1.4)
        depth -= depth.min()
        depth /= (depth.max() + 1e-8)
        return depth

    def build_point_cloud(self, rgb, depth):
        H, W = depth.shape
        fx = fy = W * 1.25
        cx, cy = W / 2, H / 2
        step = max(1, int(self.point_step))
        depth_view = depth[::step, ::step]
        rgb_view = rgb[::step, ::step]
        Hs, Ws = depth_view.shape
        xs, ys = np.meshgrid(np.arange(0, W, step)[:Ws], np.arange(0, H, step)[:Hs])
        xs, ys, z = xs.flatten(), ys.flatten(), depth_view.flatten()
        valid = z > 0.02
        xs, ys, z = xs[valid], ys[valid], z[valid]
        z = np.sqrt(z) * self.depth_scale
        x = (xs - cx) * z / fx
        y = -(ys - cy) * z / fy
        return np.stack([x, y, z], axis=1), rgb_view.reshape(-1, 3)[valid] / 255.0

    def build_mesh(self, pcd):
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_radius, max_nn=30))
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=self.mesh_depth)
        vertices_to_remove = np.asarray(densities) < np.quantile(np.asarray(densities), 0.03)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        mesh.compute_vertex_normals()
        return mesh

    # =====================================================
    # MERGE OVERLAPPING MASKS (초고속 하이브리드 병합 엔진 V2)
    # =====================================================
    def merge_overlapping_masks(self, raw_masks, depth, W, H):
        valid_masks = []
        fx = fy = W * 1.25
        cx, cy = W / 2, H / 2

        for mask in raw_masks:
            # 0.5 임계값 처리를 단일 바이트 불리언 마스크화
            binary = mask > 0.5
            area = binary.sum()
            if area < self.min_mask_area:
                continue
                
            ys, xs = np.where(binary)
            if len(xs) < 100: 
                continue
            
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            
            # 다운샘플링 스텝을 5로 확장하여 연산 최적화 극대화
            step = 5
            xs_sub, ys_sub = xs[::step], ys[::step]
            
            z = depth[ys_sub, xs_sub]
            valid = z > 0.02
            if len(z[valid]) < 10: 
                continue
                
            xs_v, ys_v, z_v = xs_sub[valid], ys_sub[valid], z[valid]
            zz = np.sqrt(z_v) * self.depth_scale
            
            xx = (xs_v - cx) * zz / fx
            yy = -(ys_v - cy) * zz / fy
            pts = np.stack([xx, yy, zz], axis=1)
            
            valid_masks.append({
                'binary': binary.astype(np.uint8),
                'area': area,
                'center': pts.mean(axis=0),
                'min_z': float(zz.min()),
                'pts': pts,
                'xs': xs, # 원본 매핑용 인덱스는 그대로 보존
                'ys': ys,
                'bbox': (x1, y1, x2, y2)
            })

        valid_masks = sorted(valid_masks, key=lambda x: x['area'], reverse=True)
        num_masks = len(valid_masks)
        visited = np.zeros(num_masks, dtype=bool)
        final_objects_data = []

        for i in range(num_masks):
            if visited[i]:
                continue
                
            base = valid_masks[i]
            current_binary = base['binary'].copy()
            current_pts = [base['pts']]
            current_xs = [base['xs']]
            current_ys = [base['ys']]
            bx1, by1, bx2, by2 = base['bbox']
            visited[i] = True
            
            for j in range(i + 1, num_masks):
                if visited[j]:
                    continue
                    
                target = valid_masks[j]
                tx1, ty1, tx2, ty2 = target['bbox']
                
                # Bounding Box 충돌 필터
                if not (bx1 <= tx2 and bx2 >= tx1 and by1 <= ty2 and by2 >= ty1):
                    continue
                
                # ROI 로컬 IoU 매칭 가속화
                ix1, iy1 = max(bx1, tx1), max(by1, ty1)
                ix2, iy2 = min(bx2, tx2), min(by2, ty2)
                inter_area = np.logical_and(
                    current_binary[iy1:iy2+1, ix1:ix2+1], 
                    target['binary'][iy1:iy2+1, ix1:ix2+1]
                ).sum()
                iou = inter_area / (target['area'] + 1e-8)
                
                dist_3d = np.linalg.norm(base['center'] - target['center'])
                depth_diff = abs(base['min_z'] - target['min_z'])

                if iou > self.merge_iou_thresh or (dist_3d < self.merge_dist_thresh and depth_diff < self.merge_depth_thresh):
                    current_binary = np.logical_or(current_binary, target['binary']).astype(np.uint8)
                    current_pts.append(target['pts'])
                    current_xs.append(target['xs'])
                    current_ys.append(target['ys'])
                    visited[j] = True
                    bx1, by1 = min(bx1, tx1), min(by1, ty1)
                    bx2, by2 = max(bx2, tx2), max(by2, ty2)
            
            combined_pts = np.vstack(current_pts)
            combined_xs = np.concatenate(current_xs)
            combined_ys = np.concatenate(current_ys)
            
            min_xyz = combined_pts.min(axis=0)
            max_xyz = combined_pts.max(axis=0)
            
            final_objects_data.append({
                'points': combined_pts,
                'xs': combined_xs,
                'ys': combined_ys,
                'center': ((min_xyz + max_xyz) / 2.0).tolist(),
                'size': (max_xyz - min_xyz).tolist(),
                'x1': bx1, 'y1': by1, 'x2': bx2, 'y2': by2
            })
            
        return final_objects_data

    # =====================================================
    # PROCESS IMAGE
    # =====================================================
    def process_image(self, image_path, output_dir="output"):
        os.makedirs(output_dir, exist_ok=True)

        image_bgr = cv2.imread(image_path)
        if image_bgr is None: raise FileNotFoundError(image_path)

        h0, w0 = image_bgr.shape[:2]
        scale = self.resize_w / w0
        new_w, new_h = self.resize_w, int(h0 * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        H, W = image_rgb.shape[:2]

        # Depth 추론 (inference_mode 적용)
        pil = Image.fromarray(image_rgb)
        with torch.inference_mode():
            depth_result = self.depth_pipe(pil)
        depth = np.array(depth_result["depth"]).astype(np.float32)
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_CUBIC)
        depth -= depth.min()
        depth /= (depth.max() + 1e-8)
        depth = self.refine_depth(depth)

        # Scene Point Cloud 빌드 및 내부 최적화
        points, colors = self.build_point_cloud(image_rgb, depth)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        o3d.io.write_point_cloud(os.path.join(output_dir, "scene.ply"), pcd)

        if self.enable_mesh:
            mesh = self.build_mesh(pcd)
            o3d.io.write_triangle_mesh(os.path.join(output_dir, "scene_mesh.ply"), mesh)

        # SAM 매개변수 최적화 (retina_masks=False 로 마스크 생성 가속화)
        with torch.inference_mode():
            sam_results = self.sam_model(image_rgb, verbose=False, retina_masks=False)

        raw_masks = []
        for result in sam_results:
            if result.masks is not None:
                raw_masks.extend(result.masks.data.cpu().numpy())

        merged_objects_data = self.merge_overlapping_masks(raw_masks, depth, W, H)

        objects = []
        semantic_crops = []

        # 객체별 색상값 일괄 획득을 위해 정규화 사전 연산
        normalized_img = image_rgb / 255.0

        for object_index, obj_data in enumerate(merged_objects_data[:self.max_objects]):
            x1, y1, x2, y2 = obj_data['x1'], obj_data['y1'], obj_data['x2'], obj_data['y2']
            crop = image_rgb[y1:y2, x1:x2]
            dominant_color = self.get_dominant_color(crop)

            pcd_file = os.path.join(output_dir, f"object_{object_index}.ply")
            obj_pcd = o3d.geometry.PointCloud()
            obj_pcd.points = o3d.utility.Vector3dVector(obj_data['points'])
            
            # 매핑 인덱스 다운샘플링으로 파일 저장 I/O 크기 최적화
            ys_sub, xs_sub = obj_data['ys'][::4], obj_data['xs'][::4]
            obj_pcd.colors = o3d.utility.Vector3dVector(normalized_img[ys_sub, xs_sub])
            o3d.io.write_point_cloud(pcd_file, obj_pcd)

            obj_summary = {
                "id": object_index,
                "center": obj_data['center'],
                "size": obj_data['size'],
                "depth": float(obj_data['center'][2]),
                "point_cloud_path": pcd_file,
                "mesh_path": None,
                "dominant_color": dominant_color,
                "semantic_labels": [],
                "crop_box": [x1, y1, x2, y2]
            }
            objects.append(obj_summary)
            semantic_crops.append(crop)

        # Batch Semantic 라벨 매핑 (SigLIP 일괄 처리)
        semantic_results = self.get_visual_semantics_batch(semantic_crops)
        for obj, labels in zip(objects, semantic_results):
            obj["semantic_labels"] = labels

        camera_pose = {"fx": W * 1.25, "fy": W * 1.25, "cx": W / 2, "cy": H / 2}
        scene_data = {
            "point_cloud_path": os.path.join(output_dir, "scene.ply"),
            "mesh_path": os.path.join(output_dir, "scene_mesh.ply") if self.enable_mesh else None,
            "depth_path": os.path.join(output_dir, "depth.png"),
            "objects": objects,
            "camera_pose": camera_pose,
            "image_width": W,
            "image_height": H,
            "coordinate_system": {
                "vision": "x=right, y=up, z=forward",
                "pybullet": "x=right, y=forward, z=up",
                "camera_mount": "overhead robot-excluded camera"
            }
        }

        with open(os.path.join(output_dir, "scene.json"), "w", encoding="utf-8") as f:
            json.dump(scene_data, f, indent=4, ensure_ascii=False)

        return scene_data

    def get_on_demand_object_info(self, target_object_id, all_objects):
        target_obj = next((obj for obj in all_objects if obj.get("id") == target_object_id), None)
        if not target_obj:
            return f"\n[실시간 알림] ID {target_object_id}번에 해당하는 물체를 작업 공간에서 찾을 수 없습니다.\n"
            
        info_feed = f"\n[실시간 수혈된 ID {target_object_id}번 물체의 상세 시각 데이터]\n"
        for key, value in target_obj.items():
            info_feed += f"- {key}: {value}\n"
        return info_feed