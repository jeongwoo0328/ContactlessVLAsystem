# =========================================================
# phi_reasoning_module.py
# COGNITIVE VISION ARCHITECTURE (Few-Shot Context Routing & Prompt Guard)
# WITH INTEGRATED SPATIAL MEMORY, MEMORY MANAGER & SPEED OPTIMIZATION
# =========================================================

import json
import os
import re
import torch
from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from skills.look import LookSkill
from skills.point import PointSkill
from skills.idle import IdleSkill
from skills.raise_low import RaiseLowSkill
from skills.motion import MotionSkill

from semantic_analyzer import SemanticAnalyzer
from memory_manager import MemoryManager

# =========================================================
# CLASS DEFINITION
# =========================================================

class AttentionRobot:

    # 인지 및 물리 제어를 위한 유효 경로 설정
    VALID_ROUTES = (
        "chat",
        "vision_chat",
        "skill",
        "vision_skill"
    )

    ROUTE_PRIORITY = {
        "chat": 0,
        "vision_chat": 1,
        "skill": 2,
        "vision_skill": 3
    }

    # =====================================================
    # INITIALIZATION (초기화 및 가속 튜닝)
    # =====================================================

    def __init__(self, pybullet=None, auto_execute=False):

        print("=" * 60)
        print("INITIALIZING ATTENTION ROBOT (SPEED & COGNITIVE HYBRID VERSION)")

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print("DEVICE:", self.device)

        # 고성능 텐서 연산 최적화 활성화
        torch.set_float32_matmul_precision("high")

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        # Microsoft Phi-3 모델 로드
        model_name = "microsoft/Phi-3-mini-4k-instruct"

        print("=" * 60)
        print("LOADING PHI-3 INSTRUCT MODEL")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 시스템 프롬프트 잘림 방지 (좌측 트런케이션 적용)
        self.tokenizer.truncation_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=(
                torch.float16
                if self.device == "cuda"
                else torch.float32
            ),
            device_map="auto",
            low_cpu_mem_usage=True
        )

        self.model.eval()

        # 로봇 하드웨어 파라미터 및 관절 제약 설정
        self.FIXED_JOINT_1 = 0.0
        self.JOINT_KEYS = [
            "joint_0",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4"
        ]

        self.JOINT_0_MIN = -2.5
        self.JOINT_0_MAX = 2.5

        self.JOINT_1_MIN = -1.8
        self.JOINT_1_MAX = 2.6

        self.JOINT_2_MIN = -3.1
        self.JOINT_2_MAX = 3.1

        self.JOINT_3_MIN = -2.8
        self.JOINT_3_MAX = 2.8

        self.JOINT_4_MIN = -1.8
        self.JOINT_4_MAX = 1.8

        self.joint_limits = {
            "joint_0": (self.JOINT_0_MIN, self.JOINT_0_MAX),
            "joint_1": (self.JOINT_1_MIN, self.JOINT_1_MAX),
            "joint_2": (self.JOINT_2_MIN, self.JOINT_2_MAX),
            "joint_3": (self.JOINT_3_MIN, self.JOINT_3_MAX),
            "joint_4": (self.JOINT_4_MIN, self.JOINT_4_MAX)
        }

        self.camera_yaw_offset = 0.0
        self.servo_direction = 1.0
        self.yaw_scale = 1.0
        self.smooth_alpha = 0.7
        self.max_history_turns = 8
        self.use_ai_routing = os.getenv(
            "VLA_AI_ROUTING",
            "1"
        ).strip().lower() in ("1", "true", "yes", "on")

        # 관절 상태 변수 초기화
        self.current_joint_0 = 0.0
        self.current_joint_1 = self.FIXED_JOINT_1
        self.current_joint_2 = 0.0
        self.current_joint_3 = 0.0
        self.current_joint_4 = 0.0

        self.current_state = {
            "joint_0": self.current_joint_0,
            "joint_1": self.current_joint_1,
            "joint_2": self.current_joint_2,
            "joint_3": self.current_joint_3,
            "joint_4": self.current_joint_4
        }

        self.target_joint_0 = 0.0
        self.target_joint_1 = self.FIXED_JOINT_1
        self.target_joint_2 = 0.0
        self.target_joint_3 = 0.0
        self.target_joint_4 = 0.0

        self.target_state = dict(self.current_state)
        self.conversation_history = []
        self.last_target_object_id = None
        self.last_route = None

        # 공간 및 장기 기억 관리자 인스턴스 연동
        self.memory_manager = MemoryManager()
        self.object_memory = {}  

        # 시뮬레이터 및 하드웨어 액션 브릿지
        self.pybullet = pybullet
        self.auto_execute = auto_execute
        self.last_action = None

        # 기본 스킬 및 별칭 정의
        self.look_skill = LookSkill(self)
        self.point_skill = PointSkill(self)
        self.idle_skill = IdleSkill(self)
        self.raise_low_skill = RaiseLowSkill(self)
        self.motion_skill = MotionSkill(self)

        self.skills = {
            "look": self.look_skill,
            "point": self.point_skill,
            "idle": self.idle_skill,
            "raise_low": self.raise_low_skill,
            "motion": self.motion_skill
        }

        self.skill_aliases = {
            "look": "look",
            "look_at_object": "look",
            "point": "point",
            "point_at_object": "point",
            "raise": "raise_low",
            "lower": "raise_low",
            "raise_low": "raise_low",
            "wave": "motion",
            "scan": "motion",
            "home": "motion",
            "rest": "motion",
            "extend": "motion",
            "fold": "motion",
            "motion": "motion",
            "idle": "idle"
        }

        # 시동 워밍업 실행 (추론 엔진 초기 지연 최소화)
        print("=" * 60)
        print("MODEL WARMUP RUN")
        try:
            self.generate("You are ready.", "hello", 4)
        except Exception as e:
            print("WARMUP ERROR:", e)

        self.semantic_analyzer = SemanticAnalyzer()
        print("=" * 60)
        print("COGNITIVE ATTENTION ROBOT READY")

    # =====================================================
    # PYBULLET BINDING
    # =====================================================

    def attach_pybullet(self, pybullet):
        self.pybullet = pybullet

    # =====================================================
    # SPATIAL OBJECT MEMORY TRACKING & FUSION (공간 기억 유지)
    # =====================================================

    def update_and_recall_objects(self, scene_data):
        scene_data = scene_data or {}
        current_objects = scene_data.get("objects", [])
        current_time = datetime.now().strftime("%H:%M:%S")

        # 실시간 좌표계 연산 및 기억 버퍼 등록
        for obj in current_objects:
            obj_id = obj.get("id")
            if obj_id is not None:
                cx, cy, cz = self.get_calibrated_coordinates(obj.get("center", [0.0, 0.0, 0.0]))
                self.object_memory[obj_id] = {
                    "id": obj_id,
                    "semantic_labels": obj.get("semantic_labels", ["object"]),
                    "dominant_color": obj.get("dominant_color", "unknown"),
                    "center": obj.get("center", [0.0, 0.0, 0.0]),
                    "calibrated_position": [cx, cy, cz],
                    "last_seen": current_time,
                    "status": "visible"
                }

        # 시야(Camera FOV)를 벗어난 객체 상태 갱신
        current_ids = {obj.get("id") for obj in current_objects if obj.get("id") is not None}
        for obj_id, obj_info in self.object_memory.items():
            if obj_id not in current_ids:
                self.object_memory[obj_id]["status"] = "out_of_view"

        # 지속 보존된 객체와 현행 탐지 객체를 병합하여 반환
        fused_objects = []
        for obj_id, obj_info in self.object_memory.items():
            fused_objects.append({
                "id": obj_info["id"],
                "semantic_labels": obj_info["semantic_labels"],
                "dominant_color": obj_info["dominant_color"],
                "center": obj_info["center"],
                "status": obj_info["status"],
                "last_seen": obj_info["last_seen"]
            })

        return {"objects": fused_objects}

    # =====================================================
    # CONVERSATION HISTORY & LONG-TERM SYNC (대화 백업)
    # =====================================================

    def get_conversation_history(self, limit=None):
        if limit is None:
            return list(self.conversation_history)
        return self.conversation_history[-int(limit):]

    def clear_conversation_history(self):
        self.conversation_history = []
        self.memory_manager.clear_recent_memory()

    def remember_turn(self, user_message, assistant_message):
        self.conversation_history.append({
            "user": str(user_message),
            "assistant": str(assistant_message)
        })
        if len(self.conversation_history) > self.max_history_turns:
            self.conversation_history = self.conversation_history[-self.max_history_turns:]

        try:
            self.memory_manager.add_memory(str(user_message), str(assistant_message))
        except Exception as e:
            print(f"[LONG MEMORY SYNC EXCEPTION] {e}")

    def build_history_text(self, memory_context=""):
        lines = []
        if memory_context:
            lines.append(str(memory_context).strip())

        for turn in self.conversation_history[-self.max_history_turns:]:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Robot: {turn['assistant']}")

        if len(lines) == 0:
            return "No prior conversation."

        return "\n".join(lines)

    # =====================================================
    # COORDINATE SYSTEM CALIBRATION (하드웨어 오차 역산)
    # =====================================================

    def get_calibrated_coordinates(self, raw_center):
        try:
            camera_x = float(raw_center[0])
            camera_y = float(raw_center[2])
            camera_z = float(raw_center[1])

            calibrated_x = camera_x
            calibrated_y = -camera_z
            calibrated_z = camera_y

            return round(calibrated_x, 3), round(calibrated_z, 3), round(calibrated_y, 3)
        except Exception:
            return round(raw_center[0], 3), round(raw_center[2], 3), round(raw_center[1], 3)

    # =====================================================
    # FEATURE ANALYZER (구조 의미 체계 연동 분석)
    # =====================================================

    def get_request_features(self, user_message, scene_data=None):
        text = self.normalize_text(user_message)
        scene_data = scene_data or {}

        # 시맨틱 의미소 토큰화 및 최종 벡터 계산 연동
        token_stream = self.semantic_analyzer.correct_and_convert(text)
        semantic_vector = self.semantic_analyzer.analyze_sentence(text)
        atom_map = self.semantic_analyzer.atoms

        # 행동 주도 원소 분석
        action_atoms = {
            atom_map.get("DO"), atom_map.get("MOVE"), atom_map.get("THINK"), 
            atom_map.get("SEE"), atom_map.get("HEAR"), atom_map.get("WANT")
        }
        has_action = any(atom in token_stream for atom in action_atoms if atom is not None)
        
        # 특정 서술 패턴 차단 보정
        if "look like" in text or "looks like" in text:
            has_action = any(atom in token_stream for atom in (action_atoms - {atom_map.get("SEE")}) if atom is not None)

        # 의문 제기 요소 분석
        question_atoms = {atom_map.get("WHERE"), atom_map.get("QUESTION")}
        has_question = any(atom in token_stream for atom in question_atoms if atom is not None)

        # 대명사 참조소 지칭 분석
        reference_atoms = {
            atom_map.get("THIS"), atom_map.get("SOMETHING"), 
            atom_map.get("HERE"), atom_map.get("THEREIS")
        }
        has_reference = any(atom in token_stream for atom in reference_atoms if atom is not None)

        has_id = re.search(r"\bid\s*[:=]?\s*(\d+)\b", text) is not None
        has_objects = bool(scene_data.get("objects"))

        scene_terms = []
        for obj in scene_data.get("objects", []):
            scene_terms.extend(
                [str(x).lower() for x in obj.get("semantic_labels", [])]
            )
            color = obj.get("dominant_color", "")
            if color:
                scene_terms.append(str(color).lower())

        has_scene_term = any(term in text for term in scene_terms if term)

        return {
            "text": text,
            "has_action": has_action,
            "has_question": has_question,
            "has_reference": has_reference,
            "has_id": has_id,
            "has_objects": has_objects,
            "has_scene_term": has_scene_term,
            "semantic_vector": semantic_vector
        }

    # =====================================================
    # TEXT & JSON ROBUST EXTRACTION (예외 복구 파서)
    # =====================================================

    def normalize_text(self, text):
        text = str(text).lower().strip()
        text = text.replace("’", "'")
        text = text.replace("“", '"')
        text = text.replace("”", '"')
        return text

    def clean_response(self, text):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        stop_strings = ["\nUSER", "\nUser", "\nAssistant", "\nHuman"]
        for s in stop_strings:
            if s in text:
                text = text.split(s)[0]
        return text.strip()

    def robust_extract_json(self, text, default_dict=None):
        if default_dict is None:
            default_dict = {}

        start = text.find("{")
        if start != -1:
            brace_count = 0
            end = None
            for i in range(start, len(text)):
                if text[i] == '{': brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            
            json_text = text[start:end] if end else text[start:]
            
            # 불안전하게 끊긴 LLM 괄호 자동 복원 수행
            if end is None:
                if json_text.count('"') % 2 != 0:
                    json_text += '"'
                open_braces = json_text.count('{')
                close_braces = json_text.count('}')
                if open_braces > close_braces:
                    json_text += "}" * (open_braces - close_braces)
                
            try:
                return json.loads(json_text)
            except Exception:
                pass

        # 완전 손상 상태에 대응한 정규식 복원
        result = dict(default_dict)
        
        route_match = re.search(r'"routes?"\s*:\s*(?:\[\s*"([^"]+)"\s*\]|"([^"]+)")', text, re.IGNORECASE)
        if route_match:
            val = route_match.group(1) or route_match.group(2)
            result["routes"] = [val] if "routes" in default_dict or "route" not in default_dict else val
            if "route" in default_dict:
                result["route"] = val
            
        intent_match = re.search(r'"intent"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
        if intent_match: result["intent"] = intent_match.group(1)
            
        target_match = re.search(r'"target_id"\s*:\s*(\d+)', text, re.IGNORECASE)
        if target_match: result["target_id"] = int(target_match.group(1))

        dir_match = re.search(r'"direction"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
        if dir_match: result["direction"] = dir_match.group(1)

        motion_match = re.search(r'"motion"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
        if motion_match: result["motion"] = motion_match.group(1)

        return result

    # =====================================================
    # INFERENCE ENGINE (최속화 배치 설정)
    # =====================================================

    @torch.inference_mode()
    def generate(self, system_prompt, user_prompt, max_new_tokens=40):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512  # 속도 단축을 위한 컨텍스트 압축
        )
        try:
            target_device = next(self.model.parameters()).device
        except Exception:
            target_device = self.device

        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=2,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.05,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=False,
            use_cache=True
        )
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return self.clean_response(response)

    # =====================================================
    # LIGHTWEIGHT PROMPT ENGINES (초경량 프롬프트 설계)
    # =====================================================

    def generate_lightweight_prompt(self, user_command, scene_data):
        """
        무겁고 복잡한 실측 좌표를 완전히 소거하여 토큰 연산 속도를 비약적으로 단축함.
        오직 객체의 ID와 핵심 명칭만을 노출하여 빠른 라우팅과 초기 예측을 유도합니다.
        """
        object_list = ""
        for obj in scene_data.get("objects", []):
            labels = obj.get("semantic_labels", [])
            label_text = "/".join([str(x) for x in labels[:3]]) if labels else "object"
            status_text = f" [{obj.get('status', 'visible')}]" if "status" in obj else ""
            object_list += f"- ID {obj.get('id')}: {label_text} ({obj.get('dominant_color', 'unknown')}){status_text}\n"
            
        prompt = f"""
Currently detected workspace objects:
{object_list}
Current History:
{self.build_history_text()}

Instruction: "{user_command}"
"""
        return prompt

    def build_scene_text(self, scene_data):
        scene_data = scene_data or {}
        lines = []
        objects = scene_data.get("objects", [])
        for obj in objects[:8]:
            cx, cy, cz = self.get_calibrated_coordinates(obj["center"])
            labels = obj.get("semantic_labels", [])
            label = "/".join([str(x) for x in labels[:3]]) if len(labels) > 0 else "object"
            color = obj.get("dominant_color", "unknown")
            status_text = f" status={obj.get('status', 'visible')}"
            lines.append(
                f"id={obj['id']} type={label} color={color} position=(left_right:{cx}, front_back:{cy}, height:{cz}){status_text}"
            )
        if len(lines) == 0:
            return "No visible or memorized objects."
        return "\n".join(lines)

    # =====================================================
    # HIGH-SPEED CONTEXT ROUTERS (규칙 및 온디맨드 추론 하이브리드)
    # =====================================================

    def normalize_routes(self, route_data):
        if isinstance(route_data, dict):
            raw_routes = route_data.get("routes", route_data.get("route", "chat"))
        else:
            raw_routes = route_data

        if isinstance(raw_routes, str):
            raw_routes = [part.strip() for part in raw_routes.split(",")]

        if not isinstance(raw_routes, list):
            raw_routes = ["chat"]

        routes = []
        for route in raw_routes:
            route = str(route).strip().lower()
            
            if any(k in route for k in ["skill", "point", "look", "motion", "raise", "lower", "wave", "scan"]):
                route = "vision_skill"
            elif any(k in route for k in ["chat", "talk", "describe", "answer"]):
                route = "vision_chat"

            if route in self.VALID_ROUTES and route not in routes:
                routes.append(route)

        if len(routes) == 0:
            routes = ["chat"]

        routes.sort(key=lambda name: self.ROUTE_PRIORITY.get(name, 99))
        return routes

    def route_has_skill(self, routes):
        return any(route in ["skill", "vision_skill"] for route in self.normalize_routes(routes))

    def build_router_context(self, user_message, scene_data=None, previous_routes=None):
        scene_data = scene_data or {}
        features = self.get_request_features(user_message, scene_data)
        referenced = self.resolve_reference_target(user_message, scene_data)

        context = {
            "user_message": str(user_message),
            "scene": self.build_scene_text(scene_data),
            "features": {
                "has_action_word": features["has_action"],
                "has_question_form": features["has_question"],
                "has_reference_word": features["has_reference"],
                "has_visible_objects": features["has_objects"],
                "mentions_visible_object": features["has_scene_term"],
                "mentions_object_id": features["has_id"]
            },
            "previous_referenced_object": self.summarize_object(referenced),
            "previous_routes": previous_routes
        }
        return json.dumps(context, ensure_ascii=False, indent=2)

    def summarize_object(self, obj):
        if not isinstance(obj, dict):
            return None
        cx, cy, cz = self.get_calibrated_coordinates(obj.get("center", [0.0, 0.0, 0.0]))
        return {
            "id": obj.get("id"),
            "labels": [str(label) for label in obj.get("semantic_labels", [])[:3]],
            "color": str(obj.get("dominant_color", "")),
            "calibrated_center": [cx, cy, cz]
        }

    def route_from_rules(self, user_message, scene_data=None):
        """
        초고속 로컬 규칙 필터링. 
        인사, 감사, 지시어의 조합을 사전 판정하여 불필요한 LLM 추론을 생략합니다.
        눈앞에 물체가 관측되면 일반 인사도 vision_chat으로 연결하여 Embodied 대화를 가능하게 만듭니다.
        """
        text = self.normalize_text(user_message)
        scene_data = scene_data or {}

        greetings = ["hello", "hi", "hey", "안녕", "하이", "thanks", "thank you", "고마워", "고맙", "bye", "goodbye", "잘가"]
        action_words = ["point", "look at", "point to", "face", "turn", "rotate", "봐", "보여", "가리켜", "집어", "향해", "돌아", "올려", "내려", "위로", "아래로"]
        question_words = ["where", "what", "which", "closest", "nearest", "어디", "뭐", "무엇", "어느", "가장", "가까운", "위치"]

        features = self.get_request_features(user_message, scene_data)
        has_action = any(w in text for w in action_words) or features["has_action"]
        has_question = any(w in text for w in question_words) or features["has_question"]
        has_id = features["has_id"]
        has_objects = features["has_objects"]

        if any(g in text for g in greetings) and not has_action and not has_id:
            return ["vision_chat"] if has_objects else ["chat"]

        if has_question and (has_action or has_id):
            return ["vision_chat", "vision_skill"] if has_objects else ["chat", "skill"]

        if has_question and not has_action:
            return ["vision_chat"] if has_objects else ["chat"]

        if has_action or has_id:
            return ["vision_skill"] if has_objects else ["skill"]

        return None

    def classify_request(self, user_message, scene_data=None):
        """
        규칙 기반 처리를 1순위로 두고, 분기가 모호할 때만 Chain-of-Thought 추론을 기동함.
        """
        # 1. 0ms 초고속 로컬 규칙 라우팅 판정
        rule_route = self.route_from_rules(user_message, scene_data)
        if rule_route is not None:
            routes = self.normalize_routes(rule_route)
            self.last_route = routes
            return routes

        # 2. 복잡도가 높은 쿼리인 경우 예외적으로 AI CoT 구동
        system_prompt = """
You are the cognitive brain of a robot arm.
Decide the route in strict JSON. Do NOT output conversational text.

Available Routes: "chat", "vision_chat", "skill", "vision_skill"
"""
        router_context = self.build_router_context(user_message, scene_data)
        
        # Restore CoT structure for high classification accuracy
        few_shot_prompt = """
[EXAMPLES]
Context:
{
  "user_message": "What color is the block next to it?"
}
Output:
{
  "reasoning": "User is asking a question about object properties, which does not require movement.",
  "route": "vision_chat"
}

Context:
{
  "user_message": "can you point it?"
}
Output:
{
  "reasoning": "User commands pointing, which is a physical motor skill.",
  "route": "vision_skill"
}

Now analyze the following real runtime context:
Context:
""" + router_context + "\nOutput:\n"

        try:
            response = self.generate(system_prompt, few_shot_prompt, max_new_tokens=60)
            data = self.robust_extract_json(response, default_dict={"route": "chat"})
            chosen_route = data.get("route", "chat")

            # 안전 가드 보정 (키워드 유무 체크)
            text_lower = self.normalize_text(user_message)
            chat_keywords = ["tell me", "what", "how", "describe", "why", "who", "where", "list", "are there"]
            skill_keywords = ["point", "look", "show", "move", "raise", "lower", "wave", "scan"]
            
            if any(k in text_lower for k in chat_keywords) and not any(k in text_lower for k in skill_keywords):
                chosen_route = "vision_chat" if scene_data and scene_data.get("objects") else "chat"
            
            routes = self.normalize_routes(chosen_route)
            self.last_route = routes
            return routes

        except Exception:
            self.last_route = ["chat"]
            return ["chat"]

    # =====================================================
    # CHAT GENERATORS
    # =====================================================

    def normal_chat(self, user_message, memory_context=""):
        system_prompt = "You are a real robot arm. Speak naturally, politely, and briefly (2-4 sentences max)."
        user_prompt = f"History:\n{self.build_history_text(memory_context)}\nUser: {user_message}"
        response = self.generate(system_prompt, user_prompt, max_new_tokens=80)
        return self.filter_bad_response(response)

    def vision_chat(self, user_message, scene_data, memory_context=""):
        scene_text = self.build_scene_text(scene_data)
        system_prompt = """
You are a real autonomous robot manipulator arm. You can physically observe and track objects in front of you.
Speak warmly, naturally, and briefly (2-3 sentences max).
When the user says hello, asks a casual question, or starts a simple chit-chat, respond naturally and feel free to casually mention one interesting object you see in front of you (without mentioning raw coordinates unless asked) to show you are paying attention.
Keep a good balance: you don't need to describe everything you see, just maintain natural interaction with your surroundings.
Do not mention that you are an AI or a language model.
"""
        user_prompt = f"History:\n{self.build_history_text(memory_context)}\nScene:\n{scene_text}\nUser: {user_message}"
        response = self.generate(system_prompt, user_prompt, max_new_tokens=100)
        return self.filter_bad_response(response)

    def ask_question(self, user_message, scene_data=None, memory_context="", route=None):
        scene_data = self.update_and_recall_objects(scene_data)

        if not memory_context:
            try:
                memory_context = self.memory_manager.build_memory_context(user_message)
            except Exception as e:
                print(f"[LONG-TERM RECALL EXCEPTION] {e}")

        if route is None:
            route = self.classify_request(user_message, scene_data)
        routes = self.normalize_routes(route)

        print("=" * 60)
        print("DISPATCHING COGNITIVE ROUTE:", routes)

        if self.route_has_skill(routes):
            response = "Understood. Starting movement sequence."
            self.remember_turn(user_message, response)
            return response

        if "vision_chat" in routes and scene_data is not None:
            self.find_best_object(user_message, scene_data)
            response = self.vision_chat(user_message, scene_data, memory_context)
        else:
            response = self.normal_chat(user_message, memory_context)

        self.remember_turn(user_message, response)
        return response

    # =====================================================
    # ON-DEMAND INTENT ANALYZER (정보 정밀 수혈 시스템)
    # =====================================================

    def generate_intent(self, user_message, scene_data):
        print("=" * 60)
        print("GENERATING SINGLE MOTOR INTENT")

        scene_data = scene_data or {}
        text = self.normalize_text(user_message)

        # -----------------------------------------------------------------
        # [복원 및 업그레이드] 고속 로컬 우선 키워드 매칭 수행
        # -----------------------------------------------------------------
        point_words = ["point", "point to", "가리켜", "가리켜줘", "가리켜 줘"]
        look_words = ["look", "look at", "봐", "보여", "향해"]
        raise_words = ["raise", "up", "higher", "lift", "올려", "위로", "높여"]
        lower_words = ["lower", "low", "down", "drop", "내려", "아래로", "낮춰"]
        
        motion_words = [
            ("wave", ["wave", "shake", "hello", "hi", "흔들", "인사"]),
            ("scan", ["scan", "search", "look around", "둘러", "스캔", "찾아"]),
            ("home", ["home", "rest", "idle", "stop", "제자리", "홈", "쉬어", "멈춰"]),
            ("extend", ["extend", "straight", "reach out", "펴", "뻗"]),
            ("fold", ["fold", "tuck", "접어", "접"])
        ]

        intent_name = "idle"
        direction = None
        motion = None
        target_id = None

        # 1. 기본 매칭 시도
        if any(w in text for w in point_words):
            intent_name = "point"
        elif any(w in text for w in look_words):
            intent_name = "look"

        # 2. 높이 조절 스킬 체크 (대상 지정 불필요)
        if any(w in text for w in lower_words):
            intent_name = "raise_low"
            direction = "lower"
        elif any(w in text for w in raise_words):
            intent_name = "raise_low"
            direction = "raise"

        # 3. 특수 모션 스킬 체크 (대상 지정 불필요)
        for name, words in motion_words:
            if any(w in text for w in words):
                intent_name = "motion"
                motion = name
                break

        # 4. 대상 물체 확인 (point / look 스킬인 경우에만 검색 유도)
        if intent_name in ["point", "look", "idle"]:
            m = re.search(r"\bid\s*[:=]?\s*(\d+)\b", text)
            if m:
                try: target_id = int(m.group(1))
                except Exception: pass

            if target_id is None:
                referenced = self.resolve_reference_target(user_message, scene_data)
                if referenced is not None: target_id = referenced.get("id")

            if target_id is None:
                best = self.find_best_object(user_message, scene_data)
                if best is not None: target_id = best.get("id")

        # 5. [0ms 완전 패스] 타겟 지정이 전혀 필요 없는 물리 행동(raise_low, motion)은 LLM 연산 생략
        if intent_name in ["raise_low", "motion"]:
            intent = {"intent": intent_name, "user_message": user_message}
            if direction: intent["direction"] = direction
            if motion: intent["motion"] = motion
            print("[0ms LOCAL SHORTCUT PASS] No LLM needed for targetless skill.")
            return intent

        # 6. 대상이 필요한 행동인데 정보가 없거나 복잡한 경우에만 초경량 AI 인터셉터 가동
        lightweight_context = self.generate_lightweight_prompt(user_message, scene_data)
        
        # Provide strict few-shot constraints to prevent conversational replies or "As an AI model" statements
        system_prompt = """
You are a precision robot controller skill planner for a 5-DOF arm.
You MUST output ONLY valid JSON or a NEED_INFO statement. Do NOT output conversational text, pleasantries, or mention you are an AI.

[OUTPUT FORMAT OPTIONS]
Option A: If target ID is known:
{"intent":"point","target_id":3}

Option B: If you need precise 3D metrics to select between multiple items:
NEED_INFO:[INTEGER_ID_OF_OBJECT]

[EXAMPLES]
Context Command: "now can you point that id 3 object?"
Output: {"intent":"point","target_id":3}

Context Command: "point to the red book"
Output: {"intent":"point","target_id":3}
"""
        try:
            response = self.generate(system_prompt, lightweight_context, max_new_tokens=30)
            print("[INTENT INTERCEPTOR RESPONSE]:", response)

            # 실시간 수혈 메커니즘 기동 (반드시 정수 ID 형태여야 함)
            if "NEED_INFO:" in response and not any(k in response for k in ["ARM", "POSITION", "JOINT"]):
                try:
                    requested_id = int(response.split("NEED_INFO:")[1].strip())
                    target_obj = self.find_object_by_id(scene_data, requested_id)
                    if target_obj:
                        # 정밀 수치 데이터 실시간 주입
                        info_feed = f"\n[DATA FEED FOR OBJECT {requested_id}]:\n" + "\n".join([f"- {k}: {v}" for k, v in target_obj.items()])
                        final_prompt = lightweight_context + f"\nSystem state: {response}\n{info_feed}"
                        
                        response = self.generate(
                            "Determine final intent and target_id using the precise values provided.",
                            final_prompt,
                            max_new_tokens=40
                        )
                except Exception:
                    pass

            # CRITICAL FIX: Use the local matching parameters as high-confidence fallbacks inside the default_dict
            # This guarantees that if the LLM fails or replies with chat/warnings, the correct local values are never lost!
            fallback_dict = {
                "intent": intent_name,
                "target_id": target_id,
                "direction": direction,
                "motion": motion
            }
            intent_data = self.robust_extract_json(response, default_dict=fallback_dict)
            
            llm_intent = intent_data.get("intent")
            if llm_intent and llm_intent != "idle":
                intent_name = llm_intent
            
            llm_target_id = intent_data.get("target_id")
            if llm_target_id is not None:
                target_id = llm_target_id
                
            llm_direction = intent_data.get("direction")
            if llm_direction:
                direction = llm_direction
                
            llm_motion = intent_data.get("motion")
            if llm_motion:
                motion = llm_motion

        except Exception as e:
            print("[INTENT EXCEPTION fallback]:", e)

        intent = {"intent": intent_name, "user_message": user_message}
        if direction: intent["direction"] = direction
        if motion: intent["motion"] = motion

        if target_id is not None:
            intent["target_id"] = target_id
            target_obj = self.find_object_by_id(scene_data, target_id)
            if target_obj is not None:
                intent["target"] = target_obj
                self.remember_target_object(target_obj)

        return intent

    # =====================================================
    # NORMALIZE INTENTS
    # =====================================================

    def normalize_intents(self, intent_data, user_message, scene_data):
        scene_data = scene_data or {}
        if isinstance(intent_data, dict):
            raw_intents = intent_data.get("intents", intent_data)
        else:
            raw_intents = intent_data

        if isinstance(raw_intents, dict):
            raw_intents = [raw_intents]
        if not isinstance(raw_intents, list):
            raw_intents = []

        valid_ids = {obj.get("id") for obj in scene_data.get("objects", [])}
        intents = []
        seen = set()

        for item in raw_intents:
            if not isinstance(item, dict):
                continue

            intent_name = self.resolve_skill_name(item.get("intent", "idle"))
            intent = {"intent": intent_name, "user_message": user_message}

            if intent_name == "raise_low":
                direction = str(item.get("direction", "raise")).lower()
                intent["direction"] = "lower" if direction in ["lower", "down", "low"] else "raise"

            if intent_name == "motion":
                motion = str(item.get("motion", "home")).lower()
                intent["motion"] = motion if motion in ["wave", "scan", "home", "extend", "fold"] else "home"

            target_id = item.get("target_id")
            try:
                if target_id is not None: target_id = int(target_id)
            except Exception:
                target_id = None

            if target_id is None and intent_name in ["look", "point"]:
                referenced = self.resolve_reference_target(user_message, scene_data)
                target_id = referenced.get("id") if referenced else None

            if target_id is None and intent_name in ["look", "point"]:
                best = self.find_best_object(user_message, scene_data)
                target_id = best.get("id") if best else None

            if target_id is not None:
                if len(valid_ids) > 0 and target_id not in valid_ids:
                    target_id = None
                else:
                    intent["target_id"] = target_id
                    target_obj = self.find_object_by_id(scene_data, target_id)
                    if target_obj:
                        intent["target"] = target_obj
                        self.remember_target_object(target_obj)

            # raise_low와 motion은 target_id가 없어도 되므로 제거 조건에서 면제합니다.
            if intent_name in ["look", "point"] and "target_id" not in intent:
                intent["intent"] = "idle"

            key = (intent.get("intent"), intent.get("target_id"), intent.get("direction"), intent.get("motion"))
            if key in seen:
                continue

            seen.add(key)
            intents.append(intent)

        non_idle = [i for i in intents if i.get("intent") != "idle"]
        return non_idle if len(non_idle) > 0 else [{"intent": "idle", "user_message": user_message}]

    def filter_bad_response(self, response):
        lower = response.lower()
        bad_patterns = ["as an ai", "language model", "virtual assistant", "i cannot", "i can't"]
        for p in bad_patterns:
            if p in lower:
                return "I can see objects in front of me."
        return response

    # =====================================================
    # MULTI-INTENT SEQUENCING PLANNER (온디맨드 인젝션 최적화)
    # =====================================================

    def generate_intents(self, user_message, scene_data):
        print("=" * 60)
        print("PLANNING MOTOR INTENT SEQUENCES")

        scene_data = scene_data or {}
        quick_intent = self.generate_intent(user_message, scene_data)
        quick_intents = self.normalize_intents(quick_intent, user_message, scene_data)

        # 0ms 로컬 매칭을 통해 바로 유의미한 비-idle 명령(raise_low, motion 등)이 검출되면 즉시 실행
        if any(intent.get("intent") != "idle" for intent in quick_intents):
            return quick_intents

        # 고도로 정밀한 연속 스킬 분석이 필요할 때만 복합 추론 수행
        lightweight_context = self.generate_lightweight_prompt(user_message, scene_data)
        skill_names = sorted(self.skills.keys())

        system_prompt = f"""
You are an advanced robot planner. Available: {json.dumps(skill_names)}
You MUST output ONLY valid JSON or a NEED_INFO statement. Do NOT output conversational chat or mention you are an AI.

[OUTPUT FORMAT OPTIONS]
Option A: Standard sequence plan:
{{"intents":[{{"intent":"look","target_id":7}}]}}

Option B: If you need coordinates:
NEED_INFO:[INTEGER_ID]
"""
        try:
            response = self.generate(system_prompt, lightweight_context, max_new_tokens=40)
            
            if "NEED_INFO:" in response and not any(k in response for k in ["ARM", "POSITION", "JOINT"]):
                try:
                    target_id = int(response.split("NEED_INFO:")[1].strip())
                    target_obj = self.find_object_by_id(scene_data, target_id)
                    if target_obj:
                        info_feed = f"\n[DETAILED OBJECT METRICS]:\n" + "\n".join([f"- {k}: {v}" for k, v in target_obj.items()])
                        final_prompt = lightweight_context + f"\nPlanner choice: {response}\n{info_feed}"
                        
                        response = self.generate(
                            "Generate final robot skill plans JSON based on detailed metrics.",
                            final_prompt,
                            max_new_tokens=60
                        )
                except Exception:
                    pass

            # CRITICAL FIX: Use the matched local quick_intent as fallback dictionary here as well
            data = self.robust_extract_json(response, default_dict=quick_intent)
            intents = self.normalize_intents(data, user_message, scene_data)

        except Exception as e:
            print("[SEQUENCE EXECUTION FALLBACK TRIGGERED]:", e)
            intents = self.normalize_intents({"intent": "idle"}, user_message, scene_data)

        return intents

    # =====================================================
    # RESOLVING & TARGETING UTILS
    # =====================================================

    def remember_target_object(self, obj):
        if isinstance(obj, dict) and "id" in obj:
            self.last_target_object_id = obj.get("id")

    def resolve_reference_target(self, user_message, scene_data):
        features = self.get_request_features(user_message, scene_data)
        if not features["has_reference"]:
            return None

        last_id = getattr(self, "last_target_object_id", None)
        return self.find_object_by_id(scene_data, last_id) if last_id is not None else None

    def find_object_by_id(self, scene_data, target_id):
        for obj in (scene_data or {}).get("objects", []):
            if obj["id"] == target_id:
                return obj
        return None

    def find_best_object(self, user_text, scene_data):
        user_text = self.normalize_text(user_text)
        user_tokens = set(re.findall(r"[a-zA-Z0-9_]+", user_text))

        best_score = 0
        best_object = None

        for obj in (scene_data or {}).get("objects", []):
            score = 0
            labels = [str(x).lower() for x in obj.get("semantic_labels", [])]
            color = obj.get("dominant_color", "").lower()

            for label in labels:
                if label in user_text:
                    score += 10
                    continue
                label_tokens = set(re.findall(r"[a-zA-Z0-9_]+", label))
                if user_tokens and label_tokens:
                    score += len(user_tokens.intersection(label_tokens)) * 4

            if color and color in user_text:
                score += 5

            object_id = str(obj["id"])
            if f"id {object_id}" in user_text:
                score += 20

            cx, cy, cz = self.get_calibrated_coordinates(obj["center"])
            if "left" in user_text and cx < -0.05: score += 15
            if "right" in user_text and cx > 0.05: score += 15
            if ("front" in user_text or "far" in user_text) and cy > 0.05: score += 15
            if ("back" in user_text or "close" in user_text) and cy < -0.05: score += 15
            if ("highest" in user_text or "tallest" in user_text or "top" in user_text) and cz > 0.05: score += 15

            score += max(0, 2 - abs(cz)) * 0.1

            if score > best_score:
                best_score = score
                best_object = obj

        if best_object is not None:
            self.remember_target_object(best_object)

        return best_object

    # =====================================================
    # ROBOT ACTUATION UTILITIES (모션 제어 역학)
    # =====================================================

    def clamp(self, value, minimum, maximum):
        return max(minimum, min(maximum, value))

    def smooth_joint(self, current, target):
        delta = abs(target - current)
        if delta < 0.01:
            return current
        return current + (target - current) * self.smooth_alpha

    def build_action(self, steps=60):
        smoothed = {}
        for key in self.current_state:
            smoothed[key] = self.smooth_joint(self.current_state[key], self.target_state[key])

        self.current_state = smoothed
        self.current_joint_0 = smoothed["joint_0"]
        self.current_joint_1 = smoothed["joint_1"]
        self.current_joint_2 = smoothed.get("joint_2", 0.0)
        self.current_joint_3 = smoothed.get("joint_3", 0.0)
        self.current_joint_4 = smoothed.get("joint_4", 0.0)

        action = dict(smoothed)
        action["steps"] = steps
        return action

    def resolve_skill_name(self, intent_name):
        return self.skill_aliases.get(str(intent_name).lower(), "idle")

    def apply_action_to_state(self, action):
        for key in self.JOINT_KEYS:
            if key in action:
                try: self.target_state[key] = float(action[key])
                except Exception: pass

        self.current_state = dict(self.target_state)
        self.current_joint_0 = self.current_state["joint_0"]
        self.current_joint_1 = self.current_state["joint_1"]
        self.current_joint_2 = self.current_state.get("joint_2", 0.0)
        self.current_joint_3 = self.current_state.get("joint_3", 0.0)
        self.current_joint_4 = self.current_state.get("joint_4", 0.0)

    def update_current_joints(self, j0, j1, j2=0.0, j3=0.0, j4=0.0, gripper=None):
        self.current_joint_0 = float(j0)
        self.current_joint_1 = float(j1)
        self.current_joint_2 = float(j2)
        self.current_joint_3 = float(j3)
        self.current_joint_4 = float(j4)

        self.current_state.update({
            "joint_0": self.current_joint_0,
            "joint_1": self.current_joint_1,
            "joint_2": self.current_joint_2,
            "joint_3": self.current_joint_3,
            "joint_4": self.current_joint_4
        })
        self.target_state = dict(self.current_state)

    def execute_action(self, action):
        self.last_action = action
        if self.auto_execute and self.pybullet is not None:
            self.pybullet.execute_action(action)
        return action

    def sanitize_action(self, action):
        safe = {}
        for key in self.JOINT_KEYS:
            current = float(getattr(self, f"current_{key}", self.current_state.get(key, 0.0)))
            low, high = self.joint_limits[key]
            try: safe[key] = self.clamp(float(action.get(key, current)), low, high)
            except Exception: safe[key] = current

        try: safe["steps"] = int(action.get("steps", 60))
        except Exception: safe["steps"] = 60

        for key in ["target_world", "target_distance"]:
            if key in action: safe[key] = action[key]
        return safe

    def sanitize_sequence_action(self, action):
        if not isinstance(action, dict) or action.get("type") != "sequence":
            return self.sanitize_action(action)

        safe_actions = [self.sanitize_action(item) for item in action.get("actions", []) if isinstance(item, dict)]
        return {"type": "sequence", "actions": safe_actions} if safe_actions else self.default_idle_action()

    def final_state_from_action(self, action):
        if isinstance(action, dict) and action.get("type") == "sequence":
            actions = action.get("actions", [])
            if len(actions) > 0 and isinstance(actions[-1], dict):
                return actions[-1]
        return action

    def default_idle_action(self):
        return {
            "joint_0": self.current_joint_0,
            "joint_1": self.FIXED_JOINT_1,
            "joint_2": 0.0,
            "joint_3": 0.0,
            "joint_4": 0.0,
            "steps": 120
        }

    # =====================================================
    # COGNITIVE PIPELINE EXECUTION ENTRY
    # =====================================================

    def generate_robot_action(self, user_message, scene_data, memory_context="", route=None):
        print("=" * 60)
        print("ACTIVATING SYSTEM CONTROL WITH HYBRID COGNITIVE MODULE")

        scene_data = self.update_and_recall_objects(scene_data)

        if not memory_context:
            try:
                memory_context = self.memory_manager.build_memory_context(user_message)
            except Exception as e:
                print(f"[LONG-TERM RECALL EXCEPTION] {e}")

        if route is None:
            route = self.classify_request(user_message, scene_data)
        routes = self.normalize_routes(route)

        # 1. 넌-피지컬(대화형) 동작 판단 시 LLM 추론 생략 및 우회
        if not self.route_has_skill(routes):
            print("[CONTROL BYPASS] Conversational query. Bypassing physical execution.")
            return {"type": "conversation", "route_fallback": routes}

        self.last_action = None
        intents = self.generate_intents(user_message, scene_data)

        is_vision_skill_active = any(r in ["vision_skill", "skill"] for r in routes)
        is_all_idle = all(intent.get("intent") == "idle" for intent in intents)

        # 2. 비정상 경로 예외 자동 보정 (Failsafe)
        if is_vision_skill_active and is_all_idle:
            print("[COGNITIVE CONVICTION FAILSAFE] Routing fallback to conversation.")
            return {"type": "conversation", "route_fallback": ["vision_chat"]}

        actions = []
        for intent in intents:
            intent_name = intent.get("intent", "idle")
            skill_name = self.resolve_skill_name(intent_name)
            skill = self.skills.get(skill_name, self.skills["idle"])

            try:
                self.last_action = None
                result = skill.execute(intent, scene_data)
            except Exception as e:
                print("[SKILL HARDWARE FAULT]:", e)
                result = None

            action = result if isinstance(result, dict) else (self.last_action if isinstance(self.last_action, dict) else self.default_idle_action())
            action = self.sanitize_sequence_action(action)
            state_action = self.final_state_from_action(action)

            for key in self.JOINT_KEYS:
                self.target_state[key] = state_action[key]

            self.apply_action_to_state(state_action)
            actions.append(action)

        if len(actions) == 1:
            return actions[0]

        flattened = []
        for action in actions:
            if isinstance(action, dict) and action.get("type") == "sequence":
                flattened.extend(action.get("actions", []))
            else:
                flattened.append(action)

        return {"type": "sequence", "actions": flattened}