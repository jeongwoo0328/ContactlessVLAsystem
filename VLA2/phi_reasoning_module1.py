# =========================================================
# attention_robot.py
# FAST + STABLE + SKILL CONNECTED VERSION
# =========================================================

import json
import os
import re
import torch

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

# =========================================================
# CLASS
# =========================================================

class AttentionRobot:

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
    # INIT
    # =====================================================

    def __init__(self, pybullet=None, auto_execute=False):

        print("=" * 60)
        print("INITIALIZING ATTENTION ROBOT")

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print("DEVICE:", self.device)

        # =================================================
        # SPEED SETTINGS
        # =================================================

        torch.set_float32_matmul_precision("high")

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        # =================================================
        # MODEL
        # =================================================

        model_name = "microsoft/Phi-3-mini-4k-instruct"

        print("=" * 60)
        print("LOADING PHI-3")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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

        # =================================================
        # ROBOT CONFIG
        # =================================================

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

        # =================================================
        # STATE
        # =================================================

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

        # =================================================
        # EXECUTION BRIDGE
        # =================================================

        self.pybullet = pybullet
        self.auto_execute = auto_execute
        self.last_action = None

        # =================================================
        # SKILLS
        # =================================================

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

        # =================================================
        # WARMUP
        # =================================================

        print("=" * 60)
        print("MODEL WARMUP")

        try:
            self.generate(
                "You are ready.",
                "hello",
                4
            )
        except Exception as e:
            print("WARMUP ERROR")
            print(e)

        self.semantic_analyzer = SemanticAnalyzer()
        print("=" * 60)
        print("ATTENTION ROBOT READY")

    # =====================================================
    # ATTACH PYBULLET
    # =====================================================

    def attach_pybullet(self, pybullet):
        self.pybullet = pybullet

    # =====================================================
    # LOCAL CONVERSATION HISTORY
    # =====================================================

    def get_conversation_history(self, limit=None):
        if limit is None:
            return list(self.conversation_history)
        return self.conversation_history[-int(limit):]

    def clear_conversation_history(self):
        self.conversation_history = []

    def remember_turn(self, user_message, assistant_message):
        self.conversation_history.append({
            "user": str(user_message),
            "assistant": str(assistant_message)
        })
        if len(self.conversation_history) > self.max_history_turns:
            self.conversation_history = self.conversation_history[-self.max_history_turns:]

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
    # REQUEST FEATURES
    # =====================================================

    def get_request_features(
            self,
            user_message,
            scene_data=None
    ):

        text = self.normalize_text(user_message)
        scene_data = scene_data or {}

        # -----------------------------------------------------------------
        # [교체 완료] 의미 원소 추출 및 수치 구조 분석 연동
        # -----------------------------------------------------------------
        # 1. 문장을 데이터베이스 단어로 정정 후 고유 의미 원소 스트림(ID 숫자 리스트)으로 변환
        token_stream = self.semantic_analyzer.correct_and_convert(text)
        
        # 2. 최종 알고리즘 연산 결과값(두 카테고리가 융합된 결합 벡터) 추출
        semantic_vector = self.semantic_analyzer.analyze_sentence(text)
        
        # JSON의 정수 ID 테이블 바인딩
        atom_map = self.semantic_analyzer.atoms

        # 3. 행동 유도 요소 추출 (DO, MOVE, THINK, SEE, HEAR, WANT 등 행동/지각 상태 원소 포함 검사)
        action_atoms = {
            atom_map.get("DO"), atom_map.get("MOVE"), atom_map.get("THINK"), 
            atom_map.get("SEE"), atom_map.get("HEAR"), atom_map.get("WANT")
        }
        has_action = any(atom in token_stream for atom in action_atoms if atom is not None)
        
        # 예외 처리 규칙 유지: 예외 질문 형태인 경우 단순 시각 확인(SEE)만으로는 물리행동 판정 차단
        if "look like" in text or "looks like" in text:
            has_action = any(atom in token_stream for atom in (action_atoms - {atom_map.get("SEE")}) if atom is not None)

        # 4. 질문 요소 추출 (WHERE 등 공간 질의 혹은 데이터베이스 내 QUESTION 원소 검사)
        question_atoms = {atom_map.get("WHERE"), atom_map.get("QUESTION")}
        has_question = any(atom in token_stream for atom in question_atoms if atom is not None)

        # 5. 지시 대명사 참조 요소 추출 (THIS, SOMETHING, HERE, THEREIS 등 탐색)
        reference_atoms = {
            atom_map.get("THIS"), atom_map.get("SOMETHING"), 
            atom_map.get("HERE"), atom_map.get("THEREIS")
        }
        has_reference = any(atom in token_stream for atom in reference_atoms if atom is not None)

        # 6. [기존 로직 유지] 객체 ID 및 주변 환경 텍스트 분석
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

        # 최종 피처 딕셔너리 반환 (semantic_vector 포함)
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
    # NORMALIZE TEXT
    # =====================================================

    def normalize_text(
            self,
            text
    ):

        text = str(text).lower().strip()
        text = text.replace("’", "'")
        text = text.replace("“", '"')
        text = text.replace("”", '"')
        return text

    # =====================================================
    # CLEAN RESPONSE
    # =====================================================

    def clean_response(
            self,
            text
    ):

        text = text.replace("```json", "")
        text = text.replace("```", "")

        stop_strings = [
            "\nUSER",
            "\nUser",
            "\nAssistant",
            "\nHuman"
        ]

        for s in stop_strings:
            if s in text:
                text = text.split(s)[0]

        return text.strip()

    # =====================================================
    # EXTRACT JSON
    # =====================================================

    def extract_json(
            self,
            text
    ):

        start = text.find("{")

        if start == -1:
            return None

        brace_count = 0
        end = None

        for i in range(start, len(text)):

            char = text[i]

            if char == "{":
                brace_count += 1

            elif char == "}":
                brace_count -= 1

                if brace_count == 0:
                    end = i + 1
                    break

        if end is None:
            json_text = text[start:]
        else:
            json_text = text[start:end]

        open_count = json_text.count("{")
        close_count = json_text.count("}")

        if open_count > close_count:
            json_text += "}" * (open_count - close_count)

        return json_text

    # =====================================================
    # GENERATE
    # =====================================================

    @torch.inference_mode()
    def generate(
            self,
            system_prompt,
            user_prompt,
            max_new_tokens=40
    ):

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
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
            max_length=512
        )

        try:
            target_device = next(self.model.parameters()).device
        except Exception:
            target_device = self.device

        inputs = {
            k: v.to(target_device)
            for k, v in inputs.items()
        }

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=5,
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
    

    def generate_lightweight_prompt(self, user_command, scene_data):
            """
            Creates a fast, lightweight description with only object IDs and names.
            Hides heavy raw coordinates initially to speed up the token processing.
            """
            object_list = ""
            for obj in scene_data.get("objects", []):
                labels = obj.get("semantic_labels", [])
                label_text = "/".join([str(x) for x in labels[:3]]) if labels else "object"
                object_list += f"- ID {obj.get('id')}: {label_text} ({obj.get('dominant_color', 'unknown')})\n"
                
            prompt = f"""
    Currently detected object candidates in the workspace:
    {object_list}

    User Instruction: "{user_command}"
    """
            return prompt

    # =====================================================
    # BUILD SCENE TEXT
    # =====================================================

    def build_scene_text(
            self,
            scene_data
    ):

        scene_data = scene_data or {}

        lines = []

        objects = scene_data.get(
            "objects",
            []
        )

        for obj in objects[:8]:

            center = obj["center"]

            x = round(center[1], 2)
            y = round(center[0], 2)
            z = round(center[2], 2)

            labels = obj.get(
                "semantic_labels",
                []
            )

            label = (
                "/".join([str(x) for x in labels[:3]])
                if len(labels) > 0
                else "object"
            )

            color = obj.get(
                "dominant_color",
                "unknown"
            )

            lines.append(
                f"id={obj['id']} "
                f"type={label} "
                f"color={color} "
                f"position=({x},{y},{z})"
            )

        if len(lines) == 0:
            return "No visible objects."

        return "\n".join(lines)

    # =====================================================
    # NORMALIZE ROUTES
    # =====================================================

    def normalize_routes(
            self,
            route_data
    ):

        if isinstance(route_data, dict):
            raw_routes = route_data.get(
                "routes",
                route_data.get("route", "chat")
            )
        else:
            raw_routes = route_data

        if isinstance(raw_routes, str):
            raw_routes = [
                part.strip()
                for part in raw_routes.split(",")
            ]

        if not isinstance(raw_routes, list):
            raw_routes = ["chat"]

        routes = []
        for route in raw_routes:
            route = str(route).strip().lower()
            if route in self.VALID_ROUTES and route not in routes:
                routes.append(route)

        if len(routes) == 0:
            routes = ["chat"]

        routes.sort(
            key=lambda name: self.ROUTE_PRIORITY.get(name, 99)
        )

        return routes

    # =====================================================
    # ROUTE HELPERS
    # =====================================================

    def route_has_skill(
            self,
            routes
    ):

        return any(
            route in ["skill", "vision_skill"]
            for route in self.normalize_routes(routes)
        )

    def route_has_vision(
            self,
            routes
    ):

        return any(
            route in ["vision_chat", "vision_skill"]
            for route in self.normalize_routes(routes)
        )

    def route_has_chat(
            self,
            routes
    ):

        return any(
            route in ["chat", "vision_chat"]
            for route in self.normalize_routes(routes)
        )

    # =====================================================
    # ROUTER CONTEXT
    # =====================================================

    def build_router_context(
            self,
            user_message,
            scene_data=None,
            previous_routes=None
    ):

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

        return json.dumps(
            context,
            ensure_ascii=False,
            indent=2
        )

    def summarize_object(
            self,
            obj
    ):

        if not isinstance(obj, dict):
            return None

        summary = {
            "id": obj.get("id"),
            "labels": [
                str(label)
                for label in obj.get("semantic_labels", [])[:3]
            ],
            "color": str(obj.get("dominant_color", "")),
        }

        center = obj.get("center", None)
        if center is not None:
            try:
                summary["center"] = [
                    round(float(value), 3)
                    for value in center[:3]
                ]
            except Exception:
                pass

        return summary

    def route_needs_phi_recheck(
            self,
            routes,
            user_message,
            scene_data=None
    ):

        features = self.get_request_features(
            user_message,
            scene_data
        )

        if not features["has_action"]:
            return False

        return not self.route_has_skill(routes)

    def recheck_route_with_phi(
            self,
            user_message,
            scene_data,
            previous_routes
    ):

        system_prompt = """
You are a robot routing auditor.

Return ONLY JSON.

VALID:
{"routes":["chat"]}
{"routes":["vision_chat"]}
{"routes":["skill"]}
{"routes":["vision_skill"]}
{"routes":["vision_chat","vision_skill"]}
{"routes":["chat","skill"]}

Rules:
- preserve the previous route if it is still correct
- if the user asks the robot to physically point, look, turn, raise, lower, or move, include skill or vision_skill
- if a physical command refers to a visible or previous object, include vision_skill
- output JSON only
- stop immediately after }
"""

        router_context = self.build_router_context(
            user_message,
            scene_data,
            previous_routes=self.normalize_routes(previous_routes)
        )

        try:
            response = self.generate(
                system_prompt,
                router_context,
                max_new_tokens=48
            )

            print("=" * 60)
            print("ROUTER RECHECK RAW")
            print(response)

            json_text = self.extract_json(response)
            if json_text is None:
                raise ValueError("NO JSON")

            return self.normalize_routes(
                json.loads(json_text)
            )

        except Exception as e:
            print("=" * 60)
            print("ROUTER RECHECK ERROR")
            print(str(e))
            return self.normalize_routes(previous_routes)

    # =====================================================
    # FILTER BAD RESPONSE
    # =====================================================

    def filter_bad_response(
            self,
            response
    ):

        lower = response.lower()

        bad_patterns = [
            "as an ai",
            "language model",
            "virtual assistant",
            "i cannot",
            "i can't",
            "i don't have access",
            "i am unable",
            "sorry, but as an ai",
            "text-based ai",
            "text based ai",
            "text-only ai",
            "text only ai"
        ]

        for p in bad_patterns:
            if p in lower:
                return "I can see objects in front of me."

        return response

    # =====================================================
    # ROUTE FROM RULES
    # =====================================================

    def route_from_rules(
            self,
            user_message,
            scene_data=None
    ):

        text = self.normalize_text(user_message)
        scene_data = scene_data or {}

        greetings = [
            "hello",
            "hi",
            "hey",
            "안녕",
            "하이",
            "thanks",
            "thank you",
            "고마워",
            "고맙",
            "bye",
            "goodbye",
            "잘가"
        ]

        action_words = [
            "point",
            "look at",
            "point to",
            "face",
            "turn",
            "rotate",
            "봐",
            "보여",
            "가리켜",
            "가리켜줘",
            "가리켜 줘",
            "집어",
            "향해",
            "돌아",
            "돌려"
        ]

        question_words = [
            "where",
            "what",
            "which",
            "closest",
            "nearest",
            "어디",
            "뭐",
            "무엇",
            "어느",
            "가장",
            "가까운",
            "위치"
        ]

        action_words.extend([
            "raise",
            "lower",
            "up",
            "down",
            "올려",
            "내려",
            "위로",
            "아래로"
        ])

        features = self.get_request_features(user_message, scene_data)
        has_action = (
            any(w in text for w in action_words)
            or features["has_action"]
        )
        has_question = (
            any(w in text for w in question_words)
            or features["has_question"]
        )
        has_id = features["has_id"]
        has_scene_term = features["has_scene_term"]
        has_objects = features["has_objects"]

        if any(g in text for g in greetings) and not has_action and not has_id:
            return ["chat"]

        if has_question and (has_action or has_id):
            if has_objects:
                return ["vision_chat", "vision_skill"]
            return ["chat", "skill"]

        # pure questions about visible objects should be answered, not executed
        if has_question and not has_action:
            if has_objects:
                return ["vision_chat"]
            return ["chat"]

        if has_action or has_id:
            if has_objects:
                return ["vision_skill"]
            return ["skill"]

        if has_scene_term:
            if has_objects:
                return ["vision_chat"]

        return None

    # =====================================================
    # CLASSIFY REQUEST
    # =====================================================

    def classify_request(
            self,
            user_message,
            scene_data=None
    ):

        if not self.use_ai_routing:
            rule_route = self.route_from_rules(
                user_message,
                scene_data
            )
            if rule_route is not None:
                routes = self.normalize_routes(rule_route)
                self.last_route = routes
                return routes

        system_prompt = """
You are a robot routing system.

Return ONLY JSON.

VALID:
{"routes":["chat"]}
{"routes":["vision_chat"]}
{"routes":["skill"]}
{"routes":["vision_skill"]}
{"routes":["vision_chat","vision_skill"]}
{"routes":["chat","skill"]}

Rules:
- choose every route needed to satisfy the user
- use chat for normal conversation
- use vision_chat for questions or answers about visible objects
- use skill for physical robot movement without object vision
- use vision_skill for physical robot movement involving visible objects
- if the user is only asking for information, do not include skill
- if the user asks the robot to point, look at, turn toward, raise, lower, or move, include skill or vision_skill
- polite question forms can still be physical commands
- "can you point it" means a physical pointing skill when a previous object exists
- "it", "this", "that", and "one" may refer to previous_referenced_object
- output JSON only
- stop immediately after }
"""

        router_context = self.build_router_context(
            user_message,
            scene_data
        )

        try:
            response = self.generate(
                system_prompt,
                router_context,
                max_new_tokens=48
            )
        except Exception as e:
            print("=" * 60)
            print("ROUTER AI ERROR")
            print(str(e))
            rule_route = self.route_from_rules(
                user_message,
                scene_data
            )
            routes = self.normalize_routes(
                rule_route
                if rule_route is not None
                else ["chat"]
            )
            self.last_route = routes
            return routes

        print("=" * 60)
        print("ROUTER RAW")
        print(response)

        try:
            json_text = self.extract_json(response)

            if json_text is None:
                raise ValueError("NO JSON")

            data = json.loads(json_text)
            routes = self.normalize_routes(data)

            if self.route_needs_phi_recheck(
                    routes,
                    user_message,
                    scene_data
            ):
                routes = self.recheck_route_with_phi(
                    user_message,
                    scene_data,
                    routes
                )

            self.last_route = routes
            return routes

        except Exception:
            route = self.route_from_rules(
                user_message,
                scene_data
            )
            if route is not None:
                routes = self.normalize_routes(route)
                self.last_route = routes
                return routes
            self.last_route = ["chat"]
            return ["chat"]

    # =====================================================
    # NORMAL CHAT
    # =====================================================

    def normal_chat(
            self,
            user_message,
            memory_context=""
    ):

        system_prompt = """
You are a real robot arm.

Never say:
- as an AI
- language model
- virtual assistant

Speak naturally and briefly.
"""

        user_prompt = f"""
Conversation history:
{self.build_history_text(memory_context)}

User:
{user_message}
"""

        response = self.generate(
            system_prompt,
            user_prompt,
            max_new_tokens=40
        )

        return self.filter_bad_response(response)

    # =====================================================
    # VISION CHAT
    # =====================================================

    def vision_chat(
            self,
            user_message,
            scene_data,
            memory_context=""
    ):
        scene_data = scene_data or {}
        history_text = self.build_history_text(memory_context)
        
        # 1. 원본 데이터를 기반으로 경량 텍스트 생성 (토큰 소모 최적화 및 잔상 방지)
        lightweight_context = self.generate_lightweight_prompt(user_message, scene_data)
        
        # 2. 1차 의도 파악용 시스템 프롬프트 (공간/위치/비교 질문 탐지)
        system_prompt_stage1 = """
You are a real robot arm assistant. 

[CRITICAL RULE]
If the user asks about ANY spatial relations, locations, or comparisons such as:
- distance/proximity (closest, farthest, near)
- horizontal placement (left, right, next to, between)
- vertical placement (highest, top, above, below, height)
You ABSOLUTELY need full 3D coordinates to answer.

In that case, you MUST respond ONLY with the exact format: NEED_INFO:[ALL]
Never explain why, and never say you lack sensors.

If the question does not involve any spatial or physical layout (e.g., simple color/name count, greetings, chatting), answer briefly and naturally.
"""

        first_user_prompt = f"""
Conversation history:
{history_text}

Scene Candidate List:
{lightweight_context}

User: {user_message}
Response:"""

        # 1차 추론: 공간 정보가 필요한지 판단 또는 단순 대답 생성
        response = self.generate(system_prompt_stage1, first_user_prompt, max_new_tokens=40)

        # 3. 3D 공간 데이터가 필요한 경우 (인터셉터 발동)
        if "NEED_INFO:" in response:
            detailed_specs = "\n[DETAILED SPATIAL 3D COORDINATES INJECTED]\n"
            
            # 모든 물체의 3D 좌표(X, Y, Z)를 상세 스펙으로 변환하여 주입
            for obj in scene_data.get("objects", []):
                tid = obj.get("id")
                labels = "/".join(obj.get("semantic_labels", []))
                # center 구조가 [X, Y, Z] 스펙이므로 통째로 제공하여 LLM이 직접 계산하게 유도
                detailed_specs += f"ID {tid} ({labels}): 3D_Position(X/Y/Z)={obj['center']}\n"
            
            # 2차 최종 대답용 시스템 프롬프트 (모듈 1의 자연스러운 로봇 페르소나 융합)
            system_prompt_final = """
You are a real robot arm. 
Answer the user's question briefly and naturally using the injected 3D coordinates.

[PHYSICAL SETUP CONTEXT]
- X coordinate represents Left(-) to Right(+)
- Y/Z coordinates represent Distance and Height (Refer to your physical setup)

[CRITICAL DIALOGUE RULES]
Never say:
- as an AI / language model
- cannot see / virtual assistant
- sensors / data constraints
Speak naturally, contextually, and briefly as a real entity in this space.
"""
            final_user_prompt = f"""
Conversation history:
{history_text}

Scene Candidate List:
{lightweight_context}
{detailed_specs}

User: {user_message}
Response:"""
            response = self.generate(system_prompt_final, final_user_prompt, max_new_tokens=40)
            
        else:
            # 4. 단순 대화나 인사일 경우, 1차 답변의 결과가 모듈 1처럼 자연스럽게 출력되도록 
            # AI 말투(as an AI 등) 필터링 및 페르소나 재정렬을 거침
            system_prompt_chat_refine = """
You are a real robot arm. Speak naturally and briefly based on the conversation.
Never mention being an AI, language model, or virtual assistant.
"""
            # 만약 1차 대답에 AI 성향의 단어가 섞여 나왔을 가능성을 방지하기 위한 안전장치
            if any(bad_word in response.lower() for bad_word in ["ai", "language model", "assistant"]):
                refine_user_prompt = f"History:\n{history_text}\nUser said: {user_message}\nFix this robot response to be natural and brief: {response}"
                response = self.generate(system_prompt_chat_refine, refine_user_prompt, max_new_tokens=40)

        # 5. 최종 불합격 답변 필터링 및 반환
        return self.filter_bad_response(response)

    # =====================================================
    # ASK QUESTION
    # =====================================================

    def ask_question(
            self,
            user_message,
            scene_data=None,
            memory_context="",
            route=None
    ):

        if route is None:
            route = self.classify_request(
                user_message,
                scene_data
            )

        routes = self.normalize_routes(route)

        print("=" * 60)
        print("ROUTE:", routes)

        if self.route_has_skill(routes):
            response = "Okay."
            self.remember_turn(user_message, response)
            return response

        if self.route_has_vision(routes):
            if scene_data is not None:
                self.find_best_object(
                    user_message,
                    scene_data
                )
            response = self.normal_chat(
                user_message,
                memory_context=memory_context
            ) if scene_data is None else self.vision_chat(
                user_message,
                scene_data,
                memory_context=memory_context
            )
            self.remember_turn(user_message, response)
            return response

        if self.route_has_chat(routes) or self.route_has_skill(routes):
            response = self.normal_chat(
                user_message,
                memory_context=memory_context
            )
            self.remember_turn(user_message, response)
            return response

        response = self.normal_chat(
            user_message,
            memory_context=memory_context
        )
        self.remember_turn(user_message, response)
        return response

    # =====================================================
    # GENERATE INTENT
    # =====================================================

    def generate_intent(
                self,
                user_message,
                scene_data
        ):

            print("=" * 60)
            print("GENERATING INTENT")

            scene_data = scene_data or {}
            text = self.normalize_text(user_message)

            point_words = [
                "point",
                "point to",
                "가리켜",
                "가리켜줘",
                "가리켜 줘"
            ]

            look_words = [
                "look",
                "look at",
                "봐",
                "보여",
                "향해"
            ]

            if any(w in text for w in point_words):
                intent_name = "point"
            elif any(w in text for w in look_words):
                intent_name = "look"
            else:
                intent_name = "idle"

            raise_words = [
                "raise",
                "up",
                "higher",
                "lift",
                "올려",
                "위로",
                "높여"
            ]

            lower_words = [
                "lower",
                "low",
                "down",
                "drop",
                "내려",
                "아래로",
                "낮춰"
            ]

            direction = None
            if any(w in text for w in lower_words):
                intent_name = "raise_low"
                direction = "lower"
            elif any(w in text for w in raise_words):
                intent_name = "raise_low"
                direction = "raise"

            motion = None
            motion_words = [
                ("wave", ["wave", "shake", "hello", "hi", "흔들", "인사"]),
                ("scan", ["scan", "search", "look around", "둘러", "스캔", "찾아"]),
                ("home", ["home", "rest", "idle", "stop", "제자리", "홈", "쉬어", "멈춰"]),
                ("extend", ["extend", "straight", "reach out", "펴", "뻗"]),
                ("fold", ["fold", "tuck", "접어", "접"])
            ]

            for name, words in motion_words:
                if any(w in text for w in words):
                    intent_name = "motion"
                    motion = name
                    break

            target_id = None

            m = re.search(r"\bid\s*[:=]?\s*(\d+)\b", text)
            if m:
                try:
                    target_id = int(m.group(1))
                except Exception:
                    target_id = None

            if target_id is None:
                referenced = self.resolve_reference_target(
                    user_message,
                    scene_data
                )
                if referenced is not None:
                    target_id = referenced.get("id", None)

            if target_id is None:
                best = self.find_best_object(
                    user_message,
                    scene_data
                )
                if best is not None:
                    target_id = best.get("id", None)

            # -----------------------------------------------------------------
            # [수정 구간] 규칙 기반 매칭 실패 시 Phi-3 온디맨드 추론 가동
            # -----------------------------------------------------------------
            if intent_name in ["look", "point"] and target_id is None:
                # 무거운 scene_text 대신 초경량 물체 ID/이름 리스트 프롬프트 생성
                lightweight_context = self.generate_lightweight_prompt(user_message, scene_data)

                system_prompt = """
    You are a robot controller skill planner.

    [CRITICAL BEHAVIOR RULES]
    If you need exact 3D positioning coordinates or dense spatial metrics of a specific object to reason about its location relative to others, 
    DO NOT output the final JSON immediately. Instead, request the raw parameters by strictly responding with ONLY:
    NEED_INFO:[OBJECT_ID]

    Otherwise, return ONLY valid JSON format directly:
    {"intent":"look","target_id":7}
    {"intent":"point","target_id":7}
    {"intent":"motion","motion":"wave"}
    {"intent":"idle"}

    Output JSON or NEED_INFO only. Stop immediately after the final token.
    """

                try:
                    # 1차 경량 질의 실행 (텍스트 크기가 극도로 작아 읽기 속도 최적화)
                    response = self.generate(
                        system_prompt,
                        lightweight_context,
                        max_new_tokens=40
                    )

                    print("=" * 60)
                    print("INTENT INTUITION RESPONSE:", response)

                    # [실시간 인터셉터] Phi-3가 특정 물체의 원천 데이터 주입을 요청한 경우
                    if "NEED_INFO:" in response:
                        try:
                            requested_id = int(response.split("NEED_INFO:")[1].strip())
                            print(f"[ON-DEMAND INJECTION] Injecting raw specs for Object ID: {requested_id} into generate_intent")
                            
                            target_obj = self.find_object_by_id(scene_data, requested_id)
                            if target_obj:
                                # 딕셔너리에 들어있는 날것의 데이터를 텍스트 블록으로 생성
                                info_feed = f"\n[RUNTIME DATA INJECTION: Detailed Specs for Object ID {requested_id}]\n"
                                for key, val in target_obj.items():
                                    info_feed += f"- {key}: {val}\n"
                                
                                # 기존 대화 컨텍스트 뒤에 3D 실측 메트릭을 이어 붙여 최종 확정 연산 유도
                                final_user_prompt = (
                                    lightweight_context 
                                    + f"\nYour Request Analysis: {response}\n" 
                                    + info_feed 
                                    + "\nNow, determine the final intent and target_id using the precise values injected above."
                                )
                                
                                system_prompt_final = """
    You are a robot controller. Convert the injected precise data into the final single action intent.
    Return ONLY valid JSON layout:
    {"intent":"point","target_id":2}
    """
                                response = self.generate(
                                    system_prompt_final,
                                    final_user_prompt,
                                    max_new_tokens=40
                                )
                                print("=" * 60)
                                print("INTENT AFTER DATA INJECTION:", response)
                        except (ValueError, IndexError):
                            print("[WARN] Failed to parse target injection ID in generate_intent. Using fallback.")

                    # 추출된 응답 스트링에서 JSON을 검출하고 파싱하는 레거시 프로세스 유지
                    json_text = self.extract_json(response)

                    if json_text is None:
                        raise ValueError("NO JSON")

                    intent_data = json.loads(json_text)
                    intent_name = intent_data.get("intent", "idle")
                    target_id = intent_data.get("target_id", None)

                except Exception as e:
                    print("=" * 60)
                    print("INTENT ERROR")
                    print(str(e))
                    intent_name = "idle"
                    target_id = None

            # -----------------------------------------------------------------
            # 후속 매칭 및 예외 차단 로직 (레거시 코드 정합성 유지)
            # -----------------------------------------------------------------
            intent = {
                "intent": intent_name
            }

            intent["user_message"] = user_message

            if direction is not None:
                intent["direction"] = direction

            if motion is not None:
                intent["motion"] = motion

            if target_id is not None:
                intent["target_id"] = target_id

                target_obj = self.find_object_by_id(
                    scene_data,
                    target_id
                )
                if target_obj is not None:
                    intent["target"] = target_obj
                    self.remember_target_object(target_obj)

            valid_ids = [
                obj["id"]
                for obj in scene_data.get("objects", [])
            ]

            if "target_id" in intent and intent["target_id"] not in valid_ids:
                intent.pop("target_id", None)
                intent.pop("target", None)
                intent["intent"] = "idle"

            print("=" * 60)
            print("FINAL INTENT")

            print(
                json.dumps(
                    intent,
                    indent=4
                )
            )

            return intent

    # =====================================================
    # NORMALIZE INTENTS
    # =====================================================

    def normalize_intents(
            self,
            intent_data,
            user_message,
            scene_data
    ):

        scene_data = scene_data or {}

        if isinstance(intent_data, dict):
            raw_intents = intent_data.get(
                "intents",
                intent_data
            )
        else:
            raw_intents = intent_data

        if isinstance(raw_intents, dict):
            raw_intents = [raw_intents]

        if not isinstance(raw_intents, list):
            raw_intents = []

        valid_ids = {
            obj.get("id")
            for obj in scene_data.get("objects", [])
        }

        intents = []
        seen = set()

        for item in raw_intents:
            if not isinstance(item, dict):
                continue

            intent_name = self.resolve_skill_name(
                item.get("intent", "idle")
            )

            intent = {
                "intent": intent_name,
                "user_message": user_message
            }

            if intent_name == "raise_low":
                direction = str(
                    item.get("direction", "raise")
                ).lower()
                if direction not in ["raise", "lower", "up", "down", "low"]:
                    direction = "raise"
                intent["direction"] = (
                    "lower"
                    if direction in ["lower", "down", "low"]
                    else "raise"
                )

            if intent_name == "motion":
                motion = str(
                    item.get("motion", "home")
                ).lower()
                if motion not in ["wave", "scan", "home", "extend", "fold"]:
                    motion = "home"
                intent["motion"] = motion

            target_id = item.get("target_id", None)
            try:
                if target_id is not None:
                    target_id = int(target_id)
            except Exception:
                target_id = None

            if target_id is None and intent_name in ["look", "point"]:
                referenced = self.resolve_reference_target(
                    user_message,
                    scene_data
                )
                if referenced is not None:
                    target_id = referenced.get("id", None)

            if target_id is None and intent_name in ["look", "point"]:
                best = self.find_best_object(
                    user_message,
                    scene_data
                )
                if best is not None:
                    target_id = best.get("id", None)

            if target_id is not None:
                if len(valid_ids) > 0 and target_id not in valid_ids:
                    target_id = None
                else:
                    intent["target_id"] = target_id
                    target_obj = self.find_object_by_id(
                        scene_data,
                        target_id
                    )
                    if target_obj is not None:
                        intent["target"] = target_obj
                        self.remember_target_object(target_obj)

            if intent_name in ["look", "point"] and "target_id" not in intent:
                intent["intent"] = "idle"

            key = (
                intent.get("intent"),
                intent.get("target_id"),
                intent.get("direction"),
                intent.get("motion")
            )

            if key in seen:
                continue

            seen.add(key)
            intents.append(intent)

        non_idle = [
            intent
            for intent in intents
            if intent.get("intent") != "idle"
        ]

        if len(non_idle) > 0:
            return non_idle

        return [{
            "intent": "idle",
            "user_message": user_message
        }]

    # =====================================================
    # GENERATE INTENTS
    # =====================================================

    def generate_intents(
                self,
                user_message,
                scene_data
        ):

            print("=" * 60)
            print("GENERATING INTENTS")

            scene_data = scene_data or {}
            quick_intent = self.generate_intent(
                user_message,
                scene_data
            )
            quick_intents = self.normalize_intents(
                quick_intent,
                user_message,
                scene_data
            )

            if any(
                    intent.get("intent") != "idle"
                    for intent in quick_intents
            ):
                print("=" * 60)
                print("FINAL INTENTS")
                print(
                    json.dumps(
                        quick_intents,
                        indent=4
                    )
                )
                return quick_intents

            # -----------------------------------------------------------------
            # [최종 수정] 1차 초경량 쿼리 프롬프트 준비 (가볍고 빠른 연산 유도)
            # -----------------------------------------------------------------
            lightweight_context = self.generate_lightweight_prompt(user_message, scene_data)
            skill_names = sorted(self.skills.keys())

            system_prompt = f"""
    You are an advanced robot skill planner. 

    Available skills: {json.dumps(skill_names)}

    [CRITICAL ATTENTION RULE]
    If you need exact 3D positioning coordinates or dense spatial metrics of a specific object to determine its precise location relative to others, 
    DO NOT execute immediately. Instead, request the raw parameters by strictly responding with ONLY:
    NEED_INFO:[OBJECT_ID]

    Otherwise, output the execution plan in the required JSON format directly:
    {{"intents":[{{"intent":"look","target_id":7}}]}}

    Output JSON or NEED_INFO only. Stop immediately after the response token.
    """

            try:
                # 1차 추론 진행 (텍스트 토큰이 대폭 감소하여 Phi-3 응답 속도가 급상승함)
                response = self.generate(
                    system_prompt,
                    lightweight_context,
                    max_new_tokens=40
                )

                print("=" * 60)
                print("INTENTS INTUITION RESPONSE:", response)

                # -----------------------------------------------------------------
                # [실시간 수혈 인터셉터] Phi-3가 특정 물체의 원천 데이터 주입을 요청한 경우
                # -----------------------------------------------------------------
                if "NEED_INFO:" in response:
                    try:
                        target_id = int(response.split("NEED_INFO:")[1].strip())
                        print(f"[ON-DEMAND INJECTION] Injecting raw visual metrics for Object ID: {target_id} into generate_intents")
                        
                        target_obj = self.find_object_by_id(scene_data, target_id)
                        if target_obj:
                            # 날것의 JSON 스펙을 상세 텍스트 피드로 전환
                            info_feed = f"\n[RUNTIME DATA INJECTION: Detailed Specs for Object ID {target_id}]\n"
                            for key, val in target_obj.items():
                                info_feed += f"- {key}: {val}\n"
                            
                            # 대화 컨텍스트 뒤에 3D 실측 데이터를 밀어넣고 2차 확정 연산 유도
                            final_user_prompt = (
                                lightweight_context 
                                + f"\nYour Analysis: {response}\n" 
                                + info_feed 
                                + "\nNow, calculate the final robot joint task sequences using the precise values above."
                            )
                            
                            system_prompt_final = f"""
    You are a robot skill planner. Convert the injected precise data into robot skill plans.
    Available skills: {json.dumps(skill_names)}
    Return ONLY valid JSON layout:
    {{"intents":[{{"intent":"point","target_id":7}},{{"intent":"raise_low","direction":"raise"}}]}}
    """
                            response = self.generate(
                                system_prompt_final,
                                final_user_prompt,
                                max_new_tokens=96
                            )
                            print("=" * 60)
                            print("INTENTS AFTER DATA INJECTION:", response)
                    except (ValueError, IndexError):
                        print("[WARN] Failed to parse targeted injection ID in generate_intents. Proceeding to fallback.")

                # 후속 JSON 파싱 처리는 기존 레거시 코드 파이프라인을 그대로 태웁니다.
                json_text = self.extract_json(response)
                if json_text is None:
                    raise ValueError("NO JSON")

                data = json.loads(json_text)
                intents = self.normalize_intents(
                    data,
                    user_message,
                    scene_data
                )

            except Exception as e:
                print("=" * 60)
                print("INTENTS ERROR")
                print(str(e))
                intents = self.normalize_intents(
                    {"intent": "idle"},
                    user_message,
                    scene_data
                )

            print("=" * 60)
            print("FINAL INTENTS")

            print(
                json.dumps(
                    intents,
                    indent=4
                )
            )

            return intents
    # =====================================================
    # TARGET MEMORY
    # =====================================================

    def remember_target_object(
            self,
            obj
    ):

        if isinstance(obj, dict) and "id" in obj:
            self.last_target_object_id = obj.get("id")

    def resolve_reference_target(
            self,
            user_message,
            scene_data
    ):

        features = self.get_request_features(
            user_message,
            scene_data
        )

        if not features["has_reference"]:
            return None

        last_target_object_id = getattr(
            self,
            "last_target_object_id",
            None
        )

        if last_target_object_id is None:
            return None

        return self.find_object_by_id(
            scene_data,
            last_target_object_id
        )

    # =====================================================
    # FIND OBJECT BY ID
    # =====================================================

    def find_object_by_id(
            self,
            scene_data,
            target_id
    ):

        for obj in (scene_data or {}).get("objects", []):

            if obj["id"] == target_id:
                return obj

        return None

    # =====================================================
    # FIND BEST OBJECT
    # =====================================================

    def find_best_object(
            self,
            user_text,
            scene_data
    ):

        user_text = self.normalize_text(user_text)
        user_tokens = set(
            re.findall(r"[a-zA-Z0-9_]+", user_text)
        )

        best_score = 0
        best_object = None

        for obj in (scene_data or {}).get("objects", []):

            score = 0

            labels = obj.get(
                "semantic_labels",
                []
            )

            labels = [
                str(x).lower()
                for x in labels
            ]

            color = obj.get(
                "dominant_color",
                ""
            ).lower()

            for label in labels:
                if label in user_text:
                    score += 10
                    continue

                label_tokens = set(
                    re.findall(r"[a-zA-Z0-9_]+", label)
                )
                if user_tokens and label_tokens:
                    score += len(user_tokens.intersection(label_tokens)) * 4

            if color and color in user_text:
                score += 5

            object_id = str(obj["id"])

            if f"id {object_id}" in user_text:
                score += 20

            z = abs(obj["center"][2])

            score += max(0, 2 - z) * 0.1

            if score > best_score:
                best_score = score
                best_object = obj

        if best_object is not None:
            self.remember_target_object(best_object)

        return best_object

    # =====================================================
    # CLAMP
    # =====================================================

    def clamp(
            self,
            value,
            minimum,
            maximum
    ):

        return max(
            minimum,
            min(
                maximum,
                value
            )
        )

    # =====================================================
    # SMOOTH JOINT
    # =====================================================

    def smooth_joint(
            self,
            current,
            target
    ):

        delta = abs(target - current)

        if delta < 0.01:
            return current

        return current + (target - current) * self.smooth_alpha

    # =====================================================
    # BUILD ACTION
    # =====================================================

    def build_action(
            self,
            steps=60
    ):

        smoothed = {}

        for key in self.current_state:
            smoothed[key] = self.smooth_joint(
                self.current_state[key],
                self.target_state[key]
            )

        self.current_state = smoothed

        self.current_joint_0 = smoothed["joint_0"]
        self.current_joint_1 = smoothed["joint_1"]
        self.current_joint_2 = smoothed.get("joint_2", 0.0)
        self.current_joint_3 = smoothed.get("joint_3", 0.0)
        self.current_joint_4 = smoothed.get("joint_4", 0.0)

        action = dict(smoothed)
        action["steps"] = steps
        return action

    # =====================================================
    # RESOLVE SKILL
    # =====================================================

    def resolve_skill_name(
            self,
            intent_name
    ):

        intent_name = str(intent_name).lower()

        return self.skill_aliases.get(intent_name, "idle")

    # =====================================================
    # APPLY ACTION TO STATE
    # =====================================================

    def apply_action_to_state(
            self,
            action
    ):

        for key in self.JOINT_KEYS:
            if key in action:
                try:
                    self.target_state[key] = float(action[key])
                except Exception:
                    pass

        self.current_state = dict(self.target_state)

        self.current_joint_0 = self.current_state["joint_0"]
        self.current_joint_1 = self.current_state["joint_1"]
        self.current_joint_2 = self.current_state.get("joint_2", 0.0)
        self.current_joint_3 = self.current_state.get("joint_3", 0.0)
        self.current_joint_4 = self.current_state.get("joint_4", 0.0)

    # =====================================================
    # UPDATE CURRENT JOINTS
    # =====================================================

    def update_current_joints(
            self,
            j0,
            j1,
            j2=0.0,
            j3=0.0,
            j4=0.0,
            gripper=None
    ):

        self.current_joint_0 = float(j0)
        self.current_joint_1 = float(j1)
        self.current_joint_2 = float(j2)
        self.current_joint_3 = float(j3)
        self.current_joint_4 = float(j4)

        self.current_state["joint_0"] = self.current_joint_0
        self.current_state["joint_1"] = self.current_joint_1
        self.current_state["joint_2"] = self.current_joint_2
        self.current_state["joint_3"] = self.current_joint_3
        self.current_state["joint_4"] = self.current_joint_4

        self.target_state = dict(self.current_state)

    # =====================================================
    # EXECUTE ACTION COMPATIBILITY
    # =====================================================

    def execute_action(
            self,
            action
    ):

        self.last_action = action

        if self.auto_execute and self.pybullet is not None:
            self.pybullet.execute_action(action)

        return action

    # =====================================================
    # SANITIZE ACTION
    # =====================================================

    def sanitize_action(
            self,
            action
    ):

        safe = {}

        for key in self.JOINT_KEYS:
            current = float(
                getattr(
                    self,
                    f"current_{key}",
                    self.current_state.get(key, 0.0)
                )
            )
            low, high = self.joint_limits[key]

            try:
                safe[key] = self.clamp(
                    float(action.get(key, current)),
                    low,
                    high
                )
            except Exception:
                safe[key] = current

        try:
            safe["steps"] = int(action.get("steps", 60))
        except Exception:
            safe["steps"] = 60

        for key in ["target_world", "target_distance"]:
            if key in action:
                safe[key] = action[key]

        return safe

    def sanitize_sequence_action(
            self,
            action
    ):

        if (
                not isinstance(action, dict)
                or action.get("type") != "sequence"
        ):
            return self.sanitize_action(action)

        safe_actions = []
        for item in action.get("actions", []):
            if isinstance(item, dict):
                safe_actions.append(
                    self.sanitize_action(item)
                )

        if len(safe_actions) == 0:
            return self.default_idle_action()

        return {
            "type": "sequence",
            "actions": safe_actions
        }

    def final_state_from_action(
            self,
            action
    ):

        if (
                isinstance(action, dict)
                and action.get("type") == "sequence"
        ):
            actions = action.get("actions", [])
            if len(actions) > 0 and isinstance(actions[-1], dict):
                return actions[-1]

        return action

    # =====================================================
    # DEFAULT IDLE ACTION
    # =====================================================

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
    # GENERATE ROBOT ACTION
    # =====================================================

    def generate_robot_action(
            self,
            user_message,
            scene_data,
            memory_context="",
            route=None
    ):

        print("=" * 60)
        print("AUTONOMOUS CONTROL")

        if route is None:
            route = self.classify_request(
                user_message,
                scene_data
            )
        routes = self.normalize_routes(route)

        print("ROUTE:", routes)

        # =================================================
        # CHAT / VISION CHAT
        # =================================================

        if not self.route_has_skill(routes):
            return {
                "type": "conversation"
            }

        # =================================================
        # SKILL / VISION SKILL
        # =================================================

        self.last_action = None

        intents = self.generate_intents(
            user_message,
            scene_data
        )

        actions = []

        for intent in intents:
            intent_name = intent.get("intent", "idle")
            skill_name = self.resolve_skill_name(intent_name)

            skill = self.skills.get(
                skill_name,
                self.skills["idle"]
            )

            try:
                self.last_action = None
                result = skill.execute(
                    intent,
                    scene_data
                )
            except Exception as e:
                print("=" * 60)
                print("SKILL ERROR")
                print(str(e))
                result = None

            if isinstance(result, dict):
                action = result
            elif isinstance(self.last_action, dict):
                action = self.last_action
            else:
                action = self.default_idle_action()

            action = self.sanitize_sequence_action(action)
            state_action = self.final_state_from_action(action)

            for key in self.JOINT_KEYS:
                self.target_state[key] = state_action[key]

            self.apply_action_to_state(state_action)
            actions.append(action)

        print("=" * 60)
        print("FINAL ACTIONS")
        print(json.dumps(actions, indent=4))

        if len(actions) == 1:
            return actions[0]

        flattened = []
        for action in actions:
            if (
                    isinstance(action, dict)
                    and action.get("type") == "sequence"
            ):
                flattened.extend(action.get("actions", []))
            else:
                flattened.append(action)

        return {
            "type": "sequence",
            "actions": flattened
        }
