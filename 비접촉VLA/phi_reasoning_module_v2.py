
"""
phi_reasoning_module_v2.py

A self-contained, drop-in replacement architecture for the original
attention robot / phi reasoning module.

Core pipeline:
    User text -> Goal -> Candidate selection -> Plan -> Atomic skill actions -> Execution

Design goals:
- preserve the high-level AttentionRobot API surface as much as possible
- replace brittle hard-coded intent routing with goal + planner layers
- support composite commands such as "turn left and point to object 12"
- make object selection robust to 1-digit / 2-digit / multi-digit IDs
- keep the module safe to import even when optional runtime dependencies are missing

Optional external modules that can be plugged in later:
- skills.look.LookSkill
- skills.point.PointSkill
- skills.idle.IdleSkill
- skills.raise_low.RaiseLowSkill
- skills.motion.MotionSkill
- semantic_analyzer.SemanticAnalyzer

If these modules are present, the class will try to use them.
If not, the file still works with the built-in fallbacks below.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:  # pragma: no cover
    AutoTokenizer = None
    AutoModelForCausalLM = None


# =========================================================
# Optional runtime integrations
# =========================================================

try:
    from semantic_analyzer import SemanticAnalyzer  # type: ignore
except Exception:  # pragma: no cover
    class SemanticAnalyzer:
        """
        Minimal fallback semantic analyzer.

        The original system appears to use a token-ID / meaning-element layer.
        This fallback keeps the interface stable when the external module is absent.
        """

        atoms = {
            "DO": 1,
            "MOVE": 2,
            "THINK": 3,
            "SEE": 4,
            "HEAR": 5,
            "WANT": 6,
            "WHERE": 7,
            "QUESTION": 8,
            "THIS": 9,
            "SOMETHING": 10,
            "HERE": 11,
            "THEREIS": 12,
            "LEFT": 13,
            "RIGHT": 14,
            "TURN": 15,
            "ROTATE": 16,
            "POINT": 17,
            "LOOK": 18,
            "RAISE": 19,
            "LOWER": 20,
        }

        def correct_and_convert(self, text: str) -> List[int]:
            text = str(text).lower()
            out = []
            for key, value in self.atoms.items():
                if key.lower() in text:
                    out.append(value)
            return out

        def analyze_sentence(self, text: str) -> List[float]:
            # stable low-dimensional placeholder
            text = str(text).lower()
            return [
                float(len(text)),
                float(sum(ch.isdigit() for ch in text)),
                float(text.count(" ")),
                float(text.count("left") + text.count("right")),
                float(text.count("?")),
            ]


def _build_fallback_skill_class(name: str):
    class _FallbackSkill:
        def __init__(self, robot: "AttentionRobot"):
            self.robot = robot

        def execute(self, intent: Dict[str, Any], scene_data: Optional[Dict[str, Any]] = None):
            return self.robot.default_idle_action()

    _FallbackSkill.__name__ = name
    return _FallbackSkill


def _try_import_skill(path: str, cls_name: str):
    try:
        module = __import__(path, fromlist=[cls_name])
        return getattr(module, cls_name)
    except Exception:
        return _build_fallback_skill_class(cls_name)


LookSkill = _try_import_skill("skills.look", "LookSkill")
PointSkill = _try_import_skill("skills.point", "PointSkill")
IdleSkill = _try_import_skill("skills.idle", "IdleSkill")
RaiseLowSkill = _try_import_skill("skills.raise_low", "RaiseLowSkill")
MotionSkill = _try_import_skill("skills.motion", "MotionSkill")


# =========================================================
# Data structures
# =========================================================

@dataclass
class Goal:
    name: str
    target_id: Optional[int] = None
    target_text: Optional[str] = None
    direction: Optional[str] = None
    angle_deg: Optional[float] = None
    motion: Optional[str] = None
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "goal": self.name,
            "raw_text": self.raw_text,
        }
        if self.target_id is not None:
            d["target_id"] = self.target_id
        if self.target_text is not None:
            d["target_text"] = self.target_text
        if self.direction is not None:
            d["direction"] = self.direction
        if self.angle_deg is not None:
            d["angle_deg"] = self.angle_deg
        if self.motion is not None:
            d["motion"] = self.motion
        return d


@dataclass
class PlanStep:
    skill: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = {"skill": self.skill}
        out.update(self.params)
        return out


# =========================================================
# Planner / retrieval helpers
# =========================================================

class GoalGenerator:
    """
    Converts user text into a normalized goal object.

    This intentionally does not rely only on keyword matching.
    It mixes:
    - command verbs
    - direction words
    - numeric target parsing
    - object reference parsing
    """

    TURN_LEFT_PATTERNS = (
        r"\bturn\s+left\b",
        r"\brotate\s+left\b",
        r"\bleft\b",
        r"왼쪽",
        r"좌측",
    )

    TURN_RIGHT_PATTERNS = (
        r"\bturn\s+right\b",
        r"\brotate\s+right\b",
        r"\bright\b",
        r"오른쪽",
        r"우측",
    )

    POINT_PATTERNS = (
        r"\bpoint\b",
        r"\bpoint\s+to\b",
        r"\bpoint\s+at\b",
        r"가리켜",
        r"가리켜줘",
    )

    LOOK_PATTERNS = (
        r"\blook\b",
        r"\blook\s+at\b",
        r"\bface\b",
        r"봐",
        r"향해",
    )

    RAISE_PATTERNS = (
        r"\braise\b",
        r"\blift\b",
        r"\bup\b",
        r"올려",
        r"위로",
    )

    LOWER_PATTERNS = (
        r"\blower\b",
        r"\bdown\b",
        r"\blower\s+it\b",
        r"내려",
        r"아래로",
    )

    MOTION_PATTERNS = {
        "wave": (r"\bwave\b", r"\bhello\b", r"\bhi\b", r"흔들", r"인사"),
        "scan": (r"\bscan\b", r"\bsearch\b", r"둘러", r"찾아"),
        "home": (r"\bhome\b", r"\bidle\b", r"\brest\b", r"제자리", r"쉬어", r"멈춰"),
        "extend": (r"\bextend\b", r"\bstraight\b", r"\breach out\b", r"뻗", r"펴"),
        "fold": (r"\bfold\b", r"\btuck\b", r"접어", r"접"),
    }

    def __init__(self, semantic_analyzer: Optional[SemanticAnalyzer] = None):
        self.semantic_analyzer = semantic_analyzer or SemanticAnalyzer()

    @staticmethod
    def normalize_text(text: str) -> str:
        text = str(text).lower().strip()
        text = text.replace("’", "'").replace("“", '"').replace("”", '"')
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _match_any(patterns: Sequence[str], text: str) -> bool:
        return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)

    @staticmethod
    def _extract_number(text: str) -> Optional[int]:
        patterns = [
            r"\bid\s*[:=]?\s*(\d+)\b",
            r"\bobject\s*[:=]?\s*(\d+)\b",
            r"\btarget\s*[:=]?\s*(\d+)\b",
            r"\b번호\s*(\d+)\b",
            r"\b(\d+)\s*번\b",
            r"\b(\d+)\b",
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    continue
        return None

    def generate(self, user_text: str, scene_data: Optional[Dict[str, Any]] = None) -> Goal:
        text = self.normalize_text(user_text)
        scene_data = scene_data or {}

        target_id = self._extract_number(text)
        target_text = None

        # preserve object mentions when user writes "this", "that", etc.
        if re.search(r"\b(this|that|it|one|저것|이것|그것)\b", text):
            target_text = re.search(r"\b(this|that|it|one|저것|이것|그것)\b", text).group(0)

        if self._match_any(self.TURN_LEFT_PATTERNS, text):
            return Goal(name="rotate", direction="left", angle_deg=30.0, raw_text=user_text)

        if self._match_any(self.TURN_RIGHT_PATTERNS, text):
            return Goal(name="rotate", direction="right", angle_deg=30.0, raw_text=user_text)

        if self._match_any(self.POINT_PATTERNS, text):
            return Goal(name="point", target_id=target_id, target_text=target_text, raw_text=user_text)

        if self._match_any(self.LOOK_PATTERNS, text):
            return Goal(name="look", target_id=target_id, target_text=target_text, raw_text=user_text)

        if self._match_any(self.RAISE_PATTERNS, text):
            return Goal(name="raise_low", direction="raise", raw_text=user_text)

        if self._match_any(self.LOWER_PATTERNS, text):
            return Goal(name="raise_low", direction="lower", raw_text=user_text)

        for motion_name, patterns in self.MOTION_PATTERNS.items():
            if self._match_any(patterns, text):
                return Goal(name="motion", motion=motion_name, raw_text=user_text)

        # fallback: if the text mentions an object number but no explicit verb,
        # treat it as a generic object interaction goal.
        if target_id is not None:
            return Goal(name="object_reference", target_id=target_id, raw_text=user_text)

        return Goal(name="idle", raw_text=user_text)


class ObjectRetriever:
    """
    Builds a shortlist of candidate objects from scene_data.
    The shortlist is designed to be small enough to reduce downstream ambiguity.
    """

    @staticmethod
    def _safe_label_set(obj: Dict[str, Any]) -> List[str]:
        labels = obj.get("semantic_labels", [])
        out = []
        for label in labels:
            if label is None:
                continue
            out.append(str(label).strip().lower())
        return out

    @staticmethod
    def _safe_color(obj: Dict[str, Any]) -> str:
        return str(obj.get("dominant_color", "")).strip().lower()

    @staticmethod
    def _safe_center(obj: Dict[str, Any]) -> Tuple[float, float, float]:
        center = obj.get("center", (0.0, 0.0, 0.0))
        try:
            return float(center[0]), float(center[1]), float(center[2])
        except Exception:
            return 0.0, 0.0, 0.0

    def find_candidates(
        self,
        goal: Goal,
        scene_data: Optional[Dict[str, Any]],
        user_text: str = "",
        max_candidates: int = 5,
    ) -> List[Dict[str, Any]]:
        scene_data = scene_data or {}
        objects = scene_data.get("objects", [])
        text = str(user_text).lower()

        if not objects:
            return []

        # Exact ID first.
        if goal.target_id is not None:
            exact = [obj for obj in objects if int(obj.get("id", -1)) == int(goal.target_id)]
            if exact:
                return exact[:max_candidates]

        # Score all visible objects.
        scored: List[Tuple[float, Dict[str, Any]]] = []
        tokens = set(re.findall(r"[a-zA-Z0-9_]+", text))

        for obj in objects:
            score = 0.0
            oid = obj.get("id", None)
            labels = self._safe_label_set(obj)
            color = self._safe_color(obj)
            x, y, z = self._safe_center(obj)

            # label / token overlap
            for label in labels:
                if not label:
                    continue
                if label in text:
                    score += 10.0
                label_tokens = set(re.findall(r"[a-zA-Z0-9_]+", label))
                if tokens and label_tokens:
                    overlap = len(tokens.intersection(label_tokens))
                    score += float(overlap) * 4.0

            # color match
            if color and color in text:
                score += 5.0

            # direct textual ID matches
            if oid is not None:
                oid_str = str(oid)
                if f"id {oid_str}" in text or f"object {oid_str}" in text or f"target {oid_str}" in text or f"{oid_str}번" in text:
                    score += 20.0

            # spatial bias: prefer central / closer objects slightly
            score += max(0.0, 2.0 - abs(z)) * 0.1
            score += max(0.0, 1.0 - abs(x)) * 0.05

            scored.append((score, obj))

        scored.sort(key=lambda item: item[0], reverse=True)
        candidates = [obj for score, obj in scored if score > 0.0][:max_candidates]

        # If nothing matched strongly, return a compact shortlist of top objects anyway.
        if not candidates:
            candidates = [obj for _, obj in scored[:max_candidates]]

        return candidates


class Planner:
    """
    Converts a Goal into an ordered list of atomic skills.
    """

    def plan(
        self,
        goal: Goal,
        candidates: Optional[List[Dict[str, Any]]] = None,
        scene_data: Optional[Dict[str, Any]] = None,
    ) -> List[PlanStep]:
        candidates = candidates or []

        if goal.name == "rotate":
            angle = float(goal.angle_deg or 30.0)
            if str(goal.direction).lower() == "right":
                angle = -abs(angle)
            else:
                angle = abs(angle)
            return [PlanStep("rotate_yaw", {"angle_deg": angle})]

        if goal.name == "look":
            if goal.target_id is not None:
                return [PlanStep("look_target", {"target_id": goal.target_id})]
            if candidates:
                return [PlanStep("look_target", {"target_id": candidates[0].get("id")})]
            return [PlanStep("look")]

        if goal.name == "point":
            steps: List[PlanStep] = []
            if goal.target_id is not None:
                steps.append(PlanStep("look_target", {"target_id": goal.target_id}))
                steps.append(PlanStep("point_target", {"target_id": goal.target_id}))
            elif candidates:
                tid = candidates[0].get("id")
                steps.append(PlanStep("look_target", {"target_id": tid}))
                steps.append(PlanStep("point_target", {"target_id": tid}))
            else:
                steps.append(PlanStep("point"))
            return steps

        if goal.name == "raise_low":
            direction = (goal.direction or "raise").lower()
            if direction not in ("raise", "lower"):
                direction = "raise"
            return [PlanStep("raise_low", {"direction": direction})]

        if goal.name == "motion":
            motion = (goal.motion or "home").lower()
            if motion not in ("wave", "scan", "home", "extend", "fold"):
                motion = "home"
            return [PlanStep("motion", {"motion": motion})]

        if goal.name == "object_reference":
            if candidates:
                tid = candidates[0].get("id")
                return [PlanStep("look_target", {"target_id": tid})]
            return [PlanStep("idle")]

        return [PlanStep("idle")]


# =========================================================
# Built-in fallback atomic skill implementations
# =========================================================

class _AtomicSkillBase:
    def __init__(self, robot: "AttentionRobot"):
        self.robot = robot

    def execute(self, intent: Dict[str, Any], scene_data: Optional[Dict[str, Any]] = None):
        return self.robot.default_idle_action()


class RotateYawSkill(_AtomicSkillBase):
    def execute(self, intent: Dict[str, Any], scene_data: Optional[Dict[str, Any]] = None):
        angle_deg = float(intent.get("angle_deg", 0.0))
        current = float(self.robot.current_joint_0)
        target = current + math.radians(angle_deg)
        action = {
            "joint_0": target,
            "joint_1": self.robot.current_joint_1,
            "joint_2": self.robot.current_joint_2,
            "joint_3": self.robot.current_joint_3,
            "joint_4": self.robot.current_joint_4,
            "steps": 80,
        }
        self.robot.apply_action_to_state(action)
        return action


class LookTargetSkill(_AtomicSkillBase):
    def execute(self, intent: Dict[str, Any], scene_data: Optional[Dict[str, Any]] = None):
        tid = intent.get("target_id")
        self.robot.last_target_object_id = tid
        action = {
            "type": "look_target",
            "target_id": tid,
            "steps": 60,
        }
        self.robot.execute_action(action)
        return action


class PointTargetSkill(_AtomicSkillBase):
    def execute(self, intent: Dict[str, Any], scene_data: Optional[Dict[str, Any]] = None):
        tid = intent.get("target_id")
        self.robot.last_target_object_id = tid
        action = {
            "type": "point_target",
            "target_id": tid,
            "steps": 60,
        }
        self.robot.execute_action(action)
        return action


class LookFallbackSkill(_AtomicSkillBase):
    def execute(self, intent: Dict[str, Any], scene_data: Optional[Dict[str, Any]] = None):
        return {
            "type": "look",
            "steps": 50,
        }


class PointFallbackSkill(_AtomicSkillBase):
    def execute(self, intent: Dict[str, Any], scene_data: Optional[Dict[str, Any]] = None):
        return {
            "type": "point",
            "steps": 50,
        }


class IdleFallbackSkill(_AtomicSkillBase):
    def execute(self, intent: Dict[str, Any], scene_data: Optional[Dict[str, Any]] = None):
        return self.robot.default_idle_action()


class RaiseLowFallbackSkill(_AtomicSkillBase):
    def execute(self, intent: Dict[str, Any], scene_data: Optional[Dict[str, Any]] = None):
        direction = str(intent.get("direction", "raise")).lower()
        action = dict(self.robot.current_state)
        delta = 0.15 if direction == "raise" else -0.15
        action["joint_1"] = self.robot.clamp(
            float(action.get("joint_1", 0.0)) + delta,
            self.robot.JOINT_1_MIN,
            self.robot.JOINT_1_MAX,
        )
        action["steps"] = 70
        self.robot.apply_action_to_state(action)
        return action


class MotionFallbackSkill(_AtomicSkillBase):
    def execute(self, intent: Dict[str, Any], scene_data: Optional[Dict[str, Any]] = None):
        motion = str(intent.get("motion", "home")).lower()
        if motion == "wave":
            return {"type": "motion", "motion": "wave", "steps": 80}
        if motion == "scan":
            return {"type": "motion", "motion": "scan", "steps": 90}
        if motion == "extend":
            return {"type": "motion", "motion": "extend", "steps": 80}
        if motion == "fold":
            return {"type": "motion", "motion": "fold", "steps": 80}
        return {"type": "motion", "motion": "home", "steps": 80}


# =========================================================
# Main module
# =========================================================

class AttentionRobot:
    """
    A more robust, compositional version of the original module.

    Public surface intentionally keeps these names:
    - classify_request
    - ask_question
    - generate_robot_action
    - generate_intents
    - generate_intent
    - build_scene_text
    - find_best_object
    - resolve_reference_target
    """

    VALID_ROUTES = (
        "chat",
        "vision_chat",
        "skill",
        "vision_skill",
    )

    ROUTE_PRIORITY = {
        "chat": 0,
        "vision_chat": 1,
        "skill": 2,
        "vision_skill": 3,
    }

    def __init__(self, pybullet=None, auto_execute: bool = False):
        print("=" * 60)
        print("INITIALIZING ATTENTION ROBOT V2")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("DEVICE:", self.device)

        torch.set_float32_matmul_precision("high")
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        self.pybullet = pybullet
        self.auto_execute = auto_execute
        self.last_action = None

        # model is optional; module remains usable even if unavailable
        self.use_local_phi = os.getenv("VLA_USE_LOCAL_PHI", "1").strip().lower() in ("1", "true", "yes", "on")
        self.model = None
        self.tokenizer = None

        if self.use_local_phi and AutoTokenizer is not None and AutoModelForCausalLM is not None:
            try:
                model_name = os.getenv("VLA_MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct")
                print("=" * 60)
                print("LOADING PHI MODEL:", model_name)

                self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
                self.model.eval()
            except Exception as e:
                print("MODEL LOAD FAILED:", e)
                self.model = None
                self.tokenizer = None

        # robot state
        self.FIXED_JOINT_1 = 0.0
        self.JOINT_KEYS = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4"]

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
            "joint_4": (self.JOINT_4_MIN, self.JOINT_4_MAX),
        }

        self.camera_yaw_offset = 0.0
        self.servo_direction = 1.0
        self.yaw_scale = 1.0
        self.smooth_alpha = 0.7
        self.max_history_turns = 8
        self.use_ai_routing = os.getenv("VLA_AI_ROUTING", "1").strip().lower() in ("1", "true", "yes", "on")

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
            "joint_4": self.current_joint_4,
        }

        self.target_state = dict(self.current_state)
        self.conversation_history: List[Dict[str, str]] = []
        self.last_target_object_id: Optional[int] = None
        self.last_route: Optional[List[str]] = None

        self.semantic_analyzer = SemanticAnalyzer()
        self.goal_generator = GoalGenerator(self.semantic_analyzer)
        self.object_retriever = ObjectRetriever()
        self.planner = Planner()

        self.look_skill = LookSkill(self) if LookSkill is not None else LookFallbackSkill(self)
        self.point_skill = PointSkill(self) if PointSkill is not None else PointFallbackSkill(self)
        self.idle_skill = IdleSkill(self) if IdleSkill is not None else IdleFallbackSkill(self)
        self.raise_low_skill = RaiseLowSkill(self) if RaiseLowSkill is not None else RaiseLowSkill(self)
        self.motion_skill = MotionSkill(self) if MotionSkill is not None else MotionFallbackSkill(self)

        self.skills = {
            "look": self.look_skill,
            "look_target": LookTargetSkill(self),
            "point": self.point_skill,
            "point_target": PointTargetSkill(self),
            "idle": self.idle_skill,
            "raise_low": self.raise_low_skill,
            "motion": self.motion_skill,
            "rotate_yaw": RotateYawSkill(self),
        }

        self.skill_aliases = {
            "look": "look",
            "look_at_object": "look",
            "look_target": "look_target",
            "point": "point",
            "point_at_object": "point",
            "point_target": "point_target",
            "raise": "raise_low",
            "lower": "raise_low",
            "raise_low": "raise_low",
            "wave": "motion",
            "scan": "motion",
            "home": "motion",
            "extend": "motion",
            "fold": "motion",
            "motion": "motion",
            "rotate": "rotate_yaw",
            "rotate_yaw": "rotate_yaw",
            "idle": "idle",
        }

        print("=" * 60)
        print("MODEL WARMUP")

        if self.model is not None and self.tokenizer is not None:
            try:
                self.generate("You are ready.", "hello", 4)
            except Exception as e:
                print("WARMUP ERROR:", e)

        print("=" * 60)
        print("ATTENTION ROBOT READY")

    # =====================================================
    # Compatibility
    # =====================================================

    def attach_pybullet(self, pybullet):
        self.pybullet = pybullet

    # =====================================================
    # History
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
            "assistant": str(assistant_message),
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
        return "\n".join(lines) if lines else "No prior conversation."

    # =====================================================
    # Text / scene helpers
    # =====================================================

    @staticmethod
    def normalize_text(text: str) -> str:
        text = str(text).lower().strip()
        text = text.replace("’", "'").replace("“", '"').replace("”", '"')
        text = re.sub(r"\s+", " ", text)
        return text

    def build_scene_text(self, scene_data):
        scene_data = scene_data or {}
        lines = []
        for obj in scene_data.get("objects", [])[:32]:
            center = obj.get("center", (0.0, 0.0, 0.0))
            try:
                x, y, z = round(float(center[0]), 2), round(float(center[1]), 2), round(float(center[2]), 2)
            except Exception:
                x = y = z = 0.0

            labels = obj.get("semantic_labels", [])
            label = "/".join(str(x) for x in labels[:3]) if labels else "object"
            color = str(obj.get("dominant_color", "unknown"))
            lines.append(f"id={obj.get('id')} type={label} color={color} position=({x},{y},{z})")
        return "\n".join(lines) if lines else "No visible objects."

    # =====================================================
    # Feature extraction
    # =====================================================

    def get_request_features(self, user_message, scene_data=None):
        text = self.normalize_text(user_message)
        scene_data = scene_data or {}

        token_stream = self.semantic_analyzer.correct_and_convert(text)
        semantic_vector = self.semantic_analyzer.analyze_sentence(text)
        atom_map = getattr(self.semantic_analyzer, "atoms", {})

        action_atoms = {
            atom_map.get("DO"),
            atom_map.get("MOVE"),
            atom_map.get("THINK"),
            atom_map.get("SEE"),
            atom_map.get("HEAR"),
            atom_map.get("WANT"),
            atom_map.get("TURN"),
            atom_map.get("ROTATE"),
            atom_map.get("POINT"),
            atom_map.get("LOOK"),
            atom_map.get("RAISE"),
            atom_map.get("LOWER"),
        }
        has_action = any(atom in token_stream for atom in action_atoms if atom is not None)

        if "look like" in text or "looks like" in text:
            has_action = any(atom in token_stream for atom in (action_atoms - {atom_map.get("SEE")}) if atom is not None)

        question_atoms = {atom_map.get("WHERE"), atom_map.get("QUESTION")}
        has_question = any(atom in token_stream for atom in question_atoms if atom is not None)

        reference_atoms = {atom_map.get("THIS"), atom_map.get("SOMETHING"), atom_map.get("HERE"), atom_map.get("THEREIS")}
        has_reference = any(atom in token_stream for atom in reference_atoms if atom is not None)

        has_id = re.search(r"\bid\s*[:=]?\s*(\d+)\b", text) is not None or re.search(r"\b(\d+)\s*번\b", text) is not None
        has_objects = bool(scene_data.get("objects"))

        scene_terms: List[str] = []
        for obj in scene_data.get("objects", []):
            scene_terms.extend([str(x).lower() for x in obj.get("semantic_labels", [])])
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
            "semantic_vector": semantic_vector,
        }

    # =====================================================
    # Model generation
    # =====================================================

    def clean_response(self, text):
        text = str(text).replace("```json", "").replace("```", "")
        for stop in ("\nUSER", "\nUser", "\nAssistant", "\nHuman"):
            if stop in text:
                text = text.split(stop)[0]
        return text.strip()

    def extract_json(self, text):
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

        json_text = text[start:] if end is None else text[start:end]
        open_count = json_text.count("{")
        close_count = json_text.count("}")
        if open_count > close_count:
            json_text += "}" * (open_count - close_count)
        return json_text

    @torch.inference_mode()
    def generate(self, system_prompt, user_prompt, max_new_tokens=40):
        if self.model is None or self.tokenizer is None:
            # Deterministic fallback: best effort assistant text.
            return "Okay."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        try:
            target_device = next(self.model.parameters()).device
        except Exception:
            target_device = self.device

        inputs = {k: v.to(target_device) for k, v in inputs.items()}

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
            use_cache=True,
        )

        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return self.clean_response(response)

    # =====================================================
    # Route normalization and helpers
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
            if route in self.VALID_ROUTES and route not in routes:
                routes.append(route)

        if not routes:
            routes = ["chat"]

        routes.sort(key=lambda name: self.ROUTE_PRIORITY.get(name, 99))
        return routes

    def route_has_skill(self, routes):
        return any(route in ("skill", "vision_skill") for route in self.normalize_routes(routes))

    def route_has_vision(self, routes):
        return any(route in ("vision_chat", "vision_skill") for route in self.normalize_routes(routes))

    def route_has_chat(self, routes):
        return any(route in ("chat", "vision_chat") for route in self.normalize_routes(routes))

    # =====================================================
    # Object resolution
    # =====================================================

    def remember_target_object(self, obj):
        if isinstance(obj, dict) and "id" in obj:
            self.last_target_object_id = obj.get("id")

    def find_object_by_id(self, scene_data, target_id):
        for obj in (scene_data or {}).get("objects", []):
            if obj.get("id") == target_id:
                return obj
        return None

    def resolve_reference_target(self, user_message, scene_data):
        features = self.get_request_features(user_message, scene_data)
        if not features["has_reference"]:
            return None
        last_target_object_id = getattr(self, "last_target_object_id", None)
        if last_target_object_id is None:
            return None
        return self.find_object_by_id(scene_data, last_target_object_id)

    def find_best_object(self, user_text, scene_data):
        user_text = self.normalize_text(user_text)
        user_tokens = set(re.findall(r"[a-zA-Z0-9_]+", user_text))

        best_score = 0.0
        best_object = None

        for obj in (scene_data or {}).get("objects", []):
            score = 0.0
            labels = [str(x).lower() for x in obj.get("semantic_labels", [])]
            color = str(obj.get("dominant_color", "")).lower()
            object_id = str(obj.get("id", ""))

            for label in labels:
                if label in user_text:
                    score += 10.0
                    continue
                label_tokens = set(re.findall(r"[a-zA-Z0-9_]+", label))
                if user_tokens and label_tokens:
                    score += len(user_tokens.intersection(label_tokens)) * 4.0

            if color and color in user_text:
                score += 5.0

            if object_id:
                if f"id {object_id}" in user_text or f"object {object_id}" in user_text or f"target {object_id}" in user_text or f"{object_id}번" in user_text:
                    score += 20.0

            center = obj.get("center", [0.0, 0.0, 0.0])
            try:
                z = abs(float(center[2]))
                score += max(0.0, 2.0 - z) * 0.1
            except Exception:
                pass

            if score > best_score:
                best_score = score
                best_object = obj

        if best_object is not None:
            self.remember_target_object(best_object)

        return best_object

    # =====================================================
    # Routing
    # =====================================================

    def route_from_rules(self, user_message, scene_data=None):
        text = self.normalize_text(user_message)
        scene_data = scene_data or {}

        greetings = ("hello", "hi", "hey", "안녕", "하이", "thanks", "thank you", "고마워", "고맙", "bye", "goodbye", "잘가")

        action_words = (
            "point", "look at", "point to", "face", "turn", "rotate",
            "봐", "보여", "가리켜", "가리켜줘", "가리켜 줘", "집어", "향해", "돌아", "돌려",
            "raise", "lower", "up", "down", "올려", "내려", "위로", "아래로",
        )

        question_words = ("where", "what", "which", "closest", "nearest", "어디", "뭐", "무엇", "어느", "가장", "가까운", "위치")

        features = self.get_request_features(user_message, scene_data)
        has_action = any(w in text for w in action_words) or features["has_action"]
        has_question = any(w in text for w in question_words) or features["has_question"]
        has_id = features["has_id"]
        has_scene_term = features["has_scene_term"]
        has_objects = features["has_objects"]

        if any(g in text for g in greetings) and not has_action and not has_id:
            return ["chat"]

        if has_question and (has_action or has_id):
            if has_objects:
                return ["vision_chat", "vision_skill"]
            return ["chat", "skill"]

        if has_question and not has_action:
            if has_objects:
                return ["vision_chat"]
            return ["chat"]

        if has_action or has_id:
            if has_objects:
                return ["vision_skill"]
            return ["skill"]

        if has_scene_term and has_objects:
            return ["vision_chat"]

        return None

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
                "mentions_object_id": features["has_id"],
            },
            "previous_referenced_object": self.summarize_object(referenced),
            "previous_routes": previous_routes,
        }
        return json.dumps(context, ensure_ascii=False, indent=2)

    def summarize_object(self, obj):
        if not isinstance(obj, dict):
            return None

        summary = {
            "id": obj.get("id"),
            "labels": [str(label) for label in obj.get("semantic_labels", [])[:3]],
            "color": str(obj.get("dominant_color", "")),
        }
        center = obj.get("center", None)
        if center is not None:
            try:
                summary["center"] = [round(float(value), 3) for value in center[:3]]
            except Exception:
                pass
        return summary

    def route_needs_phi_recheck(self, routes, user_message, scene_data=None):
        features = self.get_request_features(user_message, scene_data)
        if not features["has_action"]:
            return False
        return not self.route_has_skill(routes)

    def recheck_route_with_phi(self, user_message, scene_data, previous_routes):
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
            previous_routes=self.normalize_routes(previous_routes),
        )

        try:
            response = self.generate(system_prompt, router_context, max_new_tokens=48)
            json_text = self.extract_json(response)
            if json_text is None:
                raise ValueError("NO JSON")
            return self.normalize_routes(json.loads(json_text))
        except Exception:
            return self.normalize_routes(previous_routes)

    def classify_request(self, user_message, scene_data=None):
        if not self.use_ai_routing:
            rule_route = self.route_from_rules(user_message, scene_data)
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
        router_context = self.build_router_context(user_message, scene_data)

        try:
            response = self.generate(system_prompt, router_context, max_new_tokens=48)
            json_text = self.extract_json(response)
            if json_text is None:
                raise ValueError("NO JSON")
            data = json.loads(json_text)
            routes = self.normalize_routes(data)
            if self.route_needs_phi_recheck(routes, user_message, scene_data):
                routes = self.recheck_route_with_phi(user_message, scene_data, routes)
            self.last_route = routes
            return routes
        except Exception:
            route = self.route_from_rules(user_message, scene_data)
            if route is not None:
                routes = self.normalize_routes(route)
                self.last_route = routes
                return routes
            self.last_route = ["chat"]
            return ["chat"]

    # =====================================================
    # Conversations
    # =====================================================

    def filter_bad_response(self, response):
        lower = response.lower()
        bad_patterns = (
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
            "text only ai",
        )
        for p in bad_patterns:
            if p in lower:
                return "I can see objects in front of me."
        return response

    def normal_chat(self, user_message, memory_context=""):
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
        response = self.generate(system_prompt, user_prompt, max_new_tokens=40)
        return self.filter_bad_response(response)

    def vision_chat(self, user_message, scene_data, memory_context=""):
        scene_text = self.build_scene_text(scene_data)
        system_prompt = """
You are a real robot arm.

You can physically observe objects.

Never say:
- as an AI
- language model
- cannot see
- virtual assistant

Speak naturally and briefly.
"""
        user_prompt = f"""
Conversation history:
{self.build_history_text(memory_context)}

Scene:
{scene_text}

User:
{user_message}
"""
        response = self.generate(system_prompt, user_prompt, max_new_tokens=40)
        return self.filter_bad_response(response)

    def ask_question(self, user_message, scene_data=None, memory_context="", route=None):
        if route is None:
            route = self.classify_request(user_message, scene_data)
        routes = self.normalize_routes(route)

        print("=" * 60)
        print("ROUTE:", routes)

        if self.route_has_skill(routes):
            response = "Okay."
            self.remember_turn(user_message, response)
            return response

        if self.route_has_vision(routes):
            if scene_data is not None:
                self.find_best_object(user_message, scene_data)
            response = self.normal_chat(user_message, memory_context=memory_context) if scene_data is None else self.vision_chat(user_message, scene_data, memory_context=memory_context)
            self.remember_turn(user_message, response)
            return response

        if self.route_has_chat(routes) or self.route_has_skill(routes):
            response = self.normal_chat(user_message, memory_context=memory_context)
            self.remember_turn(user_message, response)
            return response

        response = self.normal_chat(user_message, memory_context=memory_context)
        self.remember_turn(user_message, response)
        return response

    # =====================================================
    # Goal / intent generation
    # =====================================================

    def generate_intent(self, user_message, scene_data):
        goal = self.goal_generator.generate(user_message, scene_data)
        candidates = self.object_retriever.find_candidates(goal, scene_data, user_message=user_message)
        plan = self.planner.plan(goal, candidates, scene_data)

        first = plan[0].to_dict() if plan else {"intent": "idle"}

        if goal.name in ("look", "point") and goal.target_id is None and candidates:
            first["target_id"] = candidates[0].get("id")

        if goal.name == "rotate":
            first = {
                "intent": "rotate_yaw",
                "angle_deg": goal.angle_deg if goal.angle_deg is not None else 30.0,
            }

        if goal.name == "raise_low":
            first = {
                "intent": "raise_low",
                "direction": goal.direction or "raise",
            }

        if goal.name == "motion":
            first = {
                "intent": "motion",
                "motion": goal.motion or "home",
            }

        if goal.name == "object_reference" and candidates:
            first = {
                "intent": "look_target",
                "target_id": candidates[0].get("id"),
            }

        first["user_message"] = user_message
        if goal.target_id is not None and "target_id" not in first:
            first["target_id"] = goal.target_id
        if goal.direction is not None and "direction" not in first:
            first["direction"] = goal.direction
        if goal.motion is not None and "motion" not in first:
            first["motion"] = goal.motion

        return first

    def resolve_skill_name(self, intent_name):
        intent_name = str(intent_name).lower()
        return self.skill_aliases.get(intent_name, "idle")

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
                if direction not in ("raise", "lower", "up", "down", "low"):
                    direction = "raise"
                intent["direction"] = "lower" if direction in ("lower", "down", "low") else "raise"

            if intent_name == "motion":
                motion = str(item.get("motion", "home")).lower()
                if motion not in ("wave", "scan", "home", "extend", "fold"):
                    motion = "home"
                intent["motion"] = motion

            if intent_name == "rotate_yaw":
                try:
                    intent["angle_deg"] = float(item.get("angle_deg", 30.0))
                except Exception:
                    intent["angle_deg"] = 30.0

            target_id = item.get("target_id", None)
            try:
                if target_id is not None:
                    target_id = int(target_id)
            except Exception:
                target_id = None

            if target_id is None and intent_name in ("look", "point", "look_target", "point_target"):
                referenced = self.resolve_reference_target(user_message, scene_data)
                if referenced is not None:
                    target_id = referenced.get("id", None)

            if target_id is None and intent_name in ("look", "point", "look_target", "point_target"):
                best = self.find_best_object(user_message, scene_data)
                if best is not None:
                    target_id = best.get("id", None)

            if target_id is not None:
                if valid_ids and target_id not in valid_ids:
                    target_id = None
                else:
                    intent["target_id"] = target_id
                    target_obj = self.find_object_by_id(scene_data, target_id)
                    if target_obj is not None:
                        intent["target"] = target_obj
                        self.remember_target_object(target_obj)

            if intent_name in ("look", "point", "look_target", "point_target") and "target_id" not in intent:
                intent["intent"] = "idle"

            key = (intent.get("intent"), intent.get("target_id"), intent.get("direction"), intent.get("motion"), intent.get("angle_deg"))
            if key in seen:
                continue
            seen.add(key)
            intents.append(intent)

        non_idle = [intent for intent in intents if intent.get("intent") != "idle"]
        if non_idle:
            return non_idle

        return [{"intent": "idle", "user_message": user_message}]

    def generate_intents(self, user_message, scene_data):
        scene_data = scene_data or {}

        print("=" * 60)
        print("GENERATING INTENTS")

        quick_goal = self.goal_generator.generate(user_message, scene_data)
        candidates = self.object_retriever.find_candidates(quick_goal, scene_data, user_message=user_message)
        quick_plan = self.planner.plan(quick_goal, candidates, scene_data)

        quick_intents = [step.to_dict() for step in quick_plan] if quick_plan else [{"intent": "idle"}]
        quick_intents = self.normalize_intents(quick_intents, user_message, scene_data)

        if any(intent.get("intent") != "idle" for intent in quick_intents):
            print("=" * 60)
            print("FINAL INTENTS")
            print(json.dumps(quick_intents, indent=4, ensure_ascii=False))
            return quick_intents

        # Optional LLM planner if the shortcut path yields idle.
        scene_text = self.build_scene_text(scene_data)
        skill_names = sorted(self.skills.keys())
        last_target = self.find_object_by_id(scene_data, getattr(self, "last_target_object_id", None))
        reference_text = json.dumps(self.summarize_object(last_target), ensure_ascii=False) if last_target is not None else "None"

        system_prompt = f"""
You are a robot skill planner.

Return ONLY JSON.

Available skills:
{json.dumps(skill_names)}

VALID:
{{"intents":[{{"intent":"look","target_id":7}}]}}
{{"intents":[{{"intent":"point","target_id":7}},{{"intent":"raise_low","direction":"raise"}}]}}
{{"intents":[{{"intent":"motion","motion":"wave"}}]}}
{{"intents":[{{"intent":"motion","motion":"scan"}}]}}
{{"intents":[{{"intent":"idle"}}]}}

Rules:
- choose every skill needed, in execution order
- use only available skill names
- target_id must come from the scene when a skill needs an object
- use raise_low with direction raise or lower
- use motion with motion wave, scan, home, extend, or fold for targetless body motions
- output JSON only
- no markdown
- stop immediately after }}
"""
        user_prompt = f"""
Scene:
{scene_text}

Previous referenced object:
{reference_text}

User:
{user_message}
"""
        try:
            response = self.generate(system_prompt, user_prompt, max_new_tokens=96)
            json_text = self.extract_json(response)
            if json_text is None:
                raise ValueError("NO JSON")
            data = json.loads(json_text)
            intents = self.normalize_intents(data, user_message, scene_data)

            features = self.get_request_features(user_message, scene_data)
            if features["has_action"] and len(intents) == 1 and intents[0].get("intent") == "idle":
                intents = self.normalize_intents(self.generate_intent(user_message, scene_data), user_message, scene_data)
        except Exception as e:
            print("=" * 60)
            print("INTENTS ERROR")
            print(str(e))
            intents = self.normalize_intents(self.generate_intent(user_message, scene_data), user_message, scene_data)

        print("=" * 60)
        print("FINAL INTENTS")
        print(json.dumps(intents, indent=4, ensure_ascii=False))
        return intents

    # =====================================================
    # Action/state helpers
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

    def apply_action_to_state(self, action):
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

    def update_current_joints(self, j0, j1, j2=0.0, j3=0.0, j4=0.0, gripper=None):
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
            try:
                safe[key] = self.clamp(float(action.get(key, current)), low, high)
            except Exception:
                safe[key] = current

        try:
            safe["steps"] = int(action.get("steps", 60))
        except Exception:
            safe["steps"] = 60

        for key in ("target_world", "target_distance"):
            if key in action:
                safe[key] = action[key]
        return safe

    def sanitize_sequence_action(self, action):
        if not isinstance(action, dict) or action.get("type") != "sequence":
            return self.sanitize_action(action)

        safe_actions = []
        for item in action.get("actions", []):
            if isinstance(item, dict):
                safe_actions.append(self.sanitize_action(item))

        if not safe_actions:
            return self.default_idle_action()

        return {"type": "sequence", "actions": safe_actions}

    def final_state_from_action(self, action):
        if isinstance(action, dict) and action.get("type") == "sequence":
            actions = action.get("actions", [])
            if actions and isinstance(actions[-1], dict):
                return actions[-1]
        return action

    def default_idle_action(self):
        return {
            "joint_0": self.current_joint_0,
            "joint_1": self.FIXED_JOINT_1,
            "joint_2": 0.0,
            "joint_3": 0.0,
            "joint_4": 0.0,
            "steps": 120,
        }

    # =====================================================
    # Action generation
    # =====================================================

    def generate_robot_action(self, user_message, scene_data, memory_context="", route=None):
        print("=" * 60)
        print("AUTONOMOUS CONTROL")

        if route is None:
            route = self.classify_request(user_message, scene_data)
        routes = self.normalize_routes(route)

        print("ROUTE:", routes)

        if not self.route_has_skill(routes):
            return {"type": "conversation"}

        self.last_action = None
        intents = self.generate_intents(user_message, scene_data)
        actions = []

        for intent in intents:
            intent_name = intent.get("intent", "idle")
            skill_name = self.resolve_skill_name(intent_name)
            skill = self.skills.get(skill_name, self.skills["idle"])

            try:
                self.last_action = None
                result = skill.execute(intent, scene_data)
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

            if isinstance(state_action, dict):
                for key in self.JOINT_KEYS:
                    if key in state_action:
                        self.target_state[key] = state_action[key]
                self.apply_action_to_state(state_action)

            actions.append(action)

        print("=" * 60)
        print("FINAL ACTIONS")
        print(json.dumps(actions, indent=4, ensure_ascii=False))

        if len(actions) == 1:
            return actions[0]

        flattened = []
        for action in actions:
            if isinstance(action, dict) and action.get("type") == "sequence":
                flattened.extend(action.get("actions", []))
            else:
                flattened.append(action)

        return {"type": "sequence", "actions": flattened}


# =========================================================
# Backward-compatible alias
# =========================================================

AttentionRobotV2 = AttentionRobot
