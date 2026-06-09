import json
import random
import os
from difflib import SequenceMatcher
from itertools import zip_longest

class SemanticAnalyzer:
    def __init__(self, json_path=r"C:\Users\0328j\OneDrive\Desktop\지각모델\비접촉VLA\words_data.json"):
        self.json_path = json_path
        self.atoms = {}
        self.words = {}
        self.similarity_threshold = 0.7
        self.load_database()

    def load_database(self):
        """JSON 데이터베이스를 로드하고 내부 매핑 테이블을 구축합니다."""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {self.json_path}")
            
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.atoms = data.get("atoms", {})
        raw_words = data.get("words", {})
        
        self.words = {}
        for word, atom_strings in raw_words.items():
            encoded_atoms = [self.atoms[atom] for atom in atom_strings if atom in self.atoms]
            self.words[word.lower()] = encoded_atoms

    def _get_similarity(self, a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def tokenize_sentence(self, sentence):
        """'look at', 'point to' 등 다중 단어 키를 우선 고려하여 탐욕적으로 토큰화합니다."""
        normalized = " " + " ".join(str(sentence).lower().split()) + " "
        
        # 글자 수가 긴 복합어 순으로 정렬하여 우선 매칭 ("point to"가 "point"보다 먼저 매칭되도록)
        sorted_known_words = sorted(self.words.keys(), key=len, reverse=True)
        
        matched_tokens = []
        
        # 1단계: 사전에 등록된 정확한 단어/복합어 추출
        for known_word in sorted_known_words:
            pattern = f" {known_word} "
            while pattern in normalized:
                # 임시 특수문자로 치환하여 중복 매칭 방지 및 순서 기록을 위해 표시
                normalized = normalized.replace(pattern, f" [MATCHED:{known_word}] ", 1)

        # 2단계: 원래 순서대로 복원하며 토큰 리스트 생성 및 사전에 없는 단어 오타 교정
        raw_tokens = normalized.split()
        for token in raw_tokens:
            if token.startswith("[MATCHED:") and token.endswith("]"):
                clean_word = token.replace("[MATCHED:", "").replace("]", "")
                matched_tokens.append(clean_word)
            else:
                # 오타 교정 처리
                max_sim = 0.0
                candidates = []
                for known_word in self.words.keys():
                    sim = self._get_similarity(token, known_word)
                    if sim > max_sim:
                        max_sim = sim
                        candidates = [known_word]
                    elif sim == max_sim:
                        candidates.append(known_word)

                if max_sim >= self.similarity_threshold and candidates:
                    chosen_word = random.choice(candidates)
                    matched_tokens.append(chosen_word)
                    
        return matched_tokens

    def correct_and_convert(self, sentence):
        """문장을 고유 의미 원소 숫자 스트림으로 맵핑합니다."""
        if not sentence:
            return []
            
        tokens = self.tokenize_sentence(sentence)
        corrected_stream = []

        for token in tokens:
            if token in self.words:
                corrected_stream.extend(self.words[token])
                    
        return corrected_stream

    def _calculate_category_vector(self, token_stream, target_atom_ids, weight_list, total_count):
        """특정 카테고리에 속하는 의미 원소들의 가중치와 위치 벡터를 에러 없이 결합 계산합니다."""
        positions_by_atom = {}
        all_positions = []
        
        for atom_id in target_atom_ids:
            pos_list = [idx for idx, val in enumerate(token_stream) if val == atom_id]
            positions_by_atom[atom_id] = pos_list
            all_positions.extend(pos_list)

        ppa_list = []
        lw_list = []
        for atom_id in target_atom_ids:
            # 비율(ppa) 계산
            ppa = len(positions_by_atom[atom_id]) / len(all_positions) if all_positions else 0.0
            ppa_list.append(ppa)
            
            # 위치 가중치(lw) 계산
            pos_sum = sum(positions_by_atom[atom_id])
            lw = pos_sum / total_count if total_count > 0 else 0.0
            lw_list.append(lw)

        # X 값 결합
        x_list = [lw + ppa for lw, ppa in zip(lw_list, ppa_list)]
        
        # 구조적 버그 수정: .index() 대신 인덱스 자체를 lw 기준으로 정렬
        sorted_indices = sorted(range(len(lw_list)), key=lambda k: lw_list[k])
        reordered_x = [0.0] * len(x_list)
        for new_idx, org_idx in enumerate(sorted_indices):
            reordered_x[new_idx] = x_list[org_idx]

        # 중요도 가중치 반영 (고정 차원 벡터 유지를 위해 0.0도 포함)
        final_x = [float(x_val * w) for x_val, w in zip(reordered_x, weight_list)]
                
        return final_x

    def analyze_sentence(self, sentence):
        """문장을 입력받아 고정된 차원의 수치 가중치 인터레이싱 리스트를 반환합니다."""
        token_stream = self.correct_and_convert(sentence)
        
        # 카테고리 1: 정신 상태 & 인지
        tkwfsh_atoms = ["THINK", "KNOW", "WANT", "FEEL", "SEE", "HEAR"]
        tkwfsh_ids = [self.atoms[a] for a in tkwfsh_atoms if a in self.atoms]
        weight_list1 = [0.8, 0.7, 1.0, 1.0, 0.5, 0.4]
        
        # 카테고리 2: 실체 및 대상
        iysstpb_atoms = ["I", "YOU", "SOMEONE", "SOMETHING", "SEE", "HEAR"]
        iysstpb_ids = [self.atoms[a] for a in iysstpb_atoms if a in self.atoms]
        weight_list2 = [0.8, 0.7, 1.0, 1.0, 0.5, 0.4]

        if not token_stream:
            # 빈 문장이거나 분석 불가능할 경우 고정 크기(6*2 = 12차원)의 영벡터 반환
            return [0.0] * (len(weight_list1) + len(weight_list2))

        n = len(token_stream)
        total_count = (n + 1) * n / 2

        x_list1 = self._calculate_category_vector(token_stream, tkwfsh_ids, weight_list1, total_count)
        x_list2 = self._calculate_category_vector(token_stream, iysstpb_ids, weight_list2, total_count)

        # 교차 인터레이싱 구조로 결합
        combined_list = [
            val
            for pair in zip_longest(x_list1, x_list2)
            for val in pair
            if val is not None
        ]
        
        return combined_list