# =========================================================
# memory_manager.py
# =========================================================
#
# IMPROVED SELECTIVE AI MEMORY SYSTEM
#
# 구조 유지 + 성능 개선 + 대화 흐름 유지
#
# =========================================================

import os
import json
import time

# =========================================================
# CLASS
# =========================================================

class MemoryManager:

    # =====================================================
    # INIT
    # =====================================================

    def __init__(
            self,
            memory_dir="memory"
    ):

        self.memory_dir = memory_dir

        self.conv_dir = os.path.join(
            memory_dir,
            "conversations"
        )

        os.makedirs(
            self.conv_dir,
            exist_ok=True
        )

        # =================================================
        # RECENT MEMORY
        # =================================================

        self.recent_memory = []

        # 최근 대화 최대 턴 수
        self.max_recent = 6

        # 압축 단위
        self.compress_size = 4

        print("=" * 60)
        print("MEMORY SYSTEM READY")

    # =====================================================
    # ADD MEMORY
    # =====================================================

    def add_memory(
            self,
            user,
            assistant
    ):

        memory = {

            "timestamp": time.time(),

            "user": user,

            "assistant": assistant
        }

        self.recent_memory.append(
            memory
        )

        # =================================================
        # AUTO COMPRESS
        # =================================================

        if len(self.recent_memory) >= self.max_recent:

            self.compress_old_memory()

    # =====================================================
    # COMPRESS OLD MEMORY
    # =====================================================

    def compress_old_memory(self):

        print("=" * 60)
        print("COMPRESSING MEMORY")

        # =================================================
        # COMPRESS TARGET
        # =================================================

        compress_chunk = self.recent_memory[
            :self.compress_size
        ]

        if len(compress_chunk) == 0:
            return

        # =================================================
        # BUILD FULL TEXT
        # =================================================

        conversation_text = ""

        for mem in compress_chunk:

            conversation_text += (
                f"User: {mem['user']}\n"
            )

            conversation_text += (
                f"Assistant: {mem['assistant']}\n\n"
            )

        # =================================================
        # SUMMARY
        # =================================================

        summary = self.build_summary(
            compress_chunk
        )

        # =================================================
        # TAGS
        # =================================================

        tags = self.extract_tags(
            conversation_text
        )

        # =================================================
        # SAVE DATA
        # =================================================

        memory_data = {

            "summary": summary,

            "conversation": compress_chunk,

            "tags": tags,

            "timestamp": compress_chunk[0][
                "timestamp"
            ]
        }

        # =================================================
        # MEMORY ID
        # =================================================

        memory_id = self.get_next_memory_id()

        save_path = os.path.join(
            self.conv_dir,
            f"memory_{memory_id}.json"
        )

        # =================================================
        # SAVE
        # =================================================

        try:

            with open(
                    save_path,
                    "w",
                    encoding="utf-8"
            ) as f:

                json.dump(
                    memory_data,
                    f,
                    indent=2,
                    ensure_ascii=False
                )

            print("MEMORY SAVED:")
            print(save_path)

        except Exception as e:

            print("SAVE FAILED:")
            print(e)

        # =================================================
        # REMOVE OLD
        # =================================================

        self.recent_memory = (
            self.recent_memory[
                self.compress_size:
            ]
        )

    # =====================================================
    # BUILD SUMMARY
    # =====================================================

    def build_summary(
            self,
            conversation
    ):

        summary = ""

        for mem in conversation:

            user = mem["user"][:80]
            assistant = mem["assistant"][:120]

            summary += (
                f"User asked: {user}\n"
            )

            summary += (
                f"Assistant replied: "
                f"{assistant}\n"
            )

        return summary

    # =====================================================
    # GET NEXT MEMORY ID
    # =====================================================

    def get_next_memory_id(self):

        files = os.listdir(
            self.conv_dir
        )

        ids = []

        for file_name in files:

            if not file_name.startswith(
                    "memory_"
            ):
                continue

            try:

                num = int(
                    file_name.replace(
                        "memory_",
                        ""
                    ).replace(
                        ".json",
                        ""
                    )
                )

                ids.append(num)

            except Exception:
                pass

        if len(ids) == 0:
            return 0

        return max(ids) + 1

    # =====================================================
    # TAG EXTRACTION
    # =====================================================

    def extract_tags(
            self,
            text
    ):

        text = text.lower()

        tags = []

        # =================================================
        # KEYWORDS
        # =================================================

        keywords = [

            "robot",
            "robotic",
            "arm",
            "desk",
            "object",
            "scene",
            "distance",
            "depth",
            "color",
            "shape",
            "memory",
            "planning",
            "coordinate",
            "position",
            "geometry",
            "vision",
            "spatial",
            "interaction",
            "reasoning",
            "grasp",
            "detect",
            "movement",
            "location",
            "mapping",
            "tracking",
            "camera"
        ]

        for keyword in keywords:

            if keyword in text:

                tags.append(
                    keyword
                )

        return list(set(tags))

    # =====================================================
    # SEARCH MEMORY
    # =====================================================

    def search_memory(
            self,
            query
    ):

        print("=" * 60)
        print("SEARCHING MEMORY")

        query_lower = query.lower()

        results = []

        # =================================================
        # NO MEMORY
        # =================================================

        if not os.path.exists(
                self.conv_dir
        ):

            return results

        # =================================================
        # QUERY WORDS
        # =================================================

        query_words = (
            query_lower.split()
        )

        # =================================================
        # SEARCH FILES
        # =================================================

        for file_name in os.listdir(
                self.conv_dir
        ):

            if not file_name.endswith(
                    ".json"
            ):

                continue

            path = os.path.join(
                self.conv_dir,
                file_name
            )

            try:

                with open(
                        path,
                        "r",
                        encoding="utf-8"
                ) as f:

                    data = json.load(f)

                score = 0

                # =========================================
                # TAG MATCH
                # =========================================

                for tag in data.get(
                        "tags",
                        []
                ):

                    if tag in query_lower:

                        score += 3

                # =========================================
                # SUMMARY MATCH
                # =========================================

                summary_lower = (
                    data.get(
                        "summary",
                        ""
                    ).lower()
                )

                for word in query_words:

                    if len(word) < 3:
                        continue

                    if word in summary_lower:

                        score += 1

                # =========================================
                # CONVERSATION MATCH
                # =========================================

                conversation = data.get(
                    "conversation",
                    []
                )

                for mem in conversation:

                    combined = (
                        mem["user"] +
                        " " +
                        mem["assistant"]
                    ).lower()

                    for word in query_words:

                        if len(word) < 3:
                            continue

                        if word in combined:

                            score += 2

                # =========================================
                # ADD RESULT
                # =========================================

                if score > 0:

                    results.append({

                        "score": score,

                        "summary": data[
                            "summary"
                        ]
                    })

            except Exception:

                continue

        # =================================================
        # SORT
        # =================================================

        results = sorted(
            results,
            key=lambda x: x["score"],
            reverse=True
        )

        # =================================================
        # RETURN TOP RESULTS
        # =================================================

        final_results = []

        for r in results[:3]:

            final_results.append(
                r["summary"]
            )

        print(
            "MEMORY RESULTS:",
            len(final_results)
        )

        return final_results

    # =====================================================
    # BUILD MEMORY CONTEXT
    # =====================================================

    def build_memory_context(
            self,
            current_query
    ):

        print("=" * 60)
        print("BUILDING MEMORY CONTEXT")

        context = ""

        # =================================================
        # SEARCH LONG MEMORY
        # =================================================

        searched_memories = (
            self.search_memory(
                current_query
            )
        )

        # =================================================
        # ADD SEARCHED MEMORY
        # =================================================

        if len(searched_memories) > 0:

            context += (
                "Relevant Past Memories:\n"
            )

            for mem in searched_memories:

                context += (
                    f"- {mem}\n"
                )

            context += "\n"

        # =================================================
        # RECENT MEMORY
        # =================================================

        context += (
            "Recent Conversation:\n"
        )

        if len(self.recent_memory) == 0:

            context += (
                "No recent conversation.\n"
            )

        else:

            for mem in self.recent_memory:

                context += (
                    f"User: "
                    f"{mem['user']}\n"
                )

                context += (
                    f"Assistant: "
                    f"{mem['assistant']}\n\n"
                )

        return context

    # =====================================================
    # DEBUG
    # =====================================================

    def print_recent_memory(self):

        print("=" * 60)
        print("RECENT MEMORY")
        print("=" * 60)

        for i, mem in enumerate(
                self.recent_memory
        ):

            print(f"[{i}]")

            print(
                "USER:",
                mem["user"]
            )

            print(
                "ASSISTANT:",
                mem["assistant"]
            )

            print()

    # =====================================================
    # LIST SAVED MEMORIES
    # =====================================================

    def list_saved_memories(self):

        print("=" * 60)
        print("SAVED MEMORIES")
        print("=" * 60)

        if not os.path.exists(
                self.conv_dir
        ):

            print("NO MEMORY DIRECTORY")
            return

        files = os.listdir(
            self.conv_dir
        )

        if len(files) == 0:

            print("NO SAVED MEMORIES")
            return

        for file_name in files:

            print(file_name)

    # =====================================================
    # CLEAR MEMORY
    # =====================================================

    def clear_recent_memory(self):

        self.recent_memory = []

        print("=" * 60)
        print("RECENT MEMORY CLEARED")

# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    memory = MemoryManager()

    memory.add_memory(
        "hello",
        "hello there"
    )

    memory.add_memory(
        "robot arm detected?",
        "yes robot arm detected"
    )

    memory.add_memory(
        "where is object 4",
        "object 4 is on the desk"
    )

    memory.add_memory(
        "what color is it",
        "it is dark colored"
    )

    memory.add_memory(
        "can the robot grasp it?",
        "yes grasp possible"
    )

    memory.add_memory(
        "make a movement plan",
        "movement planning generated"
    )

    print()

    context = memory.build_memory_context(
        "robot object grasp"
    )

    print(context)