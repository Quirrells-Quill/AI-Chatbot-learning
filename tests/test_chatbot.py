# tests/test_chatbot.py
"""
Automated unit test suite for the Adaptive Learning AI Chatbot & Pedagogical Engine.
Validates lesson schema integrity, student progress tracking, vector normalization math,
and Flask lesson serving endpoints.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from serve_lesson import app, LESSON_FILE


class TestLessonPlanSchema:
    """Validates generated lesson plan structure and pedagogical components."""

    def test_lesson_plan_file_exists_and_valid(self):
        assert LESSON_FILE.exists(), "final_lesson_plan.json should be present"
        with open(LESSON_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "explanation" in data
        assert "quiz" in data
        assert "title" in data["explanation"]
        assert isinstance(data["explanation"]["bullets"], list)
        assert len(data["explanation"]["bullets"]) > 0

    def test_quiz_questions_structure(self):
        with open(LESSON_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        quiz = data["quiz"]
        assert len(quiz) > 0

        valid_types = {"mcq", "tf", "short", "match", "reasoning"}
        for q in quiz:
            assert "id" in q
            assert "type" in q
            assert q["type"] in valid_types
            assert "question" in q
            assert len(q["question"]) > 5
            assert "answer_explanation" in q


class TestLearningProgress:
    """Validates student tracking and mastery telemetry."""

    def test_learning_progress_file_structure(self):
        progress_path = Path(__file__).resolve().parent.parent / "learning_progress_tk.json"
        assert progress_path.exists()

        with open(progress_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        assert isinstance(records, list)
        evaluated = [r for r in records if "score" in r]
        assert len(evaluated) > 0
        for entry in evaluated:
            assert "module" in entry
            assert 0.0 <= entry["score"] <= 1.0


class TestVectorSearchMath:
    """Validates embedding normalization and cosine similarity mechanics."""

    def test_l2_normalization_for_cosine_similarity(self):
        # 10 random 768-dim embeddings
        rng = np.random.default_rng(42)
        raw_embs = rng.normal(size=(10, 768)).astype("float32")

        # Normalize L2
        norms = np.linalg.norm(raw_embs, axis=1, keepdims=True) + 1e-10
        norm_embs = raw_embs / norms

        # Norm of each vector should equal 1.0
        calculated_norms = np.linalg.norm(norm_embs, axis=1)
        np.testing.assert_allclose(calculated_norms, np.ones(10), rtol=1e-5)

        # Dot product with self should equal 1.0
        self_sims = np.sum(norm_embs * norm_embs, axis=1)
        np.testing.assert_allclose(self_sims, np.ones(10), rtol=1e-5)


class TestFlaskLessonServer:
    """Validates the web lesson viewer routes."""

    @pytest.fixture
    def client(self):
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_index_route(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert b"Lesson Viewer" in res.data
        assert b"fetchLesson" in res.data

    def test_api_lesson_endpoint(self, client):
        res = client.get("/api/lesson")
        assert res.status_code == 200
        data = res.get_json()
        assert "explanation" in data
        assert "quiz" in data
