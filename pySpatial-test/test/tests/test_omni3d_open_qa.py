import json
import sys
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from agent.anwer import SpatialAnswer, answer_without_visual_clue
from agent.model_backend.local_qwen import parse_spatial_answer_text
from pySpatial_Interface import Scene
from mindcube import (
    apply_index_range_and_max_entries,
    compute_acc_mra,
    compute_float_mra,
    compute_float_relative_error,
    evaluate_answer_correctness,
    filter_entries_by_index_range,
    get_entry_question_index,
    load_omni3d_entries,
    validate_index_range,
)
from scripts.demo_extract_3d_positions import extract_object_names_from_omni3d_question


def test_openai_answer_falls_back_to_chat_completion_string(monkeypatch):
    calls = {"responses": 0, "chat": 0}

    class FakeResponses:
        def parse(self, **kwargs):
            calls["responses"] += 1
            raise AttributeError("'str' object has no attribute 'output'")

    class FakeCompletions:
        def create(self, **kwargs):
            calls["chat"] += 1
            return '{"reasoning": "computed", "answer": "yes"}'

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        def __init__(self, **kwargs):
            self.responses = FakeResponses()
            self.chat = FakeChat()

    class FakeOpenAIModule:
        OpenAI = FakeClient

    monkeypatch.setitem(sys.modules, "openai", FakeOpenAIModule)
    scene = Scene([], "Is the fireplace visible?", scene_id="scene_answer_fallback")

    parsed = answer_without_visual_clue(
        scene,
        api_key="test-key",
        backend="openai",
        model="closeai-model",
        base_url="https://closeai.example/v1",
    )

    assert calls == {"responses": 1, "chat": 1}
    assert parsed.answer == "yes"
    assert parsed.reasoning == "computed"


def test_spatial_answer_accepts_open_form_answers():
    assert SpatialAnswer(reasoning="computed", answer="yes").answer == "yes"
    assert SpatialAnswer(reasoning="computed", answer="0.948").answer == "0.948"
    assert SpatialAnswer(reasoning="computed", answer="the fireplace").answer == "the fireplace"


def test_local_qwen_parse_keeps_non_choice_answer():
    parsed = parse_spatial_answer_text('{"reasoning": "ratio", "answer": "0.948"}')
    assert parsed.answer == "0.948"

    parsed_text = parse_spatial_answer_text("yes")
    assert parsed_text.answer == "yes"


def test_omni3d_float_and_string_evaluation():
    assert evaluate_answer_correctness("0.95", 0.948, "float")
    assert evaluate_answer_correctness("It requires two stools to match or exceed the height of the leftmost chair", 2.0, "float")
    assert evaluate_answer_correctness(" yes. ", "yes", "str")
    assert evaluate_answer_correctness("Yes, the fireplace would still be visible", "yes", "str")
    assert evaluate_answer_correctness("No, the fireplace would not be visible", "no", "str")
    assert not evaluate_answer_correctness("no", "yes", "str")
    assert not evaluate_answer_correctness("The answer is probably yes", "yes", "str")


def test_float_mra_metrics():
    assert compute_float_relative_error("1.1", 1.0) == 0.10000000000000009
    assert compute_float_mra("0.948", 0.948) == 1.0
    assert compute_float_mra("1.1", 1.0) == 0.8
    assert compute_float_mra("no number", 1.0) is None
    assert compute_float_mra("1.0", 0.0) is None


def test_acc_mra_uses_mra_for_float_and_accuracy_for_other_types():
    results = [
        {"answer_type": "float", "expected_answer": 1.0, "generated_answer": "1.0", "float_mra": 1.0, "answer_correct": True},
        {"answer_type": "float", "expected_answer": 1.0, "generated_answer": "1.1", "float_mra": 0.8, "answer_correct": False},
        {"answer_type": "str", "expected_answer": "yes", "generated_answer": "yes", "answer_correct": True},
        {"answer_type": "int", "expected_answer": 2, "generated_answer": "3", "answer_correct": False},
    ]

    assert compute_acc_mra(results) == {"acc_mra": 0.7, "acc_mra_count": 4}


def test_entry_question_index_parsing_priority():
    assert get_entry_question_index({"raw_sample": {"question_index": "400"}, "id": "omni3d_1"}) == 400
    assert get_entry_question_index({"question_index": "401"}) == 401
    assert get_entry_question_index({"id": "omni3d_402"}) == 402
    assert get_entry_question_index({"id": "sample_402"}) is None


def test_filter_entries_by_index_range_is_inclusive():
    entries = [
        {"id": "omni3d_399"},
        {"id": "omni3d_400"},
        {"id": "omni3d_450"},
        {"id": "omni3d_500"},
        {"id": "omni3d_501"},
        {"id": "unparseable"},
    ]

    filtered = filter_entries_by_index_range(entries, start_index=400, end_index=500)

    assert [entry["id"] for entry in filtered] == ["omni3d_400", "omni3d_450", "omni3d_500"]


def test_filter_entries_by_single_sided_index_range():
    entries = [{"id": "omni3d_0"}, {"id": "omni3d_1"}, {"id": "omni3d_2"}]

    assert [entry["id"] for entry in filter_entries_by_index_range(entries, start_index=1)] == ["omni3d_1", "omni3d_2"]
    assert [entry["id"] for entry in filter_entries_by_index_range(entries, end_index=1)] == ["omni3d_0", "omni3d_1"]


def test_apply_index_range_before_max_entries():
    entries = [{"id": "omni3d_399"}, {"id": "omni3d_400"}, {"id": "omni3d_401"}, {"id": "omni3d_402"}]

    filtered = apply_index_range_and_max_entries(entries, start_index=400, end_index=402, max_entries=2)

    assert [entry["id"] for entry in filtered] == ["omni3d_400", "omni3d_401"]


def test_validate_index_range_rejects_invalid_values():
    validate_index_range(0, 0)

    try:
        validate_index_range(-1, None)
        assert False, "negative start_index should fail"
    except ValueError as exc:
        assert "--start_index must be non-negative" in str(exc)

    try:
        validate_index_range(None, -1)
        assert False, "negative end_index should fail"
    except ValueError as exc:
        assert "--end_index must be non-negative" in str(exc)

    try:
        validate_index_range(500, 400)
        assert False, "start_index > end_index should fail"
    except ValueError as exc:
        assert "--start_index must be less than or equal to --end_index" in str(exc)


def test_load_omni3d_entries_builds_scene_fields(tmpdir):
    root = Path(str(tmpdir))
    dataset = root / "annotations.json"
    dataset.write_text(json.dumps({
        "questions": [{
            "question_index": 7,
            "image_filename": "scene/image.jpg",
            "question": "Is the chair above the table?",
            "answer": "yes",
            "answer_type": "str",
        }]
    }))

    entries = load_omni3d_entries(str(dataset), "/images")

    assert entries[0]["id"] == "omni3d_7"
    assert entries[0]["images"] == ["/images/scene/image.jpg"]
    assert entries[0]["answer"] == "yes"
    assert entries[0]["answer_type"] == "str"


def test_omni3d_object_names_without_options():
    question = "What is the ratio of the height of the fireplace to the combined height of the coffee table and the sofa to the right of the coffee table?"
    assert extract_object_names_from_omni3d_question(question) == ["fireplace", "coffee table", "sofa"]
