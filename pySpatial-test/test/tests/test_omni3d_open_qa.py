import json
import sys
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from agent.anwer import SpatialAnswer
from agent.model_backend.local_qwen import parse_spatial_answer_text
from mindcube import compute_acc_mar, compute_float_mra, compute_float_relative_error, evaluate_answer_correctness, load_omni3d_entries
from scripts.demo_extract_3d_positions import extract_object_names_from_omni3d_question


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


def test_acc_mar_uses_mra_for_float_and_accuracy_for_other_types():
    results = [
        {"answer_type": "float", "expected_answer": 1.0, "generated_answer": "1.0", "float_mra": 1.0, "answer_correct": True},
        {"answer_type": "float", "expected_answer": 1.0, "generated_answer": "1.1", "float_mra": 0.8, "answer_correct": False},
        {"answer_type": "str", "expected_answer": "yes", "generated_answer": "yes", "answer_correct": True},
        {"answer_type": "int", "expected_answer": 2, "generated_answer": "3", "answer_correct": False},
    ]

    assert compute_acc_mar(results) == {"acc_mar": 0.7, "acc_mar_count": 4}


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
