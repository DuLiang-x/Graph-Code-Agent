import re
import sys
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

import mindcube
from pySpatial_Interface import Scene
from agent.codeAgent import query


VALID_CODE = '''```python
def program(input_scene: Scene):
    return {"computed_results": {"answer": "yes"}}
```'''
RAISE_CODE = '''```python
def program(input_scene: Scene):
    raise RuntimeError("bad graph call")
```'''


class RepairableAgent:
    backend = "openai"

    def __init__(self, initial_response, repair_response=VALID_CODE):
        self.initial_response = initial_response
        self.repair_response = repair_response
        self.repair_calls = []
        self.basic_qa_calls = 0

    def generate_code(self, scene):
        return self.initial_response

    def repair_code(self, scene, previous_response=None, previous_code=None, error=None):
        self.repair_calls.append({
            "previous_response": previous_response,
            "previous_code": previous_code,
            "error": error,
        })
        return self.repair_response

    def parse_LLM_response(self, scene, response):
        match = re.search(r"```python\s*(.*?)```", response or "", re.DOTALL | re.IGNORECASE)
        code = match.group(1).strip() if match else ""
        scene.code = code
        return code

    def execute(self, scene):
        namespace = {"Scene": Scene}
        exec(scene.code, namespace, namespace)
        return namespace["program"](scene)

    def answer(self, scene, visual_clue):
        class Answer:
            answer = "yes"
            reasoning = "computed"
        return Answer()

    def basic_qa(self, scene):
        self.basic_qa_calls += 1
        class Answer:
            answer = "no"
            reasoning = "fallback"
        return Answer()


def test_repair_retry_fixes_parse_failure():
    scene = Scene([], "Is the lamp above the table?", scene_id="scene_parse")
    agent = RepairableAgent("no python block")

    result = mindcube.generate_parse_execute_with_repair(agent, scene, max_code_repair_attempts=1)

    assert result["parse_success"] is True
    assert result["execution_success"] is True
    assert result["code_repair_used"] is True
    assert result["code_repair_attempts"] == 1
    assert agent.repair_calls[0]["previous_response"] == "no python block"
    assert "Code parsing failed" in agent.repair_calls[0]["error"]


def test_repair_retry_fixes_execution_failure():
    scene = Scene([], "Is the chair left of the table?", scene_id="scene_exec")
    agent = RepairableAgent(RAISE_CODE)

    result = mindcube.generate_parse_execute_with_repair(agent, scene, max_code_repair_attempts=1)

    assert result["parse_success"] is True
    assert result["execution_success"] is True
    assert result["code_repair_used"] is True
    assert result["code_repair_attempts"] == 1
    assert "bad graph call" in agent.repair_calls[0]["error"]
    assert "Traceback" in agent.repair_calls[0]["error"]


def test_repair_retry_can_be_disabled():
    scene = Scene([], "question", scene_id="scene_disabled")
    agent = RepairableAgent("no python block")

    result = mindcube.generate_parse_execute_with_repair(agent, scene, max_code_repair_attempts=0)

    assert result["parse_success"] is False
    assert result["execution_success"] is False
    assert result["code_repair_used"] is False
    assert agent.repair_calls == []


def test_process_scene_uses_repair_before_basic_qa():
    entry = {"id": "scene_process", "question": "yes/no?", "images": [], "answer": "yes", "answer_type": "str"}
    agent = RepairableAgent("no python block")

    result = mindcube.process_scene_with_agent(
        entry,
        agent,
        mode="reconstruct",
        max_code_repair_attempts=1,
    )

    assert result["execution_success"] is True
    assert result["answer_source"] == "graph"
    assert result["fallback_used"] is False
    assert result["code_repair_used"] is True
    assert result["code_repair_attempts"] == 1
    assert agent.basic_qa_calls == 0


def test_extract_chat_completion_text_accepts_string_response():
    assert query._extract_chat_completion_text("```python\npass\n```") == "```python\npass\n```"


def test_extract_chat_completion_text_accepts_dict_response():
    response = {"choices": [{"message": {"content": "```python\npass\n```"}}]}

    assert query._extract_chat_completion_text(response) == "```python\npass\n```"


def test_extract_chat_completion_text_accepts_sdk_like_response():
    class Message:
        content = "```python\npass\n```"

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    assert query._extract_chat_completion_text(Response()) == "```python\npass\n```"


def test_repair_prompt_contains_context_and_uses_same_backend(monkeypatch):
    captured = {}

    def fake_call(prompt, **kwargs):
        captured["prompt"] = prompt
        captured.update(kwargs)
        return VALID_CODE

    monkeypatch.setattr(query, "_call_code_model", fake_call)
    scene = Scene([], "Which object is closer?", scene_id="scene_prompt")
    scene.spatial_graph = type("Graph", (), {"list_nodes": lambda self: ["coffee table", "sofa"]})()

    response = query.repair_code_from_error(
        scene,
        previous_response="bad response",
        previous_code="bad_code()",
        error="Unknown object",
        api_key="key",
        backend="openai",
        model="gpt-4.1",
        base_url="https://closeai.example/v1",
        device="cuda:0",
    )

    assert response == VALID_CODE
    assert "The previously generated code failed during execution" in captured["prompt"]
    assert "Available graph object names" in captured["prompt"]
    assert "coffee table" in captured["prompt"]
    assert "bad response" in captured["prompt"]
    assert "bad_code()" in captured["prompt"]
    assert "Unknown object" in captured["prompt"]
    assert captured["backend"] == "openai"
    assert captured["model"] == "gpt-4.1"
    assert captured["base_url"] == "https://closeai.example/v1"
