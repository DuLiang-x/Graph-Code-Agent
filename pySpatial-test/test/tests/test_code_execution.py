import sys
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))
from agent.codeAgent.execute import execute_code


def test_execute_code_injects_math_module_without_explicit_import():
    program = execute_code("""
def program(input_scene):
    return math.ceil(1.2)
""")

    assert program(None) == 2

