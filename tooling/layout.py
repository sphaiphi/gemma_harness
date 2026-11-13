"""Where synthesized tools and their tests live in the workspace."""
TOOLS_DIR = "tools"
TESTS_DIR = "tests"
RUNTIME_HELPERS = ("__init__.py", "_sdk.py", "_runner.py")


def tool_rel(name: str) -> str:
    return f"{TOOLS_DIR}/{name}.py"


def test_rel(name: str) -> str:
    return f"{TESTS_DIR}/test_tool_{name}.py"