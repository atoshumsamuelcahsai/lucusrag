from rag.schemas import CodeElement
import rag.schemas.code_element as schema_mod


class FakeTextNode:
    def __init__(self, text, metadata=None):
        self.text = text
        self.metadata = metadata or {}


class FakeDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def test_to_node_builds_text_and_metadata(monkeypatch):
    monkeypatch.setattr(schema_mod, "TextNode", FakeTextNode)

    ce = CodeElement(
        type="function",
        name="add",
        docstring="Add two numbers",
        code="""
            def add(a, b):
                return a + b
        """,
        file_path="pkg/math.py",
        parameters=[
            {"name": "a", "type": "int"},
            {"name": "b", "type": "int", "default": "0"},
        ],
        return_type="int",
        decorators=["pure", "vectorized"],
        dependencies=["typing", "functools"],
        base_classes=[],
        methods=None,
        calls=["sum", "print"],
        assignments=[],
        explanation="simple add",
        is_async=False,
    )

    node = ce.to_node()
    assert isinstance(node, FakeTextNode)
    assert "FUNCTION: add" in node.text
    assert "File: pkg/math.py" in node.text
    assert "def add(a, b):" in node.text
    assert "return a + b" in node.text

    md = node.metadata
    assert md["type"] == "function"
    assert md["name"] == "add"
    assert md["file_path"] == "pkg/math.py"
    assert md["parameters"] == "a:int, b:int=0"
    assert md["return_type"] == "int"
    assert md["decorators"] == "pure,vectorized"
