from rag.schemas import CodeElement
from rag.parser import parser
from rag.parser.parser import to_node


class FakeTextNode:
    def __init__(self, text, metadata=None, id_=None):
        self.text = text
        self.metadata = metadata or {}
        self.id_ = id_


class FakeDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def test_to_node_builds_text_and_metadata(monkeypatch):
    monkeypatch.setattr(parser, "TextNode", FakeTextNode)

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
        methods=[],
        calls=["sum", "print"],
        assignments=[],
        explanation="simple add",
        is_async=False,
    )

    node = to_node(ce)
    assert isinstance(node, FakeTextNode)
    assert "FUNCTION: add" in node.text
    assert "File: pkg/math.py" in node.text
    assert "def add(a, b):" in node.text
    assert "return a + b" in node.text

    md = node.metadata
    assert md["type"] == "function"
    assert md["name"] == "add"
    assert md["file_path"] == "pkg/math.py"
    assert "parameters" in md
    assert md["parameters"] == "a:int, b:int=0"
    assert "return_type" in md
    assert md["return_type"] == "int"
    assert "decorators" in md
    assert md["decorators"] == "pure,vectorized"
