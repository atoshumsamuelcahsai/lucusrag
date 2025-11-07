import typing as t
import json
from pathlib import Path
import logging

from rag.schemas.processed_files_tracker import ProgressState
from rag.ast.builders import CodeParser, ASTParser, TreeSitterParser

logger = logging.getLogger(__name__)


def build_tree_python_files(
    root_dir: str,
    parser: CodeParser,
    state: ProgressState,
    results_dir: Path,
    progress_file: Path,
    generate_explanations: bool = False,
    llm_provider: str = "anthropic",
) -> None:
    """Process Python files and extract code elements.

    Args:
        root_dir: Root directory containing Python files
        parser: CodeParser instance to use
        state: ProgressState instance for tracking progress
        results_dir: Directory to save element JSON files
        progress_file: Path to progress JSON file
        generate_explanations: Whether to generate AI explanations
        llm_provider: LLM provider for explanations
    """
    # Process each Python file
    for python_file_path in Path(root_dir).rglob("*.py"):
        python_file_path_str = str(python_file_path)

        if state.is_file_processed(python_file_path_str):
            continue

        logger.info(f"Processing file: {python_file_path_str} .....")

        try:
            code_elements = parser.parse(
                python_file_path.read_text(), python_file_path_str
            )
        except Exception as e:
            logger.exception(f"Error parsing file {python_file_path_str}: {e}")
            continue

        for element in code_elements:
            element_id = f"{python_file_path_str}:{element.type}:{element.name}"

            if state.is_element_processed(element_id):
                continue

            try:
                # Only generate explanation if flag is True
                if generate_explanations:
                    element.generate_explanation(llm_provider)

                # Save CodeElement
                element_file = results_dir / f"{element_id.replace('/', '_')}.json"
                with open(element_file, "w", encoding="utf-8") as f:
                    json.dump(element.to_dict(), f, indent=2)

                # Update progress using ProgressState methods
                state.add_element(element_id, progress_file)

            except Exception as e:
                logger.exception(f"Error processing element {element_id}: {e}")
                state.add_failed(element_id, progress_file)

        # Mark file as complete using ProgressState method
        state.add_file(python_file_path_str, progress_file)


def get_parser(parser_type: str, project_module: str = "rag") -> CodeParser:
    """Get parser instance with project module filter.

    Args:
        parser_type: Parser type ("ast" or "tree-sitter")
        project_module: Module prefix to filter dependencies (default: "rag")

    Returns:
        CodeParser instance
    """
    # Create appropriate parser
    parser: CodeParser
    if parser_type == "ast":
        parser = ASTParser(project_module=project_module)
    elif parser_type == "tree-sitter":
        parser = TreeSitterParser(project_module=project_module)
    else:
        raise ValueError(
            f"Unknown parser type: {parser_type}. Use 'ast' or 'tree-sitter'"
        )
    return parser


def setup_processed_directory(
    output_path: str, parser_type: str
) -> t.Tuple[Path, Path]:
    """Setup output directories and return results_dir and progress_file paths.

    Args:
        output_path: Base output directory
        parser_type: Parser type ("ast" or "tree-sitter")

    Returns:
        Tuple of (results_dir, progress_file)
    """
    output_dir: Path = Path(output_path)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    results_dir: Path = output_dir / f"results_{parser_type}"
    results_dir.mkdir(exist_ok=True)
    progress_file: Path = output_dir / "progress.json"
    return results_dir, progress_file


def analyze_and_store_python_files(
    root_dir: str,
    output_path: str = "./ast_cache",
    parser_type: str = "tree-sitter",
    generate_explanations: bool = False,
    llm_provider: str = "anthropic",
    project_module: str = "rag",
) -> str:
    """Analyze Python files and extract code elements.

    Args:
        root_dir: Root directory containing Python files to analyze
        output_path: Directory to store output JSON files
        parser_type: Parser to use - "ast" (built-in) or "tree-sitter" (more accurate)
        generate_explanations: If True, generate AI explanations using configured LLM
        llm_provider: LLM provider to use for explanations (default: "anthropic")
        project_module: Module prefix to filter dependencies (default: "rag")

    Returns:
        Path to output directory

    Raises:
        ValueError: If parser_type is invalid
        ImportError: If tree-sitter is requested but not installed
    """
    parser: CodeParser = get_parser(parser_type, project_module=project_module)
    results_dir, progress_file = setup_processed_directory(output_path, parser_type)
    state: ProgressState = ProgressState.from_file(progress_file)

    # Process all Python files
    build_tree_python_files(
        root_dir=root_dir,
        parser=parser,
        state=state,
        results_dir=results_dir,
        progress_file=progress_file,
        generate_explanations=generate_explanations,
        llm_provider=llm_provider,
    )

    return str(results_dir.parent)


if __name__ == "__main__":
    import argparse

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Analyze Python codebase and extract code elements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with built-in AST parser
  python ast_builder.py --root_dir rag/ --output-dir examples/ast_cache
  
  # With tree-sitter for better accuracy
  python ast_builder.py --root_dir rag/ --parser-type tree-sitter
  
  # Generate AI explanations (requires Anthropic API key)
  python ast_builder.py --root_dir rag/ --generate-explanations
        """,
    )

    parser.add_argument(
        "--root-dir", required=True, help="Root directory of Python project to analyze"
    )
    parser.add_argument(
        "--output-dir",
        default="./ast_cache",
        help="Directory to store AST analysis (default: ./ast_cache)",
    )
    parser.add_argument(
        "--parser-type",
        default="tree-sitter",
        choices=["ast", "tree-sitter"],
        help="Parser type: ast (fast, built-in) or tree-sitter (accurate, requires install)",
    )
    parser.add_argument(
        "--generate-explanations",
        action="store_true",
        help="Generate AI explanations using configured LLM provider",
    )
    parser.add_argument(
        "--llm-provider",
        default="anthropic",
        help="LLM provider for explanations (default: anthropic)",
    )
    parser.add_argument(
        "--project-module",
        default="rag",
        help="Module prefix to filter dependencies (default: rag)",
    )

    args = parser.parse_args()

    try:
        output_path = analyze_and_store_python_files(
            root_dir=args.root_dir,
            output_path=args.output_dir,
            parser_type=args.parser_type,
            generate_explanations=args.generate_explanations,
            llm_provider=args.llm_provider,
            project_module=args.project_module,
        )
        logger.info("Analysis complete!")
        logger.info(f"Output saved to: {output_path}")

    except Exception as e:
        logger.exception(f"\n Error: {e}")
        exit(1)
