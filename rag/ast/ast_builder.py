import typing as t
import json
from pathlib import Path
import logging

import aiofiles
from rag.schemas.processed_files_tracker import ProgressState
from rag.ast.builders import CodeParser, ASTParser, TreeSitterParser
import asyncio
from rag.schemas.code_element import CodeElement
from rag.logging_config import get_logger

logger = get_logger(__name__)


async def _save_element_to_file(
    element: CodeElement, element_id: str, results_dir: Path
) -> None:
    """Save a CodeElement to a JSON file.

    Args:
        element: CodeElement to save
        element_id: Unique identifier for the element
        results_dir: Directory to save the file
    """
    element_file = (
        results_dir / f"{element_id.replace('/', '_').replace(':', '_')}.json"
    )
    async with aiofiles.open(element_file, "w", encoding="utf-8") as f:
        await f.write(json.dumps(element.to_dict(), indent=2))


async def _generate_explanation_with_rate_limit(
    element: CodeElement,
    llm_provider: str,
    semaphore: asyncio.Semaphore,
) -> None:
    """Generate explanation for an element with rate limiting.

    Args:
        element: CodeElement to generate explanation for
        llm_provider: LLM provider to use
        semaphore: Semaphore for rate limiting
    """
    async with semaphore:
        await element.generate_explanation(llm_provider)


async def _process_single_element(
    element: CodeElement,
    element_id: str,
    results_dir: Path,
    progress_file: Path,
    state: ProgressState,
    generate_explanation: bool,
    llm_provider: str,
    semaphore: t.Optional[asyncio.Semaphore],
) -> None:
    """Process a single code element: generate explanation (if needed) and save to file.

    Args:
        element: CodeElement to process
        element_id: Unique identifier for the element
        results_dir: Directory to save JSON files
        progress_file: Path to progress tracking file
        state: ProgressState instance
        generate_explanation: Whether to generate AI explanation
        llm_provider: LLM provider for explanations
        semaphore: Optional semaphore for rate limiting
    """
    try:
        if generate_explanation:
            if semaphore is None:
                raise ValueError(
                    "Semaphore is required when generate_explanation is True"
                )
            await _generate_explanation_with_rate_limit(
                element, llm_provider, semaphore
            )

        await _save_element_to_file(element, element_id, results_dir)
        state.add_element(element_id, progress_file)
    except Exception as e:
        logger.exception(f"Error processing element {element_id}: {e}")
        state.add_failed(element_id, progress_file)


async def _parse_python_file(
    file_path: Path, parser: CodeParser
) -> t.List[CodeElement]:
    """Parse a Python file and extract code elements.

    Args:
        file_path: Path to Python file
        parser: CodeParser instance

    Returns:
        List of CodeElement instances

    Raises:
        Exception: If parsing fails
    """
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        content = await f.read()
    return parser.parse(content, str(file_path))


async def _process_elements_in_file(
    elements: t.List[CodeElement],
    results_dir: Path,
    progress_file: Path,
    state: ProgressState,
    generate_explanations: bool,
    llm_provider: str,
    semaphore: t.Optional[asyncio.Semaphore],
) -> None:
    """Process all code elements in a file (concurrent or sequential).

    Args:
        elements: List of CodeElement instances to process
        results_dir: Directory to save JSON files
        progress_file: Path to progress tracking file
        state: ProgressState instance
        generate_explanations: Whether to generate explanations
        llm_provider: LLM provider for explanations
        semaphore: Optional semaphore for rate limiting
    """
    # Filter out already processed elements
    elements_to_process = [
        (element, element.id)
        for element in elements
        if not state.is_element_processed(element.id)
    ]

    if not elements_to_process:
        return

    if generate_explanations:
        # Process concurrently with rate limiting
        tasks = [
            _process_single_element(
                element=element,
                element_id=element_id,
                results_dir=results_dir,
                progress_file=progress_file,
                state=state,
                generate_explanation=True,
                llm_provider=llm_provider,
                semaphore=semaphore,
            )
            for element, element_id in elements_to_process
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    else:
        # Process sequentially (no API calls, faster for I/O)
        for element, element_id in elements_to_process:
            await _process_single_element(
                element=element,
                element_id=element_id,
                results_dir=results_dir,
                progress_file=progress_file,
                state=state,
                generate_explanation=False,
                llm_provider=llm_provider,
                semaphore=None,
            )


async def build_tree_python_files(
    root_dir: str,
    parser: CodeParser,
    state: ProgressState,
    results_dir: Path,
    progress_file: Path,
    generate_explanations: bool = False,
    llm_provider: str = "openai",
    max_concurrent: int = 5,
) -> None:
    """Process Python files and extract code elements with concurrent explanation generation.

    Args:
        root_dir: Root directory containing Python files
        parser: CodeParser instance to use
        state: ProgressState instance for tracking progress
        results_dir: Directory to save element JSON files
        progress_file: Path to progress JSON file
        generate_explanations: Whether to generate AI explanations
        llm_provider: LLM provider for explanations
        max_concurrent: Maximum number of concurrent explanation requests (default: 5)
    """
    semaphore = asyncio.Semaphore(max_concurrent) if generate_explanations else None

    for python_file_path in Path(root_dir).rglob("*.py"):
        python_file_path_str = str(python_file_path)

        if state.is_file_processed(python_file_path_str):
            continue

        logger.info(f"Processing file: {python_file_path_str} .....")

        try:
            code_elements = await _parse_python_file(python_file_path, parser)
        except Exception as e:
            logger.exception(f"Error parsing file {python_file_path_str}: {e}")
            continue

        await _process_elements_in_file(
            elements=code_elements,
            results_dir=results_dir,
            progress_file=progress_file,
            state=state,
            generate_explanations=generate_explanations,
            llm_provider=llm_provider,
            semaphore=semaphore,
        )

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


async def analyze_and_store_python_files(
    root_dir: str,
    output_path: str = "./ast_cache",
    parser_type: str = "tree-sitter",
    generate_explanations: bool = False,
    llm_provider: str = "anthropic",
    project_module: str = "rag",
    max_concurrent: int = 5,
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
    await build_tree_python_files(
        root_dir=root_dir,
        parser=parser,
        state=state,
        results_dir=results_dir,
        progress_file=progress_file,
        generate_explanations=generate_explanations,
        llm_provider=llm_provider,
        max_concurrent=max_concurrent,
    )

    return str(results_dir.parent)


if __name__ == "__main__":
    import argparse
    from rag.logging_config import configure_logging

    # Configure logging
    configure_logging(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Analyze Python codebase and extract code elements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with built-in AST parser
  python ast_builder.py --root_dir rag/ --output-dir examples/ast_cache
  
  # With tree-sitter for better accuracy
  python ast_builder.py --root_dir rag/ --parser-type tree-sitter
  
  # Generate AI explanations (requires API key)
  python ast_builder.py --root_dir rag/ --generate-explanations
  
  # Generate explanations with higher concurrency (faster, but may hit rate limits)
  python ast_builder.py --root_dir rag/ --generate-explanations --max-concurrent 10
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
        default="openai",
        help="LLM provider for explanations (default: openai)",
    )
    parser.add_argument(
        "--project-module",
        default="rag",
        help="Module prefix to filter dependencies (default: rag)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent explanation requests (default: 5, increase for faster processing)",
    )

    args = parser.parse_args()

    try:
        output_path = asyncio.run(
            analyze_and_store_python_files(
                root_dir=args.root_dir,
                output_path=args.output_dir,
                parser_type=args.parser_type,
                generate_explanations=args.generate_explanations,
                llm_provider=args.llm_provider,
                project_module=args.project_module,
                max_concurrent=args.max_concurrent,
            )
        )
        logger.info("Analysis complete!")
        logger.info(f"Output saved to: {output_path}")

    except Exception as e:
        logger.exception(f"\n Error: {e}")
        exit(1)
