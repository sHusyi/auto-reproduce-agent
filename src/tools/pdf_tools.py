"""PDF tools — download papers and read them with text + table extraction."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

from langchain_core.tools import tool


def create_pdf_tools(workspace_root: str | Path) -> list:
    """Create PDF download and reading tools."""
    ws = Path(workspace_root)
    papers_dir = ws / "papers"
    papers_dir.mkdir(exist_ok=True)

    @tool
    def download_paper(url: str, filename: str = "") -> str:
        """Download a paper PDF from a URL.

        Supports arxiv, openaccess.thecvf.com, proceedings.mlr.press, and
        most direct PDF links. The file is saved to the workspace papers/ directory.

        Args:
            url: URL to the paper PDF. Can be an arxiv abstract page
                 (e.g., https://arxiv.org/abs/1512.03385) or a direct PDF link.
            filename: Optional filename (without .pdf). Auto-detected from URL if empty.
        """
        # Convert arxiv abstract URL to PDF URL
        pdf_url = url
        arxiv_match = re.match(
            r'https?://arxiv\.org/abs/([\d.]+)(?:v\d+)?', url
        )
        if arxiv_match:
            paper_id = arxiv_match.group(1)
            pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
            if not filename:
                filename = paper_id

        if not filename:
            # Try to extract filename from URL
            parts = url.rstrip("/").split("/")
            filename = parts[-1].replace(".pdf", "") or "paper"

        if not filename.endswith(".pdf"):
            filename = filename + ".pdf"
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        # Fix: ensure .pdf extension
        if not filename.endswith(".pdf"):
            filename = filename + ".pdf"

        dest = papers_dir / filename

        try:
            # Use curl for better compatibility with various PDF hosts
            result = subprocess.run(
                ["curl", "-L", "-o", str(dest), "-A",
                 "Mozilla/5.0 (compatible; AutoReproduceAgent/0.1)", pdf_url],
                capture_output=True, text=True, timeout=120,
                cwd=str(ws),
            )
            if result.returncode != 0 or not dest.exists():
                return f"Download failed: {result.stderr[:500]}"

            size_kb = dest.stat().st_size / 1024
            if size_kb < 1:
                dest.unlink()
                return f"Downloaded file is too small ({size_kb:.1f} KB) — likely not a PDF. URL: {pdf_url}"

            return f"Downloaded: {filename} ({size_kb:.0f} KB)\nPath: papers/{filename}"
        except subprocess.TimeoutExpired:
            return f"Download timed out after 120s: {pdf_url}"

    @tool
    def read_paper(path: str, pages: str = "") -> str:
        """Read a PDF paper and extract text content and tables.

        Use this after download_paper to read the paper content. The tool
        extracts both paragraph text and tables with formatting preserved.

        Args:
            path: Path to the PDF file (e.g., 'papers/1512.03385.pdf').
            pages: Page range (e.g., '1-5', '3', or empty for all).
                   Leave empty to read the first 10 pages.
        """
        full_path = ws / path
        if not full_path.exists():
            return f"File not found: {path}"

        try:
            import pdfplumber
        except ImportError:
            return "pdfplumber is not installed. Run: uv pip install pdfplumber"

        try:
            doc = pdfplumber.open(str(full_path))
        except Exception as e:
            return f"Cannot open PDF: {e}"

        # Parse page range
        total_pages = len(doc.pages)
        if not pages:
            start, end = 0, min(10, total_pages)
        elif "-" in pages:
            parts = pages.split("-")
            start = max(0, int(parts[0]) - 1)
            end = min(total_pages, int(parts[1]))
        else:
            p = int(pages) - 1
            start, end = max(0, p), min(total_pages, p + 1)

        output_parts = [f"# {full_path.name} (pages {start+1}-{end} of {total_pages})\n"]

        for i in range(start, end):
            page = doc.pages[i]
            page_num = i + 1
            output_parts.append(f"\n## Page {page_num}\n")

            # Extract text
            text = page.extract_text()
            if text:
                output_parts.append(text)

            # Extract tables
            tables = page.extract_tables()
            if tables:
                for ti, table in enumerate(tables):
                    if table and any(any(cell for cell in row) for row in table):
                        output_parts.append(f"\n[Table {ti+1} on page {page_num}]")
                        for row in table:
                            cells = [str(c) if c else "" for c in row]
                            output_parts.append(" | ".join(cells))

        doc.close()

        result = "\n".join(output_parts)
        if len(result) > 15000:
            result = result[:15000] + f"\n\n... [truncated. Use pages='X-Y' to read specific pages]"
        return result

    return [download_paper, read_paper]
