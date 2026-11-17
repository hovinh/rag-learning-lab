"""Build the repo-feature slide deck: what this repo is, how it's
structured and how it works, with examples.

The deck is generated, not hand-edited, so it can be rebuilt and kept in
sync with the code. Slide *content* is plain data (``SLIDES`` below);
``DeckBuilder`` turns each slide spec into a PowerPoint slide. To change
the deck, edit ``SLIDES`` and rebuild. ``tests/docs/test_repo_feature_deck.py``
fails if a public ``graphrag`` name or an example script is missing from
the deck, or if the deck names a class that no longer exists.

Inline code in any text is written between backticks (`like this`) and
rendered in a monospace font.

Run from the repo root:

    .venv/Scripts/python.exe docs/repo-feature/build_deck.py
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.slide import Slide
from pptx.text.text import TextFrame, _Paragraph
from pptx.util import Emu, Inches, Pt

DEFAULT_OUTPUT = Path(__file__).with_name("repo-feature.pptx")
FOOTER = "rag-learning-lab · repo features"

# --------------------------------------------------------------------------
# Slide specs: the deck's content, as data.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TitleSlide:
    title: str
    subtitle: str
    notes: str = ""


@dataclass(frozen=True)
class BulletSlide:
    """``bullets`` starting with two spaces are rendered one level deeper."""

    title: str
    bullets: tuple[str, ...]
    subtitle: str = ""
    notes: str = ""


@dataclass(frozen=True)
class TableSlide:
    """``widths`` are relative column widths."""

    title: str
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    widths: tuple[float, ...]
    subtitle: str = ""
    caption: str = ""
    font_size: float = 12
    notes: str = ""


@dataclass(frozen=True)
class CodeSlide:
    """Code on the right, optional bullets on the left (full width when
    there are none).
    """

    title: str
    code: str
    bullets: tuple[str, ...] = ()
    subtitle: str = ""
    code_size: float = 11
    notes: str = ""


@dataclass(frozen=True)
class Layer:
    label: str
    boxes: tuple[str, ...]


@dataclass(frozen=True)
class LayerSlide:
    """Stacked architecture layers, top depends on bottom."""

    title: str
    layers: tuple[Layer, ...]
    subtitle: str = ""
    caption: str = ""
    notes: str = ""


@dataclass(frozen=True)
class ChapterSlide:
    """One book chapter: a flow diagram, key points, a code example and the
    command that runs the chapter's example script.
    """

    title: str
    subtitle: str
    flow: tuple[str, ...]
    bullets: tuple[str, ...]
    code: str
    command: str
    notes: str = ""


SlideSpec = TitleSlide | BulletSlide | TableSlide | CodeSlide | LayerSlide | ChapterSlide


def iter_text(spec: SlideSpec) -> Iterator[str]:
    """Every piece of text a slide spec carries (used by the sync test)."""
    match spec:
        case TitleSlide():
            yield from (spec.title, spec.subtitle)
        case BulletSlide():
            yield from (spec.title, spec.subtitle, *spec.bullets)
        case TableSlide():
            yield from (spec.title, spec.subtitle, spec.caption, *spec.header)
            for row in spec.rows:
                yield from row
        case CodeSlide():
            yield from (spec.title, spec.subtitle, spec.code, *spec.bullets)
        case LayerSlide():
            yield from (spec.title, spec.subtitle, spec.caption)
            for layer in spec.layers:
                yield from (layer.label, *layer.boxes)
        case ChapterSlide():
            yield from (spec.title, spec.subtitle, spec.code, spec.command)
            yield from (*spec.flow, *spec.bullets)
    yield spec.notes


# --------------------------------------------------------------------------
# Rendering.
# --------------------------------------------------------------------------


class Theme:
    """Colors and fonts. Subclass and override to restyle the deck."""

    navy = RGBColor(0x1F, 0x2A, 0x44)
    accent = RGBColor(0x0F, 0x76, 0x6E)
    accent_light = RGBColor(0xE6, 0xF2, 0xF1)
    text = RGBColor(0x1F, 0x29, 0x37)
    muted = RGBColor(0x6B, 0x72, 0x80)
    white = RGBColor(0xFF, 0xFF, 0xFF)
    code_bg = RGBColor(0xF3, 0xF4, 0xF6)
    code_border = RGBColor(0xD1, 0xD5, 0xDB)
    inline_code = RGBColor(0x9F, 0x12, 0x39)
    zebra = RGBColor(0xF8, 0xFA, 0xFC)
    layer_fills = (
        RGBColor(0xFE, 0xF3, 0xC7),
        RGBColor(0xDB, 0xEA, 0xFE),
        RGBColor(0xE6, 0xF2, 0xF1),
        RGBColor(0xED, 0xE9, 0xFE),
        RGBColor(0xF3, 0xF4, 0xF6),
    )
    font = "Calibri"
    mono = "Consolas"


class DeckBuilder:
    """Renders slide specs onto a 16:9 deck, one blank-layout slide each."""

    WIDTH = Inches(13.333)
    HEIGHT = Inches(7.5)
    MARGIN = Inches(0.5)
    HEADER_HEIGHT = Inches(1.05)
    CONTENT_TOP = Inches(1.35)
    FOOTER_TOP = Inches(7.08)
    INLINE_CODE_SCALE = 0.88  # Consolas reads larger than Calibri at the same size

    def __init__(self, theme: Theme | None = None) -> None:
        self._theme = theme or Theme()
        self._prs = Presentation()
        self._prs.slide_width = self.WIDTH
        self._prs.slide_height = self.HEIGHT
        self._blank = self._prs.slide_layouts[6]

    @property
    def content_width(self) -> int:
        return self.WIDTH - 2 * self.MARGIN

    def add_all(self, specs: Sequence[SlideSpec]) -> None:
        for spec in specs:
            self.add(spec)

    def add(self, spec: SlideSpec) -> None:
        slide = self._prs.slides.add_slide(self._blank)
        match spec:
            case TitleSlide():
                self._title_slide(slide, spec)
            case BulletSlide():
                self._bullet_slide(slide, spec)
            case TableSlide():
                self._table_slide(slide, spec)
            case CodeSlide():
                self._code_slide(slide, spec)
            case LayerSlide():
                self._layer_slide(slide, spec)
            case ChapterSlide():
                self._chapter_slide(slide, spec)
        if spec.notes:
            slide.notes_slide.notes_text_frame.text = spec.notes

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._prs.save(str(path))
        return path

    # ---- slide types -----------------------------------------------------

    def _title_slide(self, slide: Slide, spec: TitleSlide) -> None:
        t = self._theme
        self._rect(slide, 0, 0, self.WIDTH, self.HEIGHT, t.navy)
        self._rect(slide, self.MARGIN, Inches(3.55), Inches(1.4), Inches(0.08), t.accent)
        title = self._textbox(slide, self.MARGIN, Inches(2.2), self.content_width, Inches(1.3))
        self._write(title.paragraphs[0], spec.title, size=48, color=t.white, bold=True)
        sub = self._textbox(slide, self.MARGIN, Inches(3.85), self.content_width, Inches(1.6))
        self._write(
            sub.paragraphs[0], spec.subtitle, size=22, color=t.white, code_color=t.accent_light
        )
        foot = self._textbox(slide, self.MARGIN, Inches(6.6), self.content_width, Inches(0.4))
        self._write(
            foot.paragraphs[0],
            "Generated by `docs/repo-feature/build_deck.py`",
            size=12,
            color=t.accent_light,
            code_color=t.accent_light,
        )

    def _bullet_slide(self, slide: Slide, spec: BulletSlide) -> None:
        self._chrome(slide, spec.title, spec.subtitle)
        height = self.FOOTER_TOP - self.CONTENT_TOP - Inches(0.1)
        frame = self._textbox(slide, self.MARGIN, self.CONTENT_TOP, self.content_width, height)
        self._bullets(frame, spec.bullets, size=17)

    def _table_slide(self, slide: Slide, spec: TableSlide) -> None:
        t = self._theme
        self._chrome(slide, spec.title, spec.subtitle)
        caption_height = Inches(0.55) if spec.caption else 0
        top = self.CONTENT_TOP
        rows = len(spec.rows) + 1
        shape = slide.shapes.add_table(
            rows,
            len(spec.header),
            self.MARGIN,
            top,
            self.content_width,
            Inches(0.4) * rows,
        )
        table = shape.table
        total = sum(spec.widths)
        for i, weight in enumerate(spec.widths):
            table.columns[i].width = int(self.content_width * weight / total)
        for col, text in enumerate(spec.header):
            self._cell(table.cell(0, col), text, spec.font_size, t.navy, t.white, bold=True)
        for r, row in enumerate(spec.rows, start=1):
            fill = t.zebra if r % 2 == 0 else t.white
            for col, text in enumerate(row):
                self._cell(table.cell(r, col), text, spec.font_size, fill, t.text, bold=col == 0)
        if spec.caption:
            frame = self._textbox(
                slide,
                self.MARGIN,
                self.FOOTER_TOP - caption_height,
                self.content_width,
                caption_height,
            )
            self._write(frame.paragraphs[0], spec.caption, size=13, color=t.muted, italic=True)

    def _code_slide(self, slide: Slide, spec: CodeSlide) -> None:
        self._chrome(slide, spec.title, spec.subtitle)
        height = self.FOOTER_TOP - self.CONTENT_TOP - Inches(0.15)
        if spec.bullets:
            left_width = Inches(5.3)
            frame = self._textbox(slide, self.MARGIN, self.CONTENT_TOP, left_width, height)
            self._bullets(frame, spec.bullets, size=14)
            code_left = self.MARGIN + left_width + Inches(0.3)
        else:
            code_left = self.MARGIN
        code_width = self.WIDTH - self.MARGIN - code_left
        self._code(
            slide, spec.code, code_left, self.CONTENT_TOP, code_width, height, spec.code_size
        )

    def _layer_slide(self, slide: Slide, spec: LayerSlide) -> None:
        t = self._theme
        self._chrome(slide, spec.title, spec.subtitle)
        caption_height = Inches(0.5) if spec.caption else 0
        available = self.FOOTER_TOP - self.CONTENT_TOP - caption_height - Inches(0.1)
        gap = Inches(0.14)
        row_height = int((available - gap * (len(spec.layers) - 1)) / len(spec.layers))
        label_width = Inches(2.1)
        box_gap = Inches(0.12)
        for i, layer in enumerate(spec.layers):
            top = self.CONTENT_TOP + i * (row_height + gap)
            fill = t.layer_fills[i % len(t.layer_fills)]
            label = self._rect(slide, self.MARGIN, top, label_width, row_height, t.navy)
            self._shape_text(label, layer.label, size=13, color=t.white, bold=True)
            boxes_left = self.MARGIN + label_width + box_gap
            boxes_width = self.WIDTH - self.MARGIN - boxes_left
            n = len(layer.boxes)
            box_width = int((boxes_width - box_gap * (n - 1)) / n)
            for j, text in enumerate(layer.boxes):
                left = boxes_left + j * (box_width + box_gap)
                box = self._rect(
                    slide,
                    left,
                    top,
                    box_width,
                    row_height,
                    fill,
                    shape=MSO_SHAPE.ROUNDED_RECTANGLE,
                    line=t.code_border,
                )
                self._shape_text(box, text, size=12, color=t.text)
        if spec.caption:
            frame = self._textbox(
                slide,
                self.MARGIN,
                self.FOOTER_TOP - caption_height,
                self.content_width,
                caption_height,
            )
            self._write(frame.paragraphs[0], spec.caption, size=13, color=t.muted, italic=True)

    def _chapter_slide(self, slide: Slide, spec: ChapterSlide) -> None:
        t = self._theme
        self._chrome(slide, spec.title, spec.subtitle)
        flow_height = Inches(0.85)
        self._flow(slide, spec.flow, self.CONTENT_TOP - Inches(0.1), flow_height)

        body_top = self.CONTENT_TOP + flow_height + Inches(0.2)
        body_height = self.FOOTER_TOP - body_top - Inches(0.1)
        left_width = Inches(5.9)
        frame = self._textbox(slide, self.MARGIN, body_top, left_width, body_height)
        self._bullets(frame, spec.bullets, size=13)

        code_left = self.MARGIN + left_width + Inches(0.3)
        code_width = self.WIDTH - self.MARGIN - code_left
        command_height = Inches(0.5)
        code_height = self._code(
            slide,
            spec.code,
            code_left,
            body_top,
            code_width,
            body_height - command_height - Inches(0.12),
            10.5,
        )

        command = self._rect(
            slide,
            code_left,
            body_top + code_height + Inches(0.12),
            code_width,
            command_height,
            t.navy,
        )
        tf = command.text_frame
        self._margins(tf, Inches(0.12), Inches(0.04))
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.word_wrap = True
        para = tf.paragraphs[0]
        run = para.add_run()
        run.text = "$ " + spec.command
        self._font(run, size=10, color=t.white, mono=True)

    # ---- building blocks -------------------------------------------------

    def _chrome(self, slide: Slide, title: str, subtitle: str) -> None:
        """Header bar with title and optional subtitle, plus footer."""
        t = self._theme
        self._rect(slide, 0, 0, self.WIDTH, self.HEADER_HEIGHT, t.navy)
        self._rect(slide, 0, self.HEADER_HEIGHT, self.WIDTH, Inches(0.05), t.accent)
        frame = self._textbox(
            slide, self.MARGIN, Inches(0.12), self.content_width, self.HEADER_HEIGHT - Inches(0.2)
        )
        frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        self._write(frame.paragraphs[0], title, size=28, color=t.white, bold=True)
        if subtitle:
            para = frame.add_paragraph()
            self._write(para, subtitle, size=13, color=t.accent_light, code_color=t.accent_light)

        foot = self._textbox(slide, self.MARGIN, self.FOOTER_TOP, self.content_width, Inches(0.3))
        para = foot.paragraphs[0]
        self._write(para, FOOTER, size=10, color=t.muted)
        number = self._textbox(slide, self.MARGIN, self.FOOTER_TOP, self.content_width, Inches(0.3))
        para = number.paragraphs[0]
        para.alignment = PP_ALIGN.RIGHT
        self._write(para, str(len(self._prs.slides)), size=10, color=t.muted)

    def _flow(self, slide: Slide, steps: Sequence[str], top: int, height: int) -> None:
        """A left-to-right row of step boxes joined by chevrons."""
        t = self._theme
        arrow = Inches(0.26)
        gap = Inches(0.06)
        n = len(steps)
        box_width = int((self.content_width - (n - 1) * (arrow + 2 * gap)) / n)
        size = 12 if n <= 5 else 11 if n <= 7 else 10
        for i, text in enumerate(steps):
            left = self.MARGIN + i * (box_width + arrow + 2 * gap)
            box = self._rect(
                slide,
                left,
                top,
                box_width,
                height,
                t.accent_light,
                shape=MSO_SHAPE.ROUNDED_RECTANGLE,
                line=t.accent,
            )
            self._shape_text(box, text, size=size, color=t.text)
            if i < n - 1:
                self._rect(
                    slide,
                    left + box_width + gap,
                    top + (height - arrow) // 2,
                    arrow,
                    arrow,
                    t.accent,
                    shape=MSO_SHAPE.CHEVRON,
                )

    def _code(
        self, slide: Slide, code: str, left: int, top: int, width: int, height: int, size: float
    ) -> int:
        """Draws a code panel at most ``height`` tall, shrunk to fit the
        code, and returns the height it used.
        """
        t = self._theme
        lines = code.strip("\n").splitlines()
        padding = Inches(0.12)
        fitted = int(Pt(size) * 1.22 * len(lines) + 2 * padding + Inches(0.05))
        height = min(height, fitted)
        box = self._rect(slide, left, top, width, height, t.code_bg, line=t.code_border)
        tf = box.text_frame
        tf.word_wrap = False
        tf.auto_size = MSO_AUTO_SIZE.NONE
        tf.vertical_anchor = MSO_ANCHOR.TOP
        self._margins(tf, Inches(0.15), padding)
        for i, line in enumerate(lines):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.alignment = PP_ALIGN.LEFT
            run = para.add_run()
            run.text = line or " "
            self._font(run, size=size, color=t.text, mono=True)
        return height

    def _bullets(self, frame: TextFrame, items: Sequence[str], size: float) -> None:
        t = self._theme
        for i, item in enumerate(items):
            para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            level = 1 if item.startswith("  ") else 0
            para.space_before = Pt(0 if i == 0 else (3 if level else 7))
            item_size = size - 1.5 if level else size
            self._bullet_format(para, level)
            self._write(para, item.strip(), size=item_size, color=t.text)

    def _bullet_format(self, para: _Paragraph, level: int) -> None:
        """Hanging-indent bullet; python-pptx has no API for this on a
        plain text box, so set it in the paragraph XML.
        """
        indent = Inches(0.24)
        p_pr = para._p.get_or_add_pPr()
        p_pr.set("marL", str(int(indent * (level + 1) + Inches(0.05) * level)))
        p_pr.set("indent", str(-int(indent)))
        bu_clr = p_pr.makeelement(qn("a:buClr"), {})
        srgb = bu_clr.makeelement(qn("a:srgbClr"), {"val": str(self._theme.accent)})
        bu_clr.append(srgb)
        p_pr.append(bu_clr)
        bu_char = p_pr.makeelement(qn("a:buChar"), {"char": "–" if level else "•"})
        p_pr.append(bu_char)

    def _cell(
        self, cell, text: str, size: float, fill: RGBColor, color: RGBColor, *, bold: bool
    ) -> None:
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = cell.margin_right = Inches(0.08)
        cell.margin_top = cell.margin_bottom = Inches(0.04)
        tf = cell.text_frame
        tf.word_wrap = True
        self._write(tf.paragraphs[0], text, size=size, color=color, bold=bold)

    def _write(
        self,
        para: _Paragraph,
        text: str,
        *,
        size: float,
        color: RGBColor,
        bold: bool = False,
        italic: bool = False,
        code_color: RGBColor | None = None,
    ) -> None:
        """Adds ``text`` to ``para``; `backticked` spans become monospace."""
        for i, part in enumerate(text.split("`")):
            if not part:
                continue
            run = para.add_run()
            run.text = part
            is_code = i % 2 == 1
            self._font(
                run,
                size=size * self.INLINE_CODE_SCALE if is_code else size,
                color=(code_color or self._theme.inline_code) if is_code else color,
                bold=bold,
                italic=italic,
                mono=is_code,
            )

    def _font(
        self,
        run,
        *,
        size: float,
        color: RGBColor,
        bold: bool = False,
        italic: bool = False,
        mono: bool = False,
    ) -> None:
        run.font.size = Pt(size)
        run.font.name = self._theme.mono if mono else self._theme.font
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = color

    def _textbox(self, slide: Slide, left: int, top: int, width: int, height: int) -> TextFrame:
        frame = slide.shapes.add_textbox(left, top, width, height).text_frame
        frame.word_wrap = True
        frame.auto_size = MSO_AUTO_SIZE.NONE
        self._margins(frame, Inches(0.05), Inches(0.03))
        return frame

    def _rect(
        self,
        slide: Slide,
        left: int,
        top: int,
        width: int,
        height: int,
        fill: RGBColor,
        *,
        shape: MSO_SHAPE = MSO_SHAPE.RECTANGLE,
        line: RGBColor | None = None,
    ):
        rect = slide.shapes.add_shape(shape, Emu(left), Emu(top), Emu(width), Emu(height))
        if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
            rect.adjustments[0] = 0.12
        rect.shadow.inherit = False
        rect.fill.solid()
        rect.fill.fore_color.rgb = fill
        if line is None:
            rect.line.fill.background()
        else:
            rect.line.color.rgb = line
            rect.line.width = Pt(1)
        return rect

    def _shape_text(
        self, shape, text: str, *, size: float, color: RGBColor, bold: bool = False
    ) -> None:
        tf = shape.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        self._margins(tf, Inches(0.06), Inches(0.03))
        for i, line in enumerate(text.split("\n")):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.alignment = PP_ALIGN.CENTER
            self._write(para, line, size=size if i == 0 else size - 2, color=color, bold=bold)

    @staticmethod
    def _margins(frame: TextFrame, horizontal: int, vertical: int) -> None:
        frame.margin_left = frame.margin_right = horizontal
        frame.margin_top = frame.margin_bottom = vertical


# --------------------------------------------------------------------------
# The deck.
# --------------------------------------------------------------------------

SLIDES: tuple[SlideSpec, ...] = (
    TitleSlide(
        title="rag-learning-lab",
        subtitle=(
            "The Essential GraphRAG book, re-implemented as a reusable, object-oriented "
            "Python library (`graphrag`) on an embedded Kuzu graph database"
        ),
        notes="Repo-feature overview: what the repo is, how it is structured, how it works.",
    ),
    BulletSlide(
        title="Why this repo exists",
        subtitle="Two goals drive every design decision",
        bullets=(
            (
                "Re-implements the Essential GraphRAG book's notebooks `ch02`–`ch08` as a proper "
                "Python package, `graphrag`, instead of linear notebook scripts"
            ),
            "Goal 1: cover the book's concepts faithfully",
            (
                "  PDF ingestion and chunking, vector and hybrid search, step-back prompting, "
                "parent-child retrieval"
            ),
            (
                "  text2cypher, agentic RAG (query rewriting, tool routing, critique), structured "
                "extraction, Microsoft-style GraphRAG, evaluation"
            ),
            "Goal 2: rebuild it with OOP for reusability",
            (
                "  The book's module-level globals and free functions (`neo4j_driver`, `chat()`, "
                "`embed()`) become classes behind interfaces"
            ),
            "  Every piece can be swapped, mocked and unit-tested on its own",
            (
                "Neo4j is replaced by Kuzu, an embedded graph database: no server, and only "
                "`OPENAI_API_KEY` is needed at runtime"
            ),
            "`template/kg-rag` is the vendored book repo. It is reference only and never edited",
            (
                "Every departure from the book is written up in `examples/README.md`, under "
                '"Where this departs from the book"'
            ),
        ),
    ),
    TableSlide(
        title="From the book to this repo",
        subtitle="One example script per chapter, built from library classes",
        header=("Ch.", "Topic", "Example script (examples/)", "Recipe: main classes"),
        widths=(0.85, 2.5, 4.15, 4.8),
        font_size=12,
        rows=(
            (
                "ch02",
                "PDF ingestion, vector and hybrid search",
                "`vector_similarity_and_hybrid_search.py`",
                "VectorRag: PdfSource, TextChunker, OpenAIEmbedder, KuzuGraphStore",
            ),
            (
                "ch03",
                "Step-back prompting, parent-child retrieval",
                "`parent_child_stepback_rag.py`",
                "ParentChildRag: SectionSplitter, StepBackRetriever, KuzuParentChildGraphStore",
            ),
            (
                "ch04",
                "text2cypher",
                "`text2cypher_movies.py`",
                "Text2CypherQA: Text2Cypher, KuzuCypherGraphStore, MovieGraphSeeder",
            ),
            (
                "ch05",
                "Agentic RAG",
                "`agentic_rag_movies.py`",
                "AgenticRag: Agent, QueryUpdater, ToolRouter, AnswerCritic, Tool subclasses",
            ),
            (
                "ch06",
                "Structured extraction into a knowledge graph",
                "`structured_extraction_contracts.py`",
                "StructuredGraphExtraction: Contract (Pydantic), ContractGraphImporter",
            ),
            (
                "ch07",
                "Microsoft-style GraphRAG",
                "`graphrag_odyssey.py`",
                "GraphRag: EntityRelationshipExtractor, CommunityReporter, GlobalSearch, LocalSearch",
            ),
            (
                "ch08",
                "Evaluating the RAG system",
                "`evaluate_agent_movies.py`",
                "RagEvaluation: BenchmarkRunner, Evaluator, AnswerCorrectness, ContextRecall, Faithfulness",
            ),
            (
                "extra",
                "Hypothetical questions; benchmarking the text recipes",
                "`benchmark_text_recipes_odyssey.py`",
                "HypotheticalQuestionRag, RagEvaluation over 7 variants, ODYSSEY_BENCHMARK_PATH",
            ),
            (
                "shared",
                "Used by several scripts",
                "`constants.py`",
                (
                    "PDF URL, log directory and model; domain data lives in `graphrag.domains`, "
                    "e.g. `build_movie_agent()` for ch05 and ch08"
                ),
            ),
        ),
    ),
    TableSlide(
        title="Kuzu instead of Neo4j",
        subtitle="Embedded, in-process, MIT-licensed, and it speaks Cypher",
        header=("", "Neo4j (the book)", "Kuzu (this repo)"),
        widths=(2.2, 4.6, 5.5),
        font_size=13,
        rows=(
            ("Deployment", "Separate server or Docker", "In-process library: `pip install kuzu`"),
            (
                "Configuration",
                "`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`",
                (
                    "A directory: `KUZU_DB_PATH=data/kuzu_db`. `db_path_for(name)` gives each "
                    "graph its own, e.g. `data/kuzu_db_movies`, kept between runs"
                ),
            ),
            (
                "Schema",
                "Labels created implicitly on insert",
                "`CREATE NODE TABLE` / `CREATE REL TABLE` before any data goes in",
            ),
            (
                "Reading the schema",
                "APOC `apoc.meta.data()` samples the graph",
                "Read from the catalog: `show_tables()`, `table_info()`, `show_connection()`",
            ),
            (
                "Vector search",
                "Vector index",
                "`CREATE_VECTOR_INDEX` (HNSW, cosine distance)",
            ),
            ("Keyword search", "Full-text index", "`CREATE_FTS_INDEX` (BM25 over stemmed tokens)"),
            (
                "Communities",
                "GDS: `gds.graph.project`, `gds.louvain.write`",
                "`algo` extension: `project_graph`, `louvain`",
            ),
            (
                "Isolation",
                "—",
                (
                    "Only the three `Kuzu*Store` classes import `kuzu`; everything else talks to an "
                    "interface"
                ),
            ),
        ),
    ),
    CodeSlide(
        title="Cypher vs SQL: the same question",
        subtitle="`movies_info_by_actor`: an actor's movies with full cast and directors",
        code_size=10.5,
        code=(
            "// Cypher (domains/movies/tools.py)\n"
            "MATCH (a:Person)-[:ACTED_IN]->(m:Movie)\n"
            "WHERE lower(a.name) CONTAINS $actor\n"
            "OPTIONAL MATCH (m)<-[:ACTED_IN]-(c:Person)\n"
            "OPTIONAL MATCH (m)<-[:DIRECTED]-(d:Person)\n"
            "RETURN m.title,\n"
            "       collect(DISTINCT c.name) AS `cast`,\n"
            "       collect(DISTINCT d.name) AS directors\n"
            "\n"
            "-- SQL over person / movie / acted_in / directed\n"
            "SELECT m.title,\n"
            "       array_agg(DISTINCT c.name) AS cast_names,\n"
            "       array_agg(DISTINCT d.name) AS directors\n"
            "FROM person a\n"
            "JOIN acted_in ai      ON ai.person_id = a.id\n"
            "JOIN movie m          ON m.id = ai.movie_id\n"
            "LEFT JOIN acted_in ci ON ci.movie_id = m.id\n"
            "LEFT JOIN person c    ON c.id = ci.person_id\n"
            "LEFT JOIN directed di ON di.movie_id = m.id\n"
            "LEFT JOIN person d    ON d.id = di.person_id\n"
            "WHERE lower(a.name) LIKE '%' || :actor || '%'\n"
            "GROUP BY m.id, m.title;"
        ),
        bullets=(
            "Each arrow `-[:ACTED_IN]->` becomes two joins through a link table",
            (
                "Undirected edges: `(s)-[r]-(t)` instead of a `UNION` of both directions "
                "(`DescriptionSummarizer`)"
            ),
            "Variable-length paths: `-[*1..3]->` instead of a recursive CTE",
            (
                "Multi-hop retrieval (chunk → entity → community in `LocalSearch`) is one more "
                "pattern per hop, not more joins"
            ),
        ),
    ),
    BulletSlide(
        title="Why Cypher instead of SQL",
        subtitle="Inherited from the book, kept because the data is a graph",
        bullets=(
            "Inherited from the book",
            (
                "  ch04–ch08 are written in Cypher; Kuzu speaks it, so queries port with small "
                "dialect fixes (backticked `cast`, directed `MERGE`, no APOC)"
            ),
            "Relationship-shaped questions",
            (
                "  Chunk → entity → community and actor → movie ← director read as patterns, "
                "not join chains"
            ),
            "Easier for the LLM to write (text2cypher)",
            (
                "  The schema prompt is just `(:Person)-[:ACTED_IN]->(:Movie)`, with no join keys "
                "to guess; `Text2Cypher.correct()` retries with the database's error message"
            ),
            "Trade-offs",
            (
                "  Kuzu stores node and rel tables underneath: this picks a query language, "
                "not a storage model"
            ),
            "  Cypher dialects differ more than SQL's, and SQL is far more widely known",
            "  Flat aggregations (community sizes) read the same either way",
            (
                "  `CypherGraphStore` commits to Cypher, so a SQL backend would change the "
                "interface too"
            ),
        ),
    ),
    CodeSlide(
        title="Repository layout",
        code_size=12.5,
        code="""
rag-learning-lab/
├── src/graphrag/        the library, installed editable as graphrag
│   ├── llm/             LLM client, step-back rewriter, token counter
│   ├── prompts/         PromptRepository + every prompt as a .txt file
│   ├── graph/           graph-store interfaces, Kuzu stores, Text2Cypher
│   ├── ingestion/       PDF loading, section splitting, chunking
│   ├── retrieval/       embedder, entity index, global/local search
│   ├── extraction/      structured + entity extraction, communities
│   ├── agent/           agent, query planners, tool router, critic, tools
│   ├── evaluation/      benchmark runner, evaluator, RAGAS-style metrics
│   ├── pipeline/        Pipeline runner + reusable steps
│   ├── recipes/         predefined pipelines, one per chapter
│   ├── observability/   Trace, RunLogging, UsageMeter: the run logs
│   ├── domains/         domain packs: movies, contracts, odyssey
│   ├── services.py      OpenAIServices: LLM client + embedder
│   └── settings.py      Settings.from_env(), db_path_for(name)
├── examples/            one runnable script per chapter + README.md
├── tests/               pytest, mirrors src/graphrag/ package by package
├── docs/                python.md (conventions), repo-feature/ (this deck)
├── template/kg-rag/     vendored book repo: reference only
├── requirements.in/.txt pip-tools source / pinned lockfile
└── pyproject.toml       package, pytest, ruff and mypy config
""",
    ),
    LayerSlide(
        title="Architecture",
        subtitle="Dependencies point downward; only the bottom layer knows about OpenAI or Kuzu",
        layers=(
            Layer(
                "Entry points",
                (
                    "examples/*.py\none composition root per chapter",
                    "Settings.from_env()\n.env → OPENAI_API_KEY, KUZU_DB_PATH",
                    "OpenAIServices\nLLM client + embedder",
                ),
            ),
            Layer(
                "Domains, recipes",
                (
                    "domains.movies\nseeder, tools, scope, benchmark",
                    "domains.contracts\nmodel, importer, prompt",
                    "domains.odyssey\ncorpus, entity types",
                    "recipes\nVectorRag, GraphRag, AgenticRag, …",
                ),
            ),
            Layer(
                "Orchestration",
                (
                    "pipeline\nPipeline + steps",
                    "agent\nAgent, ToolRouter, AnswerCritic",
                    "evaluation\nBenchmarkRunner, Evaluator, metrics",
                    "observability\nTrace, RunLogging, UsageMeter",
                ),
            ),
            Layer(
                "Capabilities",
                (
                    "ingestion\nPdfLoader, SectionSplitter, TextChunker",
                    "graph.Text2Cypher\nquestion → Cypher",
                    "extraction\nstructured + entity graphs",
                    "retrieval\nglobal / local search",
                ),
            ),
            Layer(
                "Interfaces",
                (
                    "LLMClient",
                    "Embedder",
                    "GraphStore · CypherGraphStore\nParentChildGraphStore",
                    "PromptRepository",
                ),
            ),
            Layer(
                "Adapters",
                (
                    "OpenAILLMClient\nOpenAIEmbedder",
                    "KuzuGraphStore · KuzuCypherGraphStore\nKuzuParentChildGraphStore",
                    "prompts/*.txt",
                ),
            ),
        ),
        caption=(
            "Classes take their dependencies through the constructor, so tests swap the bottom "
            "layer for fakes or a throwaway Kuzu database."
        ),
    ),
    TableSlide(
        title="Package reference (1/2)",
        subtitle="Every public name exported from a graphrag package",
        header=("Package", "Public API (__all__)", "Role"),
        widths=(1.7, 7.0, 3.6),
        font_size=11.5,
        rows=(
            (
                "graphrag.llm",
                (
                    "LLMClient, OpenAILLMClient, Message, ToolCall, ToolDefinition, "
                    "LLMResponseError, QueryRewriter, HypotheticalQuestionGenerator, "
                    "TokenCounter, load_json_object, load_json_field"
                ),
                (
                    "Chat, tool choice and structured output; step-back rewriting; questions a "
                    "chunk answers; tiktoken counts; JSON parsing"
                ),
            ),
            (
                "graphrag.prompts",
                "PromptRepository",
                "Loads prompt text from .txt files",
            ),
            (
                "graphrag.graph",
                (
                    "GraphStore, KuzuGraphStore, CypherGraphStore, KuzuCypherGraphStore, "
                    "ParentChildGraphStore, KuzuParentChildGraphStore, ChunkMatch, ParentMatch, "
                    "GraphSchema, Property, RelationshipPattern, GraphQueryError, Text2Cypher, "
                    "CypherExample"
                ),
                (
                    "Storage and search interfaces, their Kuzu implementations, schema "
                    "models, text2cypher"
                ),
            ),
            (
                "graphrag.ingestion",
                (
                    "DocumentSource, PdfSource, TextFileSource, PdfLoader, SectionSplitter, "
                    "TextChunker, Chunk"
                ),
                "Where documents come from; split them into sections and chunks",
            ),
            (
                "graphrag.retrieval",
                (
                    "Embedder, OpenAIEmbedder, Retriever, ScoredMatch, VectorRetriever, "
                    "HybridRetriever, ParentChildRetriever, StepBackRetriever, "
                    "EntityEmbeddingIndexer, ENTITY_VECTOR_INDEX, GlobalSearch, KeyPoint, "
                    "LocalSearch, LocalContext"
                ),
                "Embeddings; question → scored matches; ch07 entity index, global/local search",
            ),
            (
                "graphrag.settings",
                "Settings",
                "Runtime config read once from the environment; `db_path_for(name)`",
            ),
            (
                "graphrag.services",
                "OpenAIServices",
                "LLM client + embedder sharing one OpenAI client and UsageMeter",
            ),
        ),
    ),
    TableSlide(
        title="Package reference (2/2)",
        subtitle="Every public name exported from a graphrag package",
        header=("Package", "Public API (__all__)", "Role"),
        widths=(1.7, 7.0, 3.6),
        font_size=11.5,
        rows=(
            (
                "graphrag.extraction",
                (
                    "StructuredExtractor, EntityRelationshipExtractor, ChunkExtraction, "
                    "ExtractedEntity, ExtractedRelationship, EntityGraphImporter, "
                    "DescriptionSummarizer, KuzuLouvainCommunityDetector, CommunityReporter, "
                    "CommunityReport"
                ),
                (
                    "Extraction into any Pydantic model; ch07 schema-free entity graph, "
                    "summaries, communities, reports"
                ),
            ),
            (
                "graphrag.agent",
                (
                    "Agent, AgentResponse, QueryPlanner, QueryUpdater, QueryDecomposer, ToolRouter, "
                    "ToolRoutingError, AnswerCritic, Tool, ToolResult, string_parameters, "
                    "AnswerGivenTool, Text2CypherTool, RetrieverTool, AgentScope"
                ),
                "ch05 agent loop and generic tools; ch08 question splitting",
            ),
            (
                "graphrag.evaluation",
                (
                    "BenchmarkCase, load_benchmark, BenchmarkRunner, Responder, AnsweredQuestion, "
                    "EvaluationSample, "
                    "Evaluator, EvaluationReport, ScoredSample, Metric, AnswerCorrectness, "
                    "ContextRecall, Faithfulness, StatementExtractor, save_samples, load_samples, "
                    "format_score, log_report"
                ),
                "ch08 benchmark answers, LLM-judge metrics, reports, saved samples",
            ),
            (
                "graphrag.observability",
                (
                    "Trace, Progress, RunLogging, TraceFormatter, UsageMeter, Usage, preview, "
                    "render_table, format_duration, format_count"
                ),
                "Sectioned, timed run logs; LLM and embedding usage counts",
            ),
            (
                "graphrag.pipeline",
                "Pipeline, PipelineStep, and every step on the next slide",
                "Runs steps over a context object",
            ),
            (
                "graphrag.recipes",
                (
                    "VectorRag, SearchMode, ParentChildRag, HypotheticalQuestionRag, "
                    "Text2CypherQA, StructuredGraphExtraction, GraphRag, AgenticRag, "
                    "RagEvaluation, RagAnswer, CypherAnswer"
                ),
                "Recipes (predefined pipelines): index once, query on every later run",
            ),
            (
                "graphrag.domains",
                "One subpackage per domain: see Domain packs",
                "Everything specific to one dataset",
            ),
        ),
    ),
    TableSlide(
        title="Pipeline steps",
        subtitle=(
            "`graphrag.pipeline`: each step declares the context fields it needs as a Protocol, "
            "so any dataclass with them can run it"
        ),
        header=("Module", "Steps", "Context fields read → written"),
        widths=(1.7, 6.0, 4.6),
        font_size=11.5,
        rows=(
            (
                "ingestion",
                "LoadDocumentsStep, SplitSectionsStep, ChunkStep, ParentChildChunkStep",
                "DocumentSource → `documents` → `chunks` / `parents`, `children_per_parent`",
            ),
            (
                "indexing",
                (
                    "EmbedChunksStep, StoreChunksStep, GenerateQuestionsStep, EmbedChildrenStep, "
                    "StoreParentChildStep"
                ),
                (
                    "`chunks` → `embeddings` → GraphStore; `chunks` → `parents` + questions as "
                    "children → ParentChildGraphStore"
                ),
            ),
            (
                "retrieval, answer",
                "RetrieveStep (any Retriever), AnswerStep",
                "`question` → `matches` → `answer`",
            ),
            (
                "cypher",
                "GenerateCypherStep, ExecuteCypherStep",
                "`question` → `cypher` → `rows`",
            ),
            (
                "extraction",
                "ExtractStructuredStep, ImportExtractedStep",
                "`documents` → `extracted` (Pydantic models) → `imported_ids`",
            ),
            (
                "graph_index",
                (
                    "ExtractEntityGraphStep, SummarizeDescriptionsStep, DetectCommunitiesStep, "
                    "ReportCommunitiesStep, IndexEntitiesStep"
                ),
                "`chunks` → the ch07 entity graph; later steps work on the graph only",
            ),
            (
                "graph_search",
                "GlobalSearchStep, LocalSearchStep",
                "`question` → `key_points` / `local_context`, `answer`",
            ),
        ),
        caption=(
            "Shared field names are what let steps chain; a script only defines the context "
            "dataclass and the order of the steps."
        ),
    ),
    TableSlide(
        title="Domain packs",
        subtitle=(
            "`graphrag.domains`: what is specific to one dataset, so the core packages stay "
            "domain-agnostic"
        ),
        header=("Pack", "Public API (__all__)", "Holds"),
        widths=(2.1, 6.4, 3.8),
        font_size=12,
        rows=(
            (
                "domains.movies",
                (
                    "MovieGraphSeeder, MOVIES_TERMINOLOGY, MOVIES_CYPHER_EXAMPLES, "
                    "MovieInfoByTitleTool, MoviesInfoByActorTool, movie_tools, "
                    "MOVIES_AGENT_SCOPE, build_movie_agent, MOVIES_BENCHMARK_PATH"
                ),
                "Graph data and schema, text2cypher sections, agent tools and scope, benchmark",
            ),
            (
                "domains.contracts",
                (
                    "Contract, ContractType, Organization, Location, ContractGraphImporter, "
                    "build_contract_extractor, LICENSE_AGREEMENT_PATH"
                ),
                "Pydantic model, its extraction prompt, its graph, a sample agreement",
            ),
            (
                "domains.odyssey",
                (
                    "OdysseyCorpus, ODYSSEY_URL, ODYSSEY_ENTITY_TYPES, ODYSSEY_BENCHMARK_PATH, "
                    "ODYSSEY_QUESTION_PURPOSE"
                ),
                (
                    "24 books (`books=` keeps the first), entity types, a Book I benchmark, the "
                    "purpose hypothetical questions serve"
                ),
            ),
        ),
        caption=(
            "A new domain is a new pack: its data or DocumentSource, prompts, models, tools and "
            "an AgentScope, wired from the generic classes."
        ),
    ),
    TableSlide(
        title="Recipes",
        subtitle=(
            "`graphrag.recipes`: predefined pipelines, each chapter's flow as one reusable class"
        ),
        header=("Recipe", "Built from", "Index (once)", "Query"),
        widths=(2.6, 3.4, 3.3, 3.0),
        font_size=11.5,
        rows=(
            (
                "VectorRag (ch02)",
                "GraphStore, Embedder, TextChunker",
                "`index(source)`: chunk, embed, store",
                "`query(q, search=` vector or hybrid`)`",
            ),
            (
                "ParentChildRag (ch03)",
                "ParentChildGraphStore, SectionSplitter",
                "`index(source)`: sections, parents, children",
                "`query(q)`, step-back by default",
            ),
            (
                "Text2CypherQA (ch04)",
                "CypherGraphStore, terminology, examples",
                "the domain seeds the graph",
                "`query(q)` → CypherAnswer",
            ),
            (
                "AgenticRag (ch05)",
                "any Tool list, AgentScope",
                "the tools' own data",
                "`query(q)` → RagAnswer",
            ),
            (
                "StructuredGraphExtraction (ch06)",
                "StructuredExtractor, an import function",
                "`index(source)` → imported ids",
                "the domain's queries",
            ),
            (
                "HypotheticalQuestionRag",
                "ParentChildGraphStore, a `purpose`",
                "`index(source)`: chunks, LLM questions per chunk",
                "`query(q)`: question-to-question match",
            ),
            (
                "GraphRag (ch07)",
                "CypherGraphStore, entity types",
                "`index(source, limit=)`: entity graph, communities",
                "`global_search(q)`, `local_search(q)`",
            ),
            (
                "RagEvaluation (ch08)",
                "Metric list (the book's three by default)",
                "—",
                "`answer(cases, respond, store=)`, `score(samples)`",
            ),
        ),
        caption=(
            "Indexes persist in their Kuzu database: `is_indexed()` says whether to index. Every "
            "`query` returns a RagAnswer (answer + contexts), so any recipe can be benchmarked."
        ),
    ),
    TableSlide(
        title="Core interfaces",
        subtitle="Callers depend on these abstract base classes, never on a concrete client",
        header=("Interface", "Contract", "Implementations", "In tests"),
        widths=(2.2, 4.0, 3.9, 2.2),
        font_size=12,
        rows=(
            (
                "LLMClient",
                "`complete`, `stream`, `choose_tools`, `parse`",
                "OpenAILLMClient",
                "ScriptedLLMClient",
            ),
            ("Embedder", "`embed(texts)`, `dimensions`", "OpenAIEmbedder", "FakeEmbedder"),
            (
                "GraphStore",
                "`replace_chunks`, `vector_search`, `keyword_search`, `hybrid_search`",
                "KuzuGraphStore",
                "Kuzu in `tmp_path`",
            ),
            (
                "ParentChildGraphStore",
                "`replace_parents`, `parent_retrieval`",
                "KuzuParentChildGraphStore",
                "Kuzu in `tmp_path`",
            ),
            (
                "CypherGraphStore",
                "`query(cypher, params)`, `get_schema()`",
                "KuzuCypherGraphStore",
                "Kuzu in `tmp_path`",
            ),
            (
                "Retriever",
                "`retrieve(question)` → ScoredMatch list",
                "VectorRetriever, HybridRetriever, ParentChildRetriever, StepBackRetriever",
                "fake stores",
            ),
            (
                "QueryPlanner",
                "`plan(question, history)` → questions to route",
                "QueryUpdater (ch05), QueryDecomposer (ch08)",
                "ScriptedLLMClient",
            ),
            (
                "Tool",
                "`name`, `description`, `parameters`, `run(arguments)` → ToolResult",
                (
                    "MovieInfoByTitleTool, MoviesInfoByActorTool, Text2CypherTool, RetrieverTool, "
                    "AnswerGivenTool"
                ),
                "real tools on a seeded graph",
            ),
            (
                "Metric",
                "`score(sample)` → float or None (not applicable)",
                "AnswerCorrectness, ContextRecall, Faithfulness",
                "ScriptedLLMClient",
            ),
        ),
        caption="Fakes live in `tests/conftest.py`. Graph tests use a real, throwaway Kuzu database.",
    ),
    CodeSlide(
        title="How a script is wired",
        subtitle="Composition root + a recipe, from `examples/vector_similarity_and_hybrid_search.py`",
        bullets=(
            (
                "`Settings.from_env()` reads `.env` once: `OPENAI_API_KEY` (required) and "
                "`KUZU_DB_PATH` (default `data/kuzu_db`)"
            ),
            (
                "Each script's `main()` is the composition root: it builds the concrete clients and "
                "injects them into constructors"
            ),
            (
                "`OpenAIServices.create(settings, usage=run.usage)` makes the LLM client and "
                "embedder, so `RunLogging` counts every call"
            ),
            (
                "`--reindex` (or an empty database, `is_indexed()` false) makes the recipe "
                "index; otherwise the index kept in Kuzu is reused"
            ),
            (
                "Inside, the `VectorRag` recipe is a Pipeline of steps from `graphrag.pipeline`; each "
                "step still logs as its own section"
            ),
            "Stores are context managers, so the database connection is always closed",
        ),
        code="""
with RunLogging(process_name, LOG_DIR,
                verbose=verbose) as run:
    load_dotenv()
    settings = Settings.from_env()
    services = OpenAIServices.create(settings,
                                     usage=run.usage)
    db_path = settings.db_path_for("ch02")

    with KuzuGraphStore(db_path, embedding_dim=
                        services.embedder.dimensions) as store:
        rag = VectorRag(store, services.embedder,
                        services.llm,
                        chunker=TextChunker(500, 40))
        if reindex or not rag.is_indexed():
            rag.index(PdfSource(PDF_URL, PDF_PATH))

        rag.query(QUESTION, search="vector")
        rag.query(QUESTION, search="hybrid")
""",
    ),
    CodeSlide(
        title="Reading a run's log",
        subtitle="`graphrag.observability`: every example logs the same indented, timed tree",
        bullets=(
            (
                "`--process-name` names `logs/<process_name>.log`, overwritten each run; "
                "`--verbose` adds full payloads to the console"
            ),
            (
                "`┌ … └` is a section (a pipeline step, a question, an agent round), closed with "
                "its time and LLM usage"
            ),
            (
                "INFO lines are summaries (`2 rows: …`); the file also keeps the full payload "
                "behind each one (DEBUG)"
            ),
            "`⚠` marks warnings; a crashed run logs its traceback and ends `✗ failed`",
            (
                "Library code logs through `Trace(__name__)` and stays silent until a script opens "
                "`RunLogging`"
            ),
            "The footer totals LLM and embedding calls and tokens, counted by `UsageMeter`",
        ),
        code_size=10.5,
        code="""
══════════════════════════════════════════════
 ch05 · Agentic RAG over the movies graph
 model     gpt-4o
══════════════════════════════════════════════
21:25:50 ┌ Question 2/2
21:25:50 │ question  How many directors and producers…
21:25:51 │ ┌ Gather
21:25:51 │ │ plan      How many directors and producers…
21:25:52 │ │ tool      text2cypher(question="…")
21:25:54 │ │ cypher    MATCH (p:Person)-[:DIRECTED]->(:Movie)
         │ │           WITH COUNT(DISTINCT p) AS directorCount
         │ │           …
21:25:54 │ │ result    directorCount=10, producerCount=4
21:25:54 │ └ 2.9s · 3 LLM calls · 1.3k tokens
21:25:55 │ critique  nothing missing
21:25:55 │ answer    There are 10 directors and 4 producers…
21:25:55 └ 4.5s · 5 LLM calls · 2.0k tokens
══════════════════════════════════════════════
 ✓ finished in 11.9s
 LLM       11 calls · 5.3k tokens (5.0k in, 285 out)
══════════════════════════════════════════════
""",
    ),
    TableSlide(
        title="Prompts are files, not string constants",
        subtitle="`PromptRepository().load(name)` reads `src/graphrag/prompts/<name>`",
        header=("Family", "Files", "Used by"),
        widths=(1.9, 6.2, 4.2),
        font_size=12,
        rows=(
            ("Answering", "`answer_system`, `answer_user`", "AnswerStep (ch02, ch03)"),
            ("Step-back", "`step_back_system`", "QueryRewriter (ch03)"),
            (
                "text2cypher",
                (
                    "`text2cypher_instructions`, `_schema`, `_terminology`, `_examples`, "
                    "`_question`, `_correction`"
                ),
                "Text2Cypher (ch04, ch05 tool)",
            ),
            (
                "Agent",
                (
                    "`agent_query_update_*`, `agent_query_decompose_system`, "
                    "`agent_tool_picker_*`, `agent_tool_answer`, `agent_answer_critique_*`, "
                    "`agent_main_*`"
                ),
                (
                    "QueryUpdater, QueryDecomposer, ToolRouter, AnswerCritic, Agent; "
                    "AgentScope fills `{source}` and `{topic}`"
                ),
            ),
            (
                "Contracts",
                "`domains/contracts/contract_extraction_system`",
                "build_contract_extractor (ch06)",
            ),
            (
                "GraphRAG",
                (
                    "`graph_extraction`, `description_summary`, `community_report`, "
                    "`global_map_system`, `global_reduce_system`, `local_search_system`"
                ),
                "ch07 extraction, summaries, reports, global and local search",
            ),
            (
                "Evaluation",
                (
                    "`eval_statements_*`, `eval_answer_correctness_*`, `eval_context_recall_*`, "
                    "`eval_faithfulness_*`"
                ),
                "StatementExtractor and the ch08 metrics",
            ),
        ),
        caption=(
            "Prompts start from the book's wording. Every change to one is explained in "
            "`examples/README.md`."
        ),
    ),
    ChapterSlide(
        title="ch02 · PDF ingestion, vector and hybrid search",
        subtitle="Re-implements `template/kg-rag/notebooks/ch02.ipynb`",
        flow=(
            "Download PDF\nPdfLoader",
            "Chunk 500 / 40\nTextChunker",
            "Embed chunks\nOpenAIEmbedder",
            "Store in Kuzu\nKuzuGraphStore",
            "Retrieve\nVector/HybridRetriever",
            "Answer\nAnswerStep",
        ),
        bullets=(
            (
                "Vector search uses Kuzu's HNSW index. It returns cosine distance, which the store "
                "flips to `score = 1 - distance` so higher is better"
            ),
            (
                "Keyword search uses Kuzu's full-text index: BM25 over lowercased, stemmed tokens "
                "with stopwords removed"
            ),
            (
                "`hybrid_search` divides each result set by its own max score, merges them and keeps "
                "the best score per chunk (`GraphStore._merge_normalized`)"
            ),
            (
                "The same question is answered twice, with vector then hybrid retrieval, to compare "
                "the two"
            ),
            "Vector search catches paraphrases; keyword search catches exact terms",
        ),
        code="""
with KuzuGraphStore(db_path, embedding_dim=1536) as store:
    store.replace_chunks(chunks, embeddings)

    q = embedder.embed([question])[0]
    vector_hits = store.vector_search(q, k=4)
    hybrid_hits = store.hybrid_search(question, q, k=4)

    for hit in hybrid_hits:          # ChunkMatch
        print(hit.score, hit.text[:80])
""",
        command=".venv/Scripts/python examples/vector_similarity_and_hybrid_search.py",
    ),
    ChapterSlide(
        title="ch03 · Step-back prompting and parent-child retrieval",
        subtitle="Re-implements `template/kg-rag/notebooks/ch03.ipynb`",
        flow=(
            "Load PDF",
            "Split sections\nSectionSplitter",
            "Parents 2000\nchildren 500",
            "Embed children only",
            "Store hierarchy\nParent→Child",
            "Step-back rewrite\nStepBackRetriever",
            "Parent retrieval → answer\nParentChildRetriever",
        ),
        bullets=(
            (
                "Sections run from 153 to 4,156 tokens (`TokenCounter`, gpt-4o's tokenizer), too "
                "uneven to retrieve as they are, so each is chunked again"
            ),
            (
                "Small child chunks are embedded and searched; the larger parent chunk they came "
                "from is what goes to the LLM"
            ),
            (
                "`parent_retrieval` vector-searches `Child` nodes, follows `HAS_CHILD` back to the "
                "`Parent` and keeps each parent's best child score"
            ),
            (
                "`QueryRewriter.step_back()` turns a specific question into a broader one, used "
                "only for retrieval"
            ),
            "The final answer still uses the original question",
        ),
        code="""
rewriter = QueryRewriter(llm)
broad = rewriter.step_back(
    "Who was Einstein's collaborator on sound reproduction?")
# -> "Who were Einstein's professional associates?"

with KuzuParentChildGraphStore(db_path,
                               embedding_dim=1536) as store:
    store.replace_parents(parents, children_per_parent,
                          child_embeddings_per_parent)
    q = embedder.embed([broad])[0]
    for match in store.parent_retrieval(q, k=4):  # ParentMatch
        print(match.score, match.text[:80])
""",
        command=".venv/Scripts/python examples/parent_child_stepback_rag.py",
    ),
    ChapterSlide(
        title="ch04 · text2cypher",
        subtitle="Re-implements `template/kg-rag/notebooks/ch04.ipynb`",
        flow=(
            "Seed movies graph\nMovieGraphSeeder",
            "Read schema\nget_schema()",
            "Build prompt\nText2Cypher",
            "LLM writes Cypher",
            "Run on Kuzu\nstore.query()",
        ),
        bullets=(
            (
                'Answers questions that need counting or aggregation ("Who directed the most '
                "movies?\"), which chunk similarity can't do"
            ),
            (
                "Bundled movies graph: 14 films, 48 people, with `ACTED_IN`, `DIRECTED`, `PRODUCED` "
                "and `WROTE`, loaded from CSV with `COPY ... FROM`"
            ),
            (
                "The schema comes straight from Kuzu's catalog into a GraphSchema (Property, "
                "RelationshipPattern) and is rendered by `to_prompt()`"
            ),
            (
                "Prompt sections: instructions, schema, terminology mapping, few-shot "
                "CypherExample pairs, then the question"
            ),
            (
                "Kuzu dialect rules (`size()`, `lower()`) and a `missing_information` reply stop "
                "the LLM from inventing data"
            ),
        ),
        code="""
with KuzuCypherGraphStore(db_path) as store:
    MovieGraphSeeder(store).seed()
    text2cypher = Text2Cypher(
        llm,
        store.get_schema(),
        terminology=MOVIES_TERMINOLOGY,
        examples=MOVIES_CYPHER_EXAMPLES,
        model="gpt-4o",
    )
    cypher = text2cypher.generate("Who directed the most movies?")
    rows = store.query(cypher)   # list[dict]; GraphQueryError if rejected
""",
        command=".venv/Scripts/python examples/text2cypher_movies.py",
    ),
    ChapterSlide(
        title="ch05 · Agentic RAG",
        subtitle="Re-implements `template/kg-rag/notebooks/ch05.ipynb`",
        flow=(
            "Question",
            "Query update\nQueryPlanner",
            "Tool routing\nToolRouter",
            "Critique\nAnswerCritic",
            "Follow-up round (once)",
            "Final answer\nAgent",
        ),
        bullets=(
            (
                "The LLM decides how to answer each question, keeping a history of "
                '"For the question X, we have the answer Y" messages'
            ),
            (
                "Tools: `movie_info_by_title` and `movies_info_by_actor` (fixed Cypher), "
                "`text2cypher` (fallback) and `respond` (answer already known, or small talk)"
            ),
            (
                "Text2CypherTool sends a rejected query back to `Text2Cypher.correct()` before "
                "giving up and returning the error as its result"
            ),
            (
                "The critic checks the history against the original question and may ask "
                "follow-up questions, for one extra round"
            ),
            "An unknown tool or a missing argument raises ToolRoutingError",
            (
                "To add a tool: subclass Tool and pass it to ToolRouter. AgentScope words the "
                "prompts for a domain (MOVIES_AGENT_SCOPE keeps the book's)"
            ),
        ),
        code="""
class MovieInfoByTitleTool(Tool):
    name = "movie_info_by_title"
    description = "Get information about a movie by providing the title"
    parameters = string_parameters("title", "The movie title")

    def run(self, arguments) -> ToolResult:
        return self._store.query(self._QUERY,
                                 {"title": arguments["title"].lower()})

agent = build_movie_agent(llm, store)   # graphrag.domains.movies
agent.run("Who's the main actor in the movie Matrix "
          "and what other movies is that person in?")
""",
        command=".venv/Scripts/python examples/agentic_rag_movies.py",
    ),
    ChapterSlide(
        title="ch06 · Structured extraction into a knowledge graph",
        subtitle="Re-implements `template/kg-rag/notebooks/ch06.ipynb`",
        flow=(
            "Load contract\nLICENSE_AGREEMENT_PATH",
            "Extract\nStructuredExtractor",
            "Import\nContractGraphImporter",
            "Query the parties",
        ),
        bullets=(
            (
                "Schema first: the entities that matter are written down as Pydantic models "
                "(Contract, Organization, Location, ContractType)"
            ),
            (
                "`LLMClient.parse()` passes the model as `response_format`, so the API constrains "
                "decoding to its JSON schema"
            ),
            (
                "Field descriptions are the instructions the LLM sees. Optional fields are "
                "`str | None` with no default, as strict mode requires"
            ),
            (
                "StructuredExtractor works with any Pydantic model; the whole ~33k-character "
                "document goes in one call"
            ),
            (
                "Organizations are merged by name and locations by address, so many contracts "
                "join up into one graph"
            ),
        ),
        code="""
# graphrag.domains.contracts: StructuredExtractor[Contract]
extractor = build_contract_extractor(llm, model="gpt-4o")
contract = extractor.extract(LICENSE_AGREEMENT_PATH.read_text())

with KuzuCypherGraphStore(db_path) as store:
    importer = ContractGraphImporter(store)
    importer.create_schema(reset=True)
    importer.import_contract(contract)

# (:Organization)-[:HAS_PARTY {role}]->(:Contract)
# (:Organization)-[:LOCATED_AT]->(:Location)
""",
        command=".venv/Scripts/python examples/structured_extraction_contracts.py",
    ),
    ChapterSlide(
        title="ch07 · Microsoft-style GraphRAG",
        subtitle="Re-implements `template/kg-rag/notebooks/ch07.ipynb` on Homer's Odyssey",
        flow=(
            "Chunk books\nOdysseyCorpus",
            "Extract entities + rels",
            "Import, merge by name",
            "Summarize\ndescriptions",
            "Louvain communities",
            "Community reports",
            "Embed entities",
            "Global / local search",
        ),
        bullets=(
            (
                "No predefined schema: the LLM gets entity types (PERSON, GOD, LOCATION, …) and "
                "returns tuple records that become ExtractedEntity and ExtractedRelationship"
            ),
            (
                "DescriptionSummarizer merges each entity's descriptions; KuzuLouvainCommunityDetector "
                "clusters entities; CommunityReporter writes a rated CommunityReport"
            ),
            (
                "GlobalSearch is a map-reduce over reports rated 5 or more: KeyPoint scores, then one "
                "merged answer"
            ),
            (
                "LocalSearch embeds the question and finds the 5 nearest entities, then gathers their "
                "chunks, community reports and strongest relationships (LocalContext)"
            ),
            "Only book 1 by default; `--books N` adds more, at 100–150 LLM calls per book",
        ),
        code="""
(:Book)-[:HAS_CHUNK]->(:Chunk {text})
(:Chunk)-[:MENTIONS]->(:Entity {name, type,
         description[], summary, community_id, embedding})
(:Entity)-[:RELATIONSHIP {description, strength}]->(:Entity)
(:Entity)-[:SUMMARIZED_RELATIONSHIP {summary, weight}]->(:Entity)
(:Entity)-[:IN_COMMUNITY]->(:Community {title, summary,
         rating, rating_explanation})

GlobalSearch(store, llm, model="gpt-4o").search(
    "What is this story about?")
LocalSearch(store, llm, embedder, model="gpt-4o").search(
    "Who is Ulysses?")
""",
        command=".venv/Scripts/python examples/graphrag_odyssey.py --books 1",
    ),
    ChapterSlide(
        title="ch08 · Evaluating the RAG system",
        subtitle="Re-implements `template/kg-rag/notebooks/ch08.ipynb`: scores the ch05 agent",
        flow=(
            "Benchmark CSV\nquestion;cypher",
            "Ground truth\nrun the Cypher",
            "Agent.respond\nanswer + contexts",
            "LLM judges\n3 metrics",
            "Report\nCSV + samples JSON",
        ),
        bullets=(
            (
                "17 benchmark rows store Cypher, not answers (`load_benchmark` → BenchmarkCase), so "
                "the ground truth follows the data"
            ),
            (
                "AnswerCorrectness: statement F1 (StatementExtractor) × 0.75 + embedding cosine "
                "similarity × 0.25"
            ),
            (
                "ContextRecall: share of ground-truth facts the tools retrieved. Faithfulness: share "
                "of the answer the contexts support"
            ),
            (
                "The RAGAS metrics are re-implemented as Metric subclasses, without depending on "
                "`ragas`. A failed question is recorded on its EvaluationSample, not hidden"
            ),
            (
                "`--limit N`, `--rescore` (re-judge saved samples), `--decompose` (QueryDecomposer "
                "splits each question first)"
            ),
        ),
        code="""
cases = load_benchmark(MOVIES_BENCHMARK_PATH)
agent = build_movie_agent(llm, store, decompose=False)
samples = BenchmarkRunner(agent.respond, store=store).run(cases)
save_samples(samples, Path("logs/ch08_samples.json"))

report = Evaluator([
    AnswerCorrectness(llm, embedder, model="gpt-4o"),
    ContextRecall(llm, model="gpt-4o"),
    Faithfulness(llm, model="gpt-4o"),
]).evaluate(samples)                 # EvaluationReport
report.means(); report.failed()      # rows: ScoredSample
""",
        command=".venv/Scripts/python examples/evaluate_agent_movies.py --limit 5",
    ),
    ChapterSlide(
        title="extra · Hypothetical question embedding",
        subtitle="Not in the book's notebooks: index each chunk under the questions it answers",
        flow=(
            "Load documents",
            "Chunk 2000\nChunkStep",
            "LLM writes questions\nGenerateQuestionsStep",
            "Embed questions only\nEmbedChildrenStep",
            "Store chunk→questions\nParentChildGraphStore",
            "Match question to\nquestions → chunk",
        ),
        bullets=(
            (
                "A user question is matched question-to-question, which embeds closer than "
                "question-to-passage"
            ),
            (
                "`purpose` (who asks, about what) steers which facts the questions cover: the "
                "index only captures what they ask about"
            ),
            (
                "Reuses ch03's store: the chunk is the parent, its questions are the children, "
                "so retrieval hands back the chunk"
            ),
            "Costs one LLM call per chunk at index time; querying costs the same as vector search",
        ),
        code="""
rag = HypotheticalQuestionRag(
    store, embedder, llm,          # KuzuParentChildGraphStore
    purpose=ODYSSEY_QUESTION_PURPOSE,
    questions_per_chunk=8,
)
if not rag.is_indexed():
    rag.index(OdysseyCorpus(path, books=1))
rag.query("Why is Neptune angry with Ulysses?")  # RagAnswer

HypotheticalQuestionGenerator(llm, purpose=...).generate(chunk)
# -> ["What caused Ulysses' men to perish?", ...]
""",
        command=(
            ".venv/Scripts/python examples/benchmark_text_recipes_odyssey.py "
            "--variants hypothetical_questions"
        ),
    ),
    TableSlide(
        title="Benchmark: the text recipes on Odyssey Book I",
        subtitle=(
            "`benchmark_text_recipes_odyssey.py`: 20 questions with written answers "
            "(13 specific, 7 broad), gpt-4o answers and judges, k=4"
        ),
        header=(
            "Variant",
            "Correct",
            "Specific",
            "Broad",
            "Recall",
            "Faithful",
            "Latency",
            "Tokens/q",
            "Index cost",
        ),
        widths=(2.9, 1.1, 1.1, 1.0, 1.0, 1.1, 1.1, 1.1, 1.9),
        font_size=11.5,
        rows=(
            (
                "VectorRag vector",
                "0.651",
                "0.679",
                "0.601",
                "0.657",
                "0.947",
                "1.9s",
                "713",
                "embeddings",
            ),
            (
                "VectorRag hybrid",
                "0.645",
                "0.702",
                "0.540",
                "0.608",
                "0.947",
                "2.1s",
                "717",
                "embeddings",
            ),
            (
                "ParentChildRag",
                "0.785",
                "0.793",
                "0.769",
                "0.806",
                "0.978",
                "2.0s",
                "2.2k",
                "embeddings",
            ),
            (
                "ParentChildRag + step-back",
                "0.749",
                "0.792",
                "0.669",
                "0.915",
                "0.988",
                "3.0s",
                "2.4k",
                "embeddings",
            ),
            (
                "HypotheticalQuestionRag",
                "0.733",
                "0.747",
                "0.707",
                "0.871",
                "1.000",
                "2.2s",
                "2.2k",
                "11 LLM calls",
            ),
            (
                "GraphRag local",
                "0.616",
                "0.622",
                "0.606",
                "0.821",
                "0.920",
                "4.7s",
                "2.3k",
                "~80 LLM calls",
            ),
            (
                "GraphRag global",
                "0.475",
                "0.442",
                "0.536",
                "0.323",
                "0.628",
                "11.3s",
                "7.5k",
                "~80 LLM calls",
            ),
        ),
        caption=(
            "One run, 20 questions: gaps under ~0.05 are noise. VectorRag gets 4×500-char chunks, "
            "the parent-based recipes 4×2000, so context size is part of their lead."
        ),
    ),
    BulletSlide(
        title="What the benchmark says",
        subtitle="Findings on one book; broad questions favour GraphRag more as the corpus grows",
        bullets=(
            "ParentChildRag is the best default: highest correctness overall and on both question types, at vector-search cost",
            "Step-back raises context recall (0.92) but costs one LLM call and lowers broad-question correctness",
            "  The broader rewrite pulls in more of the right facts, but the answer then spreads over them",
            "Hypothetical questions: perfect faithfulness and high recall, a close second, for one LLM call per chunk at index time",
            "Hybrid didn't beat plain vector here (0.645 vs 0.651): keyword matches added nothing on this corpus",
            "GraphRag lost on a single book. Four 2000-char parents already cover ~37% of it",
            "  Global answers were long and drew on the model's own Odyssey knowledge (0.63 faithful)",
            "  Its advantage needs a corpus too big for any top-k of chunks to summarize",
        ),
    ),
    TableSlide(
        title="Choosing a recipe",
        subtitle="Pick by the shape of your data first, then by cost and quality",
        header=("Your data", "Go-to recipe", "Why"),
        widths=(2.6, 3.2, 6.5),
        font_size=12.5,
        rows=(
            (
                "Unstructured text (PDF, documents)",
                "ParentChildRag",
                "Won the benchmark (0.785 correct) at 2s and 2 LLM calls; indexing only embeds",
            ),
            (
                "Text with predictable questions",
                "HypotheticalQuestionRag",
                "A `purpose` aims the index at what users ask; 1.000 faithful, 0.871 recall",
            ),
            (
                "Large corpus, whole-corpus questions",
                "GraphRag global_search",
                "No top-k of chunks can summarize it; budget ~80 LLM calls per book to index",
            ),
            (
                "Already-structured graph",
                "Text2CypherQA",
                "Exact counts, filters and aggregations, which vector search can't do",
            ),
            (
                "Several sources, open questions",
                "AgenticRag over tools",
                "Routes to text2cypher or a RetrieverTool, handles off-topic; 0.88 correct on movies",
            ),
        ),
        caption=(
            "StructuredGraphExtraction builds a graph for Text2CypherQA or AgenticRag to query; "
            "RagEvaluation scores any of them."
        ),
    ),
    TableSlide(
        title="Each recipe, one by one",
        subtitle="When to reach for it, and what it costs",
        header=("Recipe", "Use when", "Watch out for"),
        widths=(2.8, 5.0, 4.5),
        font_size=11.5,
        rows=(
            (
                "VectorRag",
                "A baseline or prototype; hybrid when questions hold exact codes or rare terms",
                "Small chunks lose context: 0.651 correct, the lowest of the chunk recipes",
            ),
            (
                "ParentChildRag",
                "The default for questions over long documents",
                "`step_back=True` adds recall but an LLM call; worth it for narrow, oddly worded questions",
            ),
            (
                "HypotheticalQuestionRag",
                "Users' questions follow a known pattern (FAQ, study, support); state it as `purpose`",
                "One LLM call per chunk to index; facts the questions skip can't be found",
            ),
            (
                "GraphRag",
                "Broad questions over a large corpus (global); entity-centred ones (local)",
                "~80 LLM calls per book to index; on one book it lost to chunk retrieval on every metric",
            ),
            (
                "Text2CypherQA",
                "Factual or aggregate questions over a graph you already have",
                "Brittle alone: bad Cypher returns empty rows; no small talk or off-topic handling",
            ),
            (
                "AgenticRag",
                "The front door over several tools, in production",
                "Several LLM calls per question; `decompose=True` only for multi-part questions",
            ),
        ),
    ),
    CodeSlide(
        title="Kuzu under the hood",
        subtitle="The Kuzu-specific Cypher, all kept inside the `Kuzu*` classes",
        bullets=(
            (
                "Tables must exist before data: every store and importer has a `create_schema()` or "
                "creates its tables on open"
            ),
            (
                "Vector index: HNSW graph with a sparse upper layer for long hops and a dense lower "
                "layer; fast, approximate"
            ),
            "FTS index: an inverted index scored with BM25",
            "Schema introspection replaces APOC: three catalog calls",
            (
                "Louvain runs from the statically linked `algo` extension, so no INSTALL or LOAD is "
                "needed"
            ),
        ),
        code_size=10.5,
        code="""
-- KuzuGraphStore
CALL CREATE_VECTOR_INDEX('Chunk', 'chunk_vector_idx',
    'embedding', metric := 'cosine');
CALL QUERY_VECTOR_INDEX('Chunk', 'chunk_vector_idx',
    $embedding, $k)
RETURN node.text, distance;
CALL CREATE_FTS_INDEX('Chunk', 'chunk_fts_idx', ['text']);
CALL QUERY_FTS_INDEX('Chunk', 'chunk_fts_idx', $query)
RETURN node.text, score;

-- KuzuCypherGraphStore.get_schema()
CALL show_tables() RETURN name, type;
CALL table_info('Movie') RETURN name, type;
CALL show_connection('DIRECTED') RETURN *;

-- KuzuLouvainCommunityDetector
CALL project_graph('entity_graph', ['Entity'], ['RELATIONSHIP']);
CALL louvain('entity_graph') WITH node, louvain_id
SET node.community_id = louvain_id;
""",
    ),
    CodeSlide(
        title="Testing and tooling",
        subtitle="Fast and offline: no test calls OpenAI",
        bullets=(
            "`tests/` mirrors `src/graphrag/` package by package",
            (
                "ScriptedLLMClient returns queued responses in order and records every call, so a "
                "test can check the prompts too"
            ),
            (
                "FakeEmbedder looks vectors up in a dict (3 dimensions, small enough to write by "
                "hand)"
            ),
            "Graph tests run against a real Kuzu database in `tmp_path`",
            (
                "ruff for lint and format (line length 100, `template/` excluded); mypy over "
                "`graphrag`, which ships `py.typed`"
            ),
            "`tests/docs/test_repo_feature_deck.py` keeps this deck in sync with the code",
            "`tests/observability/` renders traces into a buffer and checks the exact lines",
        ),
        code="""
# tests/agent/test_router.py: queue the LLM's
# tool choice, then check what the router does
def test_route_rejects_unknown_tool(llm):
    llm.tool_calls = [
        [ToolCall("answer_given", {"answer": "x"})]]

    with pytest.raises(ToolRoutingError,
                       match="answer_given"):
        ToolRouter(llm, [AnswerGivenTool()]).route("?")

# run everything
.venv/Scripts/python -m pytest
.venv/Scripts/ruff check .
.venv/Scripts/ruff format --check .
.venv/Scripts/mypy
""",
    ),
    CodeSlide(
        title="Development workflow",
        subtitle="Setup, the change process, and keeping this deck in sync",
        bullets=(
            "Every change starts with a plan, approved before any code is written (CLAUDE.md)",
            (
                "Dependencies: edit `requirements.in`, then `pip-compile`, then `pip-sync`. Never "
                "hand-edit `requirements.txt`"
            ),
            "`pip-sync` removes the editable `graphrag` install, so reinstall it afterwards",
            (
                "A change ships with its tests, its `examples/README.md` section, and an update to "
                "this deck"
            ),
            (
                "This deck is generated: edit `SLIDES` in `docs/repo-feature/build_deck.py`, then "
                "rebuild"
            ),
            (
                "The sync test fails when a public name or example script is missing from the deck, "
                "or when the deck names a class that no longer exists"
            ),
        ),
        code_size=10.5,
        code="""
# one-time setup (Python 3.11)
python -m venv .venv
.venv/Scripts/pip install pip-tools
.venv/Scripts/pip-sync requirements.txt
.venv/Scripts/pip install -e . --no-deps
cp .env.example .env        # set OPENAI_API_KEY

# adding a dependency
#   edit requirements.in, then:
.venv/Scripts/pip-compile requirements.in
.venv/Scripts/pip-sync requirements.txt
.venv/Scripts/pip install -e . --no-deps

# rebuild this deck
.venv/Scripts/python docs/repo-feature/build_deck.py
""",
    ),
)


def build_deck(output: Path = DEFAULT_OUTPUT, slides: Sequence[SlideSpec] = SLIDES) -> Path:
    builder = DeckBuilder()
    builder.add_all(slides)
    return builder.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the .pptx (default: {DEFAULT_OUTPUT.name} next to this script).",
    )
    args = parser.parse_args()
    path = build_deck(args.output)
    print(f"Wrote {len(SLIDES)} slides to {path}")


if __name__ == "__main__":
    main()
