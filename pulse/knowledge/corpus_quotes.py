"""Is each quote of a paper's file in the paper?

Every number in ``corpus/`` carries the words it was read from (``location.quote``).
The papers themselves are not in the repository, so the gate cannot check a quote
unless it is handed the paper: ``python -m pulse.admit <file> --paper <full text>``.
The full text is the publisher's or PubMed Central's XML or HTML, or plain text.

A quote is

* ``found`` when it is in the paper word for word, after both are stripped of markup
  and of typography (minus signs, plus-minus signs, kinds of space, case);
* ``numbers`` when it is not, but every number in it occurs within one short stretch
  of the paper: a table row quoted as its label and its cells, which no markup
  stripper returns verbatim;
* ``missing`` otherwise. A number that occurs nowhere in the paper is named.

A number is matched whole: 7.8 is not in a paper that prints only 37.8. A hyphen
before a number in the paper may be a minus sign or the dash of a range ("0-30
min"), so the paper is taken to print the number both with and without it.

``numbers`` is weaker than ``found``: it shows that the paper prints those numbers
together, not that the reader took them from the right row. That is the second
reader's work. What this check catches is the number nobody printed.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

FOUND, NUMBERS, MISSING = "found", "numbers", "missing"

_NEAR = 400  # characters either side of a quote's first number that must hold its others
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
_CELL_END = re.compile(r"</(td|th|tr|p|title|caption|label)>")
_TAG = re.compile(r"<[^>]+>")
_TYPOGRAPHY = (
    ("−", "-"), ("–", "-"), ("—", "-"), ("‐", "-"), ("‑", "-"),
    ("±", "+/-"), ("·", "."),
)


@dataclass(frozen=True)
class QuoteVerdict:
    label: str    # the finding's id, or what a refusal reports
    verdict: str  # FOUND | NUMBERS | MISSING
    reason: str = ""


def normalise(text: str) -> str:
    """``text`` as quotes are compared: entities decoded, compatibility forms folded
    (so every kind of space is a space), dashes and signs made plain, lower case."""
    text = unicodedata.normalize("NFKC", html.unescape(text))
    for a, b in _TYPOGRAPHY:
        text = text.replace(a, b)
    return re.sub(r"\s+", " ", text).strip().lower()


def paper_text(raw: str) -> str:
    """The comparable text of a paper given as XML, HTML or plain text. A cell, a row
    and a paragraph end in a space, so that neighbouring cells do not run together."""
    return normalise(_TAG.sub("", _CELL_END.sub(" ", raw)))


def read_paper(path: Path) -> str:
    return paper_text(Path(path).read_text(encoding="utf-8", errors="replace"))


def _printed(text: str) -> set[str]:
    """The numbers ``text`` prints, each also without a leading hyphen."""
    found = set(_NUMBER.findall(text))
    return found | {n.lstrip("-") for n in found}


def quote_verdict(quote: str, paper: str) -> tuple[str, str]:
    """``(verdict, reason)`` for one quote against ``paper`` (from ``paper_text``)."""
    q = normalise(quote)
    if q and q in paper:
        return FOUND, ""
    numbers = _NUMBER.findall(q)
    if not numbers:
        return MISSING, "not in the paper word for word, and it holds no number to look for"
    absent = sorted(set(numbers) - _printed(paper))
    if absent:
        return MISSING, f"the paper nowhere prints {', '.join(absent)}"
    first = re.compile(r"(?<![\w.])" + re.escape(numbers[0].lstrip("-")) + r"(?![\d]|\.\d)")
    for m in first.finditer(paper):
        near = _printed(paper[max(0, m.start() - _NEAR): m.start() + _NEAR])
        if all(n in near for n in numbers):
            return NUMBERS, ""
    return MISSING, "the paper prints its numbers, but nowhere together"


def check_quotes(quotes: list[tuple[str, str]], paper: str) -> list[QuoteVerdict]:
    """A verdict for each ``(label, quote)``."""
    return [QuoteVerdict(label, *quote_verdict(quote, paper)) for label, quote in quotes]
