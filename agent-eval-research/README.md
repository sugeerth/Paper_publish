# Measuring a Moving Target

A research brief on evaluating self-evolving AI agents: what self-evolution
breaks about conventional agent benchmarks, what evaluation infrastructure is
replacing them, and what a team shipping a self-improving agent should
instrument.

- `report.html` — the source document
- `report.pdf` — the rendered brief (11 pages)

## Rebuilding the PDF

```sh
make report     # from the repository root
```

Renders `report.html` through headless Chromium. There is no LaTeX dependency:
the brief is styled for print in CSS (`@page size: letter`), which is why it
builds in environments where the main paper's TeX toolchain is unavailable.

Rebuilding re-renders the document; it does not re-run the research.

## Sourcing caveat

This brief is **abstract-level evidence**. It was assembled from 21 structured
web searches on 17 September 2026. The session's network egress policy blocked
`arxiv.org`, `huggingface.co`, `alphaxiv.org`, `cacm.acm.org`, and every
secondary blog domain attempted — one of nine full-text fetches succeeded. Every
quantitative claim therefore comes from search-engine summaries of abstracts and
landing pages, not from verified full text.

Section 9 of the report states this in full and names the specific figures that
should be checked against primary sources before being cited or used to justify a
design decision. Treat the brief as a map of the literature and a starting set of
references, not as a citable secondary source.
