## Overview

Given a mathematical statement as raw text, a Python docstring, or a PDF, the tool attempts to:

- Identify the mathematical domain through weighted keyword scoring and pattern matching across seven specialized areas: Linear Algebra, Calculus, Discrete Mathematics, Number Theory, Topology, Real Analysis, and Abstract Algebra.
- Parse the statement symbolically via SymPy, detecting equations and inequalities and exposing their free variables.
- Run a Z3 satisfiability check on inequalities, reporting whether the constraint is satisfiable, unsatisfiable, or undetermined with a 5-second timeout to avoid hangs.
- Produce a structured proof outline tailored to the detected context, including quantifier-aware suggestions when universal (∀) or existential (∃) constructs are present.
- Optionally forward the statement to a locally running Ollama instance (e.g. `llama3`) for an additional natural-language analysis pass, with full response validation before the output is merged.

The tool ships in three modes a Streamlit web interface, a `ttkbootstrap` desktop GUI, and a CLI so it fits into research workflows, classroom environments, and automated pipelines equally well.

<img width="1331" height="817" alt="mathfront" src="https://github.com/user-attachments/assets/e57cce77-9570-48ae-97aa-b98af35bccea" />



## Installation

Python 3.10 or later is required. All dependencies are installable from PyPI:

```bash
pip install sympy z3-solver streamlit pypdf ttkbootstrap requests
```

The `pypdf` package replaces the deprecated `PyPDF2`, which the original codebase used. If you have `PyPDF2` in an existing environment, remove it first to avoid import conflicts.

For the optional Ollama integration, install Ollama separately from [ollama.com](https://ollama.com) and pull a model before use:

```bash
ollama pull llama3
```

## Running the tool

**Streamlit interface**

```bash
python math_proof_assistant.py --streamlit
```

Streamlit will open in your browser. Use the sidebar to select language, toggle Ollama, and choose between text input, `.py` file upload, and PDF upload.

**Desktop GUI**

```bash
python math_proof_assistant.py --gui
```

Opens the `ttkbootstrap` window. The Analyze button is disabled while a previous run is in progress, preventing thread collisions.

**Direct text input**

```bash
python math_proof_assistant.py -i "If x > 2 then x^2 > 4" -l en
```

**PDF source**

```bash
python math_proof_assistant.py -i theorem.pdf -t pdf -l pt-br -m report.md
```

**Python docstring extraction**

```bash
python math_proof_assistant.py -i proof_sketch.py -t docstring -j result.json
```

All three input modes respect the `--lang` flag (`en` or `pt-br`) and can write output to a JSON file (`-j`), a Markdown report (`-m`), or stdout when neither is specified.

## Language support

The translation is handled at the analysis layer rather than post-hoc, so the structural logic is identical in both languages; only the surface text differs. Passing `-l pt-br` (or selecting *Português Brasileiro* in the UI) switches the entire output.

## Context detection and false-positive reduction

Each domain has a table of weighted keywords, and a context is activated only when the pattern fires *or* the keyword score reaches a threshold of 3. Single word coincidences below the threshold are suppressed.

When two or more domains are simultaneously activated, the output includes an explicit disambiguation notice, since merging proof strategies from unrelated areas of mathematics generates noise rather than guidance.

## Output structure

- `parsed_expressions` — the SymPy string representation of any successfully parsed sub-expressions.
- `gaps` — conditions or premises identified as missing from the statement as written.
- `suggestions` — contextual notes on domain, detected structure (equation, inequality, implication, quantifier), and Z3 results.
- `proof_outline` — an ordered list of proof steps appropriate to the detected context.

A fifth key, `security_notices`, appears only when the sanitizer modified or truncated the input.

## limitations

Z3 handles linear arithmetic and simple polynomial inequalities well, but non-linear expressions that SymPy cannot reduce to a form Z3 accepts will return `None` from the translation layer and fall back to a symbolic-only analysis. 

If the local server is unreachable or times out after 30 seconds, analysis continues without the LLM pass and no error is surfaced to the user. The Ollama output is appended to `suggestions` but does not influence the symbolic or Z3 analysis paths.

The PDF extractor uses `pypdf`'s text layer and will produce poor results on scanned documents without an OCR pre-processing step. Heavily mathematical PDFs with LaTeX-rendered text may yield garbled Unicode depending on the font encoding used in the source file.
