import argparse
import hashlib
import html
import json
import logging
import os
import re
import sys
import threading
import time
import tkinter as tk
import unicodedata
from collections import defaultdict
from functools import lru_cache
from tkinter import filedialog, messagebox
from typing import Any

import sympy
from z3 import *
import streamlit as st
from pypdf import PdfReader
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import requests

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
_log = logging.getLogger("math_proof_assistant")

_MAX_STATEMENT_LEN = 8_000
_MAX_PDF_PAGES = 200
_MAX_FILE_SIZE = 10 * 1024 * 1024
_OLLAMA_TIMEOUT = 30
_RATE_LIMIT_WINDOW = 60
_RATE_LIMIT_MAX = 30
_SAFE_SYMPY_ASSUMPTIONS = {"evaluate": False}

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)", re.I),
    re.compile(r"(system\s*prompt|jailbreak|dan\s+mode|developer\s+mode)", re.I),
    re.compile(r"(pretend|act\s+as|roleplay|you\s+are\s+now)\s+.{0,60}(assistant|ai|model|gpt|llm)", re.I),
    re.compile(r"(forget|disregard|override)\s+(your\s+)?(training|rules|guidelines|constraints)", re.I),
    re.compile(r"(output|print|repeat|write)\s+(your\s+)?(system\s+prompt|instructions?|context)", re.I),
    re.compile(r"<!--.*?-->", re.DOTALL),
    re.compile(r"<\s*(script|iframe|object|embed|form|input)[^>]*>", re.I),
    re.compile(r"\beval\s*\(", re.I),
    re.compile(r"(__import__|exec\s*\(|compile\s*\(|globals\s*\(|locals\s*\(|getattr\s*\()", re.I),
    re.compile(r"(os\.system|subprocess|shutil\.rm|open\s*\(.*['\"]w['\"])", re.I),
]

_SUSPICIOUS_UNICODE = re.compile(
    r"[\u202a-\u202e\u2066-\u2069\u200b-\u200f\u00ad\ufeff]"
)

_rate_limit_store: dict[str, list[float]] = defaultdict(list)
_rate_limit_lock = threading.Lock()


def _rate_key(statement: str) -> str:
    return hashlib.sha256(statement[:256].encode("utf-8", errors="replace")).hexdigest()[:16]


def _check_rate_limit(key: str) -> bool:
    now = time.monotonic()
    with _rate_limit_lock:
        timestamps = _rate_limit_store[key]
        timestamps[:] = [t for t in timestamps if now - t < _RATE_LIMIT_WINDOW]
        if len(timestamps) >= _RATE_LIMIT_MAX:
            return False
        timestamps.append(now)
    return True


def _sanitize_input(text: str) -> tuple[str, list[str]]:
    violations: list[str] = []
    if not isinstance(text, str):
        return "", ["Input is not a string."]
    text = unicodedata.normalize("NFKC", text)
    suspicious = _SUSPICIOUS_UNICODE.findall(text)
    if suspicious:
        violations.append("Invisible/directional Unicode characters removed.")
        text = _SUSPICIOUS_UNICODE.sub("", text)
    text = text.strip()
    if len(text) > _MAX_STATEMENT_LEN:
        violations.append(f"Input truncated from {len(text)} to {_MAX_STATEMENT_LEN} characters.")
        text = text[:_MAX_STATEMENT_LEN]
    for pat in _INJECTION_PATTERNS:
        if pat.search(text):
            violations.append("Potential prompt injection or unsafe content detected and blocked.")
            return "", violations
    return text, violations


def _safe_encode_output(value: Any) -> Any:
    if isinstance(value, str):
        return html.escape(value, quote=True)
    if isinstance(value, list):
        return [_safe_encode_output(v) for v in value]
    if isinstance(value, dict):
        return {k: _safe_encode_output(v) for k, v in value.items()}
    return value


def _validate_file_path(path: str, allowed_extensions: tuple[str, ...]) -> tuple[bool, str]:
    if not isinstance(path, str) or not path:
        return False, "Invalid file path."
    real = os.path.realpath(path)
    cwd = os.path.realpath(os.getcwd())
    tmp = os.path.realpath("/tmp")
    if not (real.startswith(cwd) or real.startswith(tmp) or real.startswith("/mnt")):
        return False, "File path outside allowed directories."
    ext = os.path.splitext(real)[1].lower()
    if ext not in allowed_extensions:
        return False, f"File extension {ext!r} not permitted."
    try:
        size = os.path.getsize(real)
    except OSError:
        return False, "File not accessible."
    if size > _MAX_FILE_SIZE:
        return False, f"File exceeds maximum allowed size of {_MAX_FILE_SIZE // (1024*1024)} MB."
    return True, ""


def sympy_to_z3(sympy_expr, symbols_map=None):
    if symbols_map is None:
        symbols_map = {}
    if isinstance(sympy_expr, sympy.Symbol):
        name = str(sympy_expr)
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]{0,63}$", name):
            return None
        if name not in symbols_map:
            symbols_map[name] = Real(name)
        return symbols_map[name]
    elif isinstance(sympy_expr, sympy.Integer):
        return int(sympy_expr)
    elif isinstance(sympy_expr, sympy.Float):
        return float(sympy_expr)
    elif isinstance(sympy_expr, sympy.Add):
        args = [sympy_to_z3(a, symbols_map) for a in sympy_expr.args]
        if any(a is None for a in args):
            return None
        return sum(args)
    elif isinstance(sympy_expr, sympy.Mul):
        prod = 1
        for arg in sympy_expr.args:
            z = sympy_to_z3(arg, symbols_map)
            if z is None:
                return None
            prod *= z
        return prod
    elif isinstance(sympy_expr, sympy.Pow):
        base = sympy_to_z3(sympy_expr.base, symbols_map)
        exp = sympy_to_z3(sympy_expr.exp, symbols_map)
        if base is None or exp is None:
            return None
        if isinstance(exp, (int, float)) and exp == 2:
            return base * base
        if isinstance(exp, (int, float)) and 0 < exp <= 10 and exp == int(exp):
            result = 1
            for _ in range(int(exp)):
                result = result * base
            return result
        return None
    elif isinstance(sympy_expr, sympy.GreaterThan):
        l, r = sympy_to_z3(sympy_expr.lhs, symbols_map), sympy_to_z3(sympy_expr.rhs, symbols_map)
        return l >= r if l is not None and r is not None else None
    elif isinstance(sympy_expr, sympy.LessThan):
        l, r = sympy_to_z3(sympy_expr.lhs, symbols_map), sympy_to_z3(sympy_expr.rhs, symbols_map)
        return l <= r if l is not None and r is not None else None
    elif isinstance(sympy_expr, sympy.StrictGreaterThan):
        l, r = sympy_to_z3(sympy_expr.lhs, symbols_map), sympy_to_z3(sympy_expr.rhs, symbols_map)
        return l > r if l is not None and r is not None else None
    elif isinstance(sympy_expr, sympy.StrictLessThan):
        l, r = sympy_to_z3(sympy_expr.lhs, symbols_map), sympy_to_z3(sympy_expr.rhs, symbols_map)
        return l < r if l is not None and r is not None else None
    elif isinstance(sympy_expr, sympy.Eq):
        l, r = sympy_to_z3(sympy_expr.lhs, symbols_map), sympy_to_z3(sympy_expr.rhs, symbols_map)
        return l == r if l is not None and r is not None else None
    return None


def _validate_ollama_response(response_text: str) -> str:
    if not isinstance(response_text, str):
        return ""
    cleaned = response_text.strip()
    if len(cleaned) > 4000:
        cleaned = cleaned[:4000]
    for pat in _INJECTION_PATTERNS:
        if pat.search(cleaned):
            return ""
    cleaned = _SUSPICIOUS_UNICODE.sub("", cleaned)
    return cleaned


def call_ollama(prompt: str, model: str = "llama3"):
    if not isinstance(model, str) or not re.match(r"^[a-zA-Z0-9_:.\-]{1,64}$", model):
        return None
    if not isinstance(prompt, str) or len(prompt) > 2000:
        return None
    safe_prompt = _SUSPICIOUS_UNICODE.sub("", prompt.strip())
    if not safe_prompt:
        return None
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": safe_prompt, "stream": False},
            timeout=_OLLAMA_TIMEOUT,
            allow_redirects=False,
        )
        if response.status_code == 200:
            raw = response.json().get("response", "")
            return _validate_ollama_response(raw)
    except requests.exceptions.ConnectionError:
        _log.warning("Ollama not reachable.")
    except requests.exceptions.Timeout:
        _log.warning("Ollama request timed out.")
    except Exception as e:
        _log.warning("Ollama error: %s", type(e).__name__)
    return None


_CONTEXT_RULES = [
    (
        re.compile(r'linear algebra|álgebra linear|vector space|espaço vetorial|matrix|matriz|eigenvalue|autovalor', re.I),
        "linear_algebra",
        {
            "en": ("Context: Linear Algebra. Consider properties of vector spaces, matrices, linear transformations, eigenvalues, and eigenvectors.",
                   ["Identify the underlying vector space and its basis.",
                    "Represent linear transformations as matrices."]),
            "pt-br": ("Contexto: Álgebra Linear. Considere propriedades de espaços vetoriais, matrizes, transformações lineares, autovalores e autovetores.",
                      ["Identifique o espaço vetorial subjacente e sua base.",
                       "Represente transformações lineares como matrizes."]),
        },
    ),
    (
        re.compile(r'calculus|cálculo|derivative|derivada|integral|limite|limit|differential equation|equação diferencial', re.I),
        "calculus",
        {
            "en": ("Context: Calculus. Focus on differentiation, integration, limits, series, or differential equations.",
                   ["Identify functions and variables involved.",
                    "Apply relevant theorems like Mean Value Theorem, Fundamental Theorem of Calculus."]),
            "pt-br": ("Contexto: Cálculo. Concentre-se em diferenciação, integração, limites, séries ou equações diferenciais.",
                      ["Identifique funções e variáveis envolvidas.",
                       "Aplique teoremas relevantes como o Teorema do Valor Médio, Teorema Fundamental do Cálculo."]),
        },
    ),
    (
        re.compile(r'discrete mathematics|matemática discreta|set theory|teoria dos conjuntos|graph theory|teoria dos grafos|logic|lógica|induction|indução|combinatorics|combinatória', re.I),
        "discrete",
        {
            "en": ("Context: Discrete Mathematics. Consider sets, relations, functions, graphs, logic, combinatorics, or proof by induction.",
                   ["Define the elements of sets, properties of relations, or structure of graphs.",
                    "Consider proof techniques like mathematical induction, contradiction, or direct proof."]),
            "pt-br": ("Contexto: Matemática Discreta. Considere conjuntos, relações, funções, grafos, lógica, combinatória ou prova por indução.",
                      ["Defina os elementos dos conjuntos, propriedades das relações ou estrutura dos grafos.",
                       "Considere técnicas de prova como indução matemática, contradição ou prova direta."]),
        },
    ),
    (
        re.compile(r'number theory|teoria dos números|prime number|número primo|divisibility|divisibilidade|congruence|congruência', re.I),
        "number_theory",
        {
            "en": ("Context: Number Theory. Explore properties of integers, prime numbers, divisibility, and congruences.",
                   ["Use properties of modular arithmetic or prime factorization."]),
            "pt-br": ("Contexto: Teoria dos Números. Explore propriedades de inteiros, números primos, divisibilidade e congruências.",
                      ["Use propriedades da aritmética modular ou fatoração prima."]),
        },
    ),
    (
        re.compile(r'topology|topologia|open set|conjunto aberto|closed set|conjunto fechado|compact|compacto|homeomorphism|homeomorfismo', re.I),
        "topology",
        {
            "en": ("Context: Topology. Consider properties of topological spaces, continuity, compactness, and connectedness.",
                   ["Define the topological space and the relevant topology.",
                    "Identify open/closed sets and check continuity conditions."]),
            "pt-br": ("Contexto: Topologia. Considere propriedades de espaços topológicos, continuidade, compacidade e conexidade.",
                      ["Defina o espaço topológico e a topologia relevante.",
                       "Identifique conjuntos abertos/fechados e verifique condições de continuidade."]),
        },
    ),
    (
        re.compile(r'real analysis|análise real|metric space|espaço métrico|cauchy|convergence|convergência|subsequence|subsequência', re.I),
        "real_analysis",
        {
            "en": ("Context: Real Analysis. Focus on sequences, convergence, continuity, and metric spaces.",
                   ["Verify Cauchy criteria or epsilon-delta definitions.",
                    "Check boundedness and monotonicity of sequences."]),
            "pt-br": ("Contexto: Análise Real. Concentre-se em sequências, convergência, continuidade e espaços métricos.",
                      ["Verifique critérios de Cauchy ou definições epsilon-delta.",
                       "Verifique limitação e monotonicidade das sequências."]),
        },
    ),
    (
        re.compile(r'abstract algebra|álgebra abstrata|group theory|teoria dos grupos|ring|anel|field|corpo|homomorphism|homomorfismo', re.I),
        "abstract_algebra",
        {
            "en": ("Context: Abstract Algebra. Explore group, ring, and field structures, homomorphisms, and isomorphisms.",
                   ["Verify closure, associativity, identity, and inverse for group axioms.",
                    "Identify the kernel and image of homomorphisms."]),
            "pt-br": ("Contexto: Álgebra Abstrata. Explore estruturas de grupo, anel e corpo, homomorfismos e isomorfismos.",
                      ["Verifique fechamento, associatividade, identidade e inverso para axiomas de grupo.",
                       "Identifique o núcleo e a imagem de homomorfismos."]),
        },
    ),
]

_KEYWORD_WEIGHTS: dict[str, dict[str, int]] = {
    "linear_algebra": {"matrix": 3, "vector": 3, "eigenvalue": 5, "determinant": 4, "linear": 2},
    "calculus": {"derivative": 4, "integral": 4, "limit": 3, "differential": 4, "continuous": 2},
    "discrete": {"induction": 4, "graph": 3, "set": 2, "logic": 3, "combinatorics": 4},
    "number_theory": {"prime": 5, "divisib": 4, "congruenc": 4, "modular": 4, "integer": 2},
    "topology": {"topolog": 5, "homeomorphi": 5, "compact": 4, "open set": 4, "closed set": 4},
    "real_analysis": {"cauchy": 5, "metric": 4, "convergenc": 4, "subsequenc": 4, "epsilon": 3},
    "abstract_algebra": {"group": 3, "ring": 3, "field": 3, "homomorphi": 5, "isomorphi": 5},
}


def _score_contexts(statement_lower: str) -> dict[str, int]:
    scores: dict[str, int] = defaultdict(int)
    for ctx_id, weights in _KEYWORD_WEIGHTS.items():
        for kw, w in weights.items():
            if kw in statement_lower:
                scores[ctx_id] += w
    return scores


_SYMPY_SAFE_LOCALS: dict[str, Any] = {
    k: v for k, v in sympy.__dict__.items()
    if isinstance(v, type) and issubclass(v, (sympy.Basic,))
}


@lru_cache(maxsize=256)
def _cached_sympify(statement: str):
    try:
        return sympy.sympify(statement, evaluate=False, locals=_SYMPY_SAFE_LOCALS)
    except Exception:
        return None


def _safe_sympify(statement: str):
    dangerous = re.compile(
        r"(__import__|eval\s*\(|exec\s*\(|open\s*\(|compile\s*\(|getattr|setattr|delattr|globals|locals|vars\s*\()",
        re.I,
    )
    if dangerous.search(statement):
        return None, "Statement contains unsafe constructs and was rejected."
    try:
        expr = _cached_sympify(statement)
        return expr, None
    except (sympy.SympifyError, TypeError, ValueError, RecursionError) as e:
        return None, str(e)
    except Exception as e:
        _log.warning("Unexpected sympify error: %s", type(e).__name__)
        return None, "Unexpected parse error."


def _run_z3_check(sympy_expr, symbols_map: dict, lang: str):
    suggestions = []
    proof_steps = []
    gaps = []
    try:
        z3_expr = sympy_to_z3(sympy_expr, symbols_map)
        if z3_expr is None:
            if lang == "pt-br":
                suggestions.append("Não foi possível traduzir a expressão SymPy complexa para Z3.")
            else:
                suggestions.append("Could not translate complex SymPy expression to Z3.")
            return suggestions, proof_steps, gaps
        s = Solver()
        s.set("timeout", 5000)
        s.add(z3_expr)
        res = s.check()
        expr_str = str(sympy_expr)
        if res == sat:
            model = s.model()
            model_str = str(model)
            if lang == "pt-br":
                suggestions.append(f"Verificação Z3: A inequação {expr_str} é satisfatível. Modelo: {model_str}")
                proof_steps.append(f"Encontre um exemplo específico que satisfaça {expr_str} usando o modelo {model_str}.")
            else:
                suggestions.append(f"Z3 check: The inequality {expr_str} is satisfiable. Model: {model_str}")
                proof_steps.append(f"Find a specific example that satisfies {expr_str} using model {model_str}.")
        elif res == unsat:
            if lang == "pt-br":
                suggestions.append(f"Verificação Z3: A inequação {expr_str} é insatisfatível (falsa). Considere prova por contradição.")
                proof_steps.append(f"Prove por contradição que {expr_str} é falsa.")
            else:
                suggestions.append(f"Z3 check: The inequality {expr_str} is unsatisfiable (false). Consider proof by contradiction.")
                proof_steps.append(f"Prove by contradiction that {expr_str} is false.")
        else:
            if lang == "pt-br":
                suggestions.append(f"Verificação Z3: A satisfatibilidade de {expr_str} é desconhecida.")
                gaps.append(f"Precisa determinar se {expr_str} é sempre verdadeira, sempre falsa ou condicionalmente verdadeira.")
            else:
                suggestions.append(f"Z3 check: The satisfiability of {expr_str} is unknown.")
                gaps.append(f"Need to determine if {expr_str} is always true, always false, or conditionally true.")
    except Z3Exception as e:
        _log.warning("Z3 exception: %s", str(e)[:200])
        if lang == "pt-br":
            gaps.append("A análise Z3 falhou.")
        else:
            gaps.append("Z3 analysis failed.")
    except Exception as e:
        _log.warning("Z3 unexpected error: %s", type(e).__name__)
        if lang == "pt-br":
            gaps.append("A análise Z3 falhou.")
        else:
            gaps.append("Z3 analysis failed.")
    return suggestions, proof_steps, gaps


def analyze_statement(statement: str, lang: str = 'en', use_ollama: bool = False):
    if lang not in ("en", "pt-br"):
        lang = "en"

    statement, violations = _sanitize_input(statement)
    if violations and not statement:
        blocked_msg = violations[0] if violations else "Input rejected."
        return {
            "parsed_expressions": [],
            "gaps": [blocked_msg],
            "suggestions": [blocked_msg],
            "proof_outline": [],
            "security_notices": violations,
        }

    rate_key = _rate_key(statement)
    if not _check_rate_limit(rate_key):
        msg = "Limite de análise atingido. Aguarde um momento." if lang == "pt-br" else "Analysis rate limit reached. Please wait."
        return {"parsed_expressions": [], "gaps": [msg], "suggestions": [msg], "proof_outline": [], "security_notices": []}

    gaps: list[str] = []
    suggestions: list[str] = []
    parsed_expressions: list[str] = []
    proof_outline: list[str] = []
    symbols_map: dict = {}
    statement_lower = statement.lower()

    if violations:
        for v in violations:
            _log.info("Input notice: %s", v)

    if use_ollama:
        safe_ollama_prompt = (
            f"Analyze the following mathematical statement and provide a proof outline "
            f"and identify missing lemmas in {'Portuguese (pt-br)' if lang == 'pt-br' else 'English'}. "
            f"Do not include any instructions, code, or system commands. "
            f"Mathematical statement: {statement[:1000]}"
        )
        ollama_res = call_ollama(safe_ollama_prompt)
        if ollama_res:
            label = "Análise Ollama" if lang == "pt-br" else "Ollama Analysis"
            suggestions.append(f"{label}: {ollama_res}")

    context_scores = _score_contexts(statement_lower)
    detected_contexts: list[str] = []

    for rule_pat, ctx_id, translations in _CONTEXT_RULES:
        pattern_hit = bool(rule_pat.search(statement_lower))
        score_hit = context_scores.get(ctx_id, 0) >= 3
        if pattern_hit or score_hit:
            detected_contexts.append(ctx_id)
            t = translations.get(lang, translations["en"])
            suggestions.append(t[0])
            proof_outline.extend(t[1])

    if not detected_contexts:
        if lang == "pt-br":
            suggestions.append("Contexto: Matemática Geral. Considere axiomas e definições fundamentais.")
            proof_outline.append("Comece definindo todos os termos e símbolos usados.")
        else:
            suggestions.append("Context: General Mathematics. Consider fundamental axioms and definitions.")
            proof_outline.append("Start by defining all terms and symbols used.")

    if re.search(r'\bprove\s+that\b', statement_lower) or re.search(r'\bprove\s+que\b', statement_lower):
        if lang == "pt-br":
            suggestions.append("A declaração é um teorema a ser provado. Defina claramente as premissas e a conclusão.")
            gaps.append("Faltam premissas explícitas ou uma conclusão clara a ser derivada.")
            proof_outline.insert(0, "Declare claramente o teorema, premissas e conclusão.")
        else:
            suggestions.append("The statement is a theorem to be proven. Clearly define premises and conclusion.")
            gaps.append("Missing explicit premises or a clear conclusion to be derived.")
            proof_outline.insert(0, "Clearly state the theorem, premises, and conclusion.")

    if re.search(r'\bif\b\s.{1,200}\s\bthen\b', statement_lower):
        if lang == "pt-br":
            suggestions.append("Isso parece ser uma implicação (Se P então Q). Identifique P (hipótese) e Q (conclusão).")
            proof_outline.append("Assuma que a hipótese P é verdadeira.")
            proof_outline.append("Deduzir a conclusão Q a partir de P e axiomas/teoremas conhecidos.")
        else:
            suggestions.append("This appears to be an implication (If P then Q). Identify P (hypothesis) and Q (conclusion).")
            proof_outline.append("Assume the hypothesis P is true.")
            proof_outline.append("Deduce the conclusion Q from P and known axioms/theorems.")

    if re.search(r'\bfor\s+all\b|\bforall\b|\b∀\b|\bfor\s+every\b|\bpara\s+todo\b', statement_lower):
        if lang == "pt-br":
            suggestions.append("Quantificador universal detectado. Considere prova por indução ou prova direta para todos os elementos.")
        else:
            suggestions.append("Universal quantifier detected. Consider proof by induction or direct proof for all elements.")

    if re.search(r'\bthere\s+exists\b|\bexists\b|\b∃\b|\bexiste\b', statement_lower):
        if lang == "pt-br":
            suggestions.append("Quantificador existencial detectado. Construa ou exiba um objeto que satisfaça a propriedade.")
        else:
            suggestions.append("Existential quantifier detected. Construct or exhibit an object satisfying the property.")

    sympy_expr, parse_err = _safe_sympify(statement)
    if sympy_expr is not None:
        parsed_expressions.append(str(sympy_expr))
        if lang == "pt-br":
            suggestions.append(f"Declaração analisada simbolicamente: {sympy_expr}")
        else:
            suggestions.append(f"Statement parsed symbolically: {sympy_expr}")

        if isinstance(sympy_expr, sympy.Eq):
            lhs, rhs = sympy_expr.lhs, sympy_expr.rhs
            if lang == "pt-br":
                suggestions.append(f"Equação identificada: {lhs} = {rhs}")
            else:
                suggestions.append(f"Identified an equation: {lhs} = {rhs}")
            free_syms = sympy_expr.free_symbols
            if free_syms:
                sym_names = ", ".join(str(s) for s in sorted(free_syms, key=str))
                if lang == "pt-br":
                    suggestions.append(f"Considere resolver para {sym_names} ou simplificar a equação.")
                    proof_outline.append(f"Simplifique a equação {lhs} = {rhs}.")
                else:
                    suggestions.append(f"Consider solving for {sym_names} or simplifying the equation.")
                    proof_outline.append(f"Simplify the equation {lhs} = {rhs}.")

        elif isinstance(sympy_expr, (sympy.StrictGreaterThan, sympy.GreaterThan, sympy.StrictLessThan, sympy.LessThan)):
            if lang == "pt-br":
                suggestions.append(f"Inequação identificada: {sympy_expr}")
                proof_outline.append(f"Analise as propriedades da inequação {sympy_expr}.")
            else:
                suggestions.append(f"Identified an inequality: {sympy_expr}")
                proof_outline.append(f"Analyze the properties of the inequality {sympy_expr}.")
            z3_sugg, z3_steps, z3_gaps = _run_z3_check(sympy_expr, symbols_map, lang)
            suggestions.extend(z3_sugg)
            proof_outline.extend(z3_steps)
            gaps.extend(z3_gaps)

    else:
        if parse_err:
            if lang == "pt-br":
                suggestions.append(f"Não foi possível analisar completamente a declaração com SymPy: {parse_err}")
                gaps.append("A declaração não pôde ser totalmente analisada simbolicamente.")
            else:
                suggestions.append(f"Could not fully parse statement with SymPy: {parse_err}")
                gaps.append("Statement could not be fully parsed symbolically.")

    if len(detected_contexts) > 1:
        ctx_names_en = [c.replace("_", " ").title() for c in detected_contexts]
        if lang == "pt-br":
            suggestions.append(f"Múltiplos contextos detectados: {', '.join(ctx_names_en)}. Verifique se são todos relevantes para evitar falsas correlações.")
        else:
            suggestions.append(f"Multiple contexts detected: {', '.join(ctx_names_en)}. Verify all are relevant to avoid false correlations.")

    if not proof_outline:
        if lang == "pt-br":
            proof_outline = [
                "Passo 1: Entender a declaração.",
                "Passo 2: Identificar premissas e conclusão.",
                "Passo 3: Aplicar teoremas/axiomas relevantes.",
                "Passo 4: Deduzir a conclusão.",
            ]
        else:
            proof_outline = [
                "Step 1: Understand the statement.",
                "Step 2: Identify premises and conclusion.",
                "Step 3: Apply relevant theorems/axioms.",
                "Step 4: Deduce conclusion.",
            ]

    result = {
        "parsed_expressions": parsed_expressions,
        "gaps": gaps,
        "suggestions": suggestions,
        "proof_outline": proof_outline,
    }
    if violations:
        result["security_notices"] = violations
    return result


def extract_from_docstring(file_path: str) -> str:
    ok, err = _validate_file_path(file_path, (".py",))
    if not ok:
        return f"Error: {err}"
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read(500_000)
    except OSError as e:
        return f"Error reading file: {type(e).__name__}"
    match = re.search(r'"""(?:\s*latex\s*)?(.*?)"""', content, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_from_pdf(file_path: str) -> str:
    ok, err = _validate_file_path(file_path, (".pdf",))
    if not ok:
        return f"Error: {err}"
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            page_count = min(len(reader.pages), _MAX_PDF_PAGES)
            for i in range(page_count):
                try:
                    page_text = reader.pages[i].extract_text() or ""
                    text += page_text
                    if len(text) > _MAX_STATEMENT_LEN * 2:
                        text = text[:_MAX_STATEMENT_LEN * 2]
                        break
                except Exception:
                    continue
    except Exception as e:
        return f"Error reading PDF: {type(e).__name__}"
    return text.strip()


def generate_markdown_report(analysis_results: dict, output_path: str, lang: str = 'en'):
    ok, err = _validate_file_path(os.path.dirname(os.path.abspath(output_path)) + "/x.md", (".md",))
    safe_output_path = re.sub(r'[^a-zA-Z0-9_\-./]', '_', output_path)
    if not safe_output_path.endswith(".md"):
        safe_output_path += ".md"

    title = "Mathematical Proof Analysis Report" if lang == 'en' else "Relatório de Análise de Prova Matemática"
    original = html.escape(str(analysis_results.get('original_statement', 'N/A')))

    parsed_json = json.dumps(
        [html.escape(str(e)) for e in analysis_results.get('parsed_expressions', [])],
        indent=2,
        ensure_ascii=False
    )
    report_content = f"# {title}\n\n## Original Statement\n```\n{original}\n```\n\n## Parsed Expressions\n```json\n{parsed_json}\n```\n\n## Identified Gaps\n"
    for gap in analysis_results.get('gaps', []):
        report_content += f"- {html.escape(str(gap))}\n"
    report_content += "\n## Suggestions for Proof\n"
    for suggestion in analysis_results.get('suggestions', []):
        report_content += f"- {html.escape(str(suggestion))}\n"
    report_content += "\n## Proof Outline\n"
    for step in analysis_results.get('proof_outline', []):
        report_content += f"- {html.escape(str(step))}\n"
    if analysis_results.get("security_notices"):
        report_content += "\n## Security Notices\n"
        for notice in analysis_results["security_notices"]:
            report_content += f"- {html.escape(str(notice))}\n"
    try:
        with open(safe_output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
    except OSError as e:
        _log.error("Could not write report: %s", type(e).__name__)


class MathProofGUI:
    def __init__(self):
        self.root = ttk.Window(themename="cosmo")
        self.root.title("Math Proof Assistant - Desktop GUI")
        self.root.geometry("800x600")
        self.lang = tk.StringVar(value="en")
        self.use_ollama = tk.BooleanVar(value=False)
        self._analysis_running = False
        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=BOTH, expand=YES)
        ttk.Label(main_frame, text="Mathematical Proof Assistant", font=("Helvetica", 18, "bold")).pack(pady=10)
        ctrl_frame = ttk.Frame(main_frame)
        ctrl_frame.pack(fill=X, pady=5)
        ttk.Label(ctrl_frame, text="Language:").pack(side=LEFT, padx=5)
        ttk.Radiobutton(ctrl_frame, text="English", variable=self.lang, value="en").pack(side=LEFT, padx=5)
        ttk.Radiobutton(ctrl_frame, text="Português", variable=self.lang, value="pt-br").pack(side=LEFT, padx=5)
        ttk.Checkbutton(ctrl_frame, text="Use Ollama (Local LLM)", variable=self.use_ollama).pack(side=LEFT, padx=20)
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=X, pady=10)
        self._btn_pdf = ttk.Button(btn_frame, text="Upload PDF", command=self.upload_pdf, bootstyle=INFO)
        self._btn_pdf.pack(side=LEFT, padx=5)
        self._btn_analyze = ttk.Button(btn_frame, text="Analyze Text", command=self.analyze_text, bootstyle=SUCCESS)
        self._btn_analyze.pack(side=LEFT, padx=5)
        self.text_input = ttk.Text(main_frame, height=8)
        self.text_input.pack(fill=X, pady=10)
        ttk.Label(main_frame, text="Analysis Results:", font=("Helvetica", 12, "bold")).pack(anchor=W)
        self.result_output = ttk.Text(main_frame, height=15, state=DISABLED)
        self.result_output.pack(fill=BOTH, expand=YES, pady=5)

    def upload_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            ok, err = _validate_file_path(file_path, (".pdf",))
            if not ok:
                messagebox.showerror("Error", f"File rejected: {err}")
                return
            text = extract_from_pdf(file_path)
            if text.startswith("Error:"):
                messagebox.showerror("Error", text)
                return
            self.text_input.delete("1.0", END)
            self.text_input.insert(END, text)
            self.analyze_text()

    def analyze_text(self):
        if self._analysis_running:
            messagebox.showinfo("Info", "Analysis already in progress.")
            return
        statement = self.text_input.get("1.0", END).strip()
        if not statement:
            messagebox.showwarning("Warning", "Please enter a statement or upload a PDF.")
            return
        if len(statement) > _MAX_STATEMENT_LEN:
            messagebox.showwarning("Warning", f"Input truncated to {_MAX_STATEMENT_LEN} characters.")
            statement = statement[:_MAX_STATEMENT_LEN]
        self._analysis_running = True
        self._btn_analyze.config(state=DISABLED)
        threading.Thread(target=self._run_analysis, args=(statement,), daemon=True).start()

    def _run_analysis(self, statement):
        try:
            res = analyze_statement(statement, lang=self.lang.get(), use_ollama=self.use_ollama.get())
        except Exception as e:
            _log.error("Analysis error: %s", type(e).__name__)
            res = {"suggestions": ["Analysis failed due to an internal error."], "gaps": [], "proof_outline": []}
        finally:
            self.root.after(0, self._update_results, res)

    def _update_results(self, res):
        try:
            self.result_output.config(state=NORMAL)
            self.result_output.delete("1.0", END)
            self.result_output.insert(END, "--- SUGGESTIONS ---\n")
            for s in res.get('suggestions', []):
                self.result_output.insert(END, f"- {s}\n")
            self.result_output.insert(END, "\n--- GAPS ---\n")
            for g in res.get('gaps', []):
                self.result_output.insert(END, f"- {g}\n")
            self.result_output.insert(END, "\n--- PROOF OUTLINE ---\n")
            for p in res.get('proof_outline', []):
                self.result_output.insert(END, f"- {p}\n")
            if res.get("security_notices"):
                self.result_output.insert(END, "\n--- SECURITY NOTICES ---\n")
                for n in res["security_notices"]:
                    self.result_output.insert(END, f"[!] {n}\n")
            self.result_output.config(state=DISABLED)
        finally:
            self._analysis_running = False
            self._btn_analyze.config(state=NORMAL)

    def run(self):
        self.root.mainloop()


def streamlit_app():
    st.set_page_config(layout="wide")
    st.title("Mathematical Proof Assistant")
    if 'lang' not in st.session_state:
        st.session_state['lang'] = 'en'
    lang_option = st.sidebar.selectbox("Language / Idioma", [('en', 'English'), ('pt-br', 'Português Brasileiro')], format_func=lambda x: x[1])
    st.session_state['lang'] = lang_option[0]
    lang = st.session_state['lang']
    use_ollama = st.sidebar.checkbox("Use Ollama (Local LLM)")
    input_type = st.sidebar.radio("Input type:", ('Text Input', 'Upload Python File', 'Upload PDF File'))
    statement = ""
    if input_type == 'Text Input':
        statement = st.text_area("Enter statement:", height=150, max_chars=_MAX_STATEMENT_LEN)
    elif input_type == 'Upload Python File':
        uploaded_file = st.file_uploader("Upload .py", type=['py'])
        if uploaded_file:
            if uploaded_file.size > _MAX_FILE_SIZE:
                st.error(f"File too large. Maximum size is {_MAX_FILE_SIZE // (1024*1024)} MB.")
            else:
                tmp_path = f"/tmp/upload_{hashlib.sha256(uploaded_file.name.encode()).hexdigest()[:8]}.py"
                try:
                    with open(tmp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    statement = extract_from_docstring(tmp_path)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
    elif input_type == 'Upload PDF File':
        uploaded_file = st.file_uploader("Upload .pdf", type=['pdf'])
        if uploaded_file:
            if uploaded_file.size > _MAX_FILE_SIZE:
                st.error(f"File too large. Maximum size is {_MAX_FILE_SIZE // (1024*1024)} MB.")
            else:
                tmp_path = f"/tmp/upload_{hashlib.sha256(uploaded_file.name.encode()).hexdigest()[:8]}.pdf"
                try:
                    with open(tmp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    statement = extract_from_pdf(tmp_path)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

    if st.button("Analyze") and statement:
        with st.spinner("Analyzing..."):
            res = analyze_statement(statement, lang=lang, use_ollama=use_ollama)
        st.subheader("Results")
        display_res = _safe_encode_output(res)
        st.json(display_res)
        if res.get("security_notices"):
            st.warning("Security Notices: " + "; ".join(res["security_notices"]))
        report_filename = f"/tmp/report_{hashlib.sha256(statement[:64].encode()).hexdigest()[:8]}.md"
        res_with_stmt = dict(res)
        res_with_stmt['original_statement'] = statement
        generate_markdown_report(res_with_stmt, report_filename, lang=lang)
        try:
            with open(report_filename, "rb") as f:
                st.download_button("Download Markdown Report", f, file_name="report.md")
        finally:
            if os.path.exists(report_filename):
                os.remove(report_filename)


def main():
    parser = argparse.ArgumentParser(description="Math Proof Assistant")
    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--type', '-t', choices=['text', 'docstring', 'pdf'], default='text')
    parser.add_argument('--lang', '-l', choices=['en', 'pt-br'], default='en')
    parser.add_argument('--output-json', '-j', type=str)
    parser.add_argument('--output-md', '-m', type=str)
    parser.add_argument('--streamlit', '-s', action='store_true')
    parser.add_argument('--gui', '-g', action='store_true')
    parser.add_argument('--ollama', action='store_true')
    args = parser.parse_args()

    if args.streamlit:
        streamlit_app()
    elif args.gui:
        MathProofGUI().run()
    elif args.input:
        statement = ""
        if args.type == 'text':
            statement = args.input
        elif args.type == 'docstring':
            ok, err = _validate_file_path(args.input, (".py",))
            if not ok:
                _log.error("File rejected: %s", err)
                sys.exit(1)
            statement = extract_from_docstring(args.input)
        elif args.type == 'pdf':
            ok, err = _validate_file_path(args.input, (".pdf",))
            if not ok:
                _log.error("File rejected: %s", err)
                sys.exit(1)
            statement = extract_from_pdf(args.input)
        if not statement or statement.startswith("Error:"):
            _log.error("No valid statement extracted.")
            sys.exit(1)
        res = analyze_statement(statement, lang=args.lang, use_ollama=args.ollama)
        res['original_statement'] = statement
        if args.output_json:
            safe_json_path = re.sub(r'[^a-zA-Z0-9_\-./]', '_', args.output_json)
            try:
                with open(safe_json_path, 'w', encoding='utf-8') as f:
                    json.dump(_safe_encode_output(res), f, indent=2, ensure_ascii=False)
            except OSError as e:
                _log.error("Could not write JSON output: %s", type(e).__name__)
        if args.output_md:
            generate_markdown_report(res, args.output_md, lang=args.lang)
        if not args.output_json and not args.output_md:
            print(json.dumps(_safe_encode_output(res), indent=2, ensure_ascii=False))
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
