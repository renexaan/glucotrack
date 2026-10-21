"""Reusable interface pieces: shell, header, cards, tiles, callouts."""
from __future__ import annotations

import html
from contextlib import contextmanager

import pandas as pd
import streamlit as st

from . import theme

PAGES: dict = {}   # filled by streamlit_app.py so any page can link to another

MARK_SVG = (
    '<svg class="gt-mark" viewBox="0 0 34 34" role="img" aria-label="GlucoTrack logo">'
    '<rect width="34" height="34" rx="9" fill="#0E7C86"/>'
    '<path d="M17 6.5c-3.6 4.7-6.5 8-6.5 11.4a6.5 6.5 0 0 0 13 0C23.5 14.5 20.6 11.2 17 6.5z" fill="#fff" opacity=".95"/>'
    '<path d="M11.5 19.2h3l1.5-3 2.2 5.2 1.5-2.2h2.3" fill="none" stroke="#0E7C86" stroke-width="1.7" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)

CAT_LABELS = {
    "No Info": "No information", "never": "Never smoked", "former": "Former smoker",
    "current": "Current smoker", "not current": "Not current", "ever": "Ever smoked",
    "AfricanAmerican": "African American",
}


def label(value: str) -> str:
    return CAT_LABELS.get(value, value)


def esc(x) -> str:
    return html.escape(str(x))


def inject_css() -> None:
    st.markdown(f"<style>{theme.CSS}</style>", unsafe_allow_html=True)


def shell(pages: list, active=None) -> None:
    """Header (wordmark, descriptor, navigation) plus the disclaimer shown on every page."""
    inject_css()
    with st.container(key="gt-header"):
        brand, nav = st.columns([4, 9], vertical_alignment="center")
        with brand:
            st.markdown(
                f'<div class="gt-brand">{MARK_SVG}<div><div class="gt-wordmark">Gluco<span>Track</span></div>'
                f'<div class="gt-descriptor">Machine Learning Research</div></div></div>',
                unsafe_allow_html=True)
        with nav:
            with st.container(key="gt-nav", horizontal=True, horizontal_alignment="right", gap="xxsmall"):
                for page in pages:
                    # the wrapper key carries the active state so CSS never depends on Streamlit internals
                    state = "active" if active is not None and page.url_path == active.url_path else "idle"
                    with st.container(key=f"navitem-{state}-{page.url_path}"):
                        st.page_link(page)
    st.markdown(
        '<div class="gt-banner" role="note"><b>Educational research tool</b>'
        '<span>&mdash; predictions are not medical diagnoses.</span></div>',
        unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="gt-page-title" role="heading" aria-level="1">{esc(title)}</div><p class="gt-page-sub">{esc(subtitle)}</p>',
                unsafe_allow_html=True)


@contextmanager
def card(key: str, title: str | None = None, subtitle: str | None = None):
    with st.container(key=f"card-{key}"):
        if title:
            sub = f'<p class="gt-card-sub">{esc(subtitle)}</p>' if subtitle else ""
            st.markdown(f'<div><p class="gt-card-title">{esc(title)}</p>{sub}</div>', unsafe_allow_html=True)
        yield


def section(text: str) -> None:
    st.markdown(f'<p class="gt-section">{esc(text)}</p>', unsafe_allow_html=True)


def caption(text: str) -> None:
    st.markdown(f'<p class="gt-caption">{text}</p>', unsafe_allow_html=True)


def body(text: str) -> None:
    st.markdown(f'<div class="gt-body">{text}</div>', unsafe_allow_html=True)


def tiles(items: list[tuple]) -> None:
    """items: (label, value, note). One CSS grid so it wraps on narrow screens."""
    cells = "".join(
        f'<div class="gt-tile"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div>'
        + (f'<div class="n">{esc(n)}</div>' if n else "") + "</div>"
        for k, v, n in (i if len(i) == 3 else (*i, "") for i in items))
    st.markdown(f'<div class="gt-tiles">{cells}</div>', unsafe_allow_html=True)


def callout(kind: str, title: str, text: str) -> None:
    """kind: info | warn | alert | neutral. The title is always shown so meaning never rests on colour."""
    st.markdown(f'<div class="gt-callout {kind}" role="note"><span class="t">{esc(title)}</span>{text}</div>',
                unsafe_allow_html=True)


def badge(text: str, kind: str = "") -> str:
    return f'<span class="gt-badge {kind}">{esc(text)}</span>'


def empty_state(title: str, text: str) -> None:
    st.markdown(f'<div class="gt-empty"><div class="t">{esc(title)}</div><div class="d">{esc(text)}</div></div>',
                unsafe_allow_html=True)


def kv_table(rows: list[tuple[str, str]]) -> None:
    body_rows = "".join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in rows)
    st.markdown(f'<table class="gt-kv">{body_rows}</table>', unsafe_allow_html=True)


def prob_bar(p: float, threshold: float = 0.5) -> None:
    pct = max(0.0, min(1.0, p)) * 100
    st.markdown(
        f'<div class="gt-prob" role="img" aria-label="Estimated probability of the diabetic class {pct:.1f} percent">'
        f'<div class="track"><div class="fill" style="width:{pct:.2f}%"></div>'
        f'<div class="tick" style="left:{threshold*100:.0f}%"></div></div>'
        f'<div class="labels"><span>0%</span><span class="mid">decision threshold {threshold:.0%}</span><span>100%</span></div></div>',
        unsafe_allow_html=True)


def big_result(label_text: str, value: str, unit: str = "") -> None:
    u = f"<small>{esc(unit)}</small>" if unit else ""
    st.markdown(f'<div class="gt-result-k">{esc(label_text)}</div><div class="gt-result-v">{esc(value)}{u}</div>',
                unsafe_allow_html=True)


def show_table(df: pd.DataFrame, fmt: dict | None = None, highlight_max=(), highlight_min=(), height="content") -> None:
    """Consistent dataframe rendering with column formats; never overflows its card."""
    sty = df.style
    if fmt:
        sty = sty.format(fmt, na_rep="n/a")
    if highlight_max:
        sty = sty.highlight_max(subset=list(highlight_max), color=theme.TEAL_TINT)
    if highlight_min:
        sty = sty.highlight_min(subset=list(highlight_min), color=theme.TEAL_TINT)
    st.dataframe(sty, hide_index=True, width="stretch", height=height)


def csv_download(label_text: str, frame: pd.DataFrame, filename: str, key: str) -> None:
    st.download_button(label_text, frame.to_csv(index=False).encode("utf-8"), file_name=filename,
                       mime="text/csv", key=key, width="stretch")
