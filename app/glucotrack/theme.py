"""Design tokens and global CSS for the GlucoTrack interface.

Every colour, size and radius used by the app is defined once here, as CSS custom
properties and as Python constants for the charts, so the product keeps one visual
language. Selectors prefer Streamlit's documented hooks (data-testid values and
`st.container(key=...)` classes) over its internal DOM structure.
"""

# ---- palette (light-first; contrast ratios checked against white / --bg) ---------
INK = "#0F1F33"          # headings and body, 16.6:1 on white
INK_2 = "#3B4A5E"        # secondary text, 9.2:1
MUTED = "#5B6B7F"        # captions, 5.4:1 (AA for normal text)
BG = "#F7F5F0"           # warm off-white page
SURFACE = "#FFFFFF"
BORDER = "#E4E1D8"
BORDER_STRONG = "#CFCBBF"
TEAL = "#0E7C86"         # primary accent, 5.0:1 on white
TEAL_DARK = "#0A5F67"
TEAL_TINT = "#E5F2F3"
AMBER = "#9A6700"        # warning text, 5.3:1 on AMBER_TINT
AMBER_TINT = "#FDF4DC"
RED = "#B42318"          # alert text, 6.5:1 on RED_TINT
RED_TINT = "#FDEDEB"
SLATE = "#9AA7B5"        # de-emphasised chart marks

# colour-blind-safe categorical set (Okabe-Ito derived); charts pair it with symbols
CATEGORICAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7"]
CLUSTER_SYMBOLS = ["circle", "diamond", "square", "triangle-up", "cross", "x"]

FONT_STACK = "'Inter', -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {{
  --ink: {INK}; --ink-2: {INK_2}; --muted: {MUTED};
  --bg: {BG}; --surface: {SURFACE}; --border: {BORDER}; --border-strong: {BORDER_STRONG};
  --teal: {TEAL}; --teal-dark: {TEAL_DARK}; --teal-tint: {TEAL_TINT};
  --amber: {AMBER}; --amber-tint: {AMBER_TINT}; --red: {RED}; --red-tint: {RED_TINT};
  --radius-sm: 6px; --radius: 10px; --radius-lg: 14px;
  --shadow-sm: 0 1px 2px rgba(15, 31, 51, 0.05);
  --shadow: 0 1px 2px rgba(15, 31, 51, 0.04), 0 4px 14px rgba(15, 31, 51, 0.05);
  --s1: 4px; --s2: 8px; --s3: 12px; --s4: 16px; --s5: 24px; --s6: 32px; --s7: 48px;
}}

html, body, [data-testid="stApp"] {{
  background: var(--bg);
  color: var(--ink);
  font-family: {FONT_STACK};
  -webkit-font-smoothing: antialiased;
}}
/* never target bare span/div: Streamlit renders its icons as ligature text in those elements */
[data-testid="stApp"] :where(p, li, label, h1, h2, h3, h4, a, button, input, textarea, td, th, summary) {{
  font-family: {FONT_STACK};
}}

/* our own components carry the brand font explicitly (they are div/span based) */
[class^="gt-"], [class*=" gt-"], table.gt-kv, table.gt-kv td {{ font-family: {FONT_STACK}; }}

/* chrome we replace with our own header */
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], footer {{ display: none !important; }}

[data-testid="stMainBlockContainer"] {{
  max-width: 1200px;
  padding: var(--s4) var(--s5) var(--s7) var(--s5);
}}
@media (max-width: 640px) {{
  [data-testid="stMainBlockContainer"] {{ padding-left: var(--s4); padding-right: var(--s4); }}
}}

/* ---- typography ---- */
h1, h2, h3, h4 {{ color: var(--ink); letter-spacing: -0.012em; font-weight: 600; }}
.gt-page-title {{ font-size: 1.75rem; line-height: 1.2; font-weight: 650; margin: var(--s5) 0 var(--s1); color: var(--ink); }}
.gt-page-sub {{ font-size: 1rem; color: var(--ink-2); margin: 0 0 var(--s5); max-width: 62ch; line-height: 1.55; }}
.gt-section {{ font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--muted); margin: var(--s2) 0 var(--s3); }}
.gt-h3 {{ font-size: 1.05rem; font-weight: 600; margin: 0 0 var(--s1); color: var(--ink); }}
.gt-caption {{ font-size: 0.84rem; color: var(--muted); line-height: 1.5; margin: 0; }}
.gt-body {{ font-size: 0.95rem; color: var(--ink-2); line-height: 1.6; margin-bottom: var(--s3); }}
.gt-num {{ font-variant-numeric: tabular-nums; }}

/* ---- header ---- */
.st-key-gt-header {{
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg);
  padding: var(--s3) var(--s5); box-shadow: var(--shadow-sm);
}}
.gt-brand {{ display: flex; align-items: center; gap: var(--s3); min-height: 44px; }}
.gt-mark {{ width: 34px; height: 34px; flex: none; }}
.gt-wordmark {{ font-size: 1.25rem; font-weight: 700; letter-spacing: -0.02em; color: var(--ink); line-height: 1; }}
.gt-wordmark span {{ color: var(--teal); }}
.gt-descriptor {{ font-size: 0.72rem; color: var(--muted); letter-spacing: 0.06em; text-transform: uppercase; margin-top: 3px; }}

.st-key-gt-nav a[data-testid="stPageLink-NavLink"] {{
  border-radius: var(--radius-sm); padding: 8px 14px; color: var(--ink-2); font-weight: 500;
  border: 1px solid transparent; text-decoration: none; min-height: 40px; background: transparent;
}}
.st-key-gt-nav a[data-testid="stPageLink-NavLink"] p {{ color: inherit; font-weight: inherit; white-space: nowrap;
  overflow: visible; text-overflow: clip; }}
.st-key-gt-nav a[data-testid="stPageLink-NavLink"], .st-key-gt-nav [data-testid="stElementContainer"],
.st-key-gt-nav [data-testid="stPageLink"] {{ width: auto !important; max-width: none !important; overflow: visible !important; }}
.st-key-gt-nav {{ flex-wrap: wrap; row-gap: var(--s1); }}
.st-key-gt-nav a[data-testid="stPageLink-NavLink"]:hover {{ background: var(--teal-tint); color: var(--teal-dark); }}
[class*="st-key-navitem-active-"] a[data-testid="stPageLink-NavLink"] {{
  background: var(--teal-tint); color: var(--teal-dark); font-weight: 650; border-color: #BFDDE0;
  box-shadow: inset 0 -2px 0 var(--teal);
}}
a:focus-visible, button:focus-visible, input:focus-visible, [role="button"]:focus-visible,
[role="tab"]:focus-visible, [role="radio"]:focus-visible, [role="slider"]:focus-visible,
[data-baseweb="select"] *:focus-visible {{
  outline: 2px solid var(--teal) !important; outline-offset: 2px !important;
}}

/* ---- disclaimer banner ---- */
.gt-banner {{
  display: flex; align-items: center; gap: var(--s2); margin: var(--s3) 0 0;
  background: var(--amber-tint); border: 1px solid #EFDDA8; border-radius: var(--radius);
  padding: 8px var(--s4); font-size: 0.84rem; color: #5C4200; line-height: 1.4;
}}
.gt-banner b {{ font-weight: 600; }}

/* ---- cards ---- */
[class*="st-key-card-"] {{
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg);
  padding: var(--s5); box-shadow: var(--shadow-sm);
}}
[class*="st-key-card-"] > [data-testid="stVerticalBlock"] {{ gap: var(--s3); }}
.gt-card-title {{ font-size: 1.02rem; font-weight: 600; color: var(--ink); margin: 0; }}
.gt-card-sub {{ font-size: 0.86rem; color: var(--muted); margin: 2px 0 var(--s2); line-height: 1.45; }}

/* ---- tiles ---- */
.gt-tiles {{ margin-bottom: var(--s3); display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: var(--s3); }}
.gt-tile {{ background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius); padding: var(--s3) var(--s4); }}
.gt-tile .k {{ font-size: 0.74rem; color: var(--muted); font-weight: 500; letter-spacing: 0.02em; }}
.gt-tile .v {{ font-size: 1.45rem; font-weight: 650; color: var(--ink); margin-top: 2px; font-variant-numeric: tabular-nums; letter-spacing: -0.01em; }}
.gt-tile .n {{ font-size: 0.76rem; color: var(--muted); margin-top: 2px; line-height: 1.35; }}

/* ---- badges ---- */
.gt-badge {{ display: inline-flex; align-items: center; gap: 6px; font-size: 0.78rem; font-weight: 600;
  padding: 3px 10px; border-radius: 999px; border: 1px solid var(--border-strong); color: var(--ink-2); background: var(--surface); }}
.gt-badge.teal {{ background: var(--teal-tint); color: var(--teal-dark); border-color: #BFDDE0; }}
.gt-badge.amber {{ background: var(--amber-tint); color: var(--amber); border-color: #EFDDA8; }}
.gt-badge.red {{ background: var(--red-tint); color: var(--red); border-color: #F5C6C1; }}

/* ---- callouts (always carry a text label, never colour alone) ---- */
.gt-callout {{ margin-bottom: var(--s3); border-radius: var(--radius); padding: var(--s3) var(--s4); font-size: 0.88rem; line-height: 1.55;
  border: 1px solid var(--border); border-left-width: 4px; background: var(--surface); color: var(--ink-2); }}
.gt-callout .t {{ font-weight: 600; color: var(--ink); display: block; margin-bottom: 2px; }}
.gt-callout.info {{ border-left-color: var(--teal); background: #F3F9F9; }}
.gt-callout.warn {{ border-left-color: #C98A00; background: var(--amber-tint); }}
.gt-callout.alert {{ border-left-color: var(--red); background: var(--red-tint); }}
.gt-callout.neutral {{ border-left-color: var(--slate, #9AA7B5); }}

/* ---- empty state ---- */
.gt-empty {{ border: 1.5px dashed var(--border-strong); border-radius: var(--radius-lg); padding: var(--s6) var(--s5);
  text-align: center; background: #FBFAF7; }}
.gt-empty .t {{ font-weight: 600; color: var(--ink); font-size: 1.02rem; }}
.gt-empty .d {{ color: var(--muted); font-size: 0.9rem; margin: var(--s2) auto 0; max-width: 40ch; line-height: 1.5; }}

/* ---- key/value table (specificity beats the markdown table defaults) ---- */
[data-testid="stMarkdownContainer"] table.gt-kv {{ width: 100%; border-collapse: collapse; border: 0; font-size: 0.88rem; }}
[data-testid="stMarkdownContainer"] table.gt-kv td {{ padding: 7px 0; border: 0; border-bottom: 1px solid var(--border);
  vertical-align: top; background: transparent; }}
[data-testid="stMarkdownContainer"] table.gt-kv tr:last-child td {{ border-bottom: 0; }}
[data-testid="stMarkdownContainer"] table.gt-kv td:first-child {{ color: var(--muted); padding-right: var(--s4); width: 46%; }}
[data-testid="stMarkdownContainer"] table.gt-kv td:last-child {{ color: var(--ink); font-weight: 500; text-align: right;
  font-variant-numeric: tabular-nums; }}

/* ---- probability bar ---- */
.gt-prob {{ margin: var(--s2) 0 0; }}
.gt-prob .track {{ position: relative; height: 14px; border-radius: 999px; background: #ECE9E0; overflow: visible; }}
.gt-prob .fill {{ height: 100%; border-radius: 999px; background: var(--teal); }}
.gt-prob .tick {{ position: absolute; top: -4px; bottom: -4px; width: 2px; background: var(--ink); left: 50%; }}
.gt-prob .labels {{ display: flex; justify-content: space-between; font-size: 0.74rem; color: var(--muted); margin-top: 8px; }}
.gt-prob .labels .mid {{ color: var(--ink-2); }}

/* ---- big result ---- */
.gt-result-k {{ font-size: 0.78rem; color: var(--muted); font-weight: 500; }}
.gt-result-v {{ font-size: 2.6rem; font-weight: 700; line-height: 1.05; letter-spacing: -0.03em; color: var(--ink); font-variant-numeric: tabular-nums; }}
.gt-result-v small {{ font-size: 1rem; font-weight: 500; color: var(--muted); letter-spacing: 0; margin-left: 4px; }}

/* ---- widgets ---- */
[data-testid="stBaseButton-primary"] {{
  background: var(--teal); border: 1px solid var(--teal); color: #fff; font-weight: 600;
  border-radius: var(--radius); min-height: 44px; box-shadow: var(--shadow-sm);
}}
[data-testid="stBaseButton-primary"]:hover {{ background: var(--teal-dark); border-color: var(--teal-dark); color: #fff; }}
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondaryFormSubmit"] {{
  background: var(--surface); border: 1px solid var(--border-strong); color: var(--ink);
  font-weight: 500; border-radius: var(--radius); min-height: 44px;
}}
[data-testid="stBaseButton-secondary"]:hover {{ border-color: var(--teal); color: var(--teal-dark); background: var(--teal-tint); }}
[data-testid="stDownloadButton"] button {{ border-radius: var(--radius); min-height: 44px; font-weight: 500; }}

[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {{ font-size: 0.86rem; font-weight: 500; color: var(--ink-2); }}
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="base-input"] {{
  border-radius: var(--radius-sm) !important; background: var(--surface);
}}

/* tabs */
[data-baseweb="tab-list"] {{ gap: var(--s2); border-bottom: 1px solid var(--border); }}
[data-baseweb="tab"] {{ font-weight: 500; color: var(--ink-2); padding: 10px 14px; }}
[data-baseweb="tab"][aria-selected="true"] {{ color: var(--teal-dark); font-weight: 600; }}
[data-baseweb="tab-highlight"] {{ background: var(--teal) !important; height: 2px; }}

/* dataframes: keep inside their container, never overflow the page */
[data-testid="stDataFrame"] {{ border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }}
[data-testid="stPlotlyChart"], [data-testid="stImage"] {{ max-width: 100%; }}

/* expander */
[data-testid="stExpander"] {{ border: 1px solid var(--border); border-radius: var(--radius); background: var(--surface); }}

hr {{ border-color: var(--border); }}
a {{ color: var(--teal-dark); }}
"""
