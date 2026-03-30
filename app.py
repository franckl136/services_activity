import re
import unicodedata
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

st.set_page_config(
    page_title="Analyse exports Zendesk",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette contrastée (Plotly + complément)
COLOR_SEQUENCE = (
    px.colors.qualitative.Bold
    + px.colors.qualitative.Set2
    + px.colors.qualitative.Pastel
)

METADATA_NAMES = {
    "collaborateur",
    "client",
    "statut",
    "status",
    "ticket id",
    "ticket_id",
    "id",
    "#",
    "sujet",
    "subject",
    "description",
    "total",
    "totale",
    "grand total",
    "sous-total",
    "subtotal",
    "total général",
    "total general",
}

# Lignes de bas de tableau Excel (totaux / formules SUM affichées en ligne)
AGGREGATE_COLLABORATEUR_EXACT = {
    "total",
    "totale",
    "totaux",
    "sum",
    "somme",
    "subtotal",
    "sous-total",
    "sous total",
    "grand total",
    "total général",
    "total general",
    "moyenne",
    "average",
    "moyenne générale",
    "total période",
}


def is_aggregate_collaborateur_row(val) -> bool:
    """Détecte les lignes récap (Total, Sum, etc.), souvent en fin de feuille Excel."""
    if pd.isna(val):
        return False
    s = str(val).lower().strip()
    if s in ("", "nan", "none"):
        return False
    if s in AGGREGATE_COLLABORATEUR_EXACT:
        return True
    # Préfixes typiques des lignes d'agrégat
    if re.match(r"^(total|totaux|somme|sum|subtotal|moyenne|average)\b", s, re.I):
        return True
    return False


def filter_aggregate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Retire les lignes d'agrégat sur la colonne Collaborateur."""
    if "Collaborateur" not in df.columns:
        return df, 0
    mask = df["Collaborateur"].map(is_aggregate_collaborateur_row)
    n = int(mask.sum())
    return df.loc[~mask].copy(), n


def clean_collaborateur(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
    return s


def _column_name_is_date(name: str) -> bool:
    name = str(name).strip()
    if not name:
        return False
    ts = pd.to_datetime([name], errors="coerce", dayfirst=True)
    return bool(pd.notna(ts[0]))


def detect_ticket_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        cl = str(c).lower().strip()
        if "ticket" in cl and "id" in cl.replace(" ", ""):
            return c
        if cl in ("#", "id", "ticket id", "ticket_id"):
            return c
    return None


def _is_excluded_column(name: str) -> bool:
    return str(name).lower().strip() in METADATA_NAMES


def detect_date_columns(df: pd.DataFrame) -> list[str]:
    date_cols: list[str] = []
    for c in df.columns:
        if _is_excluded_column(str(c)):
            continue
        if _column_name_is_date(str(c)):
            date_cols.append(c)
            continue
    if date_cols:
        return date_cols
    for c in df.columns:
        if _is_excluded_column(str(c)):
            continue
        if detect_ticket_column(df) == c:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            date_cols.append(c)
    return date_cols


def ensure_numeric_minutes(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    return out


def build_total_temps(df: pd.DataFrame, date_cols: list[str]) -> pd.Series:
    if not date_cols:
        return pd.Series(0.0, index=df.index)
    return df[date_cols].sum(axis=1)


def interventions_par_collaborateur(df: pd.DataFrame, col_ticket: Optional[str]) -> pd.Series:
    if col_ticket and col_ticket in df.columns:
        return df.groupby("Collaborateur", dropna=False)[col_ticket].nunique()
    return df.groupby("Collaborateur", dropna=False).size()


def parse_column_to_datetime(c, ref_year: Optional[int] = None) -> pd.Timestamp:
    """Parse une en-tête de colonne « jour » (même logique partout)."""
    if isinstance(c, pd.Timestamp):
        return c
    s = str(c).strip()
    if not s or s.lower() in ("nan", "nat"):
        return pd.NaT
    # 1) Essai direct (gère déjà beaucoup de formats)
    ts = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.notna(ts):
        return ts

    # 2) Secours : mois en abrégé FR/EN (jan, fév, feb, mar, ...)
    def _norm_token(t: str) -> str:
        t = t.strip().lower().replace(".", "")
        t = unicodedata.normalize("NFKD", t)
        t = "".join(ch for ch in t if not unicodedata.combining(ch))
        return t

    month_map = {
        # FR
        "jan": 1,
        "janv": 1,
        "janvier": 1,
        "fev": 2,
        "fevr": 2,
        "fevrier": 2,
        "fév": 2,
        "févr": 2,
        "février": 2,
        "mar": 3,
        "mars": 3,
        "avr": 4,
        "avril": 4,
        "mai": 5,
        "jun": 6,
        "juin": 6,
        "jui": 7,
        "juil": 7,
        "juillet": 7,
        "aou": 8,
        "aout": 8,
        "août": 8,
        "sep": 9,
        "sept": 9,
        "septembre": 9,
        "oct": 10,
        "octobre": 10,
        "nov": 11,
        "novembre": 11,
        "dec": 12,
        "decembre": 12,
        "déc": 12,
        "décembre": 12,
        # EN
        "january": 1,
        "feb": 2,
        "february": 2,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }

    # Année de référence (si absente des en-têtes) — doit être passée explicitement ou on utilise l’année courante
    if ref_year is None:
        ref_year = int(pd.Timestamp.today().year)

    s2 = s.replace(",", " ").replace("_", " ").strip()
    m = re.search(r"\b(19|20)\d{2}\b", s2)
    year = int(m.group(0)) if m else ref_year

    # Patterns : "02 Jan", "2 jan", "Jan 02", "jan", "feb 2025", "02-feb-2025"
    patterns = [
        r"^\s*(\d{1,2})\s*[-/ ]\s*([A-Za-zÀ-ÿ]+)\s*(?:[-/ ]\s*(\d{4}))?\s*$",
        r"^\s*([A-Za-zÀ-ÿ]+)\s*[-/ ]\s*(\d{1,2})\s*(?:[-/ ]\s*(\d{4}))?\s*$",
        r"^\s*([A-Za-zÀ-ÿ]+)\s*(?:[-/ ]\s*(\d{4}))?\s*$",
    ]
    for pat in patterns:
        mm = re.match(pat, s2)
        if not mm:
            continue
        g = [x for x in mm.groups() if x is not None]
        day = 1
        month_token = ""
        year_override = None
        if len(g) == 3:
            # jour + mois + année
            if g[0].isdigit():
                day = int(g[0])
                month_token = g[1]
            else:
                month_token = g[0]
                day = int(g[1])
            year_override = int(g[2])
        elif len(g) == 2:
            if g[0].isdigit():
                day = int(g[0])
                month_token = g[1]
            else:
                month_token = g[0]
                # soit "jan 2025" soit "jan 02" (capturé par pat 2 normalement)
                if g[1].isdigit() and len(g[1]) == 4:
                    year_override = int(g[1])
                else:
                    day = int(g[1])
        elif len(g) == 1:
            month_token = g[0]

        mo = month_map.get(_norm_token(month_token))
        if not mo:
            continue
        yy = year_override if year_override is not None else year
        try:
            return pd.Timestamp(year=yy, month=mo, day=day)
        except Exception:
            return pd.NaT

    return pd.NaT


def order_date_columns(cols: list[str], ref_year: Optional[int] = None) -> list[str]:
    """Trie les en-têtes de colonnes jour par date réelle ; ordre stable si parsing impossible."""
    order_idx: list[tuple] = []
    for i, c in enumerate(cols):
        ts = parse_column_to_datetime(c, ref_year=ref_year)
        if pd.notna(ts):
            order_idx.append((ts, i, c))
        else:
            order_idx.append((pd.Timestamp.max, i, c))
    order_idx.sort(key=lambda x: (x[0], x[1]))
    return [t[2] for t in order_idx]


def jour_key_to_datetime_map(
    ordered_date_cols: list[str], ref_year: Optional[int] = None
) -> dict[str, pd.Timestamp]:
    """Clé normalisée (strip) -> timestamp pour aligner melt / graphiques."""
    m: dict[str, pd.Timestamp] = {}
    for c in ordered_date_cols:
        k = str(c).strip()
        m[k] = parse_column_to_datetime(c, ref_year=ref_year)
    return m


def apply_chronological_xaxis_day(fig, ordered_date_cols: list[str], tick_angle: int = -45) -> None:
    """Secours si l'axe date n'est pas utilisable : catégories dans l'ordre explicite."""
    labels = [str(c) for c in ordered_date_cols]
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=labels,
        tickangle=tick_angle,
    )


def daily_long_by_consultant(
    filtered: pd.DataFrame, date_cols: list[str], ordered_date_cols: list[str]
) -> pd.DataFrame:
    """Jour × Consultant avec minutes (jours dans l'ordre chronologique)."""
    id_vars = ["Collaborateur"]
    if "__ref_year" in filtered.columns:
        id_vars.append("__ref_year")
    long_df = filtered.melt(
        id_vars=id_vars,
        value_vars=date_cols,
        var_name="Jour",
        value_name="Minutes",
    )
    long_df["Minutes"] = pd.to_numeric(long_df["Minutes"], errors="coerce").fillna(0)
    long_df["Jour"] = long_df["Jour"].map(lambda x: str(x).strip())
    if "__ref_year" in long_df.columns:
        ry_default = int(pd.Timestamp.today().year)
        long_df["Date_plot"] = [
            parse_column_to_datetime(
                j, ref_year=_ref_year_scalar(ry) or ry_default
            )
            for j, ry in zip(long_df["Jour"], long_df["__ref_year"])
        ]
        daily = long_df.groupby(["Collaborateur", "Date_plot"], as_index=False)["Minutes"].sum()
        return daily.sort_values(["Collaborateur", "Date_plot"])
    daily = long_df.groupby(["Jour", "Collaborateur"], as_index=False)["Minutes"].sum()
    order_map = {str(c).strip(): i for i, c in enumerate(ordered_date_cols)}
    daily["_o"] = daily["Jour"].map(lambda x: order_map.get(x, 999))
    return daily.sort_values(["Collaborateur", "_o"]).drop(columns="_o")


def cumulative_by_consultant(
    filtered: pd.DataFrame, date_cols: list[str], ordered_date_cols: list[str]
) -> pd.DataFrame:
    """Cumul des minutes sur la période, une série par consultant."""
    daily = daily_long_by_consultant(filtered, date_cols, ordered_date_cols)
    if daily.empty:
        return pd.DataFrame(columns=["Jour", "Collaborateur", "Minutes", "Cumul_min"])
    if "Date_plot" in daily.columns:
        parts = []
        for collab, g in daily.groupby("Collaborateur", dropna=False):
            g = g.sort_values("Date_plot").copy()
            g["Cumul_min"] = g["Minutes"].cumsum()
            parts.append(g)
        return pd.concat(parts, ignore_index=True)
    parts = []
    for collab, g in daily.groupby("Collaborateur", dropna=False):
        g = g.copy()
        g["Cumul_min"] = g["Minutes"].cumsum()
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


MONTHS_FR = {
    1: "Janvier",
    2: "Février",
    3: "Mars",
    4: "Avril",
    5: "Mai",
    6: "Juin",
    7: "Juillet",
    8: "Août",
    9: "Septembre",
    10: "Octobre",
    11: "Novembre",
    12: "Décembre",
}


def ym_key_to_french_label(ym: str) -> str:
    """'2025-01' -> 'Janvier 2025' ; 'Non classé' inchangé."""
    if ym in ("Non classé", "") or (isinstance(ym, float) and pd.isna(ym)) or pd.isna(ym):
        return "Non classé"
    parts = str(ym).split("-")
    if len(parts) != 2:
        return str(ym)
    y, mo = parts[0], parts[1]
    try:
        return f"{MONTHS_FR[int(mo)]} {y}"
    except (ValueError, KeyError):
        return str(ym)


def monthly_long_by_consultant(
    filtered: pd.DataFrame, date_cols: list[str], ref_year: Optional[int] = None
) -> pd.DataFrame:
    """Agrège les colonnes-jour par mois calendaire (YYYY-MM)."""
    id_vars = ["Collaborateur"]
    if "__ref_year" in filtered.columns:
        id_vars.append("__ref_year")
    long_df = filtered.melt(
        id_vars=id_vars,
        value_vars=date_cols,
        var_name="Jour",
        value_name="Minutes",
    )
    long_df["Minutes"] = pd.to_numeric(long_df["Minutes"], errors="coerce").fillna(0)
    ry_default = ref_year if ref_year is not None else int(pd.Timestamp.today().year)
    if "__ref_year" in long_df.columns:
        long_df["Dt"] = [
            parse_column_to_datetime(
                j, ref_year=_ref_year_scalar(ry) or ry_default
            )
            for j, ry in zip(long_df["Jour"], long_df["__ref_year"])
        ]
    else:
        long_df["Dt"] = long_df["Jour"].map(
            lambda j: parse_column_to_datetime(j, ref_year=ref_year)
        )
    long_df["Mois"] = long_df["Dt"].apply(
        lambda d: f"{d.year}-{d.month:02d}" if pd.notna(d) else "Non classé"
    )
    monthly = long_df.groupby(["Mois", "Collaborateur"], as_index=False)["Minutes"].sum()
    # Ordre chronologique des mois (sans Categorical : meilleure compatibilité Plotly)
    order_ym = sorted(m for m in monthly["Mois"].unique() if m != "Non classé")
    if "Non classé" in monthly["Mois"].values:
        order_ym.append("Non classé")
    monthly["Mois_label"] = monthly["Mois"].map(ym_key_to_french_label)
    label_order = [ym_key_to_french_label(m) for m in order_ym]
    monthly["_sort"] = monthly["Mois"].map({m: i for i, m in enumerate(order_ym)})
    monthly = monthly.sort_values(["_sort", "Collaborateur"]).drop(columns="_sort")
    return monthly, label_order


def date_cols_grouped_by_year(
    date_cols: list[str], ref_year: Optional[int] = None
) -> tuple[dict[int, list[str]], list[int]]:
    """Regroupe les colonnes jour par année calendaire (année 0 = non parsé)."""
    by_year: dict[int, list[str]] = {}
    for c in date_cols:
        ts = parse_column_to_datetime(c, ref_year=ref_year)
        y = int(ts.year) if pd.notna(ts) else 0
        by_year.setdefault(y, []).append(c)
    for y in list(by_year.keys()):
        by_year[y] = order_date_columns(by_year[y], ref_year=ref_year)
    years_sorted = sorted([y for y in by_year if y > 0])
    return by_year, years_sorted


def period_bounds_from_ordered_cols(
    ordered_date_cols: list[str], ref_year: Optional[int] = None
) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    ts = [parse_column_to_datetime(c, ref_year=ref_year) for c in ordered_date_cols]
    ts_ok = [t for t in ts if pd.notna(t)]
    if not ts_ok:
        return None, None
    return min(ts_ok), max(ts_ok)


def interventions_for_date_subset(
    df: pd.DataFrame, date_cols_subset: list[str], ticket_col: Optional[str]
) -> pd.Series:
    """Tickets / lignes avec au moins une minute sur le sous-ensemble de colonnes."""
    if not date_cols_subset:
        return pd.Series(dtype=float)
    df2 = ensure_numeric_minutes(df, date_cols_subset)
    row_mask = df2[date_cols_subset].sum(axis=1) > 0
    sub = df2.loc[row_mask]
    if len(sub) == 0:
        return pd.Series(dtype=float)
    if ticket_col and ticket_col in sub.columns:
        return sub.groupby("Collaborateur", dropna=False)[ticket_col].nunique()
    return sub.groupby("Collaborateur", dropna=False).size()


def kpi_table_for_date_cols(
    filtered: pd.DataFrame, date_cols_subset: list[str], ticket_col: Optional[str]
) -> tuple[pd.DataFrame, float, int, float]:
    """KPI par consultant pour un sous-ensemble de colonnes jour."""
    if not date_cols_subset:
        kpi = pd.DataFrame()
        return kpi, 0.0, 0, 0.0
    df2 = ensure_numeric_minutes(filtered, date_cols_subset).copy()
    df2["_t"] = df2[date_cols_subset].sum(axis=1, numeric_only=True)
    inter = interventions_for_date_subset(filtered, date_cols_subset, ticket_col)
    tpc = df2.groupby("Collaborateur", dropna=False)["_t"].sum()
    inter = inter.reindex(tpc.index, fill_value=0)
    kpi = pd.DataFrame({"Temps_total": tpc, "Interventions": inter})
    kpi["Interventions"] = kpi["Interventions"].fillna(0).astype(int)
    kpi["Temps_moyen_par_ticket"] = kpi["Temps_total"] / kpi["Interventions"].replace(0, pd.NA)
    kpi = kpi.fillna(0)
    tot = float(df2["_t"].sum())
    if ticket_col and ticket_col in filtered.columns:
        sub = df2.loc[df2["_t"] > 0]
        nb = int(sub[ticket_col].nunique()) if len(sub) else 0
    else:
        nb = int((df2["_t"] > 0).sum())
    moy = tot / nb if nb else 0.0
    return kpi, tot, nb, moy


def render_monthly_charts_streamlit(
    filtered: pd.DataFrame, date_cols_subset: list[str], ref_year: Optional[int] = None
) -> None:
    """Tableau + graphiques mensuels (pour une année ou le global si besoin)."""
    monthly, mois_label_order = monthly_long_by_consultant(
        filtered, date_cols_subset, ref_year=ref_year
    )
    minutes_total = pd.to_numeric(monthly["Minutes"], errors="coerce").fillna(0).sum()
    if len(monthly) == 0 or minutes_total <= 0:
        st.info("Pas assez de minutes pour l’analyse mensuelle sur cette période.")
        return

    pivot = monthly.pivot_table(
        index="Mois",
        columns="Collaborateur",
        values="Minutes",
        aggfunc="sum",
        fill_value=0,
    )
    pivot_display = pivot.reset_index()
    pivot_display["Mois"] = pivot_display["Mois"].map(ym_key_to_french_label)
    st.dataframe(pivot_display, use_container_width=True, hide_index=True)

    m1, m2 = st.columns(2)
    fig_m_stack = px.bar(
        monthly,
        x="Mois_label",
        y="Minutes",
        color="Collaborateur",
        color_discrete_sequence=COLOR_SEQUENCE,
        title="Temps par mois (empilé par consultant)",
        category_orders={"Mois_label": mois_label_order},
    )
    fig_m_stack.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        barmode="stack",
        xaxis_title="Mois",
        yaxis_title="Minutes",
        height=420,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(l=8, r=140, t=48, b=8),
    )
    fig_m_lines = px.line(
        monthly,
        x="Mois_label",
        y="Minutes",
        color="Collaborateur",
        markers=True,
        color_discrete_sequence=COLOR_SEQUENCE,
        title="Temps par mois — une courbe par consultant",
        category_orders={"Mois_label": mois_label_order},
    )
    fig_m_lines.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        xaxis_title="Mois",
        yaxis_title="Minutes",
        height=420,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(l=8, r=140, t=48, b=8),
    )
    fig_m_lines.update_traces(line=dict(width=2))
    m1.plotly_chart(fig_m_stack, use_container_width=True)
    m2.plotly_chart(fig_m_lines, use_container_width=True)

    heat_m = pivot.copy()
    idx_sorted = sorted((m for m in heat_m.index if m != "Non classé"), key=str)
    if "Non classé" in heat_m.index:
        idx_sorted.append("Non classé")
    heat_m = heat_m.reindex(idx_sorted)
    heat_y = [ym_key_to_french_label(str(m)) for m in heat_m.index]
    fig_m_heat = px.imshow(
        heat_m.values,
        x=[str(c) for c in heat_m.columns],
        y=heat_y,
        labels=dict(x="Consultant", y="Mois", color="Min."),
        color_continuous_scale="Blues",
        aspect="auto",
        title="Heatmap — mois × consultant",
    )
    fig_m_heat.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        font=dict(color="#e6edf3"),
        height=max(280, 36 * len(heat_m.index)),
        margin=dict(l=8, r=8, t=48, b=8),
    )
    fig_m_heat.update_yaxes(categoryorder="array", categoryarray=heat_y)
    st.plotly_chart(fig_m_heat, use_container_width=True)


def render_year_detail_charts(
    fy: pd.DataFrame,
    cols_y: list[str],
    ordered_y: list[str],
    ref_year: int,
    title: str,
    ticket_col: Optional[str],
) -> None:
    """KPI, heatmap jour, lignes quotidiennes, cumul et analyse mensuelle pour un millésime."""
    st.markdown(f"### {title}")
    dt_a, dt_b = period_bounds_from_ordered_cols(ordered_y, ref_year=ref_year)
    if dt_a is not None and dt_b is not None:
        st.caption(
            f"Période : **{dt_a.strftime('%d/%m/%Y')}** → **{dt_b.strftime('%d/%m/%Y')}** · "
            f"**{len(cols_y)}** jour(s) en colonnes · **{len(fy)}** ligne(s)"
        )

    kpi_y, tot_y, nb_y, moy_y = kpi_table_for_date_cols(fy, cols_y, ticket_col)
    if len(kpi_y) == 0:
        st.warning("Pas de données agrégées pour cette année.")
        return

    y1, y2, y3 = st.columns(3)
    y1.metric("Temps (min)", f"{tot_y:,.0f}".replace(",", " "))
    y2.metric("Tickets concernés", f"{nb_y:,}".replace(",", " "))
    y3.metric("Temps moyen / ticket (min)", f"{moy_y:,.1f}".replace(",", " "))

    st.dataframe(
        kpi_y.reset_index().rename(
            columns={
                "Collaborateur": "Consultant",
                "Temps_total": "Temps (min)",
                "Interventions": "Tickets (avec temps cette année)",
                "Temps_moyen_par_ticket": "Temps moyen / ticket (min)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    kr = kpi_y.reset_index().sort_values("Temps_total", ascending=True)
    fy_bar = px.bar(
        kr,
        x="Temps_total",
        y="Collaborateur",
        orientation="h",
        color="Temps_total",
        color_continuous_scale="Tealgrn",
        labels={"Temps_total": "Minutes", "Collaborateur": ""},
        title=f"Classement du temps — {title}",
    )
    fy_bar.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        showlegend=False,
        yaxis=dict(categoryorder="total ascending"),
        height=max(320, 28 * len(kr)),
        margin=dict(l=8, r=8, t=48, b=8),
    )
    fy_pie = px.pie(
        kr.sort_values("Temps_total", ascending=False),
        values="Temps_total",
        names="Collaborateur",
        title="Part du temps par consultant",
        hole=0.45,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fy_pie.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        font=dict(color="#e6edf3"),
        margin=dict(l=8, r=8, t=48, b=8),
        showlegend=True,
    )
    cya, cyb = st.columns([1.2, 1])
    cya.plotly_chart(fy_bar, use_container_width=True)
    cyb.plotly_chart(fy_pie, use_container_width=True)

    if len(fy) and ordered_y:
        heat = (
            fy.groupby("Collaborateur", dropna=False)[ordered_y]
            .sum()
            .reindex(columns=ordered_y, fill_value=0)
        )
        x_ts = [parse_column_to_datetime(c, ref_year=ref_year) for c in heat.columns]
        if all(pd.notna(t) for t in x_ts):
            fig_heat = go.Figure(
                data=go.Heatmap(
                    z=heat.values,
                    x=x_ts,
                    y=heat.index.astype(str).tolist(),
                    colorscale="YlOrRd",
                    hovertemplate="%{y}<br>%{x|%d %b %Y}<br>%{z} min<extra></extra>",
                    colorbar=dict(title="Min."),
                )
            )
            fig_heat.update_layout(
                title=f"Charge jour × consultant — {title}",
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                font=dict(color="#e6edf3"),
                xaxis=dict(
                    type="date",
                    tickformat="%d %b %Y",
                    tickangle=-45,
                    title="Date",
                ),
                yaxis=dict(title="Consultant"),
                height=max(300, 26 * len(heat)),
                margin=dict(l=8, r=8, t=48, b=8),
            )
        else:
            day_labels = [str(c) for c in heat.columns]
            fig_heat = px.imshow(
                heat.values,
                x=day_labels,
                y=heat.index.astype(str),
                labels=dict(x="Jour", y="Consultant", color="Min."),
                color_continuous_scale="YlOrRd",
                aspect="auto",
                title=f"Charge jour × consultant — {title}",
            )
            fig_heat.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                font=dict(color="#e6edf3"),
                height=max(300, 26 * len(heat)),
                margin=dict(l=8, r=8, t=48, b=8),
            )
            apply_chronological_xaxis_day(fig_heat, ordered_y, tick_angle=-50)
        st.plotly_chart(fig_heat, use_container_width=True)

        daily_collab = daily_long_by_consultant(fy, cols_y, ordered_y)
        if "Date_plot" in daily_collab.columns:
            use_date_axis = bool(daily_collab["Date_plot"].notna().all())
        else:
            _dt_map = jour_key_to_datetime_map(ordered_y, ref_year=ref_year)
            daily_collab["Date_plot"] = daily_collab["Jour"].map(
                lambda j: _dt_map.get(str(j).strip(), pd.NaT)
            )
            use_date_axis = bool(_dt_map) and all(pd.notna(t) for t in _dt_map.values())
        if use_date_axis:
            daily_plot = daily_collab.sort_values(["Collaborateur", "Date_plot"])
            fig_lines = px.line(
                daily_plot,
                x="Date_plot",
                y="Minutes",
                color="Collaborateur",
                markers=True,
                color_discrete_sequence=COLOR_SEQUENCE,
                title=f"Activité quotidienne — {title}",
            )
            fig_lines.update_xaxes(tickformat="%d %b %Y", title="Date")
        else:
            fig_lines = px.line(
                daily_collab,
                x="Jour",
                y="Minutes",
                color="Collaborateur",
                markers=True,
                color_discrete_sequence=COLOR_SEQUENCE,
                title=f"Activité quotidienne — {title}",
            )
            apply_chronological_xaxis_day(fig_lines, ordered_y, tick_angle=-45)
        fig_lines.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#161b22",
            font=dict(color="#e6edf3"),
            yaxis_title="Minutes",
            height=440,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
            margin=dict(l=8, r=160, t=48, b=8),
        )
        fig_lines.update_traces(line=dict(width=2))
        st.plotly_chart(fig_lines, use_container_width=True)

        cum_df = cumulative_by_consultant(fy, cols_y, ordered_y)
        if len(cum_df):
            cum_df = cum_df.copy()
            if "Date_plot" not in cum_df.columns:
                _dt_map_c = jour_key_to_datetime_map(ordered_y, ref_year=ref_year)
                cum_df["Date_plot"] = cum_df["Jour"].map(
                    lambda j: _dt_map_c.get(str(j).strip(), pd.NaT)
                )
            cum_use_date = cum_df["Date_plot"].notna().all() if "Date_plot" in cum_df.columns else False
            if cum_use_date:
                cum_df = cum_df.sort_values(["Collaborateur", "Date_plot"])
                fig_cumul = px.line(
                    cum_df,
                    x="Date_plot",
                    y="Cumul_min",
                    color="Collaborateur",
                    markers=True,
                    color_discrete_sequence=COLOR_SEQUENCE,
                    title=f"Cumul sur l’année — {title}",
                )
                fig_cumul.update_xaxes(tickformat="%d %b %Y", title="Date")
            else:
                fig_cumul = px.line(
                    cum_df,
                    x="Jour",
                    y="Cumul_min",
                    color="Collaborateur",
                    markers=True,
                    color_discrete_sequence=COLOR_SEQUENCE,
                    title=f"Cumul sur l’année — {title}",
                )
                apply_chronological_xaxis_day(fig_cumul, ordered_y, tick_angle=-45)
            fig_cumul.update_traces(line=dict(width=2.5), marker=dict(size=7))
            fig_cumul.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#161b22",
                font=dict(color="#e6edf3"),
                yaxis_title="Minutes cumulées",
                height=400,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                margin=dict(l=8, r=160, t=48, b=8),
            )
            st.plotly_chart(fig_cumul, use_container_width=True)

    st.markdown(f"#### Analyse mensuelle — {title}")
    st.caption("Minutes par mois (colonnes jour de cette année uniquement).")
    render_monthly_charts_streamlit(fy, cols_y, ref_year=ref_year)
    st.markdown("---")


def _sessions_dir() -> Path:
    p = Path(__file__).resolve().parent / "saved_sessions"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _list_saved_sessions() -> list[str]:
    p = _sessions_dir()
    files = sorted(p.glob("*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
    return [f.stem for f in files]


def save_session_pickle(
    session_name: str,
    df_num: pd.DataFrame,
    date_cols: list[str],
    ticket_col: Optional[str],
    source_files: list[str],
) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", (session_name or "").strip())[:80]
    if not safe:
        safe = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "saved_at": pd.Timestamp.now().isoformat(),
        "source_files": source_files,
        "ticket_col": ticket_col,
        "date_cols": date_cols,
        "df_num": df_num,
    }
    target = _sessions_dir() / f"{safe}.pkl"
    pd.to_pickle(payload, target)
    return target


def load_session_pickle(session_stem: str) -> dict:
    path = _sessions_dir() / f"{session_stem}.pkl"
    return pd.read_pickle(path)


def fig_to_html(fig) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")


def build_report_html(
    *,
    title: str,
    period_label: str,
    df_filtered: pd.DataFrame,
    kpi_table: pd.DataFrame,
    temps_global: float,
    nb_tickets_uniques: int,
    temps_moyen: float,
    fig_bar,
    fig_pie,
    fig_grouped,
    fig_tree,
    years_in_filtered: list[int],
    date_cols: list[str],
    ticket_col: Optional[str],
) -> bytes:
    kpi_html = (
        kpi_table.reset_index()
        .rename(
            columns={
                "Collaborateur": "Consultant",
                "Temps_total": "Temps global (min)",
                "Interventions": "Nb tickets",
                "Temps_moyen_par_ticket": "Temps moyen / ticket (min)",
            }
        )
        .to_html(index=False, escape=False)
    )

    parts: list[str] = []
    parts.append(
        f"""
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; color: #111; }}
    h1,h2,h3 {{ margin: 0.6rem 0; }}
    .meta {{ color: #444; margin-bottom: 18px; }}
    .kpis {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 14px 0 18px; }}
    .kpi {{ border: 1px solid #ddd; border-radius: 10px; padding: 12px; }}
    .kpi .label {{ color:#555; font-size: 12px; }}
    .kpi .value {{ font-size: 22px; font-weight: 700; margin-top: 6px; }}
    .block {{ margin: 18px 0; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; font-size: 13px; }}
    th {{ background: #f6f7f8; text-align: left; }}
    .caption {{ color:#555; font-size: 13px; margin: 6px 0 10px; }}
    @media print {{
      body {{ margin: 12px; }}
      .no-print {{ display: none; }}
    }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="meta">Généré le {pd.Timestamp.now().strftime("%d/%m/%Y %H:%M")} · Période sélectionnée : <b>{period_label}</b> · Lignes filtrées : <b>{len(df_filtered)}</b></div>
  <div class="kpis">
    <div class="kpi"><div class="label">Temps global (min)</div><div class="value">{temps_global:,.0f}</div></div>
    <div class="kpi"><div class="label">Tickets (uniques)</div><div class="value">{nb_tickets_uniques:,}</div></div>
    <div class="kpi"><div class="label">Temps moyen / ticket (min)</div><div class="value">{temps_moyen:,.1f}</div></div>
    <div class="kpi"><div class="label">Années (noms de fichiers)</div><div class="value">{len(years_in_filtered) if years_in_filtered else 0}</div></div>
  </div>
"""
    )
    parts.append('<div class="block"><h2>Table KPI (consultants)</h2>' + kpi_html + "</div>")

    parts.append('<div class="block"><h2>Répartition du temps (global)</h2>')
    parts.append('<div class="caption">Barres, camembert, et comparaison temps total vs temps moyen par ticket.</div>')
    parts.append(fig_to_html(fig_bar))
    parts.append(fig_to_html(fig_pie))
    parts.append(fig_to_html(fig_grouped))
    parts.append("</div>")

    if fig_tree is not None:
        parts.append('<div class="block"><h2>Répartition par client (global)</h2>')
        parts.append(fig_to_html(fig_tree))
        parts.append("</div>")

    # Détail par année : on ajoute au minimum l'analyse mensuelle (plus robuste pour un rapport)
    if years_in_filtered:
        parts.append('<div class="block"><h2>Détail par année (fichier)</h2>')
        for y in years_in_filtered:
            fy = df_filtered[df_filtered["__ref_year"] == y] if "__ref_year" in df_filtered.columns else df_filtered
            if len(fy) == 0:
                continue
            parts.append(f"<h3>Année fichier {y}</h3>")
            monthly, order_lbl = monthly_long_by_consultant(fy, date_cols, ref_year=y)
            if len(monthly) and pd.to_numeric(monthly["Minutes"], errors="coerce").fillna(0).sum() > 0:
                fig_m = px.bar(
                    monthly,
                    x="Mois_label",
                    y="Minutes",
                    color="Collaborateur",
                    color_discrete_sequence=COLOR_SEQUENCE,
                    title=f"Temps par mois — {y}",
                    category_orders={"Mois_label": order_lbl},
                )
                fig_m.update_layout(margin=dict(l=8, r=8, t=48, b=8))
                parts.append(fig_to_html(fig_m))
            else:
                parts.append("<div class='caption'>Pas assez de minutes pour l’analyse mensuelle.</div>")
        parts.append("</div>")

    parts.append("</body></html>")
    html = "\n".join(parts)
    return html.encode("utf-8")


def load_uploaded_file(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith(".xls") and not name.endswith(".xlsx"):
        try:
            return pd.read_excel(uploaded, engine="xlrd")
        except Exception:
            pass
    return pd.read_excel(uploaded, engine="openpyxl")


def normalize_and_concat(uploaded_list: list) -> pd.DataFrame:
    """Charge plusieurs fichiers et les empile (même schéma attendu ; colonnes réunies si besoin)."""
    frames = []
    for up in uploaded_list:
        df = load_uploaded_file(up)
        df["_Fichier_source"] = up.name
        y = extract_year_from_filenames([up.name])
        df["__ref_year"] = y if y is not None else pd.NA
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if "__ref_year" in out.columns:
        out["__ref_year"] = pd.to_numeric(out["__ref_year"], errors="coerce").astype("Int64")
    return out


def _ref_year_scalar(ry) -> Optional[int]:
    if ry is None or (isinstance(ry, float) and pd.isna(ry)):
        return None
    try:
        return int(ry)
    except (TypeError, ValueError):
        return None


def extract_year_from_filenames(names: list[str]) -> Optional[int]:
    """Première année 2000–2100 trouvée dans les noms de fichiers (ex. rapport_2025.xlsx)."""
    pat = re.compile(r"\b(19|20)\d{2}\b")
    for n in names:
        m = pat.search(n)
        if m:
            y = int(m.group(0))
            if 2000 <= y <= 2100:
                return y
    return None


# --- UI : style léger pour sections ---
st.markdown(
    """
<style>
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }
    h1 { letter-spacing: -0.02em; }
    .block-container { padding-top: 1.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Analyse des exports Zendesk")
st.caption(
    "Un ou plusieurs exports (CSV / Excel). Colonnes jour = minutes. "
    "Les lignes et colonnes « Total » / « Sum » sont exclues du calcul. "
    "Indiquez l’année dans le **nom du fichier** (ex. `activité_2025.xlsx`) pour l’analyse mensuelle."
)

st.sidebar.header("Persistance")
_saved_sessions = _list_saved_sessions()
_load_choice = st.sidebar.selectbox(
    "Charger une session sauvegardée",
    options=["—"] + _saved_sessions,
    index=0,
    help="Permet de retrouver les données même après fermeture de Streamlit (enregistrées sur disque).",
)

uploaded_list = st.file_uploader(
    "Fichier(s) Zendesk",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
    help="Sélectionnez plusieurs fichiers pour les cumuler (lignes empilées). Incluez l’année dans le nom du fichier si les colonnes jour n’en ont pas.",
)

raw = None
_source_files: list[str] = []
_loaded_meta = {}
if uploaded_list:
    try:
        raw = normalize_and_concat(uploaded_list)
        _source_files = [f.name for f in uploaded_list]
    except Exception as e:
        st.error(f"Impossible de lire les fichiers : {e}")
        st.stop()
elif _load_choice != "—":
    try:
        _loaded_meta = load_session_pickle(_load_choice)
        raw = _loaded_meta.get("df_num")
        _source_files = _loaded_meta.get("source_files") or [f"(session) {_load_choice}"]
    except Exception as e:
        st.error(f"Impossible de charger la session : {e}")
        st.stop()
else:
    st.info("Chargez un ou plusieurs fichiers, ou rechargez une session sauvegardée.")
    st.stop()

if raw.empty:
    st.warning("Les fichiers ne contiennent aucune ligne.")
    st.stop()

col_map = {c: str(c).strip() for c in raw.columns}
df = raw.rename(columns=col_map)

collab_col = None
for candidate in ("Collaborateur", "collaborateur", "Agent", "agent", "Assigné", "assignee"):
    if candidate in df.columns:
        collab_col = candidate
        break
if collab_col is None:
    st.error(
        "Colonne **Collaborateur** introuvable. Renommez une colonne en « Collaborateur » "
        "ou ajoutez « Agent » / « Assigné »."
    )
    st.stop()

df = df.rename(columns={collab_col: "Collaborateur"})
df["Collaborateur"] = clean_collaborateur(df["Collaborateur"])

df, n_rows_aggregate = filter_aggregate_rows(df)

if "Client" not in df.columns:
    for alt in ("Organisation", "organization", "Société", "Company"):
        if alt in df.columns:
            df = df.rename(columns={alt: "Client"})
            break
if "Client" not in df.columns:
    df["Client"] = "—"

if "Statut" not in df.columns and "Status" in df.columns:
    df = df.rename(columns={"Status": "Statut"})
if "Statut" not in df.columns:
    df["Statut"] = "—"

ticket_col = detect_ticket_column(df)
date_cols = detect_date_columns(df)
df_num = ensure_numeric_minutes(df, date_cols)
df_num["Total_Temps_Ticket"] = build_total_temps(df_num, date_cols)

st.sidebar.subheader("Sauvegarder la session")
_default_session_name = pd.Timestamp.now().strftime("rapport_%Y%m%d_%H%M%S")
_session_name = st.sidebar.text_input("Nom de session", value=_default_session_name)
if st.sidebar.button("Sauvegarder (sur disque)"):
    try:
        p = save_session_pickle(
            _session_name,
            df_num=df_num,
            date_cols=date_cols,
            ticket_col=ticket_col,
            source_files=_source_files,
        )
        st.sidebar.success(f"Session sauvegardée : {p.name}")
    except Exception as e:
        st.sidebar.error(f"Échec sauvegarde : {e}")

st.sidebar.header("Période")
_years_avail = (
    sorted({int(y) for y in df_num["__ref_year"].dropna().unique()})
    if "__ref_year" in df_num.columns
    else []
)
_na_files = (
    bool(df_num["__ref_year"].isna().any()) if "__ref_year" in df_num.columns else False
)
_period_opts: list[str] = ["Toutes les années"] + [str(y) for y in _years_avail]
if _na_files:
    _period_opts.append("Sans année dans le nom du fichier")
_period_sel = st.sidebar.selectbox(
    "Année du fichier (nom du fichier)",
    options=_period_opts,
    index=0,
    help="L’année est lue dans le nom de chaque fichier (ex. `temps_2025.xlsx`). "
    "Choisissez une année pour n’analyser que ce(s) fichier(s), ou « Toutes les années » pour tout cumuler.",
)
if _period_sel == "Toutes les années":
    _mask_period = pd.Series(True, index=df_num.index)
    REF_YEAR = min(_years_avail) if _years_avail else int(pd.Timestamp.today().year)
elif _period_sel == "Sans année dans le nom du fichier":
    _mask_period = df_num["__ref_year"].isna()
    REF_YEAR = int(pd.Timestamp.today().year)
else:
    _py = int(_period_sel)
    _mask_period = df_num["__ref_year"] == _py
    REF_YEAR = _py

ordered_date_cols = order_date_columns(date_cols, ref_year=REF_YEAR) if date_cols else []

if not date_cols:
    st.warning(
        "Aucune colonne de date détectée dans les en-têtes. "
        "Vérifiez que les jours sont en colonnes (ex. « 02 Jan ») ou que des colonnes numériques existent."
    )

st.sidebar.header("Filtres")
st.sidebar.caption(
    f"**{len(uploaded_list)}** fichier(s) · **{len(df_num)}** ligne(s) utile(s)"
    + (f" · *{n_rows_aggregate} ligne(s) Total/Sum exclue(s)*" if n_rows_aggregate else "")
)
if _years_avail:
    st.sidebar.caption("Années détectées dans les noms : **" + "**, **".join(str(y) for y in _years_avail) + "**")

collab_opts = sorted(df_num["Collaborateur"].dropna().astype(str).unique().tolist())
client_opts = sorted(df_num["Client"].dropna().astype(str).unique().tolist())
statut_opts = sorted(df_num["Statut"].dropna().astype(str).unique().tolist())

sel_collab = st.sidebar.multiselect("Collaborateur", options=collab_opts, default=collab_opts)
sel_client = st.sidebar.multiselect("Client", options=client_opts, default=client_opts)
sel_statut = st.sidebar.multiselect("Statut du ticket", options=statut_opts, default=statut_opts)

filtered = df_num[
    _mask_period
    & df_num["Collaborateur"].astype(str).isin(sel_collab)
    & df_num["Client"].astype(str).isin(sel_client)
    & df_num["Statut"].astype(str).isin(sel_statut)
].copy()

interventions = interventions_par_collaborateur(filtered, ticket_col)
temps_par_collab = filtered.groupby("Collaborateur", dropna=False)["Total_Temps_Ticket"].sum()
kpi_table = pd.DataFrame({"Temps_total": temps_par_collab, "Interventions": interventions})
kpi_table["Temps_moyen_par_ticket"] = kpi_table["Temps_total"] / kpi_table["Interventions"].replace(
    0, pd.NA
)
kpi_table = kpi_table.fillna(0)

temps_global = float(filtered["Total_Temps_Ticket"].sum())
if ticket_col and ticket_col in filtered.columns:
    nb_tickets_uniques = int(filtered[ticket_col].nunique())
else:
    nb_tickets_uniques = len(filtered)
temps_moyen = temps_global / nb_tickets_uniques if nb_tickets_uniques else 0.0

_years_in_filtered = (
    sorted({int(y) for y in filtered["__ref_year"].dropna().unique()})
    if "__ref_year" in filtered.columns
    else []
)
dt_min, dt_max = period_bounds_from_ordered_cols(ordered_date_cols, REF_YEAR)

# --- Vue globale (récapitulatif) ---
st.subheader("Vue globale — récapitulatif")
if dt_min is not None and dt_max is not None:
    st.caption(
        f"Période couverte (colonnes jour) : **{dt_min.strftime('%d/%m/%Y')}** → **{dt_max.strftime('%d/%m/%Y')}** · "
        f"**{len(ordered_date_cols)}** jour(s) · **{len(kpi_table)}** consultant(s)"
    )
else:
    st.caption(
        f"Période : certaines colonnes ne sont pas datées correctement. "
        f"**{len(ordered_date_cols)}** colonne(s) jour · **{len(kpi_table)}** consultant(s)"
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Temps global (min)", f"{temps_global:,.0f}".replace(",", " "))
c2.metric("Tickets (uniques)", f"{nb_tickets_uniques:,}".replace(",", " "))
c3.metric("Temps moyen / ticket (min)", f"{temps_moyen:,.1f}".replace(",", " "))
c4.metric(
    "Années (noms de fichiers)",
    str(len(_years_avail)) if _years_avail else ("1" if _period_sel != "Toutes les années" else "—"),
)

st.dataframe(
    kpi_table.reset_index().rename(
        columns={
            "Collaborateur": "Consultant",
            "Temps_total": "Temps global (min)",
            "Interventions": "Nb tickets",
            "Temps_moyen_par_ticket": "Temps moyen / ticket (min)",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

if len(kpi_table) == 0:
    st.warning("Aucune donnée avec les filtres actuels.")
    st.stop()

# --- Comparaison consultants (vue globale) ---
st.subheader("Vue globale — répartition du temps")

kpi_reset = kpi_table.reset_index().sort_values("Temps_total", ascending=True)

fig_bar = px.bar(
    kpi_reset,
    x="Temps_total",
    y="Collaborateur",
    orientation="h",
    color="Temps_total",
    color_continuous_scale="Tealgrn",
    labels={"Temps_total": "Minutes", "Collaborateur": ""},
    title="Temps total — classement",
)
fig_bar.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#161b22",
    font=dict(color="#e6edf3"),
    showlegend=False,
    yaxis=dict(categoryorder="total ascending"),
    height=max(380, 32 * len(kpi_table)),
    margin=dict(l=8, r=8, t=48, b=8),
)
fig_bar.update_traces(marker_line_width=0)

pie_df = kpi_reset.sort_values("Temps_total", ascending=False)
fig_pie = px.pie(
    pie_df,
    values="Temps_total",
    names="Collaborateur",
    title="Part du temps par consultant",
    hole=0.45,
    color_discrete_sequence=COLOR_SEQUENCE,
)
fig_pie.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    font=dict(color="#e6edf3"),
    margin=dict(l=8, r=8, t=48, b=8),
    showlegend=True,
    legend=dict(orientation="v", yanchor="middle", y=0.5),
)

cbar, cpie = st.columns([1.2, 1])
cbar.plotly_chart(fig_bar, use_container_width=True)
cpie.plotly_chart(fig_pie, use_container_width=True)

# Grouped metrics : temps vs moyenne par ticket
comp = kpi_reset.sort_values("Temps_total", ascending=False)
fig_grouped = go.Figure()
fig_grouped.add_trace(
    go.Bar(
        name="Temps total (min)",
        x=comp["Collaborateur"],
        y=comp["Temps_total"],
        marker_color="#3fb950",
        offsetgroup=1,
    )
)
fig_grouped.add_trace(
    go.Bar(
        name="Temps moyen / ticket (min)",
        x=comp["Collaborateur"],
        y=comp["Temps_moyen_par_ticket"],
        marker_color="#58a6ff",
        offsetgroup=2,
    )
)
fig_grouped.update_layout(
    title="Comparaison directe : temps total vs temps moyen par ticket",
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#161b22",
    font=dict(color="#e6edf3"),
    barmode="group",
    xaxis_tickangle=-35,
    height=460,
    margin=dict(l=8, r=8, t=56, b=8),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_grouped, use_container_width=True)

# --- Vue globale : clients (toute la période filtrée) ---
st.subheader("Vue globale — répartition par client")
treemap_df = (
    filtered.groupby(["Client", "Collaborateur"], dropna=False)["Total_Temps_Ticket"]
    .sum()
    .reset_index()
)
treemap_df = treemap_df[treemap_df["Total_Temps_Ticket"] > 0]
if len(treemap_df):
    fig_tree = px.treemap(
        treemap_df,
        path=["Client", "Collaborateur"],
        values="Total_Temps_Ticket",
        color="Collaborateur",
        color_discrete_sequence=COLOR_SEQUENCE,
        title="Treemap — Client × Consultant (toute période)",
    )
    fig_tree.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        font=dict(color="#e6edf3"),
        margin=dict(l=8, r=8, t=48, b=8),
    )
    st.plotly_chart(fig_tree, use_container_width=True)
else:
    st.info("Pas de temps par client à afficher (treemap).")
    fig_tree = None

st.markdown("---")
st.subheader("Exporter un rapport")
_report_bytes = build_report_html(
    title="Rapport — Analyse exports Zendesk",
    period_label=_period_sel,
    df_filtered=filtered,
    kpi_table=kpi_table,
    temps_global=temps_global,
    nb_tickets_uniques=nb_tickets_uniques,
    temps_moyen=temps_moyen,
    fig_bar=fig_bar,
    fig_pie=fig_pie,
    fig_grouped=fig_grouped,
    fig_tree=fig_tree,
    years_in_filtered=_years_in_filtered,
    date_cols=date_cols,
    ticket_col=ticket_col,
)
st.download_button(
    "Télécharger le rapport (HTML)",
    data=_report_bytes,
    file_name=f"rapport_zendesk_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html",
    mime="text/html",
    help="Ouvrez le fichier HTML dans un navigateur, puis imprimez-le en PDF si besoin.",
)

st.markdown("---")
st.subheader("Détail par millésime (année dans le nom du fichier)")

if not date_cols or not len(filtered):
    st.info("Colonnes jour absentes ou aucune ligne : pas d’analyse calendrier.")
elif not _years_in_filtered:
    st.warning(
        "Aucune année détectée dans les **noms de fichiers**. "
        "Incluez une année dans chaque nom (ex. `export_2025.xlsx`) pour séparer les périodes."
    )
    st.caption("Analyse mensuelle de secours (année courante pour les en-têtes sans année) :")
    render_monthly_charts_streamlit(filtered, date_cols, ref_year=REF_YEAR)
else:
    for _year in _years_in_filtered:
        cols_y = date_cols
        ordered_y = order_date_columns(date_cols, ref_year=_year)
        fy = filtered[filtered["__ref_year"] == _year]
        if len(fy) == 0:
            continue
        render_year_detail_charts(
            fy,
            cols_y,
            ordered_y,
            ref_year=_year,
            title=f"Année fichier {_year}",
            ticket_col=ticket_col,
        )

with st.expander("Détail technique (colonnes / exclusions)"):
    st.caption(
        "Colonnes « Total » dans les **en-têtes** : ignorées pour le calcul. "
        "Lignes dont le **consultant** ressemble à Total / Sum / Somme : exclues."
    )
    st.write("Fichiers :", [f.name for f in uploaded_list])
    st.write("Colonnes jour (minutes) :", date_cols or "—")
    st.write("Colonne ticket :", ticket_col or "— (une ligne = une intervention)")
    st.write("Lignes exclues (Total/Sum) :", n_rows_aggregate)
