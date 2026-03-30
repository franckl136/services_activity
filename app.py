import hashlib
import html
import io
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


def fmt_fr(x: float, decimals: int = 2) -> str:
    """Format nombre : espaces pour milliers, décimales fixes (lisibilité)."""
    return f"{float(x):,.{decimals}f}".replace(",", " ")

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


def _is_excluded_column(name: str) -> bool:
    return str(name).lower().strip() in METADATA_NAMES


def _is_reserved_data_column(name: str) -> bool:
    """Colonnes qui ne sont pas des minutes par jour (métadonnées app, totaux export, etc.)."""
    s = str(name).lower().strip()
    if s in ("__ref_year", "_ref_year", "_fichier_source", "ref_year"):
        return True
    if s.startswith("_fichier"):
        return True
    # Totaux / cumuls souvent en fin de tableau (écrasent l’échelle des heatmaps)
    if s in ("total", "totale", "totaux", "cumul", "grand total", "total annuel", "cumul annuel"):
        return True
    if re.match(r"^(total|somme|sum|cumul|subtotal)\b", s, re.I):
        return True
    return False


def detect_date_columns(df: pd.DataFrame) -> list[str]:
    date_cols: list[str] = []
    for c in df.columns:
        if _is_excluded_column(str(c)) or _is_reserved_data_column(str(c)):
            continue
        if _column_name_is_date(str(c)):
            date_cols.append(c)
            continue
    if date_cols:
        return date_cols
    for c in df.columns:
        if _is_excluded_column(str(c)) or _is_reserved_data_column(str(c)):
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


def interventions_par_collaborateur(df: pd.DataFrame) -> pd.Series:
    return df.groupby("Collaborateur", dropna=False).size()


def interventions_par_client(df: pd.DataFrame) -> pd.Series:
    return df.groupby("Client", dropna=False).size()


def build_client_executive_table(filtered: pd.DataFrame) -> pd.DataFrame:
    """Classement clients : temps, lignes d’activité, charge relative (synthèse direction)."""
    if filtered.empty or "Client" not in filtered.columns:
        return pd.DataFrame()
    temps = filtered.groupby("Client", dropna=False)["Total_Temps_Ticket"].sum()
    inter = interventions_par_client(filtered)
    inter = inter.reindex(temps.index, fill_value=0)
    out = pd.DataFrame(
        {
            "Temps_total_min": temps,
            "Nb_lignes": inter.astype(int),
        }
    )
    out["Temps_moyen_par_ligne"] = out["Temps_total_min"] / out["Nb_lignes"].replace(0, pd.NA)
    out = out.fillna(0.0)
    tot = float(out["Temps_total_min"].sum())
    if tot > 0:
        out["Part_temps_pct"] = 100.0 * out["Temps_total_min"] / tot
        out = out.sort_values("Temps_total_min", ascending=False)
        out["Part_cumul_pct"] = out["Part_temps_pct"].cumsum()
    else:
        out["Part_temps_pct"] = 0.0
        out["Part_cumul_pct"] = 0.0
        out = out.sort_values("Temps_total_min", ascending=False)
    out.insert(0, "Rang", range(1, len(out) + 1))
    return out


def detect_subject_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl in ("sujet", "subject", "title", "titre", "résumé", "resume", "summary"):
            return c
    return None


def categorie_from_sujet(text) -> str:
    """Catégorie selon mots-clés dans le sujet (ordre : ADMIN, PARAM, TECH, sinon AUTRE)."""
    if pd.isna(text):
        return "AUTRE"
    t = str(text).lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    admin_kw = (
        "mdp",
        "session",
        "connexion",
        "acces",
        "identifiant",
        "profil",
        "creation",
        "cession",
        "utilisateur",
        "mot de passe",
        "inacessible",
        "inaccessible",
    )
    param_kw = (
        "parametrage",
        "cycle",
        "regle",
        "bdese",
        "mutation",
        "badge",
        "badgeuse",
        "calcul",
        "analytique",
        "pointage",
    )
    tech_kw = (
        "erreur",
        "fichier",
        "extraction",
        "rejet",
        "bug",
        "anomalie",
        "systeme",
        "integration",
        "impossibilite",
        "impossible",
    )
    for kw in admin_kw:
        if kw in t:
            return "ADMIN-ACCES"
    for kw in param_kw:
        if kw in t:
            return "PARAM"
    for kw in tech_kw:
        if kw in t:
            return "TECH"
    return "AUTRE"


def coef_categorie(cat: str) -> int:
    return {"ADMIN-ACCES": 1, "PARAM": 2, "TECH": 3, "AUTRE": 1}.get(cat, 1)


def enrich_complexity_columns(df: pd.DataFrame, subject_col: Optional[str]) -> pd.DataFrame:
    """Ajoute Categorie, Coef_complexite, Score_Complexite (temps × coefficient)."""
    out = df.copy()
    if subject_col and subject_col in out.columns:
        out["Categorie"] = out[subject_col].map(categorie_from_sujet)
    else:
        out["Categorie"] = "AUTRE"
    out["Coef_complexite"] = out["Categorie"].map(coef_categorie)
    out["Score_Complexite"] = pd.to_numeric(out["Total_Temps_Ticket"], errors="coerce").fillna(0) * out[
        "Coef_complexite"
    ]
    return out


def truncate_text(s: str, max_len: int = 40) -> str:
    s = (s or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def detect_ticket_column(df: pd.DataFrame) -> Optional[str]:
    """Colonne identifiant ticket (priorité aux noms usuels)."""
    priority = ("TICKET", "Ticket", "#", "ticket", "Ticket ID", "Ticket id", "ID ticket", "Id ticket")
    for c in priority:
        if c in df.columns:
            return c
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl in ("ticket id", "ticket_id", "id ticket", "n° ticket", "no ticket", "numero ticket"):
            return c
    return None


def build_complexity_treemap_hierarchy(
    df: pd.DataFrame,
    subject_col: Optional[str],
    ticket_col: Optional[str],
    sujet_truncate: int = 42,
) -> pd.DataFrame:
    """
    Une ligne par feuille (catégorie × consultant × ticket) pour px.treemap path à 3 niveaux.
    Libellé court sur la tuile ; détail complet via customdata (hover).
    """
    if df.empty or "Categorie" not in df.columns:
        return pd.DataFrame()
    base = df.copy()
    if ticket_col and ticket_col in base.columns:
        base["_tid"] = base[ticket_col].map(lambda x: str(x).strip() if pd.notna(x) else "")
    else:
        base["_tid"] = ""
    empty_tid = base["_tid"].astype(str).str.len() == 0
    if empty_tid.any():
        # Numéro stable par ligne d'export si pas d'ID ticket
        base.loc[empty_tid, "_tid"] = base.loc[empty_tid].index.map(lambda i: f"L{i+1}")

    if subject_col and subject_col in base.columns:
        base["_sujet"] = base[subject_col].fillna("").astype(str)
    else:
        base["_sujet"] = ""

    g = (
        base.groupby(["Categorie", "Collaborateur", "_tid"], dropna=False)
        .agg(
            Temps=("Total_Temps_Ticket", "sum"),
            SC=("Score_Complexite", "sum"),
            _sujet=("_sujet", "first"),
        )
        .reset_index()
    )
    g = g[g["Temps"] > 0]
    if g.empty:
        return g
    g["Score_moy"] = (g["SC"] / g["Temps"]).fillna(0)
    g["Ticket_label"] = g["_tid"].astype(str) + " — " + g["_sujet"].map(lambda s: truncate_text(s, sujet_truncate))
    _dup = g.groupby(["Categorie", "Collaborateur", "Ticket_label"], dropna=False).cumcount()
    _m = _dup > 0
    if _m.any():
        g.loc[_m, "Ticket_label"] = g.loc[_m, "Ticket_label"] + " (" + _dup[_m].astype(str) + ")"
    return g


def coloraxis_upper_for_outliers(
    s: pd.Series,
    q: float = 0.90,
    tail_ratio: float = 1.06,
) -> Optional[float]:
    """
    Plafond d'échelle continue pour limiter l'écrasement visuel par quelques outliers.
    Si max ≤ quantile(q) × tail_ratio, renvoie None (échelle naturelle).
    """
    v = pd.to_numeric(s, errors="coerce").dropna()
    if len(v) < 2:
        return None
    cap = float(v.quantile(q))
    mx = float(v.max())
    if mx <= cap * tail_ratio or cap <= 0:
        return None
    return cap


def build_complexity_treemap_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Vue synthèse : catégorie × consultant (temps total, score moyen, temps moyen / ligne)."""
    if df.empty or "Categorie" not in df.columns:
        return pd.DataFrame()
    g = (
        df.groupby(["Categorie", "Collaborateur"], dropna=False)
        .agg(
            Temps=("Total_Temps_Ticket", "sum"),
            SC=("Score_Complexite", "sum"),
            Lignes=("Total_Temps_Ticket", "size"),
        )
        .reset_index()
    )
    g = g[g["Temps"] > 0]
    if g.empty:
        return g
    g["Score_moy"] = (g["SC"] / g["Temps"]).fillna(0)
    g["Temps_moyen_ligne"] = (g["Temps"] / g["Lignes"].replace(0, pd.NA)).fillna(0.0)
    return g


def apply_treemap_complexity_ticket_hover(fig: go.Figure) -> None:
    """Infobulles : détail ticket/sujet/temps sur les feuilles ; agrégats = libellé + temps."""
    tr = fig.data[0]
    _trace_kw: dict = {
        "texttemplate": "<b>%{label}</b><br>%{value:.0f} min",
        "textfont": dict(size=10),
    }
    if tr.customdata is not None and tr.labels is not None and tr.values is not None:
        _hts: list[str] = []
        for i in range(len(tr.labels)):
            cd = tr.customdata[i]
            lab = tr.labels[i]
            val = float(tr.values[i])
            if cd is not None and len(cd) >= 3 and str(cd[0]) != "(?)":
                _hts.append(
                    f"<b>Ticket</b> {cd[0]}<br><b>Sujet</b> {cd[1]}<br><b>Temps</b> {float(cd[2]):.2f} min"
                )
            else:
                _hts.append(f"<b>{html.escape(str(lab))}</b><br>Temps : {val:.1f} min")
        _trace_kw["hovertext"] = _hts
        _trace_kw["hovertemplate"] = "%{text}<extra></extra>"
    fig.update_traces(**_trace_kw)


def pareto_80_client_share(client_exec: pd.DataFrame) -> tuple[Optional[int], Optional[float]]:
    """Nombre de clients (tri décroissant temps) pour atteindre 80 % du temps ; % du total de clients."""
    if client_exec.empty:
        return None, None
    tot = float(client_exec["Temps_total_min"].sum())
    if tot <= 0:
        return None, None
    s = client_exec.sort_values("Temps_total_min", ascending=False)
    cum = s["Temps_total_min"].cumsum() / tot * 100.0
    n = len(s)
    for i, v in enumerate(cum.values):
        if v >= 80:
            n = i + 1
            break
    n_clients = len(s)
    pct = 100.0 * n / n_clients if n_clients else 0.0
    return n, pct


def workload_minutes_by_weekday(
    filtered: pd.DataFrame, date_cols: list[str], ref_year: Optional[int]
) -> pd.DataFrame:
    """Somme des minutes par jour de la semaine (colonnes jour → date)."""
    if not date_cols or filtered.empty:
        return pd.DataFrame(columns=["jour_sem", "ordre", "minutes"])
    wd_fr = {
        0: "Lundi",
        1: "Mardi",
        2: "Mercredi",
        3: "Jeudi",
        4: "Vendredi",
        5: "Samedi",
        6: "Dimanche",
    }
    acc: dict[int, float] = {i: 0.0 for i in range(7)}
    for c in date_cols:
        ts = parse_column_to_datetime(c, ref_year=ref_year)
        if pd.isna(ts):
            continue
        w = int(ts.weekday())
        acc[w] += float(pd.to_numeric(filtered[c], errors="coerce").fillna(0).sum())
    rows = [{"ordre": i, "jour_sem": wd_fr[i], "minutes": acc[i]} for i in range(7)]
    return pd.DataFrame(rows)


def count_statut_closed_like(series: pd.Series) -> int:
    """Heuristique : lignes dont le statut évoque une clôture / résolution."""
    keys = (
        "clos",
        "closed",
        "resolu",
        "ferme",
        "complete",
        "termin",
        "solved",
    )
    s = series.astype(str).str.lower()
    return int(s.map(lambda x: any(k in x for k in keys)).sum())


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
    df: pd.DataFrame, date_cols_subset: list[str]
) -> pd.Series:
    """Lignes avec au moins une minute sur le sous-ensemble de colonnes, par consultant."""
    if not date_cols_subset:
        return pd.Series(dtype=float)
    df2 = ensure_numeric_minutes(df, date_cols_subset)
    row_mask = df2[date_cols_subset].sum(axis=1) > 0
    sub = df2.loc[row_mask]
    if len(sub) == 0:
        return pd.Series(dtype=float)
    return sub.groupby("Collaborateur", dropna=False).size()


def kpi_table_for_date_cols(
    filtered: pd.DataFrame, date_cols_subset: list[str]
) -> tuple[pd.DataFrame, float, int, float]:
    """KPI par consultant pour un sous-ensemble de colonnes jour."""
    if not date_cols_subset:
        kpi = pd.DataFrame()
        return kpi, 0.0, 0, 0.0
    df2 = ensure_numeric_minutes(filtered, date_cols_subset).copy()
    df2["_t"] = df2[date_cols_subset].sum(axis=1, numeric_only=True)
    inter = interventions_for_date_subset(filtered, date_cols_subset)
    tpc = df2.groupby("Collaborateur", dropna=False)["_t"].sum()
    inter = inter.reindex(tpc.index, fill_value=0)
    kpi = pd.DataFrame({"Temps_total": tpc, "Lignes": inter})
    kpi["Lignes"] = kpi["Lignes"].fillna(0).astype(int)
    kpi["Temps_moyen_par_ligne"] = kpi["Temps_total"] / kpi["Lignes"].replace(0, pd.NA)
    kpi = kpi.fillna(0)
    tot = float(df2["_t"].sum())
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
    _num = pivot_display.select_dtypes(include=["number"]).columns
    if len(_num):
        pivot_display[_num] = pivot_display[_num].round(2)
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
) -> None:
    """KPI, heatmap jour, lignes quotidiennes, cumul et analyse mensuelle pour un millésime."""
    # Exclure ref_year / totaux : sinon une seule colonne « annuelle » écrase l’échelle des couleurs.
    cols_y = [c for c in cols_y if not _is_reserved_data_column(str(c))]
    ordered_y = order_date_columns(cols_y, ref_year=ref_year)
    if not cols_y:
        st.warning("Aucune colonne jour exploitable après exclusion des totaux / métadonnées.")
        return

    st.markdown(f"### {title}")
    dt_a, dt_b = period_bounds_from_ordered_cols(ordered_y, ref_year=ref_year)
    if dt_a is not None and dt_b is not None:
        st.caption(
            f"Période : **{dt_a.strftime('%d/%m/%Y')}** → **{dt_b.strftime('%d/%m/%Y')}** · "
            f"**{len(cols_y)}** jour(s) en colonnes · **{len(fy)}** ligne(s)"
        )

    kpi_y, tot_y, nb_y, moy_y = kpi_table_for_date_cols(fy, cols_y)
    if len(kpi_y) == 0:
        st.warning("Pas de données agrégées pour cette année.")
        return

    y1, y2, y3 = st.columns(3)
    y1.metric("Temps (min)", fmt_fr(tot_y))
    y2.metric("Lignes (avec temps)", f"{nb_y:,}".replace(",", " "))
    y3.metric("Temps moyen / ligne (min)", fmt_fr(moy_y))

    _kpi_show = kpi_y.reset_index().rename(
        columns={
            "Collaborateur": "Consultant",
            "Temps_total": "Temps (min)",
            "Lignes": "Lignes (avec temps cette année)",
            "Temps_moyen_par_ligne": "Temps moyen / ligne (min)",
        }
    )
    _kpi_show["Temps (min)"] = _kpi_show["Temps (min)"].round(2)
    _kpi_show["Temps moyen / ligne (min)"] = _kpi_show["Temps moyen / ligne (min)"].round(2)
    st.dataframe(_kpi_show, use_container_width=True, hide_index=True)

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
    source_files: list[str],
) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", (session_name or "").strip())[:80]
    if not safe:
        safe = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "saved_at": pd.Timestamp.now().isoformat(),
        "source_files": source_files,
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
    nb_lignes: int,
    temps_moyen: float,
    fig_bar,
    fig_pie,
    fig_grouped,
    fig_tree,
    years_in_filtered: list[int],
    date_cols: list[str],
) -> bytes:
    _kpi_rpt = kpi_table.reset_index().rename(
        columns={
            "Collaborateur": "Consultant",
            "Temps_total": "Temps global (min)",
            "Lignes": "Lignes (export)",
            "Temps_moyen_par_ligne": "Temps moyen / ligne (min)",
        }
    )
    _kpi_rpt["Temps global (min)"] = _kpi_rpt["Temps global (min)"].round(2)
    _kpi_rpt["Temps moyen / ligne (min)"] = _kpi_rpt["Temps moyen / ligne (min)"].round(2)
    kpi_html = _kpi_rpt.to_html(index=False, escape=False)

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
    <div class="kpi"><div class="label">Temps global (min)</div><div class="value">{fmt_fr(temps_global)}</div></div>
    <div class="kpi"><div class="label">Lignes (export)</div><div class="value">{nb_lignes:,}</div></div>
    <div class="kpi"><div class="label">Temps moyen / ligne (min)</div><div class="value">{fmt_fr(temps_moyen)}</div></div>
    <div class="kpi"><div class="label">Années (noms de fichiers)</div><div class="value">{len(years_in_filtered) if years_in_filtered else 0}</div></div>
  </div>
"""
    )
    parts.append('<div class="block"><h2>Table KPI (consultants)</h2>' + kpi_html + "</div>")

    parts.append('<div class="block"><h2>Répartition du temps (global)</h2>')
    parts.append('<div class="caption">Barres, camembert, et comparaison temps total vs temps moyen par ligne.</div>')
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


def _load_uploaded_file_bytes(name: str, content: bytes) -> pd.DataFrame:
    """Lit un export depuis des octets (CSV / Excel)."""
    name_l = name.lower()
    buf = io.BytesIO(content)
    if name_l.endswith(".csv"):
        try:
            return pd.read_csv(buf, engine="pyarrow")
        except Exception:
            buf.seek(0)
            return pd.read_csv(buf, low_memory=False)
    if name_l.endswith(".xls") and not name_l.endswith(".xlsx"):
        try:
            return pd.read_excel(buf, engine="xlrd")
        except Exception:
            buf.seek(0)
            return pd.read_excel(buf, engine="openpyxl")
    return pd.read_excel(buf, engine="openpyxl")


@st.cache_data(show_spinner="Lecture des fichiers…", max_entries=64)
def _load_uploaded_file_cached(name: str, content_sha256: str, content: bytes) -> pd.DataFrame:
    """Cache par (nom + hash) : les reruns Streamlit ne relisent pas Excel/CSV."""
    return _load_uploaded_file_bytes(name, content)


def normalize_and_concat(uploaded_list: list) -> pd.DataFrame:
    """Charge plusieurs fichiers et les empile (même schéma attendu ; colonnes réunies si besoin)."""
    frames = []
    for up in uploaded_list:
        raw_bytes = up.getvalue()
        h = hashlib.sha256(raw_bytes).hexdigest()
        df = _load_uploaded_file_cached(up.name, h, raw_bytes)
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

st.sidebar.divider()
with st.sidebar.expander("Score de complexité (méthode)"):
    st.markdown(
        """
**Catégories** (mots-clés dans la colonne *Sujet*, sans tenir compte de la casse) :

- **ADMIN-ACCES** : mdp, session, connexion, accès, identifiant, profil, création, cession, utilisateur, mot de passe, inacessible / inaccessible  
- **PARAM** : paramétrage, cycle, règle, bdese, mutation, badge, badgeuse, calcul, analytique, pointage  
- **TECH** : erreur, fichier, extraction, rejet, bug, anomalie, système, intégration, impossibilité, impossible  
- **AUTRE** : tout le reste (ou absence de colonne sujet)

**Score** = `Temps (min) × coefficient`, avec **ADMIN=1**, **PARAM=2**, **TECH=3**, **AUTRE=1**.

Sur le treemap **synthèse** (catégorie × consultant), la couleur représente le **temps moyen par ligne** (des tickets plus longs en moyenne apparaissent plus « chauds »). Sur le **détail tickets**, la couleur utilise le **score** `Temps × coefficient`.
        """
    )

heures_presence = st.sidebar.number_input(
    "Heures de présence équipe (période, optionnel)",
    min_value=0.0,
    value=0.0,
    step=1.0,
    help="Saisissez le volume d’heures travaillées sur la période pour estimer un ratio "
    "« lignes à statut clôturé / résolu » par heure (approximatif).",
)

subject_col = detect_subject_column(df_num)
ticket_col = detect_ticket_column(df_num)
filtered = df_num[
    _mask_period
    & df_num["Collaborateur"].astype(str).isin(sel_collab)
    & df_num["Client"].astype(str).isin(sel_client)
    & df_num["Statut"].astype(str).isin(sel_statut)
].copy()
filtered = enrich_complexity_columns(filtered, subject_col)

interventions = interventions_par_collaborateur(filtered)
temps_par_collab = filtered.groupby("Collaborateur", dropna=False)["Total_Temps_Ticket"].sum()
med_par_ligne = filtered.groupby("Collaborateur", dropna=False)["Total_Temps_Ticket"].median()
kpi_table = pd.DataFrame({"Temps_total": temps_par_collab, "Lignes": interventions})
kpi_table["Temps_mediane_ligne"] = med_par_ligne.reindex(kpi_table.index).fillna(0)
kpi_table["Temps_moyen_par_ligne"] = kpi_table["Temps_total"] / kpi_table["Lignes"].replace(0, pd.NA)
kpi_table = kpi_table.fillna(0)

temps_global = float(filtered["Total_Temps_Ticket"].sum())
nb_lignes = len(filtered)
temps_moyen = temps_global / nb_lignes if nb_lignes else 0.0
temps_mediane_ligne_global = float(filtered["Total_Temps_Ticket"].median()) if nb_lignes else 0.0

client_exec = build_client_executive_table(filtered)
n_clients_pareto80, pct_clients_pareto80 = pareto_80_client_share(client_exec)
nb_lignes_closed = count_statut_closed_like(filtered["Statut"]) if "Statut" in filtered.columns else 0
ratio_closed_par_h = (nb_lignes_closed / heures_presence) if heures_presence > 0 else None

var_n1_pct: Optional[float] = None
_yref = None
if "__ref_year" in filtered.columns and len(filtered["__ref_year"].dropna()):
    yt = filtered.groupby("__ref_year", dropna=False)["Total_Temps_Ticket"].sum()
    _years_sorted = sorted(int(y) for y in yt.index if pd.notna(y))
    if len(_years_sorted) >= 2:
        y_max = _years_sorted[-1]
        y_prev = y_max - 1
        yt_by_year = {int(k): float(v) for k, v in yt.items()}
        if y_max in yt_by_year and y_prev in yt_by_year:
            t_y = yt_by_year[y_max]
            t_y1 = yt_by_year[y_prev]
            if t_y1 > 0:
                var_n1_pct = 100.0 * (t_y - t_y1) / t_y1
                _yref = (y_max, y_prev)

_years_in_filtered = (
    sorted({int(y) for y in filtered["__ref_year"].dropna().unique()})
    if "__ref_year" in filtered.columns
    else []
)
dt_min, dt_max = period_bounds_from_ordered_cols(ordered_date_cols, REF_YEAR)

if len(kpi_table) == 0:
    st.warning("Aucune donnée avec les filtres actuels.")
    st.stop()

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
c1.metric("Temps global (min)", fmt_fr(temps_global))
c2.metric("Lignes (export)", f"{nb_lignes:,}".replace(",", " "))
c3.metric("Temps moyen / ligne (min)", fmt_fr(temps_moyen))
c4.metric(
    "Années (noms de fichiers)",
    str(len(_years_avail)) if _years_avail else ("1" if _period_sel != "Toutes les années" else "—"),
)

_kpi_global = kpi_table.reset_index().rename(
    columns={
        "Collaborateur": "Consultant",
        "Temps_total": "Temps global (min)",
        "Lignes": "Lignes (export)",
        "Temps_mediane_ligne": "Temps médiane / ligne (min)",
        "Temps_moyen_par_ligne": "Temps moyen / ligne (min)",
    }
)
_kpi_global["Temps global (min)"] = _kpi_global["Temps global (min)"].round(2)
_kpi_global["Temps médiane / ligne (min)"] = _kpi_global["Temps médiane / ligne (min)"].round(2)
_kpi_global["Temps moyen / ligne (min)"] = _kpi_global["Temps moyen / ligne (min)"].round(2)
st.dataframe(_kpi_global, use_container_width=True, hide_index=True)

e1, e2, e3, e4 = st.columns(4)
e1.metric("Temps médian / ligne (global)", fmt_fr(temps_mediane_ligne_global))
if n_clients_pareto80 is not None and pct_clients_pareto80 is not None:
    e2.metric(
        "Pareto 80 % — concentration client",
        f"{n_clients_pareto80} clients",
        delta=f"{pct_clients_pareto80:.1f} % des clients pour 80 % du temps",
    )
else:
    e2.metric("Pareto 80 %", "—")
if var_n1_pct is not None and _yref is not None:
    e3.metric(
        f"Activité vs {_yref[1]} (année {_yref[0]})",
        f"{var_n1_pct:+.2f} %",
        delta="temps total (sélection)",
    )
else:
    e3.metric("Comparaison N-1", "—")
if ratio_closed_par_h is not None:
    e4.metric(
        "Lignes statut « clôturé » / heure",
        f"{ratio_closed_par_h:.2f} / h",
        delta=f"{nb_lignes_closed} lignes (heuristique) / {heures_presence:.1f} h",
    )
else:
    e4.metric("Lignes « clôturé » / heure", "—")

st.markdown("##### Treemap — complexité (catégorie × consultant)")
st.caption(
    "Taille = temps total (min). **Couleur = temps moyen par ligne** : RdYlGn_r (rouge = tickets plus longs en moyenne). "
    "L’échelle de couleur est **bornée au 90e percentile** lorsqu’un résidu (ex. une moyenne ~600 min) écrase le reste — "
    "la **valeur exacte** reste dans l’infobulle. Détail tickets ci-dessous au choix."
)
if len(filtered) and "Categorie" in filtered.columns:
    _ov = build_complexity_treemap_overview(filtered)
    if len(_ov):
        fig_cat = px.treemap(
            _ov,
            path=["Categorie", "Collaborateur"],
            values="Temps",
            color="Temps_moyen_ligne",
            color_continuous_scale="RdYlGn_r",
            labels={
                "Temps": "Minutes (total)",
                "Temps_moyen_ligne": "Temps moyen / ligne (min)",
                "Score_moy": "Coef. moyen (pondéré)",
            },
            title="Répartition du temps par catégorie et consultant",
            hover_data={
                "Score_moy": ":.2f",
                "Temps_moyen_ligne": ":.1f",
                "Lignes": True,
            },
        )
        fig_cat.update_traces(
            texttemplate="<b>%{label}</b><br>%{value:.0f} min",
            textfont=dict(size=11),
        )
        _lay_cat: dict = dict(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            font=dict(color="#e6edf3"),
            margin=dict(l=8, r=8, t=48, b=8),
        )
        _cap_tml = coloraxis_upper_for_outliers(_ov["Temps_moyen_ligne"], q=0.90)
        if _cap_tml is not None:
            _lay_cat["coloraxis"] = dict(
                cmin=0,
                cmax=_cap_tml,
                colorbar=dict(
                    title=dict(
                        text=f"Temps moy. / ligne (min)<br><sup>max affiché ≈ {_cap_tml:.0f} (P90)</sup>",
                    )
                ),
            )
        fig_cat.update_layout(**_lay_cat)
        st.plotly_chart(fig_cat, use_container_width=True)

        _seen_pairs: set[tuple[str, str]] = set()
        _pair_list: list[tuple[str, str]] = []
        for _, _r in _ov.iterrows():
            _p = (str(_r["Categorie"]), str(_r["Collaborateur"]))
            if _p not in _seen_pairs:
                _seen_pairs.add(_p)
                _pair_list.append(_p)
        _detail_choice = st.selectbox(
            "Détail des tickets (consultant)",
            options=[None] + _pair_list,
            format_func=lambda x: "—" if x is None else f"{x[0]} — {x[1]}",
            key="treemap_ticket_detail_pair",
            help="Équivalent à « zoomer » sur un consultant : les tickets s’affichent uniquement ici.",
        )
        if _detail_choice is not None:
            _dcat, _dcollab = _detail_choice
            _sub = filtered[
                (filtered["Categorie"].astype(str) == _dcat)
                & (filtered["Collaborateur"].astype(str) == _dcollab)
            ]
            _tix = build_complexity_treemap_hierarchy(_sub, subject_col, ticket_col)
            if len(_tix):
                _tg = _tix.copy()
                _tg["Score_Complexite"] = _tg["SC"]
                _tg["tid_h"] = _tg["_tid"].astype(str).map(html.escape)
                _tg["sujet_h"] = _tg["_sujet"].map(lambda s: html.escape(str(s)))
                fig_tix = px.treemap(
                    _tg,
                    path=["Ticket_label"],
                    values="Temps",
                    color="Score_Complexite",
                    color_continuous_scale="RdYlGn_r",
                    labels={"Temps": "Minutes", "Score_Complexite": "Score (min × coef.)"},
                    title=f"Tickets — {_dcat} — {_dcollab}",
                    custom_data=["tid_h", "sujet_h", "Temps"],
                )
                apply_treemap_complexity_ticket_hover(fig_tix)
                _lay_tix: dict = dict(
                    template="plotly_dark",
                    paper_bgcolor="#0e1117",
                    font=dict(color="#e6edf3"),
                    margin=dict(l=8, r=8, t=48, b=8),
                    height=520,
                )
                _cap_sc = coloraxis_upper_for_outliers(_tg["Score_Complexite"], q=0.90)
                if _cap_sc is not None:
                    _lay_tix["coloraxis"] = dict(
                        cmin=0,
                        cmax=_cap_sc,
                        colorbar=dict(
                            title=dict(
                                text=f"Score (min×coef.)<br><sup>max affiché ≈ {_cap_sc:.0f} (P90)</sup>",
                            )
                        ),
                    )
                fig_tix.update_layout(**_lay_tix)
                st.plotly_chart(fig_tix, use_container_width=True)
            else:
                st.caption("Aucun ticket à afficher pour cette sélection.")
    else:
        st.info("Pas de temps par catégorie à afficher.")
else:
    st.info("Données insuffisantes pour le treemap de complexité.")

_dcols_wd = [c for c in date_cols if not _is_reserved_data_column(str(c))]
st.markdown("##### Charge par jour de la semaine (somme des minutes)")
wd_df = workload_minutes_by_weekday(filtered, _dcols_wd, REF_YEAR)
if len(wd_df) and wd_df["minutes"].sum() > 0:
    fig_wd = px.bar(
        wd_df,
        x="jour_sem",
        y="minutes",
        category_orders={"jour_sem": list(wd_df["jour_sem"])},
        labels={"minutes": "Minutes (cumul)", "jour_sem": "Jour"},
        title="Répartition des minutes par jour de la semaine (tous consultants)",
        color="minutes",
        color_continuous_scale="Viridis",
    )
    fig_wd.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        showlegend=False,
        height=380,
        margin=dict(l=8, r=8, t=48, b=8),
    )
    st.plotly_chart(fig_wd, use_container_width=True)
    st.caption("Chaque colonne-jour du fichier est rattachée à un jour calendaire ; les minutes sont sommées par jour de la semaine (Lundi–Dimanche).")
else:
    st.caption("Pas assez de colonnes jour datées pour la répartition hebdomadaire.")

if len(client_exec) and client_exec["Temps_total_min"].sum() > 0:
    st.markdown("##### Courbe de Pareto — part cumulée du temps (clients)")
    _ce = client_exec.sort_values("Temps_total_min", ascending=False)
    _tot_c = float(_ce["Temps_total_min"].sum())
    _cum = (_ce["Temps_total_min"].cumsum() / _tot_c * 100.0).reset_index(drop=True)
    _x = list(range(1, len(_cum) + 1))
    fig_p80 = go.Figure()
    fig_p80.add_trace(
        go.Scatter(
            x=_x,
            y=_cum.values,
            mode="lines+markers",
            name="Part cumulée (%)",
            line=dict(color="#58a6ff", width=2),
            marker=dict(size=5),
        )
    )
    fig_p80.add_hline(y=80, line_dash="dash", line_color="#f85149", annotation_text="80 %")
    fig_p80.update_layout(
        title="Nombre de clients vs part cumulée du temps (le point sur la ligne rouge ≈ règle des 80 %)",
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        xaxis_title="Rang du client (du plus chargé au moins chargé)",
        yaxis_title="Part cumulée du temps (%)",
        height=400,
        margin=dict(l=8, r=8, t=56, b=8),
    )
    st.plotly_chart(fig_p80, use_container_width=True)

st.markdown("---")

# --- Synthèse direction (clients, charge, concentration) ---
st.subheader("Synthèse direction — clients & charge")
st.caption(
    "Indicateurs calculés sur la **sélection actuelle** (période, filtres client / consultant / statut). "
    "Les parts se rapportent au volume de temps total de cette sélection."
)

if client_exec.empty or temps_global <= 0:
    st.info("Pas assez de temps enregistré pour une analyse par client.")
else:
    n_clients_total = len(client_exec)
    n_clients_actifs = int((client_exec["Temps_total_min"] > 0).sum())
    top3_share = float(client_exec.head(3)["Part_temps_pct"].sum()) if n_clients_actifs else 0.0
    top5_share = float(client_exec.head(5)["Part_temps_pct"].sum()) if n_clients_actifs else 0.0
    client_sans_nom = int((client_exec.index.astype(str).str.strip().isin(("—", "-", ""))).sum())

    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Clients distincts", f"{n_clients_total:,}".replace(",", " "))
    d2.metric("Clients avec temps > 0", f"{n_clients_actifs:,}".replace(",", " "))
    d3.metric("Part du temps — Top 3 clients", f"{top3_share:.2f} %")
    d4.metric("Part du temps — Top 5 clients", f"{top5_share:.2f} %")
    d5.metric("Consultants (sélection)", f"{len(kpi_table):,}".replace(",", " "))

    if client_sans_nom:
        st.caption(
            f"**{client_sans_nom}** entrée(s) sans organisation (libellé « — ») : regroupez-les dans Zendesk pour un pilotage plus fin."
        )

    top_n_dir = st.slider("Afficher le Top N clients (table & graphiques)", 5, 50, 15, 1)
    head_cli = client_exec.head(top_n_dir).reset_index()
    disp_cli = head_cli.rename(
        columns={
            "Rang": "Rang",
            "Temps_total_min": "Temps total (min)",
            "Nb_lignes": "Lignes (export)",
            "Temps_moyen_par_ligne": "Temps moyen / ligne (min)",
            "Part_temps_pct": "Part du temps (%)",
            "Part_cumul_pct": "Part cumulée (%)",
        }
    )
    st.markdown("##### Classement des clients les plus demandants (par temps total)")
    st.dataframe(
        disp_cli.round(
            {
                "Temps total (min)": 2,
                "Lignes (export)": 0,
                "Temps moyen / ligne (min)": 2,
                "Part du temps (%)": 2,
                "Part cumulée (%)": 2,
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Une **ligne** = une ligne Excel du fichier (pas d’identifiant ticket dans les exports).")

    _csv_cli = client_exec.reset_index().drop(columns=["Rang"], errors="ignore")
    for _c in ("Temps_total_min", "Temps_moyen_par_ligne", "Part_temps_pct", "Part_cumul_pct"):
        if _c in _csv_cli.columns:
            _csv_cli[_c] = _csv_cli[_c].round(2)
    csv_clients = _csv_cli.to_csv(sep=";", decimal=",", encoding="utf-8-sig", index=False)
    st.download_button(
        "Télécharger le classement complet clients (CSV)",
        data=csv_clients.encode("utf-8-sig"),
        file_name=f"synthese_clients_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    fc1, fc2 = st.columns(2)
    fig_cli_temps = px.bar(
        head_cli.sort_values("Temps_total_min", ascending=True),
        x="Temps_total_min",
        y="Client",
        orientation="h",
        color="Temps_total_min",
        color_continuous_scale="Reds",
        labels={"Temps_total_min": "Minutes", "Client": ""},
        title=f"Top {top_n_dir} — temps total (min)",
    )
    fig_cli_temps.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        showlegend=False,
        yaxis=dict(categoryorder="total ascending"),
        height=max(360, 28 * len(head_cli)),
        margin=dict(l=8, r=8, t=48, b=8),
    )
    fig_cli_tickets = px.bar(
        head_cli.sort_values("Nb_lignes", ascending=True),
        x="Nb_lignes",
        y="Client",
        orientation="h",
        color="Nb_lignes",
        color_continuous_scale="Blues",
        labels={"Nb_lignes": "Lignes", "Client": ""},
        title=f"Top {top_n_dir} — nombre de lignes (export)",
    )
    fig_cli_tickets.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        showlegend=False,
        yaxis=dict(categoryorder="total ascending"),
        height=max(360, 28 * len(head_cli)),
        margin=dict(l=8, r=8, t=48, b=8),
    )
    fc1.plotly_chart(fig_cli_temps, use_container_width=True)
    fc2.plotly_chart(fig_cli_tickets, use_container_width=True)

    fig_pareto = go.Figure()
    csub = client_exec[client_exec["Temps_total_min"] > 0].head(25).copy()
    if len(csub):
        fig_pareto.add_trace(
            go.Bar(
                x=csub.index.astype(str),
                y=csub["Part_temps_pct"],
                name="Part du temps (%)",
                marker_color="#f85149",
            )
        )
        fig_pareto.add_trace(
            go.Scatter(
                x=csub.index.astype(str),
                y=csub["Part_cumul_pct"],
                name="Part cumulée (%)",
                yaxis="y2",
                mode="lines+markers",
                line=dict(color="#58a6ff", width=2),
                marker=dict(size=6),
            )
        )
        _ymax = max(5.0, float(csub["Part_temps_pct"].max()) * 1.15)
        fig_pareto.update_layout(
            title="Concentration de la charge — 25 premiers clients (parts & cumul)",
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#161b22",
            font=dict(color="#e6edf3"),
            xaxis=dict(title="Client", tickangle=-45),
            yaxis=dict(title="Part du temps (%)", side="left", range=[0, min(100.0, _ymax)]),
            yaxis2=dict(title="Part cumulée (%)", overlaying="y", side="right", range=[0, 105], showgrid=False),
            height=460,
            margin=dict(l=8, r=60, t=56, b=120),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

st.markdown("---")

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

# Grouped metrics : temps vs moyenne par ligne
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
        name="Temps moyen / ligne (min)",
        x=comp["Collaborateur"],
        y=comp["Temps_moyen_par_ligne"],
        marker_color="#58a6ff",
        offsetgroup=2,
    )
)
fig_grouped.update_layout(
    title="Comparaison directe : temps total vs temps moyen par ligne",
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
    nb_lignes=nb_lignes,
    temps_moyen=temps_moyen,
    fig_bar=fig_bar,
    fig_pie=fig_pie,
    fig_grouped=fig_grouped,
    fig_tree=fig_tree,
    years_in_filtered=_years_in_filtered,
    date_cols=date_cols,
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
        )

with st.expander("Détail technique (colonnes / exclusions)"):
    st.caption(
        "Colonnes « Total » dans les **en-têtes** : ignorées pour le calcul. "
        "Lignes dont le **consultant** ressemble à Total / Sum / Somme : exclues."
    )
    st.write("Fichiers :", [f.name for f in uploaded_list] if uploaded_list else _source_files)
    st.write("Colonnes jour (minutes) :", date_cols or "—")
    st.write("Lignes d’export : une ligne = une ligne du fichier (pas de n° ticket dans les XLS).")
    st.write("Lignes exclues (Total/Sum) :", n_rows_aggregate)
