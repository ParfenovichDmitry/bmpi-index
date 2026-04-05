# -*- coding: utf-8 -*-
"""
gdelt_to_bmpi.py
═══════════════════════════════════════════════════════════════════
Ekstraktor danych GDELT → dane wejściowe do kalkulatora BMPI

OBSŁUGUJE OBA FORMATY:
  *.export.CSV.zip  → GDELT Events  (61 kolumn, tab-delimited)
  *.gkg.csv.zip     → GDELT GKG 2.1 (27 kolumn, tab-delimited)
  *.CSV / *.csv     → bez kompresji

UŻYCIE:
  python gdelt_to_bmpi.py "Downloads/20260309.export.CSV.zip"
  python gdelt_to_bmpi.py "Downloads/20260309.gkg.csv.zip"
  python gdelt_to_bmpi.py "Downloads/20260309.export.CSV.zip" --date 2026-03-09
  python gdelt_to_bmpi.py "Downloads/20260309.export.CSV.zip" --save

WYJŚCIE (na ekranie + opcjonalnie CSV):
  data            : 2026-03-09
  mentions : 487
  tone     : -1.8741
  ──────────────────────────────
  → wpisz do bmpi_calculator.html
═══════════════════════════════════════════════════════════════════
"""

import sys
import os
import zipfile
import io
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────
# SŁOWA KLUCZOWE Bitcoin (filtr URL i tematów)
# ─────────────────────────────────────────────────────────────────
BTC_KEYWORDS = [
    'bitcoin', 'btc', 'cryptocurrency', 'crypto currency',
    'satoshi', 'blockchain', 'coinbase', 'binance',
    'cryptomarket', 'digital currency', 'digitalcurrency',
]

# Dla GKG: tematy GDELT związane z krypto
GKG_THEMES = [
    'WEB_BITCOIN', 'ECON_CRYPTOCURRENCY', 'ECON_DIGITALCURRENCY',
    'BITCOIN', 'CRYPTO', 'BTC',
]

# ─────────────────────────────────────────────────────────────────
# NAGŁÓWKI KOLUMN
# ─────────────────────────────────────────────────────────────────

# GDELT Events 2.0 — 61 kolumn
EVENTS_COLS = [
    'GLOBALEVENTID','SQLDATE','MonthYear','Year','FractionDate',
    'Actor1Code','Actor1Name','Actor1CountryCode','Actor1KnownGroupCode',
    'Actor1EthnicCode','Actor1Religion1Code','Actor1Religion2Code',
    'Actor1Type1Code','Actor1Type2Code','Actor1Type3Code',
    'Actor2Code','Actor2Name','Actor2CountryCode','Actor2KnownGroupCode',
    'Actor2EthnicCode','Actor2Religion1Code','Actor2Religion2Code',
    'Actor2Type1Code','Actor2Type2Code','Actor2Type3Code',
    'IsRootEvent','EventCode','EventBaseCode','EventRootCode',
    'QuadClass','GoldsteinScale',
    'NumMentions','NumSources','NumArticles','AvgTone',   # ← kol 32,33,34,35
    'Actor1Geo_Type','Actor1Geo_FullName','Actor1Geo_CountryCode',
    'Actor1Geo_ADM1Code','Actor1Geo_Lat','Actor1Geo_Long','Actor1Geo_FeatureID',
    'Actor2Geo_Type','Actor2Geo_FullName','Actor2Geo_CountryCode',
    'Actor2Geo_ADM1Code','Actor2Geo_Lat','Actor2Geo_Long','Actor2Geo_FeatureID',
    'ActionGeo_Type','ActionGeo_FullName','ActionGeo_CountryCode',
    'ActionGeo_ADM1Code','ActionGeo_Lat','ActionGeo_Long','ActionGeo_FeatureID',
    'DATEADDED','SOURCEURL',                              # ← kol 60,61
]

# GDELT GKG 2.1 — 27 kolumn
GKG_COLS = [
    'GKGRECORDID','V2DATE','V2SOURCECOLLECTIONIDENTIFIER',
    'V2SOURCECOMMONNAME','V2DOCUMENTIDENTIFIER',
    'V2COUNTS','V2COUNTS_ADV',
    'V2THEMES',                # ← kol 8 (0-indexed: 7)
    'V2ENHANCEDLOCATIONS',
    'V2PERSONS','V2ORGS',
    'V2TONE',                  # ← kol 12 (0-indexed: 11) — "avg,pos,neg,pol,act,self,emo"
    'V2CSEVENTIDS',
    'V2GCAM',
    'V2SHARINGIMAGE',
    'V2RELATEDIMAGES',
    'V2SOCIALIMAGEEMBEDS',
    'V2SOCIALVIDEOEMBEDS',
    'V2QUOTATIONS',
    'V2ALLNAMES',
    'V2AMOUNTS',
    'V2TRANSLATIONINFO',
    'V2EXTRASXML',
    'V2GCAM2',
    'V2GCAM3',
    'V2GCAM4',
    'V2GCAM5',
]


# ─────────────────────────────────────────────────────────────────
# WYKRYWANIE FORMATU
# ─────────────────────────────────────────────────────────────────

def detect_format(filename: str) -> str:
    """
    Zwraca 'events' lub 'gkg' na podstawie nazwy pliku.
    """
    fname = filename.lower()
    if '.gkg.' in fname or 'gkg' in fname:
        return 'gkg'
    if '.export.' in fname or 'export' in fname or 'events' in fname:
        return 'events'
    # Próbuj po liczbie kolumn
    return 'unknown'


# ─────────────────────────────────────────────────────────────────
# ŁADOWANIE PLIKU
# ─────────────────────────────────────────────────────────────────

def load_file(path: str) -> tuple[pd.DataFrame, str]:
    """
    Wczytuje plik ZIP lub CSV do DataFrame.
    Zwraca (df, format_name).
    """
    p = Path(path)
    fmt = detect_format(p.name)

    print(f"  Plik:   {p.name}")
    print(f"  Format: {fmt.upper()} {'(Events)' if fmt=='events' else '(GKG)' if fmt=='gkg' else '(nieznany)'}")
    print(f"  Rozmiar: {p.stat().st_size / 1024 / 1024:.1f} MB")
    print()

    # Otwórz plik
    if path.lower().endswith('.zip'):
        with zipfile.ZipFile(path, 'r') as z:
            names = z.namelist()
            csv_name = next((n for n in names if n.lower().endswith('.csv')), names[0])
            print(f"  Plik w ZIP: {csv_name}")
            with z.open(csv_name) as f:
                raw = f.read()
        fileobj = io.BytesIO(raw)
    else:
        fileobj = open(path, 'rb')

    # Wczytaj z odpowiednimi kolumnami
    read_kwargs = dict(
        sep='\t',
        header=None,
        on_bad_lines='skip',
        low_memory=False,
        encoding='utf-8',
        encoding_errors='replace',
    )

    if fmt == 'events':
        read_kwargs['names'] = EVENTS_COLS
    elif fmt == 'gkg':
        read_kwargs['names'] = GKG_COLS
    # unknown → brak nazw, wykryjemy po liczbie kolumn

    df = pd.read_csv(fileobj, **read_kwargs)

    # Wykryj format po liczbie kolumn jeśli unknown
    if fmt == 'unknown':
        ncols = len(df.columns)
        if ncols >= 58:
            fmt = 'events'
            df.columns = EVENTS_COLS[:ncols]
            print(f"  Auto-wykryto: EVENTS ({ncols} kolumn)")
        elif ncols >= 20:
            fmt = 'gkg'
            df.columns = GKG_COLS[:ncols]
            print(f"  Auto-wykryto: GKG ({ncols} kolumn)")
        else:
            raise ValueError(f"Nieznany format — {ncols} kolumn. "
                             f"Oczekiwano 61 (Events) lub 27 (GKG).")

    print(f"  Wczytano: {len(df):,} wierszy × {len(df.columns)} kolumn")
    return df, fmt


# ─────────────────────────────────────────────────────────────────
# FILTROWANIE BITCOIN
# ─────────────────────────────────────────────────────────────────

def filter_bitcoin_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtruje wiersze Events dotyczące Bitcoina/krypto.
    Sprawdza: SOURCEURL, Actor1Name, Actor2Name.
    """
    pattern = '|'.join(BTC_KEYWORDS)

    masks = []

    # Sprawdź SOURCEURL
    if 'SOURCEURL' in df.columns:
        m = df['SOURCEURL'].str.contains(pattern, case=False, na=False)
        masks.append(m)
        print(f"  Filtr SOURCEURL: {m.sum():,} trafień")

    # Sprawdź Actor1Name i Actor2Name
    for col in ['Actor1Name', 'Actor2Name']:
        if col in df.columns:
            m = df[col].str.contains(pattern, case=False, na=False)
            masks.append(m)
            hits = m.sum()
            if hits > 0:
                print(f"  Filtr {col}: {hits:,} trafień")

    if not masks:
        raise ValueError("Brak kolumny SOURCEURL lub Actor*Name — sprawdź format pliku.")

    combined = masks[0]
    for m in masks[1:]:
        combined = combined | m

    result = df[combined].copy()
    print(f"  → Łącznie BTC: {len(result):,} wierszy")
    return result


def filter_bitcoin_gkg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtruje wiersze GKG dotyczące Bitcoina/krypto.
    Sprawdza: V2THEMES, V2SOURCECOMMONNAME (URL).
    """
    pattern_themes = '|'.join(GKG_THEMES)
    pattern_url    = '|'.join(BTC_KEYWORDS)

    masks = []

    if 'V2THEMES' in df.columns:
        m = df['V2THEMES'].str.contains(pattern_themes, case=False, na=False)
        masks.append(m)
        print(f"  Filtr V2THEMES: {m.sum():,} trafień")

    if 'V2SOURCECOMMONNAME' in df.columns:
        m = df['V2SOURCECOMMONNAME'].str.contains(pattern_url, case=False, na=False)
        masks.append(m)
        hits = m.sum()
        if hits > 0:
            print(f"  Filtr SOURCEURL: {hits:,} trafień")

    if not masks:
        raise ValueError("Brak kolumn V2THEMES / V2SOURCECOMMONNAME.")

    combined = masks[0]
    for m in masks[1:]:
        combined = combined | m

    result = df[combined].copy()
    print(f"  → Łącznie BTC: {len(result):,} wierszy")
    return result


# ─────────────────────────────────────────────────────────────────
# AGREGACJA → DANE DO BMPI
# ─────────────────────────────────────────────────────────────────

def aggregate_events(df: pd.DataFrame, date_override: str = None) -> pd.DataFrame:
    """
    Z danych Events wyciąga: data, mentions, tone.
    """
    # Data z SQLDATE (format YYYYMMDD)
    if 'SQLDATE' in df.columns:
        df['data'] = pd.to_datetime(
            df['SQLDATE'].astype(str).str[:8], format='%Y%m%d', errors='coerce'
        )
    elif date_override:
        df['data'] = pd.Timestamp(date_override)
    else:
        df['data'] = pd.Timestamp('today').normalize()

    # NumMentions i AvgTone
    df['NumMentions'] = pd.to_numeric(df['NumMentions'], errors='coerce').fillna(1)
    df['AvgTone']     = pd.to_numeric(df['AvgTone'],     errors='coerce')

    # Ważona średnia tonu (wagi = NumMentions)
    daily = df.groupby('data').apply(lambda g: pd.Series({
        'mentions': g['NumMentions'].sum(),
        'tone':     np.average(
            g['AvgTone'].dropna(),
            weights=g.loc[g['AvgTone'].notna(), 'NumMentions']
        ) if g['AvgTone'].notna().sum() > 0 else np.nan,
        'n_artykulow': len(g),
    })).reset_index()

    daily['zrodlo'] = 'gdelt_events'
    return daily


def aggregate_gkg(df: pd.DataFrame, date_override: str = None) -> pd.DataFrame:
    """
    Z danych GKG wyciąga: data, mentions, tone.
    Zgodne z formatem balanced.csv z Twojego projektu.
    """
    # Data z V2DATE (format YYYYMMDDHHMMSS)
    if 'V2DATE' in df.columns:
        df['data'] = pd.to_datetime(
            df['V2DATE'].astype(str).str[:8], format='%Y%m%d', errors='coerce'
        )
    elif date_override:
        df['data'] = pd.Timestamp(date_override)
    else:
        df['data'] = pd.Timestamp('today').normalize()

    # AvgTone = pierwsza wartość z V2TONE (format: "avg,pos,neg,pol,act,self,emo")
    if 'V2TONE' in df.columns:
        df['avg_tone'] = (
            df['V2TONE']
            .astype(str)
            .str.split(',')
            .str[0]
            .pipe(pd.to_numeric, errors='coerce')
        )
    else:
        df['avg_tone'] = np.nan

    # Agregacja per dzień
    daily = df.groupby('data').agg(
        mentions=('GKGRECORDID', 'count'),
        tone=('avg_tone', 'mean'),
        n_artykulow=('GKGRECORDID', 'count'),
    ).reset_index()

    daily['zrodlo'] = 'gdelt_gkg'
    return daily


# ─────────────────────────────────────────────────────────────────
# OBLICZ BMPI
# ─────────────────────────────────────────────────────────────────

# Parametry kalibracji z datasetu 2015–2026 (n=2220 dni)
CALIB = {
    'mu_wzm':   379.0,
    'sd_wzm':   305.8,
    'mu_tone': -0.9121,
    'sd_tone':  0.7139,
}
WEIGHTS = {'s1': 0.25, 's2': 0.20}

ZONES = [
    (0.000, 0.470, '🟢 MINIMAL',       'Organiczny ruch. Niskie ryzyko manipulacji.'),
    (0.470, 0.530, '🔵 BASELINE',     'Standardowa aktywność medialna. Monitoruj.'),
    (0.530, 0.590, '🟡 ELEVATED',  'Wzmożona narracja. Zalecana ostrożność.'),
    (0.590, 0.650, '🟠 HIGH',        'Wysoka ekstremalna presja medialna. Reversal ryzyko ~55%.'),
    (0.650, 1.001, '🔴 EXTREME',  'Ekstremalny szum (top 5% hist.). Half-life 2 dni.'),
]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def compute_bmpi(wzm: float, tone: float) -> dict:
    z1 = np.clip((wzm  - CALIB['mu_wzm'])  / CALIB['sd_wzm'],  -3, 3)
    z2 = np.clip((tone - CALIB['mu_tone']) / CALIB['sd_tone'], -3, 3)
    raw  = WEIGHTS['s1'] * z1 + WEIGHTS['s2'] * z2
    bmpi = float(sigmoid(raw))

    zone_name = zone_desc = ''
    for lo, hi, name, desc in ZONES:
        if lo <= bmpi < hi:
            zone_name = name
            zone_desc = desc
            break

    # Percentyl historyczny
    pct_breaks = [
        (0.425, 5),(0.426,10),(0.434,20),(0.438,25),(0.445,30),
        (0.461,40),(0.478,50),(0.495,60),(0.516,70),(0.528,75),
        (0.548,80),(0.571,85),(0.598,90),(0.648,95),(0.805,99),
    ]
    pct = 99
    for thr, p in pct_breaks:
        if bmpi <= thr:
            pct = p
            break

    return {
        'bmpi':      bmpi,
        'z_volume':   float(z1),
        'z_tone':   float(z2),
        'zone':      zone_name,
        'zone_desc': zone_desc,
        'percentyl': pct,
        'top_pct':   100 - pct,
    }


# ─────────────────────────────────────────────────────────────────
# WYDRUK WYNIKÓW
# ─────────────────────────────────────────────────────────────────

def print_results(daily: pd.DataFrame, fmt: str):
    SEP = '─' * 52

    print()
    print('╔' + '═' * 50 + '╗')
    print('║  WYNIKI EKSTRAKCJI GDELT → BMPI' + ' ' * 18 + '║')
    print('╠' + '═' * 50 + '╣')

    for _, row in daily.iterrows():
        wzm  = row.get('mentions', 0)
        tone = row.get('tone', 0)
        date_str = str(row['data'])[:10] if 'data' in row else '—'

        if pd.isna(wzm) or pd.isna(tone):
            print(f'║  {date_str}  Brak danych BTC w tym pliku' + ' ' * 10 + '║')
            continue

        bmpi_res = compute_bmpi(float(wzm), float(tone))

        print(f'║  DATA:              {date_str:<28}║')
        print(f'║  Źródło:            {fmt.upper():<28}║')
        print('╠' + '─' * 50 + '╣')
        print(f'║  mentions:   {wzm:<28.0f}║')
        print(f'║  tone:       {tone:<28.4f}║')
        if 'n_artykulow' in row:
            print(f'║  n artykułów:       {row["n_artykulow"]:<28.0f}║')
        print('╠' + '─' * 50 + '╣')
        print(f'║  z_volume (S1):   {bmpi_res["z_volume"]:<28.4f}║')
        print(f'║  z_tone     (S2):   {bmpi_res["z_tone"]:<28.4f}║')
        print('╠' + '═' * 50 + '╣')
        bmpi_str = f'{bmpi_res["bmpi"]:.4f}'
        print(f'║  BMPI:              {bmpi_str:<28}║')
        zone_str = bmpi_res["zone"]
        print(f'║  Strefa:            {zone_str:<28}║')
        pct_str  = f'wyżej niż {bmpi_res["percentyl"]}% dni hist. (top {bmpi_res["top_pct"]}%)'
        print(f'║  Percentyl:         {pct_str:<28}║')
        print('╠' + '─' * 50 + '╣')
        desc = bmpi_res["zone_desc"]
        # Zawijanie opisu
        words = desc.split()
        line = ''
        for w in words:
            if len(line) + len(w) + 1 > 46:
                print(f'║  {line:<48}║')
                line = w
            else:
                line = (line + ' ' + w).strip()
        if line:
            print(f'║  {line:<48}║')

    print('╚' + '═' * 50 + '╝')
    print()
    print('→ Wpisz te wartości do bmpi_calculator.html')
    print()


# ─────────────────────────────────────────────────────────────────
# SAVE CSV
# ─────────────────────────────────────────────────────────────────

def save_csv(daily: pd.DataFrame, source_path: str):
    p    = Path(source_path)
    stem = p.stem.replace('.export', '').replace('.gkg', '')
    out  = p.parent / f"{stem}_bmpi_input.csv"
    cols = ['data', 'mentions', 'tone', 'zrodlo']
    daily[[c for c in cols if c in daily.columns]].to_csv(out, index=False)
    print(f'  Zapisano: {out}')
    return str(out)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Ekstraktor GDELT → dane wejściowe BMPI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  python gdelt_to_bmpi.py "C:/Downloads/20260309.export.CSV.zip"
  python gdelt_to_bmpi.py "C:/Downloads/20260309.gkg.csv.zip"
  python gdelt_to_bmpi.py "C:/Downloads/20260309.export.CSV.zip" --save
  python gdelt_to_bmpi.py "C:/Downloads/20260309.export.CSV.zip" --date 2026-03-09
        """
    )
    parser.add_argument('path',  help='Ścieżka do pliku ZIP lub CSV z GDELT')
    parser.add_argument('--date', default=None,
                        help='Data (YYYY-MM-DD) — jeśli nie ma w pliku')
    parser.add_argument('--save', action='store_true',
                        help='Zapisz wynik do CSV obok pliku wejściowego')
    parser.add_argument('--keywords', nargs='+',
                        help='Dodatkowe słowa kluczowe (domyślnie: bitcoin btc crypto...)')

    args = parser.parse_args()

    # Dodatkowe słowa kluczowe
    if args.keywords:
        BTC_KEYWORDS.extend([k.lower() for k in args.keywords])

    print()
    print('╔' + '═' * 50 + '╗')
    print('║  GDELT → BMPI Extractor                        ║')
    print('╚' + '═' * 50 + '╝')
    print()

    # 1. Sprawdź czy plik istnieje
    if not os.path.exists(args.path):
        print(f'  [BŁĄD] Plik nie istnieje: {args.path}')
        print()
        print('  Sprawdź ścieżkę. Windows: użyj cudzysłowów i \\ lub /')
        print('  Przykład: "Downloads/20260309.export.CSV.zip"')
        sys.exit(1)

    # 2. Wczytaj
    print('[ 1/4 ] Wczytywanie pliku...')
    df, fmt = load_file(args.path)

    # 3. Filtruj Bitcoin
    print('[ 2/4 ] Filtrowanie artykułów Bitcoin/Crypto...')
    if fmt == 'events':
        btc = filter_bitcoin_events(df)
    else:
        btc = filter_bitcoin_gkg(df)

    if len(btc) == 0:
        print()
        print('  [UWAGA] Brak artykułów BTC w tym pliku!')
        print('  Możliwe przyczyny:')
        print('  1. To plik z dnia bez aktywności BTC w mediach')
        print('  2. Plik jest uszkodzony')
        print('  3. Spróbuj: --keywords bitcoin btc satoshi')
        sys.exit(0)

    # 4. Agreguj
    print('[ 3/4 ] Agregacja danych dziennych...')
    if fmt == 'events':
        daily = aggregate_events(btc, args.date)
    else:
        daily = aggregate_gkg(btc, args.date)

    # 5. Wyniki
    print('[ 4/4 ] Obliczanie BMPI...')
    print_results(daily, fmt)

    # 6. Opcjonalny zapis
    if args.save:
        print('Zapisuję CSV...')
        save_csv(daily, args.path)

    # 7. Prosty wydruk do kopiowania
    for _, row in daily.iterrows():
        wzm  = row.get('mentions', 0)
        tone = row.get('tone', 0)
        if not pd.isna(wzm) and not pd.isna(tone):
            bmpi = compute_bmpi(float(wzm), float(tone))
            print('─' * 40)
            print('DO KOPIOWANIA (kalkulator BMPI):')
            print(f'  Liczba wzmianek : {int(wzm)}')
            print(f'  Średni ton      : {tone:.4f}')
            print(f'  BMPI            : {bmpi["bmpi"]:.4f}')
            print(f'  Strefa          : {bmpi["zone"]}')
            print('─' * 40)


if __name__ == '__main__':
    main()