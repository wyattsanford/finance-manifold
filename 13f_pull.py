# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 12:58:12 2026

@author: Justin.Sanford
"""

# 13f_pull.py
# Pull 13F filings from SEC EDGAR for a cohort of long-equity managers
# CIKs verified against actual EDGAR filings
# Outputs: one parquet per filing + combined all_filings.parquet

import requests
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
import xml.etree.ElementTree as ET

DATA_DIR    = Path(r'C:\Users\Justin.Sanford\finance\data')
FILINGS_DIR = DATA_DIR / 'filings'
DATA_DIR.mkdir(parents=True, exist_ok=True)
FILINGS_DIR.mkdir(exist_ok=True)

CHECKPOINT_FILE = DATA_DIR / 'filings_checkpoint.json'

# Verified CIKs — confirmed against actual EDGAR 13F filings
# Note: some managers filed under different entities over time
COHORT = {
    # Verified
    'Baupost':         '1061768',   # Baupost Group LLC/MA
    'Appaloosa':       '1656456',   # Appaloosa LP (post-2015 entity)
    'Appaloosa_old':   '1006438',   # Appaloosa Management LP (pre-2015)
    'Viking_Global':   '1103804',   # Viking Global Investors LP
    'Lone_Pine':       '1061165',   # Lone Pine Capital LLC
    'Greenlight':      '1079114',   # Greenlight Capital Inc
    'Soros_Fund':      '1029160',   # Soros Fund Management LLC
    'Pershing_Square': '1336528',   # Pershing Square Capital Management
    'Third_Point':     '1040570',   # Third Point LLC
    'Tiger_Global':    '1167483',   # Tiger Global Management LLC
    'Icahn':           '921669',    # Icahn Capital Management LP
    'Paulson':         '1035674',   # Paulson & Co Inc
    'Farallon':        '1047235',   # Farallon Capital Management LLC
    'Jana_Partners':   '1159159',   # Jana Partners LLC
    'Blue_Ridge':      '1056831',   # Blue Ridge Capital LLC
    'Glenview':        '1138118',   # Glenview Capital Management LLC
    'Duquesne':        '813671',    # Duquesne Family Office LLC
    'ValueAct':        '1418814',   # ValueAct Holdings LP
    'Maverick':        '1061188',   # Maverick Capital Ltd
    'Coatue':          '1336477',   # Coatue Management LLC
}

HEADERS = {
    'User-Agent': 'Justin Sanford justin.sanford@dlhcorp.com',
    'Accept-Encoding': 'gzip, deflate',
}

START_YEAR = 2000
END_YEAR   = 2023

# ── Index fetch ───────────────────────────────────────────────────────────────
def get_filings_index(cik):
    """Get all 13F-HR filing metadata for a CIK."""
    cik_clean  = str(int(cik))           # strip leading zeros for int
    cik_padded = cik_clean.zfill(10)     # pad to 10 digits for API
    url = f'https://data.sec.gov/submissions/CIK{cik_padded}.json'

    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code == 404:
        print(f"    CIK {cik} not found (404)")
        return []
    r.raise_for_status()
    data = r.json()

    recent = data.get('filings', {}).get('recent', {})
    if not recent:
        return []

    forms      = recent.get('form', [])
    dates      = recent.get('filingDate', [])
    accessions = recent.get('accessionNumber', [])

    results = []
    for form, date, acc in zip(forms, dates, accessions):
        if form in ('13F-HR', '13F-HR/A'):
            year = int(date[:4])
            if START_YEAR <= year <= END_YEAR:
                results.append({
                    'form':      form,
                    'date':      date,
                    'accession': acc.replace('-', ''),
                    'cik':       cik_clean,
                })
    return results

# ── XML parse ─────────────────────────────────────────────────────────────────
def parse_13f_xml(cik, accession):
    cik_int  = int(cik)
    base_url = (f'https://www.sec.gov/Archives/edgar/data/'
                f'{cik_int}/{accession}/')

    idx_url = base_url + 'index.json'
    r = requests.get(idx_url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        return None

    try:
        idx = r.json()
    except Exception:
        return None

    items = idx.get('directory', {}).get('item', [])

    info_table_url = None
    for item in items:
        name = item.get('name', '').lower()
        if (('infotable' in name or 'informationtable' in name)
                and name.endswith('.xml')):
            info_table_url = base_url + item['name']
            break

    if not info_table_url:
        for item in items:
            name = item.get('name', '').lower()
            if (name.endswith('.xml')
                    and 'index' not in name
                    and 'primary' not in name):
                info_table_url = base_url + item['name']
                break

    if not info_table_url:
        return None

    r = requests.get(info_table_url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        return None

    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        return None

    # Detect namespace from root tag
    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}')[0] + '}'

    # Try both with and without namespace
    records = []
    for try_ns in [ns, '']:
        info_tables = root.findall(f'.//{try_ns}infoTable')
        if info_tables:
            for tbl in info_tables:
                def get(tag):
                    el = tbl.find(f'{try_ns}{tag}')
                    if el is not None and el.text:
                        return el.text.strip()
                    return ''

                def get_nested(parent_tag, child_tag):
                    parent = tbl.find(f'{try_ns}{parent_tag}')
                    if parent is not None:
                        child = parent.find(f'{try_ns}{child_tag}')
                        if child is not None and child.text:
                            return child.text.strip()
                    return ''

                records.append({
                    'nameOfIssuer':  get('nameOfIssuer'),
                    'cusip':         get('cusip'),
                    'value':         get('value'),
                    'sshPrnamt':     get_nested('shrsOrPrnAmt', 'sshPrnamt'),
                    'sshPrnamtType': get_nested('shrsOrPrnAmt', 'sshPrnamtType'),
                    'putCall':       get('putCall'),
                })
            break

    if not records:
        return None

    df = pd.DataFrame(records)
    df['value']     = pd.to_numeric(df['value'],     errors='coerce')
    df['sshPrnamt'] = pd.to_numeric(df['sshPrnamt'], errors='coerce')
    df = df[df['cusip'].str.len() >= 8]
    df = df.dropna(subset=['value'])
    df = df[df['value'] > 0]

    return df if len(df) > 0 else None

# ── Main pull ─────────────────────────────────────────────────────────────────
if CHECKPOINT_FILE.exists():
    with open(CHECKPOINT_FILE) as f:
        checkpoint = json.load(f)
    completed = set(checkpoint['completed'])
    print(f"Resuming — {len(completed)} filings done")
else:
    checkpoint = {'completed': [], 'skipped': []}
    completed  = set()

manifest = []

for manager_name, cik in COHORT.items():
    print(f"\n=== {manager_name} (CIK: {cik}) ===")

    try:
        filings = get_filings_index(cik)
        print(f"  Found {len(filings)} 13F filings "
              f"({START_YEAR}--{END_YEAR})")
        time.sleep(0.3)
    except Exception as e:
        print(f"  FAILED index: {e}")
        continue

    if not filings:
        print(f"  No filings — skipping")
        continue

    n_ok  = 0
    n_bad = 0

    for filing in filings:
        key = f"{manager_name}_{filing['accession']}"
        if key in completed:
            continue

        try:
            df = parse_13f_xml(filing['cik'], filing['accession'])
            time.sleep(0.12)  # stay well under 10 req/sec SEC limit

            if df is None or len(df) == 0:
                n_bad += 1
                checkpoint['skipped'].append(key)
                completed.add(key)
                checkpoint['completed'].append(key)
                with open(CHECKPOINT_FILE, 'w') as f:
                    json.dump(checkpoint, f, indent=2)
                continue

            # Add metadata
            quarter = pd.to_datetime(
                filing['date']).to_period('Q').strftime('%YQ%q')
            df['manager']   = manager_name
            df['cik']       = cik
            df['date']      = filing['date']
            df['quarter']   = quarter
            df['accession'] = filing['accession']

            out = (FILINGS_DIR /
                   f"{manager_name}_{filing['date']}.parquet")
            df.to_parquet(out)

            total_val = df['value'].sum()
            manifest.append({
                'manager':     manager_name,
                'date':        filing['date'],
                'quarter':     quarter,
                'n_holdings':  len(df),
                'total_value': float(total_val),
            })

            n_ok += 1
            completed.add(key)
            checkpoint['completed'].append(key)
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump(checkpoint, f, indent=2)

            print(f"  {filing['date']} — "
                  f"{len(df):4d} holdings | "
                  f"${total_val/1e3:7.1f}B")

        except Exception as e:
            n_bad += 1
            print(f"  FAILED {filing['date']}: {e}")
            checkpoint['skipped'].append(key)
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            time.sleep(1.0)
            continue

    print(f"  Summary: {n_ok} ok, {n_bad} failed")

# ── Combine ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMBINING ALL FILINGS")
print("=" * 60)

all_files = sorted(FILINGS_DIR.glob('*.parquet'))

if all_files:
    dfs = []
    for f in all_files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"  Could not read {f.name}: {e}")

    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        full_df.to_parquet(DATA_DIR / 'all_filings.parquet')

        print(f"\nTotal observations:  {len(full_df):,}")
        print(f"Managers:            {full_df['manager'].nunique()}")
        print(f"Date range:          "
              f"{full_df['date'].min()} — {full_df['date'].max()}")
        print(f"Unique CUSIPs:       {full_df['cusip'].nunique():,}")

        print(f"\nFilings per manager (sorted by coverage):")
        summary = (full_df.groupby('manager')
                   .agg(
                       n_quarters   =('quarter',   'nunique'),
                       n_rows       =('nameOfIssuer','count'),
                       date_min     =('date',      'min'),
                       date_max     =('date',      'max'),
                   )
                   .sort_values('n_quarters', ascending=False))
        print(summary.to_string())
    else:
        print("No readable files found")
else:
    print("No parquet files found in", FILINGS_DIR)

# Save manifest
with open(DATA_DIR / 'filings_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)

print("\nDone — delete checkpoint and rerun if CIKs fail")
print("Next: run prices_pull.py")