# scripts/fetch_bnm_api.py
import os, sys, time, json, math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Tuple

import requests
import pandas as pd
from dateutil.relativedelta import relativedelta

BASE = "https://api.bnm.gov.my/public"
HEADERS = {
    # 必须带这个 Accept 头
    "Accept": "application/vnd.BNM.API.v1+json",
    "User-Agent": "myopr-watch/1.0 (+github.com/MysticNeD/myopr-watch)"
}

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = (ROOT / "data").resolve()
RAW_DIR = (DATA_DIR / "raw").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

def get_json(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Call BNM Open API and return JSON; raise on HTTP errors."""
    url = f"{BASE}{path}"
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} for {url} params={params} body={resp.text[:300]}")
    return resp.json()

def _safe_list(val):
    """
    Return a list of data rows from API response.
    - If val is already a list -> return it
    - If val is a dict with "data":
        - if data is list -> return data
        - if data is dict -> wrap it into [data]
    - Else return empty list
    """
    if isinstance(val, list):
        return val
    if isinstance(val, dict) and "data" in val:
        data = val["data"]
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    return []

def _dump_raw(name: str, obj: Any):
    fp = RAW_DIR / f"{name}.json"
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"Saved raw JSON: {fp}")

def _write_csv(df: pd.DataFrame, outfile: Path):
    df.to_csv(outfile, index=False, encoding="utf-8")

def _normalize_date(s: Any) -> Any:
    # 尝试把各种日期字段规范成 YYYY-MM-DD
    if s is None:
        return None
    try:
        return pd.to_datetime(s).date().isoformat()
    except Exception:
        return s

def iter_years(n_years: int = 1):
    """
    Yield years from current year going back n_years-1.
    Example: if now is 2025 and n_years=5 -> yields 2025,2024,2023,2022,2021
    """
    now = datetime.now(timezone.utc)
    cur_year = now.year
    for i in range(n_years):
        yield cur_year - i


def fetch_opr(lookback_years: int = 1) -> pd.DataFrame:
    """
    Robust OPR fetcher:
    - Try /opr (may return single dict or list)
    - If single or insufficient, try /opr/year/{year} for lookback_years
    - Merge any existing data/raw/opr_raw*.json fragments
    - Merge with existing data/oprs.csv (if present) to avoid wiping history; prefer API/new data for same dates
    - Output DataFrame with columns: date, new_opr_level, change_in_opr, year
    """
    rows: list[dict] = []

    # 1) try top-level /opr
    try:
        raw = get_json("/opr")
        _dump_raw("opr_raw", raw)
        rows.extend(_safe_list(raw))
    except Exception as e:
        print(f"[opr] top-level /opr fetch failed: {e}", file=sys.stderr)

    # 2) If rows empty or we only got single/latest, try per-year fetch for lookback_years
    if not rows or (isinstance(rows, list) and len(rows) < 2):
        for y in iter_years(lookback_years):
            try:
                rawy = get_json(f"/opr/year/{y}")
                _dump_raw(f"opr_raw_{y}", rawy)
                rows.extend(_safe_list(rawy))
            except Exception as e:
                print(f"[opr] /opr/year/{y} fetch failed: {e}", file=sys.stderr)

    # 3) also merge any pre-existing raw files in data/raw/ (opr_raw*.json)
    try:
        for f in RAW_DIR.glob("opr_raw*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    js = json.load(fh)
                rows.extend(_safe_list(js))
            except Exception as e:
                print(f"[opr] failed reading raw file {f.name}: {e}", file=sys.stderr)
    except Exception:
        pass

    # 4) Normalize rows into DataFrame with expected column names
    out = []
    for r in rows:
        date = r.get("announcement_date") or r.get("date") or r.get("effective_date") or r.get("as_at")
        rate = r.get("new_opr_level") or r.get("rate") or r.get("opr") or r.get("value")
        change = r.get("change_in_opr") or r.get("change") or r.get("diff")
        year = r.get("year")
        out.append({
            "date": _normalize_date(date),
            "new_opr_level": float(rate) if rate is not None else None,
            "change_in_opr": float(change) if change is not None else None,
            "year": int(year) if year is not None else (pd.to_datetime(date).year if date else None)
        })

    df_api = pd.DataFrame(out)

    if "date" in df_api.columns:
        df_api = df_api.dropna(subset=["date"]).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    else:
        df_api = pd.DataFrame(columns=["date", "new_opr_level", "change_in_opr", "year"])

    # 5) Merge with existing data/oprs.csv if exists
    existing_path = DATA_DIR / "oprs.csv"
    if existing_path.exists():
        try:
            existing = pd.read_csv(existing_path)  # 先读，不 parse_dates
            existing.columns = [c.strip() for c in existing.columns]  # 去掉空格
            if "date" in existing.columns:
                existing["date"] = pd.to_datetime(existing["date"], errors="coerce").dt.date
            else:
                raise ValueError("existing oprs.csv 没有 'date' 列")

            if "rate" in existing.columns and "new_opr_level" not in existing.columns:
                existing = existing.rename(columns={"rate": "new_opr_level"})
            existing["date"] = pd.to_datetime(existing["date"]).dt.date
            combined = pd.concat([existing, df_api], ignore_index=True, sort=False)
            combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
            if "new_opr_level" in combined.columns:
                combined["new_opr_level"] = pd.to_numeric(combined["new_opr_level"], errors="coerce")
            if "change_in_opr" in combined.columns:
                combined["change_in_opr"] = pd.to_numeric(combined["change_in_opr"], errors="coerce")
            if "year" not in combined.columns:
                combined["year"] = pd.to_datetime(combined["date"]).dt.year
            return combined[["date", "new_opr_level", "change_in_opr", "year"]]
        except Exception as e:
            print(f"[opr] error merging with existing oprs.csv: {e}", file=sys.stderr)
            return df_api[["date", "new_opr_level", "change_in_opr", "year"]]
    else:
        if "year" not in df_api.columns and "date" in df_api.columns:
            df_api["year"] = pd.to_datetime(df_api["date"]).dt.year
        return df_api[["date", "new_opr_level", "change_in_opr", "year"]]




def iter_months(n_months: int) -> Iterable[Tuple[int, int]]:
    """Yield (year, month) for current month going back n_months-1."""
    now = datetime.now(timezone.utc)
    cur = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    for i in range(n_months):
        d = cur - relativedelta(months=i)
        yield d.year, d.month


def fetch_myor(n_months: int = 1) -> pd.DataFrame:
    """
    MYOR-I: /my-overnight-rate-i
    支持 /my-overnight-rate-i/month/{month}/year/{year}?reverse=true
    规范为 columns: date, myor, aggregate_volume
    """
    all_rows: List[Dict[str, Any]] = []

    # 1) 尝试按月从 API 拉取（按传入的 n_months）
    for y, m in iter_months(n_months):
        try:
            raw = get_json(f"/my-overnight-rate-i/month/{m}/year/{y}", params={"reverse": "true"})
            _dump_raw(f"myor_raw_{y}_{m}", raw)
            all_rows.extend(_safe_list(raw))
        except Exception as e:
            # 记录但不抛出，继续下一月
            print(f"[myor] API fetch failed for {y}-{m}: {e}", file=sys.stderr)

    # 2) 再去 data/raw/ 找历史分片文件并合并（以防脚本以前保存了多个月的 raw）
    try:
        for f in RAW_DIR.glob("myor_raw*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    js = json.load(fh)
                all_rows.extend(_safe_list(js))
            except Exception as e:
                print(f"[myor] failed to read raw file {f.name}: {e}", file=sys.stderr)
    except Exception:
        pass

    # 3) 若仍然为空，再退化为一次性取全部
    if not all_rows:
        try:
            raw = get_json("/my-overnight-rate-i")
            _dump_raw("myor_raw_fallback", raw)
            all_rows = _safe_list(raw)
        except Exception as e:
            print(f"[myor] fallback full API fetch failed: {e}", file=sys.stderr)
            all_rows = []

    # 4) 规范字段，兼容各种命名（你的样本使用 reference_date/reference_rate/aggregate_volume）
    out = []
    for r in all_rows:
        date = r.get("reference_date") or r.get("date") or r.get("announcement_date") or r.get("effective_date") or r.get("as_at")
        val = r.get("reference_rate") or r.get("rate") or r.get("myor") or r.get("value")
        vol = r.get("aggregate_volume") or r.get("aggregate_vol") or r.get("aggregate") or r.get("volume")
        out.append({
            "date": _normalize_date(date),
            "myor": (float(val) if val is not None else None),
            "aggregate_volume": (float(vol) if vol is not None else None)
        })

    df = pd.DataFrame(out)
    if "date" in df.columns:
        df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["date", "myor", "aggregate_volume"])
    return df


def fetch_interbank_rates(n_months: int = 1, product: str = "interbank") -> pd.DataFrame:
    """
    Interbank rates: /interest-rate
    Output: normalized long-form DataFrame with columns: date, tenor, rate
    Handles two response shapes:
      - long-form: rows with keys ['date','tenor','rate', ...]
      - wide-form: rows with keys ['date','overnight','1_week','1_month',...]
    """
    rows: List[Dict[str, Any]] = []

    # 1) 按月拉 API（尝试两种可能的路径）
    for y, m in iter_months(n_months):
        tried = False
        try:
            raw = get_json(f"/interest-rate/year/{y}/month/{m}", params={"product": product})
            _dump_raw(f"interbank_rates_{y}_{m}", raw)
            rows.extend(_safe_list(raw))
            tried = True
        except Exception:
            pass
        if not tried:
            try:
                raw = get_json(f"/interest-rate/month/{m}/year/{y}", params={"product": product})
                _dump_raw(f"interbank_rates_alt_{y}_{m}", raw)
                rows.extend(_safe_list(raw))
            except Exception:
                print(f"[interbank_rates] API fetch failed for {y}-{m}", file=sys.stderr)
                continue

    # 2) 合并 data/raw 中同类分片（如果存在）
    try:
        for f in RAW_DIR.glob("interbank_rates*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    js = json.load(fh)
                rows.extend(_safe_list(js))
            except Exception as e:
                print(f"[interbank_rates] failed to read raw file {f.name}: {e}", file=sys.stderr)
        for f in RAW_DIR.glob("interest-rate*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    js = json.load(fh)
                rows.extend(_safe_list(js))
            except Exception as e:
                print(f"[interbank_rates] failed to read raw file {f.name}: {e}", file=sys.stderr)
    except Exception:
        pass

    if not rows:
        return pd.DataFrame(columns=["date", "tenor", "rate"])

    # 3) Detect shape and normalize:
    # If first row contains 'tenor' key -> assume long-form
    sample0 = rows[0] if len(rows) > 0 else {}
    long_form = "tenor" in sample0 or "maturity" in sample0 or "tenure" in sample0

    normalized = []
    if long_form:
        # rows are already long-form
        for r in rows:
            date = r.get("date") or r.get("reference_date") or r.get("effective_date") or r.get("as_at")
            tenor = r.get("tenor") or r.get("maturity") or r.get("tenure")
            rate = r.get("rate") or r.get("reference_rate") or r.get("value")
            normalized.append({
                "date": _normalize_date(date),
                "tenor": tenor,
                "rate": (float(rate) if rate is not None else None)
            })
    else:
        # wide-form: each row has date + multiple tenor columns (overnight, 1_week, 1_month, etc.)
        for r in rows:
            date = r.get("date") or r.get("reference_date") or r.get("effective_date") or r.get("as_at")
            for k, v in r.items():
                if k is None:
                    continue
                # skip date-like keys
                if k.lower() in {"date", "reference_date", "effective_date", "as_at"}:
                    continue
                # skip non-tenor meta keys
                if isinstance(v, dict) or k.lower().startswith("meta") or k.lower().startswith("product"):
                    continue
                # consider this a tenor column if value is numeric or null (API uses null for missing)
                if v is None or isinstance(v, (int, float)) or (isinstance(v, str) and v.replace(".", "", 1).replace("-", "", 1).isdigit()):
                    # map key to a readable tenor string (keep as-is)
                    tenor_label = str(k)
                    # for consistency, convert '1_week' -> '1_week', 'overnight' -> 'overnight'
                    try:
                        rate_val = float(v) if v is not None else None
                    except Exception:
                        rate_val = None
                    normalized.append({
                        "date": _normalize_date(date),
                        "tenor": tenor_label,
                        "rate": rate_val
                    })
                # else skip (non-numeric text fields)
    df = pd.DataFrame(normalized)
    if {"date", "tenor"}.issubset(df.columns):
        df = df.dropna(subset=["date", "tenor"]).sort_values(["date", "tenor"]).drop_duplicates(subset=["date", "tenor"], keep="last").reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["date", "tenor", "rate"])
    return df


def fetch_interbank_volumes(n_months: int = 1, product: str = "interbank") -> pd.DataFrame:
    """
    Interbank volumes: /interest-volume
    Output: long-form DataFrame with columns: date, tenor, volume
    Handles wide-form where columns are tenor names (overnight, 1_week, ...) and values are volumes.
    """
    rows: List[Dict[str, Any]] = []

    # 1) 按月拉 API（尝试两种路径）
    for y, m in iter_months(n_months):
        tried = False
        try:
            raw = get_json(f"/interest-volume/year/{y}/month/{m}", params={"product": product})
            _dump_raw(f"interbank_vol_{y}_{m}", raw)
            rows.extend(_safe_list(raw))
            tried = True
        except Exception:
            pass
        if not tried:
            try:
                raw = get_json(f"/interest-volume/month/{m}/year/{y}", params={"product": product})
                _dump_raw(f"interbank_vol_alt_{y}_{m}", raw)
                rows.extend(_safe_list(raw))
            except Exception:
                print(f"[interbank_volumes] API fetch failed for {y}-{m}", file=sys.stderr)
                continue

    # 2) 合并 data/raw 中现有分片
    try:
        for f in RAW_DIR.glob("interbank_vol*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    js = json.load(fh)
                rows.extend(_safe_list(js))
            except Exception as e:
                print(f"[interbank_volumes] failed to read raw file {f.name}: {e}", file=sys.stderr)
        for f in RAW_DIR.glob("interest-volume*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    js = json.load(fh)
                rows.extend(_safe_list(js))
            except Exception as e:
                print(f"[interbank_volumes] failed to read raw file {f.name}: {e}", file=sys.stderr)
    except Exception:
        pass

    if not rows:
        return pd.DataFrame(columns=["date", "tenor", "volume"])

    # 3) normalize: long-form vs wide-form detection
    sample0 = rows[0] if len(rows) > 0 else {}
    long_form = "tenor" in sample0 or "maturity" in sample0 or "tenure" in sample0

    normalized = []
    if long_form:
        for r in rows:
            date = r.get("date") or r.get("reference_date") or r.get("effective_date") or r.get("as_at")
            tenor = r.get("tenor") or r.get("maturity") or r.get("tenure")
            vol = r.get("volume") or r.get("aggregate_volume") or r.get("value")
            normalized.append({
                "date": _normalize_date(date),
                "tenor": tenor,
                "volume": (float(vol) if vol is not None else None)
            })
    else:
        # wide form: each row has date + multiple tenor columns (values are volumes)
        for r in rows:
            date = r.get("date") or r.get("reference_date") or r.get("effective_date") or r.get("as_at")
            for k, v in r.items():
                if k is None:
                    continue
                if k.lower() in {"date", "reference_date", "effective_date", "as_at"}:
                    continue
                if isinstance(v, dict) or k.lower().startswith("meta") or k.lower().startswith("product"):
                    continue
                # treat numeric or null values as volume cells
                if v is None or isinstance(v, (int, float)) or (isinstance(v, str) and v.replace(",", "").replace(".", "", 1).replace("-", "", 1).isdigit()):
                    tenor_label = str(k)
                    try:
                        vol_val = float(str(v).replace(",", "")) if v is not None else None
                    except Exception:
                        vol_val = None
                    normalized.append({
                        "date": _normalize_date(date),
                        "tenor": tenor_label,
                        "volume": vol_val
                    })
    df = pd.DataFrame(normalized)
    if {"date", "tenor"}.issubset(df.columns):
        df = df.dropna(subset=["date", "tenor"]).sort_values(["date", "tenor"]).drop_duplicates(subset=["date", "tenor"], keep="last").reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["date", "tenor", "volume"])
    return df


def fetch_fast_liquidity_position() -> pd.DataFrame:
    """
    Fetch /fast/liquidity-position and return a normalized DataFrame.
    We also write raw JSON to data/raw/fast_liquidity_raw.json for inspection.
    """
    raw = get_json("/fast/liquidity-position")
    _dump_raw("fast_liquidity_raw", raw)

    rows = raw.get("data") if isinstance(raw, dict) and "data" in raw else raw
    if not isinstance(rows, list):
        # If API returns a single dict under 'data', wrap it into list
        if isinstance(rows, dict):
            rows = [rows]
        else:
            raise RuntimeError("Unexpected response shape for liquidity-position")

    # Flatten nested JSON to flat table
    df = pd.json_normalize(rows)

    # Heuristic cleanup: ensure there's a 'date' (or similar) column
    date_cols = [c for c in df.columns if "date" in c.lower() or "as_at" in c.lower()]
    if date_cols:
        df = df.rename(columns={date_cols[0]: "date"})
    # Convert date-like columns
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        except Exception:
            pass

    out_path = DATA_DIR / "fast.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved CSV: {out_path} ({len(df)} rows, {list(df.columns)[:10]}...)")
    return df


def main():
    print("Fetching from BNM Open API…")
    try:
        opr = fetch_opr()
        _write_csv(opr, DATA_DIR / "oprs.csv")
        print(f"Saved {DATA_DIR / 'oprs.csv'} ({len(opr)} rows)")
    except Exception as e:
        print("Error fetching OPR:", e, file=sys.stderr)

    try:
        myor = fetch_myor(n_months=1)
        _write_csv(myor, DATA_DIR / "myor.csv")
        print(f"Saved {DATA_DIR / 'myor.csv'} ({len(myor)} rows)")
    except Exception as e:
        print("Error fetching MYOR:", e, file=sys.stderr)

    try:
        ib_rates = fetch_interbank_rates(n_months=1, product="interbank")
        _write_csv(ib_rates, DATA_DIR / "interbank_rates.csv")
        print(f"Saved {DATA_DIR / 'interbank_rates.csv'} ({len(ib_rates)} rows)")
    except Exception as e:
        print("Error fetching interbank rates:", e, file=sys.stderr)

    try:
        ib_vols = fetch_interbank_volumes(n_months=1, product="interbank")
        _write_csv(ib_vols, DATA_DIR / "interbank_volumes.csv")
        print(f"Saved {DATA_DIR / 'interbank_volumes.csv'} ({len(ib_vols)} rows)")
    except Exception as e:
        print("Error fetching interbank volumes:", e, file=sys.stderr)

    # FAST：
    print("Fetching FAST liquidity position from BNM OpenAPI...")
    try:
        df = fetch_fast_liquidity_position()
    except requests.HTTPError as e:
        print("HTTP error when fetching FAST liquidity-position:", e)
    except Exception as e:
        print("Error fetching/processing FAST liquidity-position:", e)

    # write last_fetch
    with open(DATA_DIR / "last_fetch.txt", "w", encoding="utf-8") as f:
        f.write(datetime.now(timezone.utc).isoformat())

if __name__ == "__main__":
    main()
