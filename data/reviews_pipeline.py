#!/usr/bin/env python3
import pandas as pd, numpy as np, re, html, json, math, hashlib, os, argparse
from datetime import datetime, timezone

def strip_html(s):
    if pd.isna(s): return s
    no_tags = re.sub(r"<[^>]+>", " ", str(s))
    return html.unescape(re.sub(r"\s+", " ", no_tags)).strip()

def normalize_text(s):
    if pd.isna(s): return s
    s = str(s).replace("\u200b","")
    return re.sub(r"\s+"," ",s).strip()

def parse_photo_field(x):
    if pd.isna(x): return []
    val = x
    if isinstance(val, (list, tuple)):
        urls = []
        for v in val:
            if isinstance(v, str):
                urls.append(v.strip())
            elif isinstance(v, dict) and "url" in v:
                u = v["url"]
                if isinstance(u, list): urls.extend([str(i).strip() for i in u])
                else: urls.append(str(u).strip())
        return [{"url":[u]} for u in urls if u]
    if isinstance(val, dict):
        if "url" in val:
            u = val["url"]
            urls = [u] if isinstance(u, str) else list(u)
            return [{"url":[str(i).strip()]} for i in urls if i]
        return []
    s = str(val).strip()
    try:
        decoded = json.loads(s)
        return parse_photo_field(decoded)
    except Exception:
        pass
    parts = re.split(r"[,\s;|]+", s)
    urls = [p for p in parts if re.match(r"^https?://", p)]
    return [{"url":[u]} for u in urls]

def pic_count_from_pics(pics):
    return sum(1 for _ in pics) if isinstance(pics, list) else 0

def safe_int(x):
    try: return int(float(x))
    except Exception: return np.nan

def to_epoch_ms(x):
    if pd.isna(x) or x == "": return np.nan
    if isinstance(x,(int,float)) and not np.isnan(x) and x>1e10: return int(x)
    fmts = ["%Y-%m-%d %H:%M:%S","%Y-%m-%d","%d-%m-%Y","%m/%d/%Y","%Y/%m/%d","%d %b %Y","%d %B %Y"]
    for fmt in fmts:
        try:
            dt = datetime.strptime(str(x), fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp()*1000)
        except Exception:
            continue
    return np.nan

def haversine_km(lat1, lon1, lat2, lon2):
    if any(pd.isna(v) for v in [lat1,lon1,lat2,lon2]): return np.nan
    R = 6371.0088
    phi1,phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2-lat1)
    dlambda = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2*math.atan2(math.sqrt(a),math.sqrt(1-a))
    return R*c

OVERWORDS = {"amazing","incredible","unbelievable","perfect","flawless","best","worst","scam","fraud","ripoff",
             "life-changing","mind-blowing","must-try","never","always","terrible","horrible","awful","disgusting",
             "phenomenal","fantastic","awesome","pathetic","disgraceful","magical"}

def text_features(s):
    if pd.isna(s):
        return pd.Series({"text_len":0,"excl_count":0,"all_caps_words":0,"overword_count":0,"lowercase_i_as_pronoun":0,"multi_excl_or_q":0,"trailing_ellipsis":0})
    st = str(s)
    words = re.findall(r"[A-Za-z']+", st)
    all_caps = sum(1 for w in words if len(w)>2 and w.isupper())
    excl = st.count("!")
    overcnt = sum(1 for w in words if w.lower() in OVERWORDS)
    lower_i = len(re.findall(r"(?:^|[^A-Za-z])i(?:'m| am|[^A-Za-z])", st))
    multi_punc = 1 if re.search(r"([!?])\1{1,}", st) else 0
    trailing_ellipsis = 1 if re.search(r"\.\.\.\s*$", st) else 0
    return pd.Series({"text_len":len(st),"excl_count":excl,"all_caps_words":all_caps,"overword_count":overcnt,
                      "lowercase_i_as_pronoun":lower_i,"multi_excl_or_q":multi_punc,"trailing_ellipsis":trailing_ellipsis})

def detect_rating_anomalies(df):
    key = "gmap_id" if ("gmap_id" in df.columns and df["gmap_id"].notna().any()) else "business_name"
    parts = []
    for bid, grp in df.groupby(key):
        grp = grp.copy()
        if grp["time"].notna().sum() > 2:
            grp = grp.sort_values("time")
            g = grp.set_index(pd.to_datetime(grp["time"], unit="ms", errors="coerce"))
            daily = g["rating"].resample("D").mean().to_frame("rating")
            roll_mean = daily["rating"].rolling(window=28, min_periods=7).mean()
            roll_std = daily["rating"].rolling(window=28, min_periods=7).std()
            grp["date"] = pd.to_datetime(grp["time"], unit="ms", errors="coerce").dt.normalize()
            grp = grp.merge(roll_mean.rename("roll_mean"), left_on="date", right_index=True, how="left")
            grp = grp.merge(roll_std.rename("roll_std"), left_on="date", right_index=True, how="left")
            grp["ma_dev"] = (grp["rating"] - grp["roll_mean"]) / grp["roll_std"]
            grp["rating_anomaly_flag"] = (grp["ma_dev"].abs() >= 2).astype("Int64")
            grp = grp.drop(columns=["date"], errors="ignore")
        else:
            mu = grp["rating"].mean()
            sd = grp["rating"].std(ddof=0) or 1.0
            grp["ma_dev"] = (grp["rating"] - mu) / sd
            grp["roll_mean"] = mu
            grp["roll_std"] = sd
            grp["rating_anomaly_flag"] = (grp["ma_dev"].abs() >= 2).astype("Int64")
        parts.append(grp)
    return pd.concat(parts, axis=0)

def compute_drift(df):
    df = df.copy()
    df["drift_speed_kmh"] = np.nan
    df["drift_impossible_flag"] = pd.NA
    if df[["business_lat","business_lng"]].notna().sum().sum() == 0 or df["time"].notna().sum() == 0:
        df["drift_impossible_flag"] = pd.NA
        return df
    key = "user_id" if "user_id" in df.columns else "name"
    for uid, grp in df.groupby(key):
        grp = grp.copy().sort_values("time")
        lat = grp["business_lat"].values
        lng = grp["business_lng"].values
        t = grp["time"].values
        speeds = [np.nan]
        for i in range(1, len(grp)):
            d_km = haversine_km(lat[i-1], lng[i-1], lat[i], lng[i])
            dt_hr = (t[i] - t[i-1]) / (1000.0 * 3600.0) if not (pd.isna(t[i]) or pd.isna(t[i-1])) else np.nan
            speed = d_km / dt_hr if (dt_hr and dt_hr > 0 and not pd.isna(d_km)) else np.nan
            speeds.append(speed)
        grp["drift_speed_kmh"] = speeds
        grp["drift_impossible_flag"] = grp["drift_speed_kmh"].apply(lambda v: 1 if (isinstance(v,(int,float)) and v>900) else (0 if not pd.isna(v) else pd.NA)).astype("Int64")
        df.loc[grp.index, ["drift_speed_kmh","drift_impossible_flag"]] = grp[["drift_speed_kmh","drift_impossible_flag"]]
    return df

def run_pipeline(reviews_csv, out_clean_csv, out_anoms_csv, business_meta_csv=None, user_meta_csv=None):
    df = pd.read_csv(reviews_csv, low_memory=False)

    # Normalize strings
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(normalize_text)

    # Clean text
    df["text_clean"] = df["text"].apply(strip_html) if "text" in df.columns else None

    # Deduplicate
    subset_cols = [c for c in ["business_name","author_name","text_clean"] if c in df.columns]
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols, keep="first")

    # Photos -> pics + pic_count
    if "photo" in df.columns:
        df["pics"] = df["photo"].apply(parse_photo_field)
        df["pic_count"] = df["pics"].apply(pic_count_from_pics).astype("Int64")
    else:
        df["pics"] = [[] for _ in range(len(df))]
        df["pic_count"] = pd.array([0]*len(df), dtype="Int64")

    # Ratings
    if "rating" in df.columns:
        df["rating"] = df["rating"].apply(safe_int).clip(lower=1, upper=5)

    # Business metadata
    if business_meta_csv and os.path.exists(business_meta_csv):
        bm = pd.read_csv(business_meta_csv)
        df = df.merge(bm, on="business_name", how="left")
    else:
        for c in ["gmap_id","business_lat","business_lng","city","country"]:
            if c not in df.columns:
                df[c] = pd.NA if c=="gmap_id" or c in ["city","country"] else np.nan

    # User metadata
    if user_meta_csv and os.path.exists(user_meta_csv):
        um = pd.read_csv(user_meta_csv)
        df = df.merge(um, on="author_name", how="left")
    else:
        if "author_name" in df.columns:
            df["user_id"] = df["author_name"].apply(lambda n: hashlib.sha1(n.encode("utf-8")).hexdigest()[:16] if pd.notna(n) and n!="" else pd.NA)
        for c in ["home_lat","home_lng"]:
            if c not in df.columns: df[c] = np.nan

    # Time
    tcol = next((c for c in df.columns if c.lower() in ("time","review_time","created_at","timestamp")), None)
    df["time"] = df[tcol].apply(to_epoch_ms) if tcol else np.nan

    # Unified schema renaming

    if "resp" not in df.columns:
        df["resp"] = None

    rename_map = {"author_name":"name","text_clean":"text"}
    for k,v in rename_map.items():
        if k in df.columns: df[v] = df[k] if v not in df.columns else df[v]
    if "name" not in df.columns: df["name"] = pd.NA
    # Keep only expected columns + extras
    keep = ["user_id","name","time","rating","text","pics","resp","gmap_id","business_name","business_lat","business_lng","city","country","pic_count"]
    extras = [c for c in df.columns if c not in keep]
    df = df[keep + extras]

    # Text features
    feats = df["text"].apply(text_features)
    df = pd.concat([df, feats], axis=1)

    # Anomalies
    df = detect_rating_anomalies(df)
    df = compute_drift(df)

    # Save outputs
    df.to_csv(out_clean_csv, index=False)
    anom = df[(df["rating_anomaly_flag"]==1) | (df["drift_impossible_flag"]==1)]
    anom.to_csv(out_anoms_csv, index=False)

    # Return small stats
    return {
        "rows": len(df),
        "unique_businesses": df["business_name"].nunique(dropna=False) if "business_name" in df.columns else None,
        "with_time": int(df["time"].notna().sum()),
        "with_coords": int((df[["business_lat","business_lng"]].notna().all(axis=1)).sum()) if "business_lat" in df.columns else 0,
        "rating_anomalies": int((df["rating_anomaly_flag"]==1).sum()),
        "drift_flags": int((df["drift_impossible_flag"]==1).sum())
    }

def main():
    ap = argparse.ArgumentParser(description="Google Reviews ETL + Anomaly Detection")
    ap.add_argument("--reviews_csv", required=True, help="Path to reviews.csv")
    ap.add_argument("--out_clean_csv", default="reviews_clean.csv")
    ap.add_argument("--out_anoms_csv", default="review_anomalies.csv")
    ap.add_argument("--business_meta_csv", default=None, help="Optional business metadata CSV")
    ap.add_argument("--user_meta_csv", default=None, help="Optional user metadata CSV")
    args = ap.parse_args()
    stats = run_pipeline(args.reviews_csv, args.out_clean_csv, args.out_anoms_csv, args.business_meta_csv, args.user_meta_csv)
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
