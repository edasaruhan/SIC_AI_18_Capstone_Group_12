import numpy as np
import pandas as pd

DEFAULT = {
    "H_CAT": 3.0,
    "W_CAT": {"kumar_h": 1.0, "oyun_h": 1.0, "sosyal_video_h": 0.7,
              "genel_web_h": 0.3, "iletisim_h": 0.15},
    "H_SLEEP": 1.0,
    "W_SLEEP": {"kumar_sleep_h": 1.2, "oyun_sleep_h": 1.2, "sosyal_video_sleep_h": 1.0,
                "genel_web_sleep_h": 0.6, "iletisim_sleep_h": 0.8, "dusuk_sleep_h": 0.3},
    "BASE_DAYS": 14,
    "PTS": {"cat": 20, "sleep": 25, "dev": 20, "loop": 15},
}


def compute_components(m, p=DEFAULT):
    m = m.sort_values(["pid", "date"]).reset_index(drop=True).copy()
    pts = p["PTS"]
    m["cat_pts"] = pts["cat"] * np.clip(
        sum(w * m[c] for c, w in p["W_CAT"].items()) / p["H_CAT"], 0, 1)
    m["sleep_pts"] = pts["sleep"] * np.clip(
        sum(w * m[c] for c, w in p["W_SLEEP"].items()) / p["H_SLEEP"], 0, 1)

    m["day_idx"] = m.groupby("pid").cumcount()
    base = (m[m["day_idx"] < p["BASE_DAYS"]].groupby("pid")["screen_h"]
            .agg(base_med="median", base_mad=lambda x: (x - x.median()).abs().median()))
    m = m.merge(base, on="pid", how="left")
    z = (m["screen_h"] - m["base_med"]) / (1.4826 * m["base_mad"] + 0.5)
    m["dev_pts"] = np.where(m["day_idx"] >= p["BASE_DAYS"],
                            pts["dev"] * np.clip(z / 3, 0, 1), 0.0)

    as2 = m["as2"].where(m["app_h"] >= 0.1)
    loop_a = np.clip((as2 - 1.0) / 2.0, 0, 1).fillna(0)
    loop_b = np.clip((m["sess_short_share"] - 0.4) / 0.4, 0, 1).fillna(0)
    m["loop_pts"] = pts["loop"] * np.clip(0.5 * loop_a + 0.5 * loop_b, 0, 1)

    m["risk"] = m[["cat_pts", "sleep_pts", "dev_pts", "loop_pts"]].sum(axis=1) / sum(pts.values())
    return m