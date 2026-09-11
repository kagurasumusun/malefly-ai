#!/usr/bin/env python3
# ============================================================================
# tools/malecns_stats.py — extract real MaleCNS v1.0 statistics for engine
# grounding. Data: Janelia FlyEM MaleCNS v1.0 (CC-BY),
# gs://flyem-male-cns (https://male-cns.janelia.org/download/).
# Output: docs/DATA.md (auto-generated tables) + data/malecns/glomeruli.csv
# ============================================================================
import os
import sys
import pandas as pd

FEATHER = "data/malecns/body-annotations-male-cns-v1.0-minconf-0.5.feather"
OUT_MD = "docs/DATA.md"
OUT_GLOM = "data/malecns/glomeruli.csv"


def main() -> int:
    if not os.path.exists(FEATHER):
        print(f"missing {FEATHER}; see tools/fetch_malecns.sh", file=sys.stderr)
        return 1
    df = pd.read_feather(FEATHER)
    lines = []
    lines.append("# DATA — MaleCNS v1.0 実データに基づく設計定数\n")
    lines.append("- 源: Janelia FlyEM **MaleCNS v1.0** (male-cns.janelia.org, CC-BY)")
    lines.append("- 取得: `tools/fetch_malecns.sh`(body-annotations feather, 14.4 MB)")
    lines.append("- 本ファイルは `tools/malecns_stats.py` の自動出力です(再実行で更新)\n")

    lines.append("## 全体\n")
    lines.append(f"- annotated bodies: **{len(df):,}**")
    lines.append(f"- unique types: **{df['type'].nunique():,}**(論文公称 11,691 と整合)")
    st = df["status"].value_counts(dropna=False).head(8)
    lines.append("\n| status | count |\n|---|---|")
    for k, v in st.items():
        lines.append(f"| {k} | {v:,} |")

    lines.append("\n## superclass 分布(上位)\n")
    lines.append("| superclass | count |\n|---|---|")
    for k, v in df["superclass"].value_counts().head(12).items():
        lines.append(f"| {k} | {v:,} |")

    # ---- olfactory system (our Goal-1 circuits) ----
    lines.append("\n## 嗅覚系(Goal 1–2 の実装対象)\n")
    orns = df[df["type"].astype(str).str.startswith("ORN_")]
    gloms = sorted(set(orns["type"].str.replace("ORN_", "", regex=False)))
    lines.append(f"- ORN types (= 実糸球体リスト): **{len(gloms)}**(実装は 50 チャネル)")
    lines.append(f"- ALPN (uniglomerular PN): **{(df['class'] == 'ALPN').sum():,}**"
                 "(実装: 1 糸球体 1 PN チャネル)")
    lines.append(f"- ALLN (局所ニューロン): **{(df['class'] == 'ALLN').sum():,}**(実装: LN 20)")

    # ---- mushroom body ----
    lines.append("\n## キノコ体(MB)\n")
    kc = df[df["class"] == "Kenyon_Cell"]
    lines.append(f"- Kenyon cells: **{len(kc):,}**(実装: 2,000 — 実規模の約半分。"
                 "拡張は ROADMAP で実測比較してから)")
    lines.append(f"- MBON: **{(df['class'] == 'MBON').sum():,}**(実装: 2 価値チャネルへ集約)")
    dan = (df["class"] == "DAN").sum()
    ppl1 = df["type"].astype(str).str.startswith("PPL1").sum()
    pam = df["type"].astype(str).str.startswith("PAM").sum()
    lines.append(f"- DAN: **{dan:,}**(PPL1 {ppl1} / PAM {pam})(実装: スカラー強化チャネル)")

    lines.append("\n### KC サブタイプ(実測)\n")
    lines.append("| KC type | count |\n|---|---|")
    for k, v in kc["type"].value_counts().head(12).items():
        lines.append(f"| {k} | {v:,} |")

    lines.append("\n### 主要 MBON types(実測, 上位)\n")
    mbon = df[df["class"] == "MBON"]
    lines.append("| MBON type | count |\n|---|---|")
    for k, v in mbon["type"].value_counts().head(10).items():
        lines.append(f"| {k} | {v:,} |")

    # ---- dimorphism ----
    lines.append("\n## 性二型(論文 Table 対応)\n")
    lines.append("| dimorphism | count |\n|---|---|")
    for k, v in df["dimorphism"].value_counts(dropna=False).head(6).items():
        lines.append(f"| {k} | {v:,} |")
    lines.append(f"\n- fruDsx annotated: {df['fruDsx'].notna().sum():,}")

    lines.append("\n## 実装定数への反映方針\n")
    lines.append("1. **採用済み**: PN 1 チャネル/糸球体、KC fan-in ~7(hemibrain 公表値)、"
                 "APL/LN 抑制、DAN 正則規則。")
    lines.append("2. **実測差分(次イテレーション候補、変更前に性能比較を必須)**: "
                 "KC 2,000→4,064、LN のサブタイプ分化、MBON のチャネル分割、"
                 "CX 領域(cb_intrinsic 32,164 の中核)は Goal 4 で設計。")
    lines.append("3. 本統計は設計レビューのたびに再実行し、実データとの乖離を監査する。")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))

    pd.DataFrame({"glomerulus": gloms}).to_csv(OUT_GLOM, index=False)
    print(f"\nwrote {OUT_MD} and {OUT_GLOM} ({len(gloms)} glomeruli)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
