# DATA — MaleCNS v1.0 実データに基づく設計定数

- 源: Janelia FlyEM **MaleCNS v1.0** (male-cns.janelia.org, CC-BY)
- 取得: `tools/fetch_malecns.sh`(body-annotations feather, 14.4 MB)
- 本ファイルは `tools/malecns_stats.py` の自動出力です(再実行で更新)

## 全体

- annotated bodies: **211,577**
- unique types: **11,751**(論文公称 11,691 と整合)

| status | count |
|---|---|
| Traced | 165,122 |
| Orphan | 15,925 |
| Glia | 11,864 |
| Unimportant | 10,751 |
| None | 5,472 |
| Assign | 1,832 |
| Anchor | 611 |

## superclass 分布(上位)

| superclass | count |
|---|---|
| ol_intrinsic | 89,403 |
| cb_intrinsic | 32,164 |
| vnc_intrinsic | 13,161 |
| visual_projection | 9,201 |
| vnc_sensory | 6,370 |
| ol_sensory | 6,098 |
| cb_sensory | 4,868 |
| ascending_neuron | 1,846 |
| descending_neuron | 1,314 |
| vnc_motor | 708 |
| visual_centrifugal | 563 |
| sensory_ascending | 537 |

## 嗅覚系(Goal 1–2 の実装対象)

- ORN types (= 実糸球体リスト): **53**(実装は 50 チャネル)
- ALPN (uniglomerular PN): **686**(実装: 1 糸球体 1 PN チャネル)
- ALLN (局所ニューロン): **420**(実装: LN 20)

## キノコ体(MB)

- Kenyon cells: **4,064**(実装: 2,000 — 実規模の約半分。拡張は ROADMAP で実測比較してから)
- MBON: **97**(実装: 2 価値チャネルへ集約)
- DAN: **340**(PPL1 16 / PAM 316)(実装: スカラー強化チャネル)

### KC サブタイプ(実測)

| KC type | count |
|---|---|
| KCg-m | 1,342 |
| KCab-s | 657 |
| KCab-m | 536 |
| KCab-c | 488 |
| KCa'b'-ap2 | 291 |
| KCg-d | 206 |
| KCa'b'-m | 205 |
| KCa'b'-ap1 | 199 |
| KCab-p | 129 |
| KCg-s2 | 2 |
| KCg-s1 | 2 |
| KCg-s3 | 2 |

### 主要 MBON types(実測, 上位)

| MBON type | count |
|---|---|
| MBON10 | 9 |
| MBON09 | 4 |
| MBON07 | 4 |
| MBON15 | 4 |
| MBON14 | 4 |
| MBON15-like | 4 |
| MBON25-like | 4 |
| MBON19 | 4 |
| MBON12 | 4 |
| MBON35 | 2 |

## 性二型(論文 Table 対応)

| dimorphism | count |
|---|---|
| None | 209,209 |
| male-specific | 1,258 |
| sexually dimorphic | 771 |
| potentially sexually dimorphic | 177 |
| potentially male-specific | 162 |

- fruDsx annotated: 5,012

## 実装定数への反映方針

1. **採用済み**: PN 1 チャネル/糸球体、KC fan-in ~7(hemibrain 公表値)、APL/LN 抑制、DAN 正則規則。
2. **実測差分(次イテレーション候補、変更前に性能比較を必須)**: KC 2,000→4,064、LN のサブタイプ分化、MBON のチャネル分割、CX 領域(cb_intrinsic 32,164 の中核)は Goal 4 で設計。
3. 本統計は設計レビューのたびに再実行し、実データとの乖離を監査する。
