# RESEARCH — MaleCNS 以外の脳研究プロジェクトからの統合

> 原則: 「生物学的に存在する」は採用理由にならない。
> 各プロジェクトからは **AI 能力(学習・記憶・汎化・安定性・計算効率)に
> 有効な原理のみ**を選び、効果を測定して採用/棄却する。
> 状態の 3 分類: `採用済(実測)` / `採用予定(仮説・実験計画あり)` / `棄却(理由明記)`。

## 1. Janelia FlyEM / MaleCNS・hemibrain・MANC・fafb — 基盤構造の供給源

- 内容: オス成虫 CNS 166,691 ニューロン / 11,691 細胞型の完全コネクトーム
  (Cell 2026)、hemibrain (2020)、MANC=オス VNC (2023)、fafb=メス全脳 (2025)。
- 供給される構造原理: 疎ランダム拡張(PN→KC fan-in ~7)と照合検出、
  正則化された抑制(APL/LN)、教師信号 = 可塑性ゲート(DAN)、
  価値の 2 分枝読み出し(MBON)、感覚→行動の完全経路。
- **採用済(実測)**: AL/MB/MBON/DAN 構造(Goal 1 で学習が実測済み、RESULTS §2)。
- 採用予定: CX の corollary discharge(E3)、LH innate 経路、VNC 運動回路の選択構造。

## 2. Blue Brain Project (BBP) — 細胞型分化と抑制サブクラス

- 内容: ラット S1 のデジタル再構成 — 約 31,000 ニューロン、**207 形態電気型**、
  3,700 万シナプス(Markram et al., Cell 2015)。抑制性細胞の運動学サブタイプ
  (fast/slow、disinhibitory 回路)まで分類。
  https://portal.blueblue…(NMC Portal)、https://pubmed.ncbi.nlm.nih.gov/36213546/
- **採用予定(仮説)**: LN プールを抑制サブクラスに分化(高速 fenestra 局所抑制 /
  遅い広域抑制 / 脱抑制)。期待: 利得制御の帯域拡大とパターン分離の改善。
  実験: 単一 LN vs 分化 LN で分離度・学習速度を比較し、**改善がなければ棄却**。
- **棄却(現段階)**: 形態 + 多区画 biophysical 詳細の全面導入。
  理由: 現マイルストーンの能力指標への寄与が未実証で、計算コストが 2–3 桁増。
  ラダー L3 として再評価を保留。

## 3. Allen Institute / BRAIN Initiative Cell Census Network (BICCN)

- 内容: マウス全脳の完全細胞型アトラス — **5,300+ 細胞型、3,000 万細胞プロファイル**
  (Nature 10 連作, 2023)。https://alleninstitute.org/news/scientists-unveil-first-complete-cellular-map-of-adult-mouse-brain/
  https://biccn.org/science/whole-mouse-brain
- **採用済(思想)**: 「集団を型(taxonomy)で組織する」— 我々の population 設定は
  型ごとの config 構造として実装済み(MB/AL 各 config)。人口統計(型分布)の
  定量報告を RESULTS の標準項目化。
- **採用予定(仮説)**: 型ごとのパラメータ分化(τ_m、閾値、時定数を型内共通・型間分散)。
  実験: 均一 vs 型分化でタスク性能・頑健性を比較。

## 4. Human Brain Project / EBRAINS — ループで閉じた評価とマルチスケール

- 内容: Neurorobotics Platform(仮想身体での closed-loop 実験標準)、
  NEST↔The Virtual Brain のマルチスケール共シミュレーション、
  7.7 万ニューロンモデルの公開実行基盤。
  https://www.humanbrainproject.eu/en/science-development/focus-areas/neurorobotics/
  https://ebrains.eu/news-and-events/2021/ebrains-shares-access-to-improved-laptop-to-supercomputer-brain-simulator
- **採用済(思想)**: 「脳モデルは環境との閉ループで評価する」— 本プロジェクトは
  常に環境層 (env/) とセットで評価する(Goal 1–4 すべて closed-loop)。
- **採用予定**: マクロ(領域レベル平均場)とミクロ(スパイキング回路)の
  共シミュレーション分割 — 大規模化(Goal 4 以降)で領域間ダイナミクスを
  安く持つ手法として評価。
- **棄却**: 外部プラットフォームへの依存(ゼロ依存原則。思想のみ輸入)。

## 5. 革新脳 Brain/MINDS(+Beyond)

- 内容: マーモセット全脳のマルチスケール(μ/meso/macro)アトラスと統合データ基盤
  (Okano et al., Neuron 2016; 3D atlas, Sci Data 2018)。
- **採用済(思想)**: 階層的アトラス設計 — 領域モジュールを標準インターフェースで
  積み上げる拡張方式(本リポジトリの circuits/ レイヤー設計はこれに準拠)。
- 採用予定: 種横断の「相同構造」対応表 — 将来哺乳類構造(皮質様回路)へ拡張する際の
  設計指針。

## 6. 理論(自己・感情の出現を検証可能にする枠組み)

| 理論 | 我々への適用 | 状態 |
|---|---|---|
| corollary discharge / efference copy(ハエ CX で実在) | 自己由来入力の差し引き → E3 自己/他者区別 | 採用予定(実験計画済) |
| attention schema (Graziano) | 自己状態の内部モデル = E4 の自己状態回路 | 採用予定 |
| 予測処理 (Friston) | 自己予測誤差を状態変数として保持 | 採用予定 |
| somatic marker (Damasio) | 感情状態が意思決定閾値を変調(E2 の変調経路) | 採用予定 |
| global workspace (Dehaene) | 価値・覚醒の大域放送(変調の全領域適用) | 仮説(設計に反射、要実験) |

すべて「実装結果 = 成立」ではなく、ROADMAP E 段階の各判定基準で測定する。

## 7. 統合のルール(このリポジトリ内での決定手順)

1. 機構候補を挙げる(どのプロジェクト由来か明記)。
2. 期待される AI 能力への効果を 1 文で仮説化。
3. 最小実装 → 実験 → 比較(変更前後必須)。
4. 改善が実測されたら採用、されなければ棄却して RESULTS.md に記録。
   **「生物学に存在するから」は理由にならない。**
