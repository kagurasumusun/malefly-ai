# RESEARCH_CODE — 脳型構造 AI コーディングエージェントの神経基盤と研究文献

本ドキュメントは、本リポジトリにおける**脳型構造 AI コーディングエージェント**の設計・実装・検証にあたり参照した神経科学・コネクトーム・認知計算論・統計的言語/コード学習の文献および実装マッピングの対応を記録したものである。

---

## 1. 脳領域とコード処理の構造写像 (Structural Mapping)

ショウジョウバエ MaleCNS コネクトームおよび哺乳類高次脳構造（海馬・前頭前野・基底核）から抽象化した回路配置:

```
[コード入力 (AST / CFG / トークン)]
            │ 256-channel Sensory Recoding
      ┌─────▼──────┐
      │ Code       │ 疎表現展開 (Antennal Lobe / Glomerular Recoding)
      │ Sensorium  │ 構文木(AST)・制御フロー(CFG)のマルチチャネルベクトル変換
      └─────┬──────┘
            ├──────────────────────────┐
      ┌─────▼──────┐             ┌─────▼──────┐
      │ Mushroom   │ 価値学習    │ Prefrontal │ Persistent firing
      │ Body (MB)  │ RPE可塑性   │ Cortex     │ 遅延保持・スコープ作業記憶
      └─────┬──────┘ (DAN)       └─────┬──────┘ (DLPFC)
            │                          │
      ┌─────▼──────┐             ┌─────▼──────┐
      │ Hippocampus│ DG->CA3    │ Basal      │ 行動選択 & 候補評価
      │ (HPC)      │ SWR Replay  │ Ganglia    │ (Softmax / Actor-Critic)
      └────────────┘             └────────────┘
```

---

## 2. 参照文献と実装箇所の対応表

| 文献 | 生物学的・計算論的発見 | 本 AI の実装箇所 | 回路・メカニズム実装 |
|---|---|---|---|
| **Saffran, Aslin & Newport (Science 1996)** | 乳幼児の音節間遷移確率 (TP) による統計的語分節・言語習得 | `src/circuits/code_sensorium.hpp` | 構文トークンおよび AST 節間の痕跡タイマー型 Hebb 可塑性 ($P(\text{next} \mid \text{curr})$) |
| **McClelland, McNaughton & O'Reilly (Psychol Rev 1995)** | 相補的学習システム (CLS): 海馬高速 1-shot エピソード記憶 vs 新皮質/MB 低速統計学習 | `src/circuits/code_brain.hpp`, `src/circuits/hippocampus.hpp` | HPC 1-shot エピソードスタンプ (DG/CA3) と Mushroom Body (MBON) RPE 価値学習の並列統合 |
| **Buzsáki (Neuron 2015)** | 静止中の Sharp-Wave Ripple (SWR) リプレイによる記憶固定 | `src/circuits/code_brain.hpp` | コード試行間の静寂期における CA3 アセンブリ再点火と固定化プロセス (`replay_rest_phase()`) |
| **Goldman-Rakic (Neuron 1995)** | DLPFC における遅延試行中の持続放電 (Persistent Activity) と作業記憶 | `src/circuits/pfc.hpp`, `src/circuits/code_brain.hpp` | 長い試行間・ネストスコープを跨いで入力ベクトル表現をアトラクタ保持する PFC レイヤー |
| **Schultz, Dayan & Montague (Science 1997)** | 中脳ドーパミン神経による報酬予測誤差 (RPE) 信号 | `src/circuits/mushroom_body.hpp` | コード実行・サンドボックス検証結果 (Pass/Fail) に基づく DAN ドーパミン RPE ゲート可塑性 |

---

## 3. DevOps 実測主義と統計評価指標 (DevOps Metrics)

本 AI は以下の定量的指標を用いて多角的に評価する:

1. **実行通過率 (Sandbox Pass Rate):** パイソン実行サンドボックスにおける構文・ロジック・テストアサーション通過率。
2. **学習曲線 (RPE Loss / Reward Curve):** 試行回数に対する報酬・誤差シグナルの収束挙動。
3. **神経ダイナミクス (Neural Dynamics):** PFC Persistent Firing 発火率、Hippocampal SWR Replay 発生数、Mushroom Body KC 疎性比率。
