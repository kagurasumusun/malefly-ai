# SYSTEM_SPEC — 脳型構造 AI コーディングエージェント 要件定義・企画・システム仕様書

## 1. プロジェクト目的と要件定義 (Requirements)

本プロジェクトの目的は、深層ニューラルネットワーク (LLM 等) のパラメタ追加に依拠せず、**生物学的脳構造 (ショウジョウバエ MaleCNS コネクトーム + 霊長類高次脳構造)** のダイナミクス・可塑性・記憶・比較評価系から動く**実用的脳型 AI コーディングエージェント**をゼロから C++17 で構築することである。

### 必須要件 (Core Requirements):
1. **構造的コード表現 (Structural Code Representation):**
   - 単なるテキスト文字列ではなく、抽象構文木 (AST) および制御フロー (CFG) の構造を 256 チャネルの神経入力ベクトルに変換・展開する。
2. **階層的脳領域統合 (Multi-Region Brain Integration):**
   - **PFC (前頭前野):** ネスト構造・スコープを跨ぐ持続的発火 (Persistent Firing) による作業記憶保持。
   - **Hippocampus (海馬):** DG パターン分離 $\rightarrow$ CA3 One-shot エピソード記憶 $\rightarrow$ 静寂期の Sharp-Wave Ripple (SWR) リプレイによるオフライン記憶固定。
   - **Mushroom Body (MB):** ドーパミン (DAN) による報酬予測誤差 (RPE) 可塑性ゲート。
3. **実環境サンドボックス検証 (Real Execution Verification):**
   - パイソン サンドボックスプロセスにてコード候補を実行検証し、構文・アサーション通過・例外エラーに基づいてドーパミン強化シグナル (Pass = +1.0, Fail = -1.0) をフィードバックする。
4. **DevOps型 比較実験・統計検証 (DevOps Continuous Statistical Evaluation):**
   - 4つの回路バリアント (A: 標準Hebbian, B: RPEドーパミンゲート, C: PFC+海馬リプレイ, D: 統合脳) を複数 seed で自動並列評価し、学習曲線と通過率を統計化する。

---

## 2. 認知システム構成図 (System Architecture)

```
 [ コード入力 ]
       │
 ┌─────▼────────────────────────────────────────┐
 │ CodeASTEncoder / CodeSensorium (256-channel) │
 └─────┬──────────────────┬─────────────────────┘
       │                  │
 ┌─────▼──────┐     ┌─────▼──────┐
 │ PFC        │     │ Mushroom   │ 報酬予測誤差 (RPE)
 │ (作業記憶) │     │ Body (MB)  │ (DAN Dopamine Modulation)
 └─────┬──────┘     └─────┬──────┘
       │                  │
 ┌─────▼──────┐     ┌─────▼──────┐
 │ Hippocampus│     │ Basal      │ 行動選択
 │ (SWRReplay)│     │ Ganglia    │ (Softmax / Candidate Readout)
 └────────────┘     └─────┬──────┘
                          │
                    ┌─────▼──────┐
                    │ CodeSandbox│ リアル Python 実行・アサーション検証
                    └────────────┘
```

---

## 3. 開発・検証フェーズ (DevOps Cycle)

1. **要件定義 & 企画:** 本文書 (`docs/SYSTEM_SPEC.md`)
2. **研究 & 文献調査:** 神経科学・コネクトーム文献照合 (`docs/RESEARCH_CODE.md`)
3. **システム化 & 回路実装:** `src/circuits/code_brain.hpp`, `code_ast.hpp`, `code_brain_variants.hpp`
4. **サンドボックス & コーパス構築:** `src/env/code_sandbox.hpp`, `src/circuits/code_corpus.hpp`
5. **DevOps 比較ベンチマーク:** `src/experiments/code_brain_devops.cpp`
