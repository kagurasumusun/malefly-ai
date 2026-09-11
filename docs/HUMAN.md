# HUMAN.md — 人の脳・記憶・学習の文献と実装の照合表

> サルレベル指令（「いろいろなコネクトームからサルや人に近づけるために必要な要素を取り入れる。
> 人の脳・記憶・学習などの論文も徹底的に調べて照らし合わせる」）に基づく、
> 文献 → 実装 → 測定結果 の対応表。結果はすべて実際のランから（捏造禁止）。

## 1. 採用した文献と実装の対応

| 文献 | 示したこと | 実装箇所 | 実装内容 | 測定結果 (monkey1 seed300, 現HEAD) |
|---|---|---|---|---|
| McClelland, McNaughton & O'Reilly 1995 (Psychol Rev) + Kumaran et al. 2016 (CTDL, PMC6964152) | 海馬: 高速・疎・一回学習。新皮質(ここではMB): 低速・統計的学習。新奇性が格納を司る | `src/circuits/hippocampus.hpp`, `brain.hpp` MB | HPC = one-shot エピソード記憶（DG→CA3 疎アセンブリ + US時 stamp-in）、MB = 低速弁別学習。2系統を並列に同一 PN ストリームへ接続 | B: completion 0.167, エピソード 525; C: 獲得 +0.080（>0 を達成） |
| Rolls 2013 (Nat Rev Neurosci, PMC3812781) | DG の疎展開でパターン分離; 苔状線維の randomizing; CA3 反回帰（希釈）で自動連想→パターン完成; LTP + ヘテロシナプス LTD; CA3 疎性 ~2–5% | `hippocampus.hpp` | DG n1500 (fanin 5) → CA3 n1200 (mossy fanin 40, 希釈反回帰 fanin 60, one-shot η=0.55, cap, homeo LTD)。CA3 疎性を閾値/量子で ~2–8% に調整 | CA3 frac: odor ~0.03–0.08, baseline ~0.002。part cue (65%) → valence 再生 v+=193 / v−=0 |
| Frey & Morris 1997 (Nature) — synaptic tagging & capture | 弱い入力でも「強い事象(US)」が近接して起きると、進行中の活動にタグが付き LTP が起きる | `hippocampus.hpp` step() encode gate | US live (us_hold 800ms) 中は反回帰書き込みの不応期を無視して stamp-in（us_refrac 500ms = US 1回あたり実質 1 スタンプ） | A+ stamp 時 episodes 増加 + valence 結合成立（B で v+=193 再生） |
| Li, Cullen, Anwyl & Rowan 2003 (Nat Neurosci) | 新奇事象が海馬 LTP を促進（変調性 US が可塑性を開く） | 同上 | encode gate の US オーバーライド機構 | 同上 |
| Brown & Aggleton 2001 (Nat Rev Neurosci) | 認知は familiarity(周囲皮質系) と recollection(海馬系) の二過程 | `hippocampus.hpp` perirhinal (PR) 層 | CA3→PR (n64, fanin 30, 無音初期化)。encode 時の stamp で「体験済み結合」を形成し、cue で PR が発火 = familiarity。活動量 EMA ではなく**結合量ベースの match 信号** | cue65 familiarity 0.44 vs 新奇 ≈0（活動量指標だと区別不能だった問題を修正） |
| Funahashi, Bruce & Goldman-Rakic 1989 (J Neurophysiol); Curtis & D'Esposito 2003 | DLPFC: 刺激選択的な遅延中持続放電 = 作業記憶 | `src/circuits/pfc.hpp` | n400 bank, KC fanin, τm 0.040s。遅延中の持続活動を測定 | 部分的: 維持は ~1–1.75s で崩壊。D=1s での保持は現在**不合格**（次項 §3） |
| Compte 2000 (Cereb Cortex); Wang 2001 (Nature) | 持続活動には (a) NMDA 型の遅い興奮 (b) E-I バランス（興奮/抑制）が必須。無抑制だと飽和・同期で崩壊 | `pfc.hpp` | τexc (NMDA様), E-I フィードバック抑制 (n_inh 100, E→I→E) 追加。†飽和運用点の再較正: 全結合量を閾値スケールへ (ge≈120 → 閾値 ~0.02 の 6000 倍飽和を検出・修正) | 遅延活動は存在するが再帰応募で漸増 → probe 判別信号がまだ弱い |
| Miller & Desimone 1994 (Neuron) | IT での match enhancement: 標本と一致する probe が再活性化する。累積 familiarity ではなく試行内 match | `monkey1.cpp` DMS rule, `pfc.hpp` fam_fast | fast EMA (τ=0.15s) の match 信号、絶対閾値 0.45 | D=0 では信号出現（match 0.73 max vs lure 0.002）だが D≥1s で消失（維持崩壊のため） |
| Markov et al. 2014 (Cereb Cortex) — macaque FLNe | 領域間結合重みは対数正規分布（5桁）| 全シナプス生成 (`make_random_fanin_dist`) | lognormal 重み σ=0.30, fan-in 分布 | 全モジュールで使用中（MICrONS 由来の採用と整合） |

## 2. 接続体からの採用（既存 §RESEARCH.md 6b との関係）

- **MaleCNS**: 全体ワイヤリング基盤（KC 4,064/MBON 97/DAN 340 相当の比率感覚）。
- **Honeybee VUMmx1 (Hammer & Menzel 1995)**: US = RPE 1 細胞。`drive_vum/drive_dan` により HPC が予測 US を VUM に内部駆動（B/C のループ）。
- **MICrONS**: lognormal 重み・fan-in 分布（前表 Markov と相互整合）。

## 3. 現在の判定（honest state, seed300）

| ベンチ | 判定 | 値 |
|---|---|---|
| A. DMS (D=1s >0.7, distractor active>rest) | **FAIL** | 0.35 / active 0.55 / rest 0.45 / D=4s 0.35。信号は D=0 で存在、D≥1s は維持崩壊 |
| B. エピソード完成 + valence 再生 | **valence PASS / completion 弱** | v+=193, v−=0（方向正しい）。completion 0.167 |
| C. 検索練習による固定 | **PASS (>0)** | +0.080 (0.023→0.678 から較正後の値へ変動あり。詳細は results/monkey1.json) |
| D. 間隔効果 (>0) | **FAIL** | −0.073 (massed 0.436 vs spaced 0.363) |

## 4. 次に調査・採用予定の文献（キュー）

-基底核 RL（DAN-Striatum、家禽≠霊長: Graybiel 2008, Samejima 2005）
- 睡眠中リプレイと系統的固定（Wilson & McNaughton 1994; Rasch & Born 2013）→ C の 60s gap を「sleep 相」に
- 新皮質の低速統合（Fusi, Drew & Abbott 2005; Benna & Fusi 2016）→ MB のカスケード記憶
- 皮質階層（Markov 2014 の FLNe 階層、Mesulam）→ 領域間の feedforward/feedback 非対称
- DMS 用: 疎コーディングの再設計（KC fanin ↓、coincidence threshold、onset アセンブリ縛りは実装済み → 次は入力疎性）
