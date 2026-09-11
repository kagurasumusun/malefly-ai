# malefly-ai — MaleCNS 由来の構造化神経回路 AI

MaleCNS(オス成虫ショウジョウバエ中枢神経系コネクトーム, 166,691 ニューロン /
11,691 細胞型, Janelia FlyEM+Cambridge+Google, Cell 2026)を**設計書・構造的基盤**として、
C/C++ でゼロから構築する脳型 AI。

方針は深層NNの積み重ねではなく、**構造・ダイナミクス・可塑性が一体化した神経回路**
そのものに計算・記憶・学習・制御を担わせること。最終目標は、経験で自分の行動を変え、
未知環境にその場で適応し、最終的に自己認識・感情・自我までもが**構造から芽生える**
継続的な人工主体(定義と測定で検証する。断定はしない)。

```
           匂い (50糸球体チャネル, 実糸球体名は data/malecns/glomeruli.csv)
              │
        ┌─────▼──────┐   利得制御(LN, 振動同期)
        │ Antennal   │◄──┐            「構造そのものが計算する」:
        │ Lobe (AL)  │───┘            ・疎ランダム拡張 PN→KC fan-in 7 がパターン分離
        └─────┬──────┘                ・KC 高閾値が照合検出
              │ 4,000本の固定結合(CSR)  ・APL 抑制が疎性(~7%)を強制
        ┌─────▼──────┐                ・KC→MBON 可塑性重みが記憶そのもの
        │ Mushroom   │  2,000 KC      ・DAN 教師信号が正則(Aso 2014)で書き込む
        │ Body (MB)  │──MBON(2価値)──► 行動選択 (Approach/Avoid)
        └─────▲──────┘                 │
              │ DAN (+1報酬 / -1罰) ◄──┘ 環境が結果を返す
        ┌─────┴──────┐
        │  World     │  bandit → 記憶課題 → 逆転学習 → センソリモータ (Goal順)
        └────────────┘
```

## 現状(すべて実測。詳しくは docs/RESULTS.md)

| Goal | 内容 | 状態 | 主要実測値 |
|---|---|---|---|
| 1 | 自律的な価値学習(バンディット) | **達成**(4 seeds) | p(A)=0.66→**0.86–0.90**、p(B)=0.67→**0.21–0.29**、最終ブロック正答 **1.00**、構造的一般化 r=**0.92–0.99** |
| 2 | 記憶(保持/干渉/遅延選択) | **達成** | 120s無強化後も劣化なし(Δ≤0.005)、干渉下で並列維持、**作業記憶 D=4s で正答1.00**(KC亜型分化により1s→4s+へ延長)、ソフトウェア記憶との差別化も実測 |
| 3 | 未知環境への適応(価値逆転、再訓練禁止) | **達成** | **逆転 3/3 seeds**(t70=66/69/39)、再逆転で高速化の兆候(49<66, 27<39)、ablation で覚醒変調・KC型分化の寄与を分離測定 |
| 4 | 感覚→判断→行動統合(CX 導入) | 設計済 | ROADMAP §Goal 4 |
| E | 感情・自己・自我(構造から) | E1実装・**E2覚醒を実装・実測**・E3/E4設計 | ROADMAP §E(定義→構造→実験→判定を先置き) |

単体テスト: **14,192 checks, 0 failures**。同 seed は bit 単位で再現(run_hash 検証)。

## 5機関プロジェクトの統合(すべて実測付き採用判定)

| 採用源 | 組み込んだ機構 | 実測された効果 |
|---|---|---|
| MaleCNS(実データ) | KC亜型比率・回路骨格 | 全Goalの基盤、STM延長 |
| Blue Brain Project | 抑制サブクラス分化(fast/slow LN) | 利得制御の二重化、疎性維持 |
| Allen BICCN | 細胞型タキソノミー(実データ比率) | **作業記憶 1s→4s+** |
| HBP/EBRAINS | マクロ覚醒↔ミクロ回路(E2) | 反復逆転適応の成立に必要 |
| BRAIN Initiative | RPEゲート可塑性 | 保留( Goal 4 で再測定) |
| 革新脳 Brain/MINDS | 標準Regionインターフェース | LHモジュール実装、CX拡張の道 |

詳細: `docs/RESEARCH.md`(採用/予定/棄却の根拠)、`docs/RESULTS.md` §6–8。

## ビルド & 実行(ninja のみ。依存は C++17 標準 + vendored nlohmann/json)

```sh
ninja                       # build/malefly + build/test_core
./build/test_core           # 単体テスト
./build/malefly goal1 --seed 42     # Goal 1: 価値学習 (results/goal1.json|csv)
./build/malefly goal2 --seed 100    # Goal 2: 記憶 (results/goal2.json|csv)
python3 tools/plot_results.py       # 図生成 (results/figures/*.png)
sh tools/fetch_malecns.sh           # MaleCNS v1.0 実データ取得 + 統計 (docs/DATA.md)
```

## 実データ接地

MaleCNS v1.0 の公開実データ(CC-BY, male-cns.janelia.org)を使用:
- `data/malecns/body-annotations-*.feather` — **211,577 ニューロンの実アノテーション**
- 実測: KC **4,064** / MBON **97** / DAN **340**(PPL1 16・PAM 316)/ ALPN **686** /
  ALLN **420** / 実糸球体 **53** — `docs/DATA.md` に自動集計
- 実装(2,000 KC 等)と実データの乖離は文書化し、拡張は必ず実測比較の上で行う

## MaleCNS 以外の統合源

Blue Brain Project / Allen BICCN / HBP-EBRAINS / 革新脳 Brain/MINDS からは
**AI 能力に有効な原理のみ**を選別(採用・予定・棄却の根拠は `docs/RESEARCH.md`)。

## ドキュメント

- `docs/PRINCIPLES.md` 絶対原則(実測主義・再現性・偽装禁止)
- `docs/MALECNS.md` 設計書としての MaleCNS 事実と実装写像
- `docs/DESIGN.md` アーキテクチャとモデル階層ラダー
- `docs/ROADMAP.md` Goal 1–4 と E 段階(感情・自己・自我)の判定基準
- `docs/RESEARCH.md` 外部プロジェクト統合の記録
- `docs/RESULTS.md` **全実測結果(変更前後比較を含む)**
- `docs/DATA.md` MaleCNS 実データ統計(自動生成)

## ライセンス

MIT(コード)。MaleCNS データは CC-BY(Janelia FlyEM ほか)。
