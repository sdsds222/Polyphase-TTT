# Polyphase TTT reproduction

**Article:** Polyphase TTT: Training-Free Stabilization across Arbitrary Chunk Boundaries  
**Journal:** Applied Intelligence  
**Author:** Dian Yu  
**Affiliation:** School of Computer Science and Engineering, UNSW Sydney, Sydney, NSW, Australia  
**Corresponding email:** dian.yu2@student.unsw.edu.au

This repository accompanies **Polyphase TTT: Training-Free Stabilization across Arbitrary Chunk Boundaries**. The work studies a robustness issue in frozen TTT-MLP sequence models: predictions can change when the same fixed-width chunk grid is shifted to a different internal origin, even though the input sequence and checkpoint remain unchanged.

Polyphase TTT is a training-free inference wrapper that reduces this chunk-origin sensitivity while preserving the original checkpoint and fast-weight update rule. This repository provides the code and locked outputs used to reproduce the experiments reported in the manuscript.

## Contents

- `source/`: minimal experiment and analysis code.
- `results/`: locked raw JSON outputs used in the manuscript.
- `requirements.txt`: the tested Python environment.

The paper gives the algorithm and experimental rationale; this file gives a concise reproduction path for the reported results.

## Setup

Python 3.11, CUDA, and a recent NVIDIA GPU are recommended. Install the tested dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the public checkpoints into the directory shown below:

```bash
mkdir -p source/third_party
hf download Test-Time-Training/ttt-mlp-350m-books-2k --local-dir source/third_party/ttt_mlp_350m
hf download Test-Time-Training/ttt-mlp-760m-pile-8k --local-dir source/third_party/ttt_mlp_760m_hf
```

Place the official LAMBADA test JSONL at `source/data/lambada/lambada_test.jsonl`. SQuAD v1.1 and HotpotQA are downloaded automatically by `qa_data.py` when absent. The expected paths are `source/data/squad/dev-v1.1.json` and `source/data/hotpotqa/hotpot_dev_distractor_v1.json`.

`ttt_ckpt_util.py` repairs two non-persistent buffers omitted from the public checkpoint state dictionary: RoPE `inv_freq` and the TTT `token_idx` schedule. All runs must load checkpoints through `load_ttt`.

## Primary LAMBADA evaluation

Run from `code/source`:

```bash
python -u ttt_polyphase_lambada_final.py --model-dir third_party/ttt_mlp_350m --data data/lambada/lambada_test.jsonl --questions 384 --seed 20261006 --exclude-runs 20261004:64,20261005:96 --max-tokens 512 --threshold 0.04 --residual-strength 0.25 --out ../results/reproduced_350m_n384.json

python -u ttt_polyphase_lambada_final.py --model-dir third_party/ttt_mlp_760m_hf --data data/lambada/lambada_test.jsonl --questions 384 --seed 20261006 --exclude-runs 20261004:64,20261005:96 --max-tokens 512 --threshold 0.04 --residual-strength 0.25 --out ../results/reproduced_760m_n384.json
```

The confidence cascade uses origins 0, 8, 4, and 12 in that order, threshold 0.04, and residual strength 0.25. The locked reference outputs are `ttt_polyphase_lambada_final_350m_n384.json` and `ttt_polyphase_lambada_final_760m_n384.json`.

## Complete-origin diagnostic

```bash
python -u ttt_polyphase_lambada.py --model-dir third_party/ttt_mlp_350m --data data/lambada/lambada_test.jsonl --mode phase --questions 96 --seed 20261005 --exclude-seed 20261004 --exclude-questions 64 --phase-offsets 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --residual-strength 0.25 --out ../results/reproduced_all16_n96.json
```

## External QA checks

The following commands reproduce the largest held-out SQuAD and HotpotQA runs. Earlier selection runs are retained in `results/` so that `polyphase_disjoint_final_analysis.py` can reconstruct the strictly disjoint evaluation sets.

```bash
python -u ttt_polyphase_real_binding.py --model-dir third_party/ttt_mlp_350m --dataset squad --data data/squad/dev-v1.1.json --answer-kinds proper,number,other --groups-per-kind 8 --length 512 --seed 20261008 --residual-strength 0.25 --out ../results/reproduced_squad_seed4_n96.json

python -u ttt_polyphase_real_binding.py --model-dir third_party/ttt_mlp_760m_hf --dataset hotpotqa --data data/hotpotqa/hotpot_dev_distractor_v1.json --answer-kinds proper,other --groups-per-kind 18 --exclude-specs 20261009:2:1024,20261010:6:1024 --length 1024 --seed 20261012 --residual-strength 0.25 --out ../results/reproduced_hotpot_expand_n144.json
```

To rerun the locked aggregation, run:

```bash
python polyphase_disjoint_final_analysis.py
```

## Figures and integrity

`make_figures.py` reads the locked files in `results/` and writes publication figures to `figures/`. Run `python source/make_figures.py` from `code/`. `SHA256SUMS.txt` records hashes for all files in this package.

The main paper reports exact-match decisions, paired prediction/correctness flips, latency, phase count, and peak allocated CUDA memory. Do not replace paired flip analysis with aggregate accuracy alone: opposite example-level changes can cancel in the mean.
