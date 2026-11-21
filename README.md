# B-ACE 2.0 paper

LaTeX draft for *B-ACE 2.0: An Open, Physically Grounded Framework for Beyond-Visual-Range
Air Combat Research*. The simulator lives in [B-ACE-2.0](https://github.com/andrekuros/B-ACE-2.0).

## Build the PDF

```bash
pdflatex bace20 && bibtex bace20 && pdflatex bace20 && pdflatex bace20
```

## Regenerate the data, figures and numbers

In a clone of B-ACE-2.0 (with `pip install maturin numpy pettingzoo gymnasium torch` and
`maturin develop --release -m crates/bace-py/Cargo.toml`):

```bash
python -m bace.studies all --out runs/paper
for A in 1 2; do for S in 0 1 2; do
  python -m bace.ppo --agents $A --seed $S --steps 4000000 --out runs/paper/rl/ippo_${A}v${A}_s$S
  python -c "from pathlib import Path; from bace.ppo import select_checkpoint; select_checkpoint(Path('runs/paper/rl/ippo_${A}v${A}_s$S'))"
done; done
python -m bace.studies rl_eval --policies runs/paper/rl/*/best.pt --out runs/paper
python -m bace.studies rl_train --runs runs/paper/rl/* --out runs/paper
```

Copy `runs/paper/*.json` into `data/` (the WEZ file without its `outcomes` lists), then:

```bash
pip install matplotlib
python make_figures.py      # figures/*.pdf, data/numbers.json, numbers.tex
```

`numbers.tex` defines every number quoted in the text as a macro, so the draft cannot drift
from the data. `make_figures.py` runs no simulation.

## Contents

| File | Role |
|------|------|
| `bace20.tex`, `refs.bib`, `numbers.tex` | Draft, bibliography, generated numbers |
| `figures/` | Figures (PDF for the paper, PNG previews); `lab_*.png` are Lab screenshots |
| `data/*.json` | Study outputs from `bace.studies` |
| `data/numbers.json` | Every number used in the text |
| `make_figures.py` | Figures and numbers from `data/` |

## License

MIT, see [LICENSE](LICENSE). Same authors as B-ACE 2.0.
