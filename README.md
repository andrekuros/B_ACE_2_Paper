# B-ACE 2.0 paper

[![Open in Overleaf](https://img.shields.io/badge/Open%20in-Overleaf-47A141?logo=overleaf&logoColor=white)](https://www.overleaf.com/docs?snip_uri=https%3A%2F%2Fraw.githubusercontent.com%2Fandrekuros%2FB_ACE_2_Paper%2Fmain%2Foverleaf%2Fbace20-overleaf.zip&engine=pdflatex)
[![latex](https://github.com/andrekuros/B_ACE_2_Paper/actions/workflows/latex.yml/badge.svg)](https://github.com/andrekuros/B_ACE_2_Paper/actions/workflows/latex.yml)

LaTeX draft for *B-ACE 2.0: An Open, Physically Grounded Framework for Beyond-Visual-Range
Air Combat Research*. The simulator lives in [B-ACE-2.0](https://github.com/andrekuros/B-ACE-2.0).

## Open in Overleaf

Click **Open in Overleaf** above: Overleaf creates a new project in your account from
`overleaf/bace20-overleaf.zip` (sources, bibliography, `numbers.tex` and figures) and
compiles it with pdfLaTeX. Without the badge: Overleaf, New Project, Upload Project, and
choose that zip. Overleaf projects made this way are copies; to sync edits with this
repository use Overleaf's GitHub integration (Menu, GitHub) or download the project and
commit the changed files here.

After changing figures or numbers, rebuild the zip with `python make_overleaf.py`; CI
fails if it is stale.

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
| `make_overleaf.py`, `overleaf/` | Flat zip of the LaTeX project for Overleaf |

## License

MIT, see [LICENSE](LICENSE). Same authors as B-ACE 2.0.
