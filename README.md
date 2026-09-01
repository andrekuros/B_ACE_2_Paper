# B-ACE 2.0 working paper

Shareable LaTeX draft for B-ACE 2.0 (not a camera-ready I/ITSEC template).

The simulator lives in a separate repository: [B-ACE-2.0](https://github.com/andrekuros/B-ACE-2.0).
This repo holds the draft, figures, and the script that regenerates them.

## Build the PDF

```bash
pdflatex bace20
bibtex bace20
pdflatex bace20
pdflatex bace20
```

Output: [`bace20.pdf`](bace20.pdf).

## Regenerate figures from the sim

Install the framework first (from a clone of B-ACE-2.0):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install maturin pytest pettingzoo gymnasium numpy matplotlib
maturin develop -m crates/bace-py/Cargo.toml
```

Then, in this paper repo (with that environment active):

```bash
python make_figures.py
```

Paper-scale experiment profiles (`wez`, `fsm`, `marl_core`, `marl_striker`, `marl_tracks`) are defined in the framework under `configs/experiments/` and run as:

```bash
python -m bace.experiment wez --profile paper
python -m bace.experiment fsm --profile paper
python -m bace.experiment marl_core --profile paper
```

`make_figures.py` invokes those recipes itself.

## Contents

| File | Role |
|------|------|
| `bace20.tex` | Draft |
| `refs.bib` | Bibliography |
| `bace20.pdf` | Compiled draft |
| `figures/*.pdf` | WEZ hit/fit/traces, FSM search/occupancy, MARL curves/eval, tactical 4v4 |
| `data/paper_results.json` | Numbers used in the text |
| `make_figures.py` | Regenerates figures (requires `bace`) |

## License

MIT — see [LICENSE](LICENSE). Same authors as B-ACE 2.0.
