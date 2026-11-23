# Transformer digit reversal

Train a small PyTorch encoder–decoder to turn `38` into `83 <eos>`. The notebook replays training updates and shows attention, gradients, and Adam with real values.

[Live demo](https://singhcoder.github.io/ai-engineering/digit-reversal/) · [Notebook](digit-reversal.ipynb)

## Run locally

Use Python 3.12 from this directory.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/jupyter notebook digit-reversal.ipynb
```

Run all cells. Open **Cross-attention → Scores · head 1**, then click a cell. Use the update slider and **Forward / Backward** to follow its math. Edit the configuration cell to change the architecture.

Run `.venv/bin/python train.py` for the terminal version or `.venv/bin/python export_demo.py` to rebuild the demo.

The architecture follows [Attention Is All You Need](https://arxiv.org/abs/1706.03762). The viewer uses styles adapted from [Google PAIR Explorables](https://pair.withgoogle.com/explorables/data-leak/) under the [Apache 2.0 license](LICENSE-PAIR).
