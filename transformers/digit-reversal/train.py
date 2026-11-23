"""Train a small 2017-style encoder–decoder Transformer to reverse digits."""

import argparse
import math
import sys

import torch
from torch import nn
from torch.nn import functional as F

DIGITS = 10
LENGTH = 2
BOS = DIGITS       # Decoder input token marking the beginning of the answer.
EOS = DIGITS + 1   # Decoder token marking the end of the answer.
WIDTH = 4         # Features carried by each token between layers.
HEADS = 2          # Each head uses WIDTH / HEADS features.
INNER = 8         # Features inside the feed-forward layer.
BLOCKS = 1         # Number of blocks on each side.
BATCH = 64         # Examples per optimizer update.
STEPS = 2000
RATE = 0.003       # Adam learning rate.
HOLDOUT = (3, 8)
SEED = 7


def passthrough(name, value, *parents, **info):
    """Ordinary training uses the tensor directly. A trace also records its inputs."""
    return value


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        # Each linear layer computes input @ weight.T + bias.
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self._heads = heads

    def forward(self, query_input, key_value_input, show, mask=None):
        # Self-attention takes both inputs from one stream.
        # Cross-attention takes queries from the decoder and keys/values from the encoder.
        q = show("Query", self.query(query_input), query_input, op=self.query)
        k = show("Key", self.key(key_value_input), key_value_input, op=self.key)
        v = show("Value", self.value(key_value_input), key_value_input, op=self.value)

        # Layout is [batch, positions, features]. Split only the features.
        queries = q.chunk(self._heads, dim=-1)
        keys = k.chunk(self._heads, dim=-1)
        values = v.chunk(self._heads, dim=-1)
        heads = []
        for head, (qh, kh, vh) in enumerate(zip(queries, keys, values), 1):
            # Record which features belong to each head so the viewer can trace them.
            start = (head - 1) * qh.size(-1)
            qh = show(f"Query · head {head}", qh, q, kind="slice", start=start)
            kh = show(f"Key · head {head}", kh, k, kind="slice", start=start)
            vh = show(f"Value · head {head}", vh, v, kind="slice", start=start)
            # Compare every query to every key. The result has one row per query
            # and one column per key. For source 38 this is 3 × 2 in training.
            # Scaling keeps dot products from growing with the head width.
            raw_scores = qh @ kh.transpose(-2, -1)
            scores = show(f"Scores · head {head}", raw_scores / math.sqrt(qh.size(-1)),
                          qh, kh, kind="dot")

            # A masked score gets zero probability after softmax.
            if mask is not None:
                scores = show(f"Causal mask · head {head}", scores.masked_fill(mask, -torch.inf),
                              scores, kind="mask")

            # Each query row sums to one. Use it to form a weighted sum of value rows.
            weights = show(f"Attention weights · head {head}", scores.softmax(-1),
                           scores, kind="softmax")
            heads.append(show(f"Weighted values · head {head}", weights @ vh,
                              weights, vh, kind="weighted"))

        # Join the head features and let the output projection mix them.
        joined = show("Concatenate heads", torch.cat(heads, -1), *heads, kind="join")
        return show("Attention output", self.output(joined), joined, op=self.output)


def add_norm(x, change, norm, show):
    # Keep the input through the residual path, then normalize each token's features.
    x = show("Residual addition", x + change, x, change, kind="add")
    return show("Layer normalization", norm(x), x, op=norm, kind="norm")


class Block(nn.Module):
    def __init__(self, width, heads, inner, side):
        super().__init__()
        self.self_attention = Attention(width, heads)
        self.self_norm = nn.LayerNorm(width)
        if side == "decoder":
            self.cross_attention = Attention(width, heads)
            self.cross_norm = nn.LayerNorm(width)
        self.expand = nn.Linear(width, inner)
        self.contract = nn.Linear(inner, width)
        self.output_norm = nn.LayerNorm(width)

    def forward(self, x, show, block_name, encoder_output=None):
        def at(section):
            # Bind the diagram label once so each operation only needs a short name.
            def record(name, value, *parents, **info):
                return show(name, value, *parents, group=f"{block_name} · {section}", **info)

            return record

        # Row i can see columns 0 through i. Only decoder self-attention uses this mask.
        mask = None
        if encoder_output is not None:
            positions = x.size(1)
            mask = torch.ones(positions, positions, device=x.device, dtype=torch.bool).triu(1)

        # A self-attention layer gets queries and keys/values from the same tokens.
        record = at("Self-attention")
        change = self.self_attention(x, x, record, mask)
        x = add_norm(x, change, self.self_norm, record)
        if encoder_output is not None:
            # Every decoder position can read all source positions through cross-attention.
            record = at("Cross-attention")
            change = self.cross_attention(x, encoder_output, record)
            x = add_norm(x, change, self.cross_norm, record)

        # Apply the same two-layer network separately to each token.
        record = at("Feed-forward")
        wide = record("Expand", self.expand(x), x, op=self.expand)
        active = record("ReLU", wide.relu(), wide, kind="relu")
        narrow = record("Contract", self.contract(active), active, op=self.contract)
        return add_norm(x, narrow, self.output_norm, record)


class Transformer(nn.Module):
    def __init__(self, width=WIDTH, heads=HEADS, inner=INNER, blocks=BLOCKS):
        super().__init__()
        if heads < 1 or width % heads or width % 2 or blocks < 1:
            raise ValueError("Use an even width divisible by the head count and at least one block.")
        self.embedding = nn.Embedding(DIGITS + 2, width)
        self.encoder = nn.ModuleList(Block(width, heads, inner, "encoder") for _ in range(blocks))
        self.decoder = nn.ModuleList(Block(width, heads, inner, "decoder") for _ in range(blocks))
        # Predict ten digits or EOS. BOS is only an input.
        self.output = nn.Linear(width, DIGITS + 1)
        self._width = width

        # Give each position fixed sine/cosine features at multiple frequencies.
        # A buffer travels with model state but receives no optimizer updates.
        positions = torch.arange(LENGTH + 1)[:, None]
        freq = 10000 ** (-torch.arange(0, width, 2) / width)
        angles = positions * freq
        self.register_buffer("position", torch.stack((angles.sin(), angles.cos()), -1).flatten(1))

    def _embed(self, tokens, show, group):
        # Convert token IDs [batch, positions] to vectors [batch, positions, WIDTH].
        digit = show("Token embedding", self.embedding(tokens) * math.sqrt(self._width),
                     op=self.embedding, kind="embedding", group=group, tokens=tokens)
        position = show("Sinusoidal positions", self.position[:tokens.size(1)].expand_as(digit),
                        kind="position", group=group)
        return show("Embedding + position", digit + position, digit, position,
                    kind="add", group=group)

    def encode(self, source, show=passthrough):
        # Return one contextual vector per source digit.
        x = self._embed(source, show, "Encoder input")
        for i, block in enumerate(self.encoder, 1):
            x = block(x, show, f"Encoder {i}")
        return x

    def decode(self, prefix, encoder_output, show=passthrough):
        # Each prefix position produces scores for the next token.
        x = self._embed(prefix, show, "Decoder input")
        for i, block in enumerate(self.decoder, 1):
            x = block(x, show, f"Decoder {i}", encoder_output)
        return show("Next-token scores", self.output(x), x, op=self.output, group="Prediction")

    def forward(self, source, prefix, show=passthrough):
        return self.decode(prefix, self.encode(source, show), show)


def practice(model, source, show=passthrough):
    # Align inputs and targets one position apart so each row predicts the next token.
    # Source 38 gives inputs BOS 8 3 and targets 8 3 EOS.
    reverse = source.flip(1)
    prefix = torch.cat((torch.full_like(source[:, :1], BOS), reverse), 1)
    # Output classes 0–9 represent digits. Class 10 represents EOS.
    target = torch.cat((reverse, torch.full_like(source[:, :1], DIGITS)), 1)
    scores = model(source, prefix, show)
    # PyTorch expects [batch, classes, positions]. Keep one loss per target token.
    losses = F.cross_entropy(scores.transpose(1, 2), target, reduction="none")
    return prefix, target, losses


@torch.no_grad()
def generate(model, source):
    # Encode once. The decoder recomputes projected K and V on each pass.
    encoder_output = model.encode(source)
    prefix = torch.full_like(source[:, :1], BOS)
    predictions = []
    # Stop at EOS or at the maximum number of predictions this task allows.
    for _ in range(LENGTH + 1):
        # This function handles one example. The final prefix row predicts the next token.
        probs = model.decode(prefix, encoder_output)[0, -1].softmax(0)
        guess = probs.argmax().item()
        token = EOS if guess == DIGITS else guess
        predictions.append((token, probs))
        if token == EOS:
            break
        prefix = torch.cat((prefix, prefix.new_tensor([[token]])), 1)
    return predictions


def draw_batch():
    source = torch.randint(DIGITS, (BATCH, LENGTH))
    # If the demo input is drawn, change its first digit to keep it held out.
    source[(source == torch.tensor(HOLDOUT)).all(1), 0] = (HOLDOUT[0] + 1) % DIGITS
    return source


def label(token):
    return {BOS: "<bos>", EOS: "<eos>"}.get(token, str(token))


def show_trace(pages):
    print("Enter: next | b: back | f: full tensor | n: next update | q: resume training")
    index = 0
    while index < len(pages):
        name, value = pages[index]
        print(f"\n[{index + 1}/{len(pages)}] {name} · shape {tuple(value.shape)}")
        print(value.detach())
        command = input("> ").strip() if sys.stdin.isatty() else ""
        if command in ("q", "n"):
            return command
        if command == "b":
            index = max(0, index - 1)
        elif command == "f":
            torch.set_printoptions(profile="full")
            print(value.detach())
            torch.set_printoptions(profile="default")
        elif not command:
            index += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--trace", type=int, metavar="STEP", help="inspect one training update")
    parser.add_argument("--sample", type=int, default=1, help="example in the batch, counting from 1")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.trace is not None and not 1 <= args.trace <= args.steps:
        parser.error("--trace must be within the training run")
    if not 1 <= args.sample <= BATCH:
        parser.error(f"--sample must be 1 to {BATCH}")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    model = Transformer()
    adam = torch.optim.Adam(model.parameters(), lr=RATE)
    example = torch.tensor([HOLDOUT])
    source_text = " ".join(map(label, HOLDOUT))
    expected = " ".join(map(label, (*reversed(HOLDOUT), EOS)))
    print(f"Source: {source_text} | Expected: {expected}")
    for step in range(1, args.steps + 1):
        source = draw_batch()
        trace = [] if step == args.trace else None
        sample = args.sample - 1

        def record(name, value, *parents, group, **info):
            # Ordinary autograd discards intermediate gradients. Keep them for this trace.
            if value.requires_grad:
                value.retain_grad()
            trace.append((f"{group} · {name}", value))
            return value

        prefix, _, losses = practice(model, source, record if trace is not None else passthrough)
        # Average the batch's token losses, clear old gradients, then differentiate.
        loss = losses.mean()
        adam.zero_grad()
        loss.backward()

        # Save the output weights here so the trace can show the effect of this Adam step.
        before = model.output.weight.detach().clone() if trace is not None else None
        adam.step()
        # Check the updated weights using the same held-out input each time.
        prediction = " ".join(label(token) for token, _ in generate(model, example))
        print(f"{step:3}  batch loss {loss.item():.5f}  prediction {prediction}", flush=True)
        if trace is not None:
            print(f"Trace of update {step}, example {args.sample}")
            print(f"Decoder prefix: {' '.join(map(label, prefix[sample].tolist()))}")
            pages = [("Encoder source", source[sample]), ("Decoder prefix", prefix[sample])]
            pages.extend((name, value[sample]) for name, value in trace)
            pages.append(("Cross-entropy per target token", losses[sample]))
            pages.extend((f"Gradient · {name}", value.grad[sample])
                         for name, value in reversed(trace) if value.grad is not None)
            pages.append(("Adam update · output projection weight", model.output.weight.detach() - before))
            if show_trace(pages) == "n":
                args.trace = step + 1


if __name__ == "__main__":
    main()
