"""Small full-softmax word embeddings with explicit gradients and SGD."""
import math
import random
from collections import Counter

BASE = ('the cat eats fish', 'the dog eats fish', 'the cat drinks water',
        'the dog drinks water', 'the bird flies home', 'the bird sings loudly')
EDIT = tuple(s.replace('eats fish', 'flies home').replace('drinks water', 'sings loudly')
             if s.startswith('the dog ') else s for s in BASE)
STEPS = (0, 1, 10, 30, 60, 120, 240, 480, 960, 1920)
RATE = 0.12


def examples(sentences, vocab, mode, radius):
    """Yield (input IDs, target ID); never cross sentence boundaries."""
    if mode not in ('cbow', 'skip') or radius < 1:
        raise ValueError('Choose cbow or skip and a positive radius.')
    ids = {w: i for i, w in enumerate(vocab)}
    for sentence in sentences:
        row = [ids[w] for w in sentence.split()]
        for i, center in enumerate(row):
            context = row[max(0, i-radius):i] + row[i+1:i+radius+1]
            if mode == 'cbow' and context:
                yield context, center
            elif mode == 'skip':
                for neighbor in context:
                    yield [center], neighbor


def dot(a, b):
    return sum(x*y for x, y in zip(a,b))


def forward(E, U, inputs, target):
    """E and U both store one row per word; neither uses a bias."""
    h = [sum(E[i][d] for i in inputs)/len(inputs) for d in range(len(E[0]))]
    z = [dot(u,h) for u in U]
    peak = max(z)
    terms = [math.exp(x-peak) for x in z]
    denom = sum(terms)
    p = [x/denom for x in terms]
    return h, z, p, peak + math.log(denom) - z[target]


def gradients(E,U,inputs,target):
    h, _, p, _ = forward(E,U,inputs,target)
    err = [v - (j == target) for j,v in enumerate(p)]
    dh = [sum(err[j]*u[d] for j,u in enumerate(U)) for d in range(len(h))]
    counts = Counter(inputs)
    dE = [[counts[i]/len(inputs)*v for v in dh] for i in range(len(E))]
    dU = [[g*v for v in h] for g in err]
    return dE, dU


def train(sentences, mode='cbow', radius=1, seed=3, width=2,
          steps=STEPS, rate=RATE, vocab=None):
    """Checkpoint j has already seen j updates."""
    vocab = sorted(set(vocab) if vocab is not None else
                   set(word for s in sentences for word in s.split()))
    data = list(examples(sentences,vocab,mode,radius))
    if not data:
        raise ValueError('Include a sentence with at least two words.')
    rng = random.Random(seed)
    E,U = ([[rng.uniform(-.5,.5) for _ in range(width)] for _ in vocab] for _ in range(2))
    checkpoints = []
    end = max(steps)
    for step in range(end+1):
        pair = rng.choice(data) if step < end else None
        if step in steps:
            checkpoints.append({
                'step':step, 'next_pair':pair, 'E':[row[:] for row in E], 'U':[row[:] for row in U],
                'loss':sum(forward(E,U,x,y)[-1] for x,y in data)/len(data),
            })
        if pair is None:
            break
        x,y = pair
        for table, grad in zip((E,U), gradients(E,U,x,y)):
            for row, g in zip(table,grad):
                for d, value in enumerate(g):
                    row[d] -= rate * value
    return vocab,checkpoints


def cosine(a,b):
    length = math.sqrt(dot(a,a)*dot(b,b))
    return dot(a,b)/length if length else 0.0
