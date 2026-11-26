"""Rebuild the standalone notebook and the browser checkpoint data."""
import json
from pathlib import Path
from model import BASE, EDIT, RATE, train

HERE = Path(__file__).resolve().parent
vocab = sorted(set(' '.join(BASE + EDIT).split()))
corpora = {'base':BASE,'edit':EDIT}
runs = {variant:{mode:{r:train(sentences,mode,r,vocab=vocab)[1] for r in (1,2)}
                 for mode in ('cbow','skip')} for variant,sentences in corpora.items()}
out = HERE.parent / 'docs/visualizations/word2vec/checkpoints.js'
out.write_text('const WORD2VEC_DATA = '+json.dumps(
    {'vocab':vocab,'corpora':corpora,'runs':runs,'rate':RATE},separators=(',',':'))+';\n')

cells = []
def md(s):
    cells.append(dict(cell_type='markdown',metadata={},source=s))
def code(s):
    cells.append(dict(cell_type='code',metadata={},source=s,execution_count=None,outputs=[]))

md('# How Word2vec learns from neighbors\n\nTrain CBOW or Skip-gram on six sentences. Only the Python standard library is needed in the kernel. Edit the corpus or window and rerun from the configuration cell. [Companion explorable](https://singhcoder.github.io/ai-engineering/visualizations/word2vec/).')
md('## 1. Define the training pairs and model\n\nCBOW averages context vectors; Skip-gram uses the center vector and trains on each neighbor separately. This teaching version uses full softmax and plain SGD.')
code((HERE/'model.py').read_text())
md('## 2. Change the corpus and train\n\nSwitch to edit to change only where dog appears. The two presets share the same vocabulary and initial weights.')
code('''variant = 'base'  # 'base' or 'edit'
mode = 'cbow'     # 'cbow' or 'skip'
radius = 1
sentences = list(BASE if variant == 'base' else EDIT)
# sentences[1] = 'the dog sings loudly'  # Try your own edit.
vocab = sorted(set(' '.join(BASE + EDIT + tuple(sentences)).split()))
vocab, checkpoints = train(sentences, mode, radius, vocab=vocab)
for c in checkpoints:
    print(f"step {c['step']:4d}  mean training loss {c['loss']:.5f}")''')
md('## 3. Reproduce one SGD update\n\nPick any pair at a checkpoint. This probe need not be the next random training example. Change the table, row or feature to inspect a different weight.')
code('''checkpoint = 0  # index, not update number
example = 1
x, y = list(examples(sentences, vocab, mode, radius))[example]
# x, y = checkpoints[checkpoint]['next_pair']  # Use the actual SGD pair instead.
E, U = (checkpoints[checkpoint][k] for k in ('E', 'U'))
h, z, p, loss = forward(E,U,x,y)
dE, dU = gradients(E,U,x,y)
print('inputs:', [(vocab[i],E[i]) for i in x], 'target:', vocab[y])
print('hidden:', h, 'loss:', loss)
print('word     output vector       logit      probability')
for w, u, a, b in zip(vocab, U, z, p):
    print(f'{w:8} {u[0]: .6f} {u[1]: .6f}  {a: .6f}  {b:.6f}')
table, grad, word, feature = U, dU, vocab[y], 0
i = vocab.index(word)
old, g = table[i][feature], grad[i][feature]
print(f'{word}[{feature}]: {old:.9g} - {RATE} × {g:.9g} = {old-RATE*g:.9g}')''')
md('## 4. Compare counting with learned neighbors\n\nThe trigram baseline counts exact continuations. The context baseline compares counts from both directions. The learned baseline compares the input embedding rows.')
code('''query = 'dog'
def context_counts(word):
    counts = Counter()
    for s in sentences:
        words = s.split()
        for i,w in enumerate(words):
            if w == word:
                counts.update(words[max(0,i-radius):i]+words[i+1:i+radius+1])
    return [counts[w] for w in vocab]

trigrams = Counter(tuple(words[i:i+3]) for s in sentences
                   for words in [s.split()] for i in range(len(words)-2))
print('After “the '+query+'”:', {w:c for (a,b,w),c in trigrams.items() if (a,b)==('the',query)})
for j in (0,-1):
    E = checkpoints[j]['E']
    q = vocab.index(query)
    scores = [(w, cosine(context_counts(query),context_counts(w)),
               cosine(E[q],E[i])) for i,w in enumerate(vocab) if i!=q]
    print('step',checkpoints[j]['step'], sorted(scores,key=lambda row:-row[-1])[:5])''')
nb = dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
          language_info=dict(name='python')),nbformat=4,nbformat_minor=5)
for i,cell in enumerate(cells):
    cell['id'] = f'word2vec-{i}'
(HERE/'word2vec.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
