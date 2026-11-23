"""Build the standalone GitHub Pages replay from real PyTorch checkpoints."""

import base64
import gzip
import json
from pathlib import Path

import nbformat

import train

HERE = Path(__file__).resolve().parent
PAGE = HERE.parents[1] / "docs" / "index.html"
MILESTONES = (1, 2, 10, 25, 50, 100, 200, 500, 1000, 1500)


def read_cell(notebook, marker):
    return next(cell.source for cell in notebook.cells if cell.cell_type == "code" and marker in cell.source)


def train_and_replay(notebook):
    # Use the terminal model's small configuration in this public demonstration.
    scope = {}
    exec(read_cell(notebook, "from copy import deepcopy"), scope)
    for setting in ("DIGITS", "LENGTH", "BOS", "EOS", "WIDTH", "HEADS", "INNER",
                    "BLOCKS", "BATCH", "STEPS", "RATE", "HOLDOUT", "SEED"):
        scope[setting] = getattr(train, setting)

    for marker in ("class Attention(nn.Module)", "class Transformer(nn.Module)",
                   "def practice(", "history = []", "def replay("):
        exec(read_cell(notebook, marker), scope)

    history = scope["history"]
    updates = sorted({step for step in (*MILESTONES, len(history)) if step <= len(history)})
    digits = "".join(map(str, scope["HOLDOUT"]))
    packets = {step: scope["replay"](step, 0, digits) for step in updates}
    final_guess = [item["guess"] for item in packets[updates[-1]]["guesses"]]
    expected = [*reversed(scope["HOLDOUT"]), scope["EOS"]]
    if final_guess != expected:
        raise RuntimeError(f"The final demo prediction was {final_guess}, expected {expected}.")

    return {
        "history": [history[step - 1][-1] for step in updates],
        "updates": updates,
        "packets": packets,
        "initial_tensors": {name: values.tolist() for name, values in history[0][0].items()},
        "batch": scope["BATCH"],
        "length": scope["LENGTH"],
        "seed": scope["SEED"],
    }


def viewer_module(notebook):
    source = read_cell(notebook, 'diagram_js = r"""').split('"""', 2)[1]
    # Static replay selects a sampled checkpoint by slider index.
    replacements = {
        "const history = model.get(\"history\");": """
  const history = model.get("history");
  const updates = model.get("updates");
  $("example-setup").hidden = true;
  $("loss-plot").open = true;
  $("generated").open = true;
""",
        "const loss=history[update-1];": "const loss=history[update-1], checkpoint=updates[update-1];",
        '"Update "+update+" · loss "': '"Update "+checkpoint+" · loss "',
        '$("stepno").textContent=update;': '$("stepno").textContent=checkpoint;',
        'queued={update:+$("step").value,sample,digits};':
            'queued={update:updates[+$("step").value-1],sample,digits};',
        '$("step").value=packet.update;': '$("step").value=updates.indexOf(packet.update)+1;',
        'mark(packet.update);': 'mark(+$("step").value);',
        "'<text x=\"470\" y=\"91\">Training update → '+history.length+'</text>';":
            "'<text x=\"438\" y=\"91\">Checkpoint → '+updates.at(-1)+'</text>';",
        "export default {render};": "",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError(f"Notebook viewer changed at {old!r}.")
        source = source.replace(old, new)
    return "\n".join(line.rstrip() for line in source.splitlines())


def build_page():
    notebook = nbformat.read(HERE / "digit-reversal.ipynb", 4)
    data = train_and_replay(notebook)
    style = {}
    exec(read_cell(notebook, "figure_fonts ="), style)
    data["css"] = style["figure_css"]
    data["fonts"] = style["figure_fonts"]
    payload = base64.b64encode(gzip.compress(json.dumps(data, separators=(",", ":")).encode(), mtime=0)).decode()

    html = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Transformer digit reversal</title>
<style>body{max-width:900px;margin:24px auto;padding:0 16px;font:16px/1.55 system-ui;color:#3c4043}</style>
<p><a href="https://github.com/SinghCoder/ai-engineering/tree/main/transformers/digit-reversal">Notebook and code</a></p>
<p>Replay a trained PyTorch model at saved checkpoints. The math uses input <code>38</code> in a replayed batch. The model weights come from training batches that held out <code>38</code>.</p>
<div id="viewer"><p>Loading training replay…</p></div>
<script id="payload" type="application/octet-stream">__PAYLOAD__</script>
<script type="module">
__VIEWER__
const el=document.getElementById("viewer");
try {
  const payload=document.getElementById("payload").textContent;
  const bytes=Uint8Array.from(atob(payload),c=>c.charCodeAt(0));
  const decompressed=new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  const state=await new Response(decompressed).json();
  const listeners=new Map();
  state.packet=state.packets[state.updates.at(-1)];
  const model={
    get:key=>state[key],
    set:(key,value)=>{state[key]=value;},
    on:(key,callback)=>listeners.set(key,callback),
    off:key=>listeners.delete(key),
    save_changes(){
      const selection=state.selection;
      state.packet={...state.packets[selection.update],request:selection.id};
      queueMicrotask(()=>listeners.get("change:packet")?.());
    }
  };
  el.replaceChildren();
  render({model,el});
} catch(error) {
  el.replaceChildren();
  el.textContent="Could not load the training replay. See the notebook and code above.";
  console.error(error);
}
</script>
</html>
"""
    PAGE.parent.mkdir(exist_ok=True)
    PAGE.write_text(html.replace("__PAYLOAD__", payload).replace("__VIEWER__", viewer_module(notebook)))
    print(f"{PAGE.name} · {len(data['updates'])} checkpoints · {PAGE.stat().st_size:,} bytes")


if __name__ == "__main__":
    build_page()
