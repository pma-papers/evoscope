"""Convert the tutorial scripts (percent format: '# %%' code cells, '# %% [markdown]'
markdown cells) into Jupyter notebooks in tutorials/notebooks/.  No dependencies."""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE, TARGET = ROOT/"tutorials", ROOT/"tutorials/notebooks"


def cells(text):
    out, kind, lines = [], None, []
    for line in text.splitlines():
        marker = re.match(r"# %%( \[markdown\])?\s*$", line)
        if marker:
            if kind is not None:
                out.append((kind, lines))
            kind, lines = ("markdown" if marker.group(1) else "code"), []
        elif kind is not None:
            lines.append(line)
    if kind is not None:
        out.append((kind, lines))
    return out


def notebook(text):
    nb_cells = []
    for kind, lines in cells(text):
        while lines and not lines[-1].strip():
            lines.pop()
        if kind == "markdown":
            lines = [re.sub(r"^# ?", "", l) for l in lines]
            lines = [l.replace("](../docs/figures/", "](../../docs/figures/") for l in lines]
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": [l+"\n" for l in lines[:-1]]+lines[-1:]})
        else:
            nb_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                             "source": [l+"\n" for l in lines[:-1]]+lines[-1:]})
    return {"cells": nb_cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}


if __name__ == "__main__":
    TARGET.mkdir(exist_ok=True)
    for path in sorted(SOURCE.glob("0*.py")):
        out = TARGET/(path.stem+".ipynb")
        out.write_text(json.dumps(notebook(path.read_text()), indent=1)+"\n")
        print("wrote", out.relative_to(ROOT))
