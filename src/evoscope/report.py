"""Reports: the result of a check in numbers and in plain language.

Every check returns a :class:`Report` made of :class:`Finding` objects.  A finding says
which property was checked, which result of the paper it belongs to, the key numbers,
a verdict, and what the verdict means for the algorithm being developed.

Verdicts
--------
consistent    the numbers behave as the theory requires on the populations examined;
suspect       they partly do: look at the plot and at the populations where it fails;
violated      they clearly do not; the corresponding result of the paper does not apply;
not resolved  the numbers are too noisy or too coarse to decide; use more replicates,
              smaller steps, or more populations.

A numerical check is evidence about the populations it examined, never a proof: a
"consistent" verdict means that nothing contradicts the condition there.
"""
from dataclasses import dataclass, field

__all__ = ["Finding", "Report", "VERDICTS"]

VERDICTS = ("consistent", "suspect", "violated", "not resolved")
_SEVERITY = {"consistent": 0, "not resolved": 1, "suspect": 2, "violated": 3, "info": -1}


@dataclass
class Finding:
    check: str
    reference: str
    statistic: str
    verdict: str
    meaning: str
    data: dict = field(default_factory=dict, repr=False)


@dataclass
class Report:
    title: str
    findings: list
    notes: list = field(default_factory=list)
    plotter: object = field(default=None, repr=False)

    @property
    def verdict(self):
        """The most severe verdict among the findings."""
        graded = [f.verdict for f in self.findings if f.verdict in _SEVERITY and f.verdict != "info"]
        return max(graded, key=_SEVERITY.get) if graded else "info"

    def __getitem__(self, check):
        for f in self.findings:
            if f.check == check:
                return f
        raise KeyError(check)

    def summary(self, width=100):
        """A plain-text table followed by the plain-language meaning of every finding."""
        lines = [self.title, "="*min(width, len(self.title))]
        w1 = max(len(f.check) for f in self.findings)
        for f in self.findings:
            lines.append(f"{f.check:<{w1}}  {f.verdict:<13} {f.statistic}")
        lines.append("")
        for f in self.findings:
            lines.append(f"* {f.check} [{f.verdict}] ({f.reference})")
            lines.extend("    "+line for line in _wrap(f.meaning, width-4))
        for note in self.notes:
            lines.extend(_wrap("Note: "+note, width))
        return "\n".join(lines)

    def __str__(self):
        return self.summary()

    def to_markdown(self):
        rows = ["| Check | Verdict | Numbers | Paper |", "|---|---|---|---|"]
        rows += [f"| {f.check} | {f.verdict} | {f.statistic} | {f.reference} |" for f in self.findings]
        text = [f"### {self.title}", "", *rows, ""]
        text += [f"- **{f.check}** ({f.verdict}): {f.meaning}" for f in self.findings]
        text += [f"\n*Note:* {n}" for n in self.notes]
        return "\n".join(text)

    def plot(self, path=None):
        """Draw the diagnostic panels of the report (if the check provides them)."""
        if self.plotter is None:
            raise ValueError("this report has no plot")
        fig = self.plotter(self)
        if path is not None:
            fig.savefig(path)
        return fig


def _wrap(text, width):
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line)+len(word)+1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return lines
