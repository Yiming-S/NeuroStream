"""
NeuroStream — BCI Real-Time Motor Imagery Streaming Demo
=========================================================
Entry point only.  All logic lives in the sub-modules:

    config.py          BCIConfig dataclass
    data_engine.py     MOABB data loading + EA alignment
    model.py           CSP + LDA/SVM sklearn pipeline
    streaming.py       Generator-based trial delivery
    ui/plots.py        Standalone canvas drawing functions
    ui/app_view.py     tkinter AppUI (root.after driven, never blocks)
"""

import tkinter as tk

from ui.app_view import AppUI


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")   # no MNE plot windows stealing focus

    root = tk.Tk()
    root.geometry("1050x720")
    root.minsize(860, 600)

    try:
        root.tk.call(
            "::tk::unsupported::MacWindowStyle", "style", root._w,
            "document", "closeBox collapseBox resizeBox",
        )
    except Exception:
        pass

    AppUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
