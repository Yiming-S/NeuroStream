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

import sys
import tkinter as tk
from pathlib import Path

from ui.app_view import AppUI

_BASE_DIR = Path(__file__).resolve().parent
_ICON_PATH = _BASE_DIR / "ui" / "img" / "neurostream.png"


def _set_macos_dock_icon() -> None:
    """Update the macOS Dock icon from the bundled PNG asset."""
    if sys.platform != "darwin":
        return
    try:
        from AppKit import NSApplication, NSImage  # type: ignore[import]
        ns_app = NSApplication.sharedApplication()
        ns_icon = NSImage.alloc().initWithContentsOfFile_(str(_ICON_PATH))
        if ns_icon:
            ns_app.setApplicationIconImage_(ns_icon)
    except Exception:
        pass


def _set_window_icon(root: tk.Tk) -> None:
    """Set the runtime window/taskbar icon for Tk-managed windows."""
    try:
        from PIL import Image, ImageTk
        img = Image.open(str(_ICON_PATH))
        icon = ImageTk.PhotoImage(img)
        root._app_icon = icon  # keep a reference alive for Tk
        root.iconphoto(True, icon)
    except Exception:
        pass


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")   # no MNE plot windows stealing focus

    root = tk.Tk()
    _set_window_icon(root)
    _set_macos_dock_icon()
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
