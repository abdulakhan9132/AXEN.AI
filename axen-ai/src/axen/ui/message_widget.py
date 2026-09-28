from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from .theme import COLORS, FONT


class MessageWidget(ttk.Frame):
    def __init__(self, parent: tk.Misc, role: str, content: str, on_copy: Callable[[str], None]) -> None:
        super().__init__(parent)
        ttk.Label(self, text="You" if role == "user" else "AXEN", foreground=COLORS["accent"]).pack(anchor="w")
        body = tk.Text(
            self,
            height=max(1, min(8, content.count("\n") + 1)),
            wrap="word",
            font=FONT,
            bg=COLORS["user"] if role == "user" else COLORS["assistant"],
            fg=COLORS["text"],
            relief="flat",
            padx=10,
            pady=8,
        )
        body.insert("1.0", content)
        body.configure(state="disabled")
        body.pack(fill="x", pady=(2, 4))
        if role == "assistant":
            ttk.Button(self, text="Copy", command=lambda: on_copy(content)).pack(anchor="e", pady=(0, 8))
