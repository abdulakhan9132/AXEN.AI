from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path

from axen.inference.chat_engine import AXENChatEngine

from .chat_window import ChatWindow
from .state import ChatRepository


def launch_desktop(
    checkpoint_path: str | Path = "checkpoints/overfit.pt",
    database_path: str | Path = "data/axen_chats.sqlite3",
) -> None:
    logging.basicConfig(filename="logs/desktop.log", level=logging.INFO)
    root = tk.Tk()
    ChatWindow(root, ChatRepository(database_path), AXENChatEngine(checkpoint_path))
    root.mainloop()
