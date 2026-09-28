from __future__ import annotations

import logging
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from axen.inference.chat_engine import AXENChatEngine

from .message_widget import MessageWidget
from .state import ChatRepository, ChatSession, Message
from .theme import COLORS, FONT, TITLE_FONT

LOGGER = logging.getLogger(__name__)


class ChatWindow:
    def __init__(self, root: tk.Tk, repository: ChatRepository, engine: AXENChatEngine) -> None:
        self.root = root
        self.repository = repository
        self.engine = engine
        self.session = self.repository.create_session()
        self.stop_event = threading.Event()
        self.generating = False
        self.loading = True
        self.root.title("AXEN AI")
        self.root.geometry("1050x700")
        self.root.minsize(760, 500)
        self._configure_style()
        self._build()
        self._refresh_status()

    def _configure_style(self) -> None:
        self.root.configure(bg=COLORS["background"])
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=COLORS["background"], foreground=COLORS["text"], font=FONT)
        style.configure("Sidebar.TFrame", background=COLORS["sidebar"])
        style.configure("Panel.TFrame", background=COLORS["panel"])
        style.configure("Title.TLabel", background=COLORS["background"], foreground=COLORS["text"], font=TITLE_FONT)
        style.configure("Muted.TLabel", background=COLORS["background"], foreground=COLORS["muted"])
        style.configure("Accent.TButton", foreground=COLORS["background"], background=COLORS["accent"])

    def _build(self) -> None:
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=16, pady=(12, 8))
        ttk.Label(header, text="AXEN AI", style="Title.TLabel").pack(side="left")
        self.status = ttk.Label(header, style="Muted.TLabel")
        self.status.pack(side="right")

        sidebar = ttk.Frame(self.root, style="Sidebar.TFrame", width=220)
        sidebar.pack(side="left", fill="y", padx=(10, 6), pady=(0, 10))
        sidebar.pack_propagate(False)
        ttk.Button(sidebar, text="+ New Chat", command=self.new_chat).pack(fill="x", padx=10, pady=10)
        self.session_list = tk.Listbox(
            sidebar, bg=COLORS["sidebar"], fg=COLORS["text"], selectbackground=COLORS["user"], relief="flat"
        )
        self.session_list.pack(fill="both", expand=True, padx=8, pady=4)
        self.session_list.bind("<<ListboxSelect>>", self._select_session)
        ttk.Button(sidebar, text="Model info", command=self.show_model_info).pack(fill="x", padx=10, pady=4)
        ttk.Button(sidebar, text="Clear conversation", command=self.clear_conversation).pack(fill="x", padx=10, pady=(4, 10))

        main = ttk.Frame(self.root)
        main.pack(side="left", fill="both", expand=True, padx=(6, 10), pady=(0, 10))
        self.messages_canvas = tk.Canvas(main, bg=COLORS["background"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(main, orient="vertical", command=self.messages_canvas.yview)
        self.messages_frame = ttk.Frame(self.messages_canvas)
        self.messages_frame.bind(
            "<Configure>", lambda _event: self.messages_canvas.configure(scrollregion=self.messages_canvas.bbox("all"))
        )
        self.messages_canvas.create_window((0, 0), window=self.messages_frame, anchor="nw")
        self.messages_canvas.configure(yscrollcommand=scrollbar.set)
        self.messages_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        input_frame = ttk.Frame(main, style="Panel.TFrame")
        input_frame.pack(fill="x", pady=(10, 0))
        self.input = tk.Text(input_frame, height=4, wrap="word", font=FONT, bg=COLORS["panel"], fg=COLORS["text"], relief="flat")
        self.input.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        self.input.bind("<Return>", self._on_enter)
        self.input.bind("<Control-n>", lambda _event: self.new_chat())
        self.input.bind("<Control-l>", lambda _event: self.input.delete("1.0", "end"))
        self.input.bind("<Escape>", lambda _event: self.stop_generation())
        self.send_button = ttk.Button(input_frame, text="➤", style="Accent.TButton", command=self.send_message)
        self.send_button.pack(side="right", padx=8, pady=8)
        self._refresh_session_list()

    def _on_enter(self, event: tk.Event) -> str | None:
        if event.state & 0x0001:
            return None
        self.send_message()
        return "break"

    def _refresh_status(self) -> None:
        self.status.configure(text="● LOADING", foreground=COLORS["muted"])
        self.engine.load_model()
        info = self.engine.model_info()
        self.loading = False
        self.status.configure(
            text=f"● {info['status']}",
            foreground=COLORS["accent"] if self.engine.is_model_ready() else COLORS["error"],
        )
        self._render_session()

    def _refresh_session_list(self) -> None:
        self.session_list.delete(0, "end")
        sessions = self.repository.list_sessions()
        for session in sessions:
            self.session_list.insert("end", session.title)
        selected = next((index for index, item in enumerate(sessions) if item.id == self.session.id), None)
        if selected is not None:
            self.session_list.selection_set(selected)

    def _select_session(self, _event: tk.Event) -> None:
        selection = self.session_list.curselection()
        if not selection:
            return
        sessions = self.repository.list_sessions()
        loaded = self.repository.load_session(sessions[selection[0]].id)
        if loaded is not None:
            self.session = loaded
            self._render_session()

    def _render_session(self) -> None:
        for child in self.messages_frame.winfo_children():
            child.destroy()
        for message in self.session.messages:
            MessageWidget(self.messages_frame, message.role, message.content, self._copy).pack(fill="x", padx=12, pady=4)
        self.messages_canvas.update_idletasks()
        self.messages_canvas.yview_moveto(1.0)

    def new_chat(self) -> None:
        self.session = self.repository.create_session()
        self._refresh_session_list()
        self._render_session()

    def clear_conversation(self) -> None:
        self.session.messages.clear()
        self.repository.save_session(self.session)
        self._render_session()

    def _copy(self, content: str) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(content)

    def show_model_info(self) -> None:
        info = self.engine.model_info()
        details = "\n".join(f"{key.title()}: {value}" for key, value in info.items() if key != "error" and value)
        messagebox.showinfo("AXEN model information", details)

    def send_message(self) -> None:
        if self.generating or self.loading:
            return
        content = self.input.get("1.0", "end-1c").strip()
        if not content:
            messagebox.showwarning("Message required", "Enter a message before sending.")
            return
        if not self.engine.is_model_ready():
            messagebox.showwarning(
                "Model not ready",
                "AXEN model is not ready yet.\n\nStatus: NOT TRAINED\n\nComplete training before chatting.",
            )
            return
        self.input.delete("1.0", "end")
        if self.session.title == "New Chat":
            self.session.title = content[:40]
        self.session.messages.append(Message("user", content))
        self.repository.save_session(self.session)
        self._refresh_session_list()
        self._render_session()
        self.generating = True
        self.stop_event.clear()
        self.status.configure(text="● GENERATING", foreground=COLORS["accent"])
        self.send_button.configure(text="STOP", command=self.stop_generation)
        threading.Thread(target=self._generate, args=(content,), daemon=True).start()

    def _generate(self, prompt: str) -> None:
        try:
            response = self.engine.generate(prompt, stop_event=self.stop_event)
            self.root.after(0, self._generation_done, response)
        except Exception as exc:
            LOGGER.exception("AXEN generation failed")
            self.root.after(0, self._generation_failed, str(exc))

    def _generation_done(self, response: str) -> None:
        self.generating = False
        self.status.configure(text="● READY", foreground=COLORS["accent"])
        self.send_button.configure(text="➤", command=self.send_message)
        if response:
            self.session.messages.append(Message("assistant", response))
            self.repository.save_session(self.session)
            self._render_session()

    def _generation_failed(self, error: str) -> None:
        self.generating = False
        self.status.configure(text="● ERROR", foreground=COLORS["error"])
        self.send_button.configure(text="➤", command=self.send_message)
        self.session.messages.append(
            Message("assistant", "⚠ Generation failed.\n\nThe model encountered an internal inference error.\n\nDetails:\n" + error)
        )
        self.repository.save_session(self.session)
        self._render_session()

    def stop_generation(self) -> None:
        if self.generating:
            self.stop_event.set()
            self.status.configure(text="● READY", foreground=COLORS["accent"])
