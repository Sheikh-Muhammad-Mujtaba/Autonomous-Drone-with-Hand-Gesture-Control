"""Single-window tabbed GUI for the flight script.

Replaces the FOUR separate windows (pygame keyboard window + three
cv2.imshow windows) with ONE tkinter window that has a tab per view:

    Drone View   |   part View   |   Position Map

The class duck-types the small part of the KeyPressModule API the
control loop uses (getKey / getKeyPressedOnce), so it can be passed
straight to command_dispatcher.from_keyboard() in place of kp —
no dispatcher changes needed.
"""

import threading

import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

_KEYNAME_TO_KEYSYM = {
    "LEFT": "Left",
    "RIGHT": "Right",
    "UP": "Up",
    "DOWN": "Down",
}

# Pressing these ends the flight loop (q = land-and-quit, same as before).
_QUIT_KEYS = {"q", "escape"}


class UnifiedGUI:
    def __init__(self, title="Drone Reporter", width=900, height=660):
        self._root = tk.Tk()
        self._root.title(title)
        self._root.geometry(f"{width}x{height}")

        style = ttk.Style(self._root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TNotebook", background="#1e1e1e", borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background="#2d2d2d",
            foreground="#e0e0e0",
            padding=(18, 8),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#0f6cbd")],
            foreground=[("selected", "#ffffff")],
        )
        self._root.configure(bg="#1e1e1e")

        self._notebook = ttk.Notebook(self._root)
        self._notebook.pack(fill="both", expand=True)

        self._status = tk.Label(
            self._root, text="", anchor="w",
            bg="#0f6cbd", fg="#ffffff",
            font=("Segoe UI", 9), padx=8, pady=3,
        )
        self._status.pack(side="bottom", fill="x")
        self.set_reporter_state(True)

        self._labels = {}
        self._photos = {}
        for name in ("Drone View", "part View", "Position Map"):
            tab = tk.Frame(self._notebook, bg="black")
            self._notebook.add(tab, text=name)
            label = tk.Label(tab, bg="black")
            label.pack(fill="both", expand=True)
            self._labels[name] = label
            self._photos[name] = None

        self._down = set()
        self._prev = {}
        self._quit = False
        self._pending = []
        self._fit_done = False
        self._lock = threading.Lock()

        self._root.bind_all("<KeyPress>", self._on_press)
        self._root.bind_all("<KeyRelease>", self._on_release)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Keyboard API — KeyPressModule-compatible (names: LEFT/RIGHT/UP/DOWN
    # and plain letters like w/s/a/d/q/e/x/r)
    # ------------------------------------------------------------------
    @staticmethod
    def _sym(key_name):
        return _KEYNAME_TO_KEYSYM.get(key_name, key_name.lower())

    @staticmethod
    def _norm(event):
        sym = event.keysym
        return sym.lower() if len(sym) == 1 else sym

    def _on_press(self, event):
        sym = self._norm(event)
        self._down.add(sym)
        if sym in _QUIT_KEYS:
            self._quit = True

    def _on_release(self, event):
        self._down.discard(self._norm(event))

    def getKey(self, key_name):
        """True while the key is held (pumps the GUI event queue)."""
        self.update()
        return self._sym(key_name) in self._down

    def getKeyPressedOnce(self, key_name):
        """True only on the up -> down transition (edge detection)."""
        self.update()
        sym = self._sym(key_name)
        is_down = sym in self._down
        was_down = self._prev.get(key_name, False)
        self._prev[key_name] = is_down
        return is_down and not was_down

    # ------------------------------------------------------------------
    # Frame / window API
    # ------------------------------------------------------------------
    def show(self, tab_name, bgr_frame):
        """Post a BGR frame to a tab; it is rendered on the next update()."""
        if bgr_frame is None:
            return
        with self._lock:
            self._pending.append((tab_name, bgr_frame))

    def update(self):
        """Process GUI events and render posted frames (main thread only)."""
        with self._lock:
            frames, self._pending = self._pending, []
        for name, bgr in frames:
            self._render(name, bgr)
        try:
            self._root.update()
        except tk.TclError:
            self._quit = True
        if not self._fit_done and frames:
            for name, bgr in frames:
                if name == "Drone View":
                    self._fit_done = True
                    self._fit_to_video(bgr.shape[1], bgr.shape[0])
                    break

    def _fit_to_video(self, vid_w, vid_h):
        """One-shot resize: make the video area match the video's aspect
        ratio so the frame fills the tab exactly — no bars, no crop."""
        if vid_w <= 0 or vid_h <= 0:
            return
        try:
            label = self._labels.get("Drone View")
            if label is None:
                return
            cw = max(label.winfo_width(), 50)
            ch = max(label.winfo_height(), 50)
            ww = max(self._root.winfo_width(), 100)
            wh = max(self._root.winfo_height(), 100)
            chrome_w = max(ww - cw, 0)   # borders + scrollbar slivers
            chrome_h = max(wh - ch, 0)   # tab bar + status bar
            aspect = vid_w / float(vid_h)
            new_ch = int(round(cw / aspect))
            new_wh = new_ch + chrome_h
            new_ww = ww
            screen_h = self._root.winfo_screenheight()
            if new_wh > screen_h - 100:  # clamp so it fits on screen
                new_wh = screen_h - 100
                new_ch = max(new_wh - chrome_h, 100)
                new_ww = int(round(new_ch * aspect)) + chrome_w
            self._root.geometry(f"{new_ww}x{new_wh}")
        except tk.TclError:
            pass

    def _render(self, tab_name, bgr):
        label = self._labels.get(tab_name)
        if label is None:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        max_w = label.winfo_width()
        max_h = label.winfo_height()
        if max_w > 50 and max_h > 50:
            h, w = rgb.shape[:2]
            # Cover-crop: zoom until the frame COMPLETELY fills the tab
            # (no black bars), cropping whatever overflows the edges.
            scale = max(max_w / float(w), max_h / float(h))
            tw = max(int(round(w * scale)), max_w)
            th = max(int(round(h * scale)), max_h)
            if (tw, th) != (w, h):
                rgb = cv2.resize(
                    rgb, (tw, th),
                    interpolation=(cv2.INTER_AREA if scale < 1.0
                                   else cv2.INTER_LINEAR),
                )
            x0 = (tw - max_w) // 2
            y0 = (th - max_h) // 2
            rgb = rgb[y0:y0 + max_h, x0:x0 + max_w]
        self._photos[tab_name] = ImageTk.PhotoImage(Image.fromarray(rgb))
        label.configure(image=self._photos[tab_name])

    @property
    def should_quit(self):
        return self._quit

    def set_reporter_state(self, enabled):
        """Update the bottom status bar (called on init and on O toggles)."""
        state = "ON (reporter only)" if enabled else "OFF (bypass — anyone)"
        self._status.configure(
            text=f"Reporter Guard: {state}    |    "
                 f"O: toggle reporter    Q: land + quit"
        )

    def _on_close(self):
        self._quit = True

    def destroy(self):
        try:
            self._root.destroy()
        except tk.TclError:
            pass
