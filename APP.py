import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys
import numpy as np
import random
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import warnings
from PIL import Image, ImageTk, ImageDraw  
warnings.filterwarnings('ignore')


def create_emsi_logo_image(width=200, height=80):
    """Create EMSI logo programmatically"""
    img = Image.new('RGBA', (width, height), (22, 27, 34, 255))
    draw = ImageDraw.Draw(img)
    
    
    green = (34, 139, 96)  
    cyan = (0, 212, 255)   
    white = (230, 237, 243)  
    
    
    for i in range(3):
        x = 15 + i * 20
        draw.rectangle([x, 40, x + 12, 75], fill=green, outline=cyan, width=1)
        draw.rectangle([x+2, 50, x+10, 75], fill=cyan, outline=white, width=1)
    
    
    text = "EMSI"
    try:
        font_title = ImageFont.truetype("arial.ttf", 24)
        font_sub = ImageFont.truetype("arial.ttf", 9)
    except:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
    
    
    draw.text((70, 10), "MOROCCAN SCHOOL OF", fill=green, font=font_sub)
    draw.text((70, 25), "ENGINEERING SCIENCES", fill=white, font=font_sub)
    draw.text((70, 43), "EMSI", fill=cyan, font=font_title)
    draw.text((70, 63), "HONOURS | UNITED UNIVERSITIES", fill=(139, 148, 158), font=font_sub)
    
    return img

try:
    from PIL import ImageFont
except:
    ImageFont = None


COLORS = {
    "bg_dark": "#0e1726",
    "bg_panel": "#111b2e",
    "bg_card": "#152238",
    "text_white": "#E6EDF3",
    "text_muted": "#8B949E",
    "border": "#30363D",
    "regression": "#00D4FF",
    "clustering": "#FF6B6B",
    "forest": "#56E39F",
    "timeseries": "#FFD93D",
    "neurones": "#C77DFF",
    "validation": "#FF9F1C",
}

FONT_TITLE = ("Courier New", 22, "bold")
FONT_SUB = ("Courier New", 13, "bold")
FONT_LABEL = ("Courier New", 10)
FONT_MONO = ("Courier New", 9)
FONT_BTN = ("Courier New", 10, "bold")

APP_NAME = "Study and Understanding"
SPLASH_ACCENT = "#00D4FF"
SPLASH_BG = COLORS["bg_dark"]
SPLASH_CARD_BG = COLORS["bg_panel"]
SPLASH_INFO_BG = COLORS["bg_card"]
LOGO_MAX_W = 270
LOGO_MAX_H = 78


def _is_frozen():
    return getattr(sys, "frozen", False)


def _bundle_dir():
    """PyInstaller extract folder (onefile) or project folder (script)."""
    if _is_frozen():
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _exe_dir():
    """Folder containing APP.exe (or APP.py when run as script)."""
    if _is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _logo_search_roots():
    """Directories to search for logo files (exe folder first, then bundle)."""
    roots = []
    for d in (_exe_dir(), _bundle_dir()):
        if d and d not in roots:
            roots.append(d)
    return roots


def _find_logo_path():
    """Resolve EMSI logo file from bundle, exe folder, or project assets."""
    rel_paths = [
        "logo emsi.png",
        os.path.join("src", "assets", "emsi_logo.png"),
        os.path.join("src", "assets", "emsi_logo.jpg"),
        os.path.join("assets", "emsi_logo.png"),
        os.path.join("assets", "emsi_logo.jpg"),
        "emsi_logo.png",
        "logo.png",
    ]
    for root in _logo_search_roots():
        for rel in rel_paths:
            path = os.path.join(root, rel)
            if os.path.isfile(path):
                return path

    search_dirs = []
    for root in _logo_search_roots():
        for sub in ("src/assets", "assets", ""):
            folder = os.path.join(root, sub) if sub else root
            if folder not in search_dirs:
                search_dirs.append(folder)
    exts = (".png", ".jpg", ".jpeg", ".webp", ".gif")
    for folder in search_dirs:
        if not os.path.isdir(folder):
            continue
        named = []
        for name in sorted(os.listdir(folder)):
            low = name.lower()
            if not low.endswith(exts):
                continue
            if "emsi" in low or "logo" in low:
                return os.path.join(folder, name)
            named.append(name)
        if len(named) == 1:
            return os.path.join(folder, named[0])
    return None


def _fit_logo_image(img, max_w, max_h):
    """Scale image to fit within max_w x max_h without distortion."""
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    scale = min(max_w / w, max_h / h)
    if scale >= 1.0:
        return img
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def _flatten_logo_on_card(img):
    """Blend transparent logo onto the card background for clean contrast."""
    img = img.convert("RGBA")
    r = int(SPLASH_CARD_BG[1:3], 16)
    g = int(SPLASH_CARD_BG[3:5], 16)
    b = int(SPLASH_CARD_BG[5:7], 16)
    flat = Image.new("RGBA", img.size, (r, g, b, 255))
    flat.paste(img, mask=img.split()[3])
    return flat


def _load_emsi_logo(max_w=LOGO_MAX_W, max_h=LOGO_MAX_H):
    """Load EMSI logo from assets folder; fall back to generated placeholder."""
    logo_path = _find_logo_path()
    if logo_path:
        img = Image.open(logo_path).convert("RGBA")
        img = _flatten_logo_on_card(img)
    else:
        img = create_emsi_logo_image(max_w, max_h)
    return _fit_logo_image(img, max_w, max_h)


class SplashScreen(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} — AI & Machine Learning Platform")
        self._launch = False
        self._angle = 0
        self._particles = []
        self._screen_w = self.winfo_screenwidth()
        self._screen_h = self.winfo_screenheight()
        self.geometry(f"{self._screen_w}x{self._screen_h}+0+0")
        self.resizable(True, True)
        self.minsize(800, 600)
        self.configure(bg=SPLASH_BG)
        self._init_particles()
        self._build()
        self._animate()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Configure>", self._on_resize)

    def _on_close(self):
        self.destroy()

    def _on_resize(self, event=None):
        if event and event.widget is not self:
            return
        self._screen_w = self.winfo_width()
        self._screen_h = self.winfo_height()
        self.canvas.config(width=self._screen_w, height=self._screen_h)

    def _init_particles(self):
        for _ in range(80):
            self._particles.append({
                "x": random.uniform(0, self._screen_w),
                "y": random.uniform(0, self._screen_h),
                "vx": random.uniform(-0.35, 0.35),
                "vy": random.uniform(-0.35, 0.35),
                "r": random.uniform(1, 2.5),
                "color": random.choice(
                    [SPLASH_ACCENT, "#7B61FF", "#56E39F", "#FF9F43", "#FF6B6B"]
                ),
            })

    def _build(self):
        self._build_background()
        self._build_card()
        self._build_footer()

    def _build_background(self):
        self.canvas = tk.Canvas(
            self, width=self._screen_w, height=self._screen_h,
            bg=SPLASH_BG, highlightthickness=0,
        )
        self.canvas.place(x=0, y=0, relwidth=1, relheight=1)

    def _build_card(self):
        card = tk.Frame(
            self, bg=SPLASH_CARD_BG,
            highlightthickness=2,
            highlightbackground=SPLASH_ACCENT,
        )
        card.place(relx=0.5, rely=0.48, anchor="center", width=760, height=420)
        self._card = card

        self._build_header(card)
        self._build_info_box(card)
        self._build_start_button(card)

    def _build_header(self, parent):
        header = tk.Frame(parent, bg=SPLASH_CARD_BG)
        header.pack(fill="x", padx=28, pady=(26, 18))
        header.columnconfigure(1, weight=1)

        logo_slot = tk.Frame(
            header, bg=SPLASH_CARD_BG,
            width=LOGO_MAX_W, height=LOGO_MAX_H,
        )
        logo_slot.grid(row=0, column=0, sticky="nw")
        logo_slot.grid_propagate(False)

        self._logo_label = tk.Label(
            logo_slot, bg=SPLASH_CARD_BG,
            borderwidth=0, highlightthickness=0,
        )
        self._logo_label.place(x=0, y=0, anchor="nw")
        self._apply_header_logo()

        titles = tk.Frame(header, bg=SPLASH_CARD_BG)
        titles.grid(row=0, column=1, sticky="nw", padx=(22, 0), pady=(12, 0))

        tk.Label(
            titles,
            text=APP_NAME,
            bg=SPLASH_CARD_BG, fg=COLORS["text_white"],
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w")
        tk.Label(
            titles,
            text="\u2b21  AI & MACHINE LEARNING PLATFORM",
            bg=SPLASH_CARD_BG, fg=SPLASH_ACCENT,
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", pady=(8, 0))
        tk.Label(
            titles,
            text="Deep Learning & Data Science",
            bg=SPLASH_CARD_BG, fg=COLORS["text_muted"],
            font=("Segoe UI", 11),
        ).pack(anchor="w", pady=(6, 0))

    def _apply_header_logo(self):
        try:
            logo_pil = _load_emsi_logo(LOGO_MAX_W, LOGO_MAX_H)
            self.logo_img = ImageTk.PhotoImage(logo_pil)
            self._logo_label.config(image=self.logo_img)
        except Exception as e:
            print(f"Logo load: {e}")

    def _build_info_box(self, parent):
        info_box = tk.Frame(
            parent, bg=SPLASH_INFO_BG,
            highlightthickness=1,
            highlightbackground=SPLASH_ACCENT,
        )
        info_box.pack(fill="x", padx=28, pady=(0, 24))

        row = tk.Frame(info_box, bg=SPLASH_INFO_BG)
        row.pack(fill="x", padx=24, pady=18)

        prof = tk.Frame(row, bg=SPLASH_INFO_BG)
        prof.pack(side="left", expand=True)
        tk.Label(
            prof,
            text="\U0001f468\u200d\U0001f3eb  Professor: EL MKHALET MOUNA",
            bg=SPLASH_INFO_BG, fg=COLORS["text_white"],
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")

        student = tk.Frame(row, bg=SPLASH_INFO_BG)
        student.pack(side="right", expand=True)
        tk.Label(
            student,
            text="\U0001f393  Student: ETTAHIRI RAYAN",
            bg=SPLASH_INFO_BG, fg=COLORS["text_white"],
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="e")

    def _build_start_button(self, parent):
        btn_frame = tk.Frame(parent, bg=SPLASH_CARD_BG)
        btn_frame.pack(fill="x", padx=28, pady=(0, 28))

        self.launch_btn = tk.Button(
            btn_frame,
            text="\u25ba  START",
            command=self._launch_app,
            bg=SPLASH_ACCENT, fg="#0a0f18",
            font=("Segoe UI", 14, "bold"),
            relief="flat", bd=0, padx=20, pady=14,
            activebackground="#33e0ff",
            activeforeground="#0a0f18",
            cursor="hand2",
        )
        self.launch_btn.pack(fill="x")
        self.launch_btn.bind("<Enter>", lambda e: self.launch_btn.config(bg="#33e0ff"))
        self.launch_btn.bind("<Leave>", lambda e: self.launch_btn.config(bg=SPLASH_ACCENT))

    def _build_footer(self):
        tk.Label(
            self,
            text="v2.0 \u2022 EMSI 2025",
            bg=SPLASH_BG, fg=COLORS["text_muted"],
            font=("Segoe UI", 9),
        ).place(relx=0.5, rely=0.97, anchor="s")

    def _animate(self):
        if not self.winfo_exists():
            return
        w = max(self.winfo_width(), 1)
        h = max(self.winfo_height(), 1)
        self.canvas.delete("particle")
        self._angle += 0.25

        hex_outline = "#1a2744"
        for row in range(0, h + 60, 56):
            for col in range(0, w + 70, 64):
                x, y = col + (row % 2) * 32, row
                size = 16
                pts = []
                for a in range(6):
                    ang = math.radians(60 * a + self._angle * 0.08)
                    pts += [x + size * math.cos(ang), y + size * math.sin(ang)]
                self.canvas.create_polygon(
                    pts, outline=hex_outline, fill="", tags="particle", width=1,
                )

        for p in self._particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            if p["x"] < 0 or p["x"] > w:
                p["vx"] *= -1
            if p["y"] < 0 or p["y"] > h:
                p["vy"] *= -1
            self.canvas.create_oval(
                p["x"] - p["r"], p["y"] - p["r"],
                p["x"] + p["r"], p["y"] + p["r"],
                fill=p["color"], outline="", tags="particle",
            )

        cx, cy = w / 2, h / 2
        ring_r = min(w, h) * 0.38
        for i in range(10):
            a = math.radians(self._angle * 1.2 + i * 36)
            rx = cx + ring_r * math.cos(a)
            ry = cy + ring_r * math.sin(a) * 0.28
            colors = [SPLASH_ACCENT, "#7B61FF", "#56E39F", "#FF9F43"]
            self.canvas.create_oval(
                rx - 2, ry - 2, rx + 2, ry + 2,
                fill=colors[i % 4], outline="", tags="particle",
            )
        self.after(35, self._animate)

    def _launch_app(self):
        self._launch = True
        self.destroy()

def gen_random_data(n=200, x1_range=(-10, 10), x2_range=(-5, 15), x3_range=(0, 20)):
    np.random.seed(42)
    X1 = np.random.uniform(*x1_range, n)
    X2 = np.random.uniform(*x2_range, n)
    X3 = np.random.uniform(*x3_range, n)
    noise = np.random.randn(n) * 2
    Y = 3.5*X1 - 2.1*X2 + 1.8*X3 + noise
    return X1, X2, X3, Y

def styled_entry(parent, width=8):
    e = tk.Entry(parent, width=width, bg=COLORS["bg_card"],
                fg=COLORS["text_white"], insertbackground=COLORS["text_white"],
                font=FONT_MONO, relief="flat", bd=4,
                highlightthickness=1, highlightcolor=COLORS["border"],
                highlightbackground=COLORS["border"])
    return e

def styled_button(parent, text, color, command):
    btn = tk.Button(parent, text=text, command=command,
                    bg=color, fg="#0D1117", font=FONT_BTN,
                    relief="flat", bd=0, padx=14, pady=7,
                    activebackground=color, cursor="hand2")
    return btn

def embed_figure(parent, fig, bg_color):
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    w = canvas.get_tk_widget()
    w.config(bg=bg_color)
    w.pack(fill="both", expand=True, padx=4, pady=4)
    return canvas

def style_3d_ax(ax, accent):
    """Apply dark theme to a 3D axis."""
    ax.set_facecolor(COLORS["bg_card"])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(COLORS["border"])
    ax.yaxis.pane.set_edgecolor(COLORS["border"])
    ax.zaxis.pane.set_edgecolor(COLORS["border"])
    ax.tick_params(colors=COLORS["text_muted"], labelsize=7)
    ax.xaxis.label.set_color(COLORS["text_muted"])
    ax.yaxis.label.set_color(COLORS["text_muted"])
    ax.zaxis.label.set_color(COLORS["text_muted"])
    ax.title.set_color(accent)

class TabBase(tk.Frame):
    def __init__(self, parent, accent):
        super().__init__(parent, bg=COLORS["bg_dark"])
        self.accent = accent
        self._view_3d = False
        self._build()
    
    def _build(self):
        pass
    
    def _section_title(self, parent, text):
        f = tk.Frame(parent, bg=COLORS["bg_panel"])
        f.pack(fill="x", pady=(10, 4))
        tk.Label(f, text=text, bg=COLORS["bg_panel"],
                fg=self.accent, font=FONT_SUB).pack(anchor="w", padx=10)
        tk.Frame(f, bg=self.accent, height=2).pack(fill="x", padx=10)
        return f
    
    def _result_box(self, parent, height=8):
        frame = tk.Frame(parent, bg=COLORS["bg_card"],
                        highlightthickness=1,
                        highlightbackground=COLORS["border"])
        frame.pack(fill="x", padx=10, pady=4)
        txt = tk.Text(frame, height=height, bg=COLORS["bg_card"],
                    fg=COLORS["text_white"], font=FONT_MONO,
                    relief="flat", bd=6, state="disabled",
                    wrap="word", insertbackground="white")
        txt.pack(fill="both")
        return txt
    
    def _write_result(self, widget, text):
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.config(state="disabled")
    
    def _view_toggle_btn(self, parent, run_callback):
        """Add 2D/3D toggle button."""
        f = tk.Frame(parent, bg=COLORS["bg_panel"])
        f.pack(fill="x", padx=10, pady=4)
        def toggle():
            self._view_3d = not self._view_3d
            lbl = " Vue 2D" if self._view_3d else " Vue 3D"
            toggle_btn.config(text=lbl)
            run_callback()
        toggle_btn = tk.Button(f, text=" Vue 3D",
                                command=toggle,
                                bg=COLORS["bg_card"], fg=self.accent,
                                font=("Courier New", 9, "bold"),
                                relief="flat", bd=0, padx=10, pady=5,
                                highlightthickness=1,
                                highlightbackground=self.accent,
                                cursor="hand2")
        toggle_btn.pack(side="left")
        return toggle_btn

class RegressionTab(TabBase):   
    def _build(self):
        left = tk.Frame(self, bg=COLORS["bg_panel"], width=250)
        left.pack(side="left", fill="y", padx=(8,4), pady=8)
        left.pack_propagate(False)
        tk.Label(left, text=" RÉGRESSION", bg=COLORS["bg_panel"],
                fg=self.accent, font=FONT_TITLE).pack(pady=(18,4))
        tk.Label(left, text="Linéaire Multiple", bg=COLORS["bg_panel"],
                fg=COLORS["text_muted"], font=FONT_LABEL).pack()
        
        self._section_title(left, "Paramètres X1")
        f1 = tk.Frame(left, bg=COLORS["bg_panel"])
        f1.pack(fill="x", padx=10)
        tk.Label(f1, text="Min:", bg=COLORS["bg_panel"], fg=COLORS["text_muted"], font=FONT_LABEL).grid(row=0, column=0, sticky="w")
        self.x1_min = styled_entry(f1, 6)
        self.x1_min.insert(0, "-10")
        self.x1_min.grid(row=0, column=1, padx=5)
        tk.Label(f1, text="Max:", bg=COLORS["bg_panel"], fg=COLORS["text_muted"], font=FONT_LABEL).grid(row=0, column=2, sticky="w", padx=(10,0))
        self.x1_max = styled_entry(f1, 6)
        self.x1_max.insert(0, "10")
        self.x1_max.grid(row=0, column=3, padx=5)
        
        self._section_title(left, "Paramètres X2")
        f2 = tk.Frame(left, bg=COLORS["bg_panel"])
        f2.pack(fill="x", padx=10)
        tk.Label(f2, text="Min:", bg=COLORS["bg_panel"], fg=COLORS["text_muted"], font=FONT_LABEL).grid(row=0, column=0, sticky="w")
        self.x2_min = styled_entry(f2, 6)
        self.x2_min.insert(0, "-5")
        self.x2_min.grid(row=0, column=1, padx=5)
        tk.Label(f2, text="Max:", bg=COLORS["bg_panel"], fg=COLORS["text_muted"], font=FONT_LABEL).grid(row=0, column=2, sticky="w", padx=(10,0))
        self.x2_max = styled_entry(f2, 6)
        self.x2_max.insert(0, "15")
        self.x2_max.grid(row=0, column=3, padx=5)
        
        self._section_title(left, "Affichage")
        self._view_toggle_btn(left, self._run_if_data)
        
        self._section_title(left, "Résultats")
        self.res = self._result_box(left, 8)
        styled_button(left, " EXÉCUTER", self.accent, self._run).pack(pady=12)
        
        self.right = tk.Frame(self, bg=COLORS["bg_dark"])
        self.right.pack(side="left", fill="both", expand=True, padx=(4,8), pady=8)
        self._last_data = None
    
    def _run_if_data(self):
        if self._last_data: 
            self._draw(*self._last_data)
    
    def _run(self):
        try:
            x1_min = float(self.x1_min.get())
            x1_max = float(self.x1_max.get())
            x2_min = float(self.x2_min.get())
            x2_max = float(self.x2_max.get())
        except ValueError:
            messagebox.showerror("Erreur", "Valeurs invalides")
            return
        
        X1, X2, X3, Y = gen_random_data(200, (x1_min, x1_max), (x2_min, x2_max))
        X = np.column_stack([X1, X2, X3])
        model = LinearRegression().fit(X, Y)
        Y_pred = model.predict(X)
        r2 = r2_score(Y, Y_pred)
        mse = mean_squared_error(Y, Y_pred)
        a, b, c = model.coef_
        d = model.intercept_
        
        txt = (f"Modèle: Y = {d:.4f} + {a:.4f}·X1 + {b:.4f}·X2 + {c:.4f}·X3\n"
               f"{'─'*38}\nR² = {r2:.4f}\nMSE = {mse:.4f}\nRMSE= {np.sqrt(mse):.4f}\n\n"
               f"Coefficients:\n β0={d:.4f}\n β1={a:.4f}\n β2={b:.4f}\n β3={c:.4f}\n\n"
               f"N = 200 échantillons\nFeatures: X1, X2, X3")
        self._write_result(self.res, txt)
        self._last_data = (X1, X2, X3, Y, Y_pred, model)
        self._draw(X1, X2, X3, Y, Y_pred, model)
    
    def _draw(self, X1, X2, X3, Y, Y_pred, model):
        for w in self.right.winfo_children(): 
            w.destroy()
        
        fig = Figure(figsize=(7, 5), facecolor=COLORS["bg_dark"])
        if self._view_3d:
            ax = fig.add_subplot(111, projection='3d', facecolor=COLORS["bg_card"])
            ax.scatter(X1, X2, Y, color=self.accent, alpha=0.4, s=15, label="Données")
            x1g = np.linspace(X1.min(), X1.max(), 20)
            x2g = np.linspace(X2.min(), X2.max(), 20)
            XX1, XX2 = np.meshgrid(x1g, x2g)
            XX3 = np.full_like(XX1, X3.mean())
            ZZ = model.predict(np.column_stack([XX1.ravel(), XX2.ravel(), XX3.ravel()])).reshape(XX1.shape)
            ax.plot_surface(XX1, XX2, ZZ, alpha=0.3, color="#FF6B6B", edgecolor="none")
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            ax.set_zlabel("Y")
            ax.set_title("Plan de Régression 3D", fontsize=11)
            style_3d_ax(ax, self.accent)
            ax.view_init(elev=25, azim=135)
        else:
            ax = fig.add_subplot(111, facecolor=COLORS["bg_card"])
            ax.scatter(Y, Y_pred, color=self.accent, alpha=0.5, s=20, label="Prédictions")
            mn, mx = min(Y.min(), Y_pred.min()), max(Y.max(), Y_pred.max())
            ax.plot([mn, mx], [mn, mx], color="#FF6B6B", lw=2, label="Idéal")
            ax.set_xlabel("Valeurs réelles", color=COLORS["text_muted"])
            ax.set_ylabel("Valeurs prédites", color=COLORS["text_muted"])
            ax.set_title("Régression Linéaire Multiple", color=self.accent, fontsize=13)
            ax.tick_params(colors=COLORS["text_muted"])
            for spine in ax.spines.values(): 
                spine.set_edgecolor(COLORS["border"])
            ax.legend(facecolor=COLORS["bg_panel"], labelcolor=COLORS["text_white"])
        
        fig.tight_layout()
        embed_figure(self.right, fig, COLORS["bg_dark"])

class ClusteringTab(TabBase):
    def _build(self):
        left = tk.Frame(self, bg=COLORS["bg_panel"], width=250)
        left.pack(side="left", fill="y", padx=(8,4), pady=8)
        left.pack_propagate(False)
        tk.Label(left, text=" CLUSTERING", bg=COLORS["bg_panel"],
                fg=self.accent, font=FONT_TITLE).pack(pady=(18,4))
        tk.Label(left, text="K-Means", bg=COLORS["bg_panel"],
                fg=COLORS["text_muted"], font=FONT_LABEL).pack()
        
        self._section_title(left, "Paramètres")
        f = tk.Frame(left, bg=COLORS["bg_panel"])
        f.pack(fill="x", padx=10)
        tk.Label(f, text="Clusters K:", bg=COLORS["bg_panel"], fg=COLORS["text_muted"], font=FONT_LABEL).grid(row=0, column=0, sticky="w")
        self.k_val = styled_entry(f, 6)
        self.k_val.insert(0, "3")
        self.k_val.grid(row=0, column=1, padx=5)
        tk.Label(f, text="Taille N:", bg=COLORS["bg_panel"], fg=COLORS["text_muted"], font=FONT_LABEL).grid(row=1, column=0, sticky="w", pady=(5,0))
        self.n_val = styled_entry(f, 6)
        self.n_val.insert(0, "300")
        self.n_val.grid(row=1, column=1, padx=5, pady=(5,0))
        
        self._section_title(left, "Affichage")
        self._view_toggle_btn(left, self._run_if_data)
        
        self._section_title(left, "Résultats")
        self.res = self._result_box(left, 8)
        styled_button(left, " CLUSTERING", self.accent, self._run).pack(pady=12)
        
        self.right = tk.Frame(self, bg=COLORS["bg_dark"])
        self.right.pack(side="left", fill="both", expand=True, padx=(4,8), pady=8)
        self._last_data = None
    
    def _run_if_data(self):
        if self._last_data: 
            self._draw(*self._last_data)
    
    def _run(self):
        try:
            k = int(self.k_val.get())
            n = int(self.n_val.get())
        except ValueError:
            messagebox.showerror("Erreur", "Valeurs invalides")
            return
        
        X1, X2, X3, _ = gen_random_data(n)
        X = np.column_stack([X1, X2, X3])
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        centers = model.cluster_centers_
        inertia = model.inertia_
        
        txt = f"K-Means k={k} N={n}\n{'─'*34}\nInertie: {inertia:.2f}\n\nCentres:\n"
        for i, c in enumerate(centers):
            txt += f" Cluster {i+1}: [{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}]\n"
        sizes = np.bincount(labels)
        txt += "\nTailles:\n"
        for i, s in enumerate(sizes):
            txt += f" Cluster {i+1}: {s} points\n"
        
        self._write_result(self.res, txt)
        self._last_data = (X, labels, centers, k)
        self._draw(X, labels, centers, k)
    
    def _draw(self, X, labels, centers, k):
        for w in self.right.winfo_children(): 
            w.destroy()
        
        cluster_colors = ["#FF6B6B","#56E39F","#00D4FF","#FFD93D","#C77DFF"]
        fig = Figure(figsize=(7, 5), facecolor=COLORS["bg_dark"])
        
        if self._view_3d:
            ax = fig.add_subplot(111, projection='3d')
            for i in range(k):
                mask = labels == i
                ax.scatter(X[mask,0], X[mask,1], X[mask,2],
                          color=cluster_colors[i%len(cluster_colors)],
                          alpha=0.5, s=14, label=f"C{i+1}")
            ax.scatter(centers[:,0], centers[:,1], centers[:,2],
                      color="white", marker="*", s=250, zorder=5, label="Centres")
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            ax.set_zlabel("X3")
            ax.set_title(f"K-Means 3D (k={k})", fontsize=11)
            style_3d_ax(ax, self.accent)
            ax.view_init(elev=20, azim=45)
            ax.legend(facecolor=COLORS["bg_panel"], labelcolor=COLORS["text_white"], fontsize=8)
        else:
            ax = fig.add_subplot(111, facecolor=COLORS["bg_card"])
            for i in range(k):
                mask = labels == i
                ax.scatter(X[mask,0], X[mask,1], color=cluster_colors[i%len(cluster_colors)],
                          alpha=0.6, s=18, label=f"Cluster {i+1}")
            ax.scatter(centers[:,0], centers[:,1], color="white", marker="*", s=200, zorder=5)
            ax.set_xlabel("X1", color=COLORS["text_muted"])
            ax.set_ylabel("X2", color=COLORS["text_muted"])
            ax.set_title(f"K-Means Clustering (k={k})", color=self.accent, fontsize=13)
            ax.tick_params(colors=COLORS["text_muted"])
            for spine in ax.spines.values(): 
                spine.set_edgecolor(COLORS["border"])
            ax.legend(facecolor=COLORS["bg_panel"], labelcolor=COLORS["text_white"])
        
        fig.tight_layout()
        embed_figure(self.right, fig, COLORS["bg_dark"])

class ForestTab(TabBase):
    def _build(self):
        left = tk.Frame(self, bg=COLORS["bg_panel"], width=300)
        left.pack(side="left", fill="y", padx=(8,4), pady=8)
        left.pack_propagate(False)
        
        self._section_title(left, "Paramètres Random Forest")
        f_params = tk.Frame(left, bg=COLORS["bg_panel"])
        f_params.pack(fill="x", padx=10, pady=5)
        
        self.entries = {}
        for var, dmin, dmax in [("Variable X1", "-10", "10"), ("Variable X2", "-5", "15"), ("Variable X3", "0", "20")]:
            tk.Label(f_params, text=var, bg=COLORS["bg_panel"], fg=COLORS["text_white"], font=FONT_LABEL).pack(anchor="w")
            row = tk.Frame(f_params, bg=COLORS["bg_panel"])
            row.pack(fill="x")
            emin = styled_entry(row, 6); emin.insert(0, dmin); emin.pack(side="left", padx=2)
            emax = styled_entry(row, 6); emax.insert(0, dmax); emax.pack(side="left", padx=2)
            self.entries[var] = (emin, emax)

        tk.Label(f_params, text="Nombre d'arbres:", bg=COLORS["bg_panel"], fg=COLORS["text_muted"], font=FONT_MONO).pack(anchor="w", pady=(10,0))
        self.n_trees = styled_entry(f_params, 15); self.n_trees.insert(0, "100"); self.n_trees.pack(anchor="w", padx=10)
        tk.Label(f_params, text="Nombre de classes:", bg=COLORS["bg_panel"], fg=COLORS["text_muted"], font=FONT_MONO).pack(anchor="w")
        self.n_cls = styled_entry(f_params, 15); self.n_cls.insert(0, "3"); self.n_cls.pack(anchor="w", padx=10)

        self._section_title(left, "Métriques de performance")
        self.perf_res = self._result_box(left, 5)
        self._section_title(left, "Importance des variables")
        self.imp_res = self._result_box(left, 4)
        styled_button(left, "EXÉCUTER L'ANALYSE", self.accent, self._run).pack(pady=15)

        self.right = tk.Frame(self, bg=COLORS["bg_dark"])
        self.right.pack(side="left", fill="both", expand=True, padx=(4,8), pady=8)

    def _run(self):
        try:
            n_trees, n_cls = int(self.n_trees.get()), int(self.n_cls.get())
            X1, X2, X3, Y = gen_random_data(400)
            X = np.column_stack([X1, X2, X3])
            y_cls = np.digitize(Y, np.percentile(Y, np.linspace(0, 100, n_cls+1))[1:-1])
            model = RandomForestClassifier(n_estimators=n_trees, random_state=42).fit(X, y_cls)
            y_pred = model.predict(X)
            
            self._write_result(self.perf_res, f"Accuracy: {accuracy_score(y_cls, y_pred):.4f}\nF1 Score: {f1_score(y_cls, y_pred, average='weighted'):.4f}")
            imp = model.feature_importances_
            self._write_result(self.imp_res, f"X1: {imp[0]:.2%}\nX2: {imp[1]:.2%}\nX3: {imp[2]:.2%}")
            self._draw(X, y_cls, y_pred, imp, model, n_cls)
        except Exception as e: messagebox.showerror("Erreur", str(e))

    def _draw(self, X, y_cls, y_pred, imp, model, n_cls):
        for w in self.right.winfo_children(): w.destroy()
        fig = plt.figure(figsize=(9, 6), facecolor=COLORS["bg_dark"])
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0]); ax1.bar(["X1", "X2", "X3"], imp, color="#3498db")
        ax2 = fig.add_subplot(gs[0, 1]); sns.heatmap(confusion_matrix(y_cls, y_pred), annot=True, cmap='Blues', ax=ax2, cbar=False)
        ax3 = fig.add_subplot(gs[1, :]); 
        x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
        y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel(), np.full(xx.ravel().shape, X[:,2].mean())]).reshape(xx.shape)
        ax3.contourf(xx, yy, Z, alpha=0.3, cmap='viridis'); ax3.scatter(X[:,0], X[:,1], c=y_cls, s=10, cmap='viridis')
        fig.tight_layout(); embed_figure(self.right, fig, COLORS["bg_dark"])

class TimeSeriesTab(TabBase):
    def _build(self):
        left = tk.Frame(self, bg=COLORS["bg_panel"], width=250)
        left.pack(side="left", fill="y", padx=(8,4), pady=8)
        left.pack_propagate(False)
        tk.Label(left, text=" TIME SERIES", bg=COLORS["bg_panel"],
                fg=self.accent, font=("Courier New", 16, "bold")).pack(pady=(18,4))
        tk.Label(left, text="ARIMA simplifié", bg=COLORS["bg_panel"],
                fg=COLORS["text_muted"], font=FONT_LABEL).pack()
        
        self._section_title(left, "Paramètres")
        f = tk.Frame(left, bg=COLORS["bg_panel"])
        f.pack(fill="x", padx=10)
        tk.Label(f, text="N points:", bg=COLORS["bg_panel"], fg=COLORS["text_muted"], font=FONT_LABEL).grid(row=0, column=0, sticky="w")
        self.n_pts = styled_entry(f, 6)
        self.n_pts.insert(0, "200")
        self.n_pts.grid(row=0, column=1, padx=5)
        tk.Label(f, text="Prévision:", bg=COLORS["bg_panel"], fg=COLORS["text_muted"], font=FONT_LABEL).grid(row=1, column=0, sticky="w", pady=(5,0))
        self.n_pred = styled_entry(f, 6)
        self.n_pred.insert(0, "30")
        self.n_pred.grid(row=1, column=1, padx=5, pady=(5,0))
        tk.Label(f, text="Min valeur:", bg=COLORS["bg_panel"], fg=COLORS["text_muted"], font=FONT_LABEL).grid(row=2, column=0, sticky="w", pady=(5,0))
        self.v_min = styled_entry(f, 6)
        self.v_min.insert(0, "0")
        self.v_min.grid(row=2, column=1, padx=5, pady=(5,0))
        tk.Label(f, text="Max valeur:", bg=COLORS["bg_panel"], fg=COLORS["text_muted"], font=FONT_LABEL).grid(row=3, column=0, sticky="w", pady=(5,0))
        self.v_max = styled_entry(f, 6)
        self.v_max.insert(0, "20")
        self.v_max.grid(row=3, column=1, padx=5, pady=(5,0))
        
        self._section_title(left, "Affichage")
        self._view_toggle_btn(left, self._run_if_data)
        
        self._section_title(left, "Résultats")
        self.res = self._result_box(left, 6)
        styled_button(left, " ANALYSER", self.accent, self._run).pack(pady=12)
        
        self.right = tk.Frame(self, bg=COLORS["bg_dark"])
        self.right.pack(side="left", fill="both", expand=True, padx=(4,8), pady=8)
        self._last_data = None
    
    def _run_if_data(self):
        if self._last_data: 
            self._draw(*self._last_data)
    
    def _run(self):
        try:
            n = int(self.n_pts.get())
            n_f = int(self.n_pred.get())
            vmin = float(self.v_min.get())
            vmax = float(self.v_max.get())
        except ValueError:
            messagebox.showerror("Erreur", "Valeurs invalides")
            return
        
        np.random.seed(7)
        t = np.arange(n)
        trend = np.linspace(vmin, vmax, n)
        season = 3 * np.sin(2 * np.pi * t / 20)
        noise = np.random.randn(n) * 1.5
        series = trend + season + noise
        
        window = 10
        ma = np.convolve(series, np.ones(window)/window, mode="valid")
        last_slope = (ma[-1] - ma[-5]) / 5
        forecast = np.array([ma[-1] + last_slope*i + np.random.randn()*1.2 for i in range(1, n_f+1)])
        std_v = np.std(series)
        
        txt = (f"Série Temporelle\n{'─'*34}\n"
               f"N = {n} points\nPrévision = {n_f} pas\n\n"
               f"Statistiques:\n Moy: {np.mean(series):.4f}\n Std: {std_v:.4f}\n"
               f" Min: {series.min():.4f}\n Max: {series.max():.4f}\n\n"
               f"Prévision (premiers):\n"
               + "\n".join(f" t+{i+1}: {v:.4f}" for i, v in enumerate(forecast[:5]))
               + "\n ...")
        
        self._write_result(self.res, txt)
        self._last_data = (t, series, forecast, n, n_f, std_v)
        self._draw(t, series, forecast, n, n_f, std_v)
    
    def _draw(self, t, series, forecast, n, n_f, std_v):
        for w in self.right.winfo_children(): 
            w.destroy()
        
        fig = Figure(figsize=(7, 5), facecolor=COLORS["bg_dark"])
        t_fore = np.arange(n, n + n_f)
        
        if self._view_3d:
            ax = fig.add_subplot(111, projection='3d')
            lag1 = series[1:-1]
            lag2 = series[:-2]
            lag0 = series[2:]
            ax.scatter(lag2, lag1, lag0, c=np.arange(len(lag0)),
                      cmap='plasma', s=8, alpha=0.7)
            ax.set_xlabel("t-2")
            ax.set_ylabel("t-1")
            ax.set_zlabel("t")
            ax.set_title("Espace de Phase 3D\n(Attractor)", fontsize=10)
            style_3d_ax(ax, self.accent)
            ax.view_init(elev=20, azim=55)
        else:
            ax = fig.add_subplot(111, facecolor=COLORS["bg_card"])
            ax.plot(t, series, color=self.accent, lw=1.2, alpha=0.8, label="Série")
            ax.plot(t_fore, forecast, color="#FF6B6B", lw=2, linestyle="--", label="Prévision")
            ax.fill_between(t_fore,
                           forecast - std_v*0.8, forecast + std_v*0.8,
                           alpha=0.15, color="#FF6B6B")
            ax.axvline(n-1, color=COLORS["border"], lw=1, linestyle=":")
            ax.set_xlabel("Temps", color=COLORS["text_muted"])
            ax.set_ylabel("Valeur", color=COLORS["text_muted"])
            ax.set_title("Série Temporelle + Prévision", color=self.accent, fontsize=13)
            ax.tick_params(colors=COLORS["text_muted"])
            for spine in ax.spines.values(): 
                spine.set_edgecolor(COLORS["border"])
            ax.legend(facecolor=COLORS["bg_panel"], labelcolor=COLORS["text_white"])
        
        fig.tight_layout()
        embed_figure(self.right, fig, COLORS["bg_dark"])

class NeuralTab(TabBase):
    def _build(self):
        left = tk.Frame(self, bg=COLORS["bg_panel"], width=250)
        left.pack(side="left", fill="y", padx=(8,4), pady=8)
        left.pack_propagate(False)
        tk.Label(left, text=" NEURONES", bg=COLORS["bg_panel"],
                fg=self.accent, font=FONT_TITLE).pack(pady=(18,4))
        tk.Label(left, text="MLP Régression", bg=COLORS["bg_panel"],
                fg=COLORS["text_muted"], font=FONT_LABEL).pack()
        
        self._section_title(left, "Architecture")
        f = tk.Frame(left, bg=COLORS["bg_panel"])
        f.pack(fill="x", padx=10)
        tk.Label(f, text="Couche 1:", bg=COLORS["bg_panel"], fg=COLORS["text_muted"], font=FONT_LABEL).grid(row=0, column=0, sticky="w")
        self.l1 = styled_entry(f, 6)
        self.l1.insert(0, "64")
        self.l1.grid(row=0, column=1, padx=5)
        tk.Label(f, text="Couche 2:", bg=COLORS["bg_panel"], fg=COLORS["text_muted"], font=FONT_LABEL).grid(row=1, column=0, sticky="w", pady=(5,0))
        self.l2 = styled_entry(f, 6)
        self.l2.insert(0, "32")
        self.l2.grid(row=1, column=1, padx=5, pady=(5,0))
        tk.Label(f, text="Itérations:", bg=COLORS["bg_panel"], fg=COLORS["text_muted"], font=FONT_LABEL).grid(row=2, column=0, sticky="w", pady=(5,0))
        self.iters = styled_entry(f, 6)
        self.iters.insert(0, "300")
        self.iters.grid(row=2, column=1, padx=5, pady=(5,0))
        
        self._section_title(left, "Affichage")
        self._view_toggle_btn(left, self._run_if_data)
        
        self._section_title(left, "Résultats")
        self.res = self._result_box(left, 8)
        styled_button(left, " ENTRAÎNER", self.accent, self._run).pack(pady=12)
        
        self.right = tk.Frame(self, bg=COLORS["bg_dark"])
        self.right.pack(side="left", fill="both", expand=True, padx=(4,8), pady=8)
        self._last_data = None
    
    def _run_if_data(self):
        if self._last_data: 
            self._draw(*self._last_data)
    
    def _run(self):
        try:
            l1 = int(self.l1.get())
            l2 = int(self.l2.get())
            iters = int(self.iters.get())
        except ValueError:
            messagebox.showerror("Erreur", "Valeurs invalides")
            return
        
        X1, X2, X3, Y = gen_random_data(200)
        X = np.column_stack([X1, X2, X3])
        sc = StandardScaler()
        X_s = sc.fit_transform(X)
        split = 160
        X_tr, X_te = X_s[:split], X_s[split:]
        y_tr, y_te = Y[:split], Y[split:]
        
        model = MLPRegressor(hidden_layer_sizes=(l1, l2), max_iter=iters,
                            random_state=42, early_stopping=True, validation_fraction=0.1)
        model.fit(X_tr, y_tr)
        Y_pred = model.predict(X_te)
        mse = mean_squared_error(y_te, Y_pred)
        r2 = r2_score(y_te, Y_pred)
        
        txt = (f"MLP Régression\n{'─'*34}\n"
               f"Architecture: 3→{l1}→{l2}→1\nItérations: {model.n_iter_}/{iters}\n\n"
               f"Performance test:\n R² = {r2:.4f}\n MSE = {mse:.4f}\n RMSE = {np.sqrt(mse):.4f}\n"
               f"Train: {split} Test: {len(X)-split}\nLoss finale: {model.loss_:.6f}")
        
        self._write_result(self.res, txt)
        self._last_data = (X_te, y_te, Y_pred, model, l1, l2)
        self._draw(X_te, y_te, Y_pred, model, l1, l2)
    
    def _draw(self, X_te, y_te, Y_pred, model, l1, l2):
        for w in self.right.winfo_children(): 
            w.destroy()
        
        fig = Figure(figsize=(7, 5), facecolor=COLORS["bg_dark"])
        
        if self._view_3d:
            ax = fig.add_subplot(111, projection='3d')
            w_range = np.linspace(-2, 2, 30)
            b_range = np.linspace(-2, 2, 30)
            WW, BB = np.meshgrid(w_range, b_range)
            ZZ = np.sin(WW**2 + BB**2) * np.exp(-0.1*(WW**2+BB**2)) + 0.5*(WW**2+BB**2)*0.1
            ax.plot_surface(WW, BB, ZZ, cmap='plasma', alpha=0.7, edgecolor='none')
            ax.set_xlabel("W")
            ax.set_ylabel("b")
            ax.set_zlabel("Loss")
            ax.set_title("Surface de Loss 3D\n(Paysage d'Optimisation)", fontsize=10)
            style_3d_ax(ax, self.accent)
            ax.view_init(elev=30, azim=60)
        else:
            ax1 = fig.add_subplot(121, facecolor=COLORS["bg_card"])
            ax1.scatter(y_te, Y_pred, color=self.accent, alpha=0.6, s=20)
            mn, mx = min(y_te.min(), Y_pred.min()), max(y_te.max(), Y_pred.max())
            ax1.plot([mn, mx], [mn, mx], "#FF6B6B", lw=2)
            ax1.set_xlabel("Réel", color=COLORS["text_muted"])
            ax1.set_ylabel("Prédit", color=COLORS["text_muted"])
            ax1.set_title("Prédictions", color=self.accent, fontsize=11)
            ax1.tick_params(colors=COLORS["text_muted"])
            for spine in ax1.spines.values(): 
                spine.set_edgecolor(COLORS["border"])
            
            ax2 = fig.add_subplot(122, facecolor=COLORS["bg_card"])
            ax2.plot(model.loss_curve_, color=self.accent, lw=1.5)
            ax2.set_xlabel("Itération", color=COLORS["text_muted"])
            ax2.set_ylabel("Loss", color=COLORS["text_muted"])
            ax2.set_title("Courbe d'Apprentissage", color=self.accent, fontsize=11)
            ax2.tick_params(colors=COLORS["text_muted"])
            for spine in ax2.spines.values(): 
                spine.set_edgecolor(COLORS["border"])
        
        fig.tight_layout()
        embed_figure(self.right, fig, COLORS["bg_dark"])

class ValidationTab(TabBase):
    def _build(self):
        left = tk.Frame(self, bg=COLORS["bg_panel"], width=250)
        left.pack(side="left", fill="y", padx=(8,4), pady=8)
        left.pack_propagate(False)
        tk.Label(left, text=" VALIDATION", bg=COLORS["bg_panel"],
                fg=self.accent, font=FONT_TITLE).pack(pady=(18,4))
        tk.Label(left, text="Croisée K-Fold", bg=COLORS["bg_panel"],
                fg=COLORS["text_muted"], font=FONT_LABEL).pack()
        
        self._section_title(left, "Paramètres")
        f = tk.Frame(left, bg=COLORS["bg_panel"])
        f.pack(fill="x", padx=10)
        tk.Label(f, text="K folds:", bg=COLORS["bg_panel"], fg=COLORS["text_muted"], font=FONT_LABEL).grid(row=0, column=0, sticky="w")
        self.kfolds = styled_entry(f, 6)
        self.kfolds.insert(0, "5")
        self.kfolds.grid(row=0, column=1, padx=5)
        tk.Label(f, text="N points:", bg=COLORS["bg_panel"], fg=COLORS["text_muted"], font=FONT_LABEL).grid(row=1, column=0, sticky="w", pady=(5,0))
        self.n_pts = styled_entry(f, 6)
        self.n_pts.insert(0, "200")
        self.n_pts.grid(row=1, column=1, padx=5, pady=(5,0))
        
        self._section_title(left, "Affichage")
        self._view_toggle_btn(left, self._run_if_data)
        
        self._section_title(left, "Résultats")
        self.res = self._result_box(left, 10)
        styled_button(left, " COMPARER", self.accent, self._run).pack(pady=12)
        
        self.right = tk.Frame(self, bg=COLORS["bg_dark"])
        self.right.pack(side="left", fill="both", expand=True, padx=(4,8), pady=8)
        self._last_data = None
    
    def _run_if_data(self):
        if self._last_data: 
            self._draw(*self._last_data)
    
    def _run(self):
        try:
            k = int(self.kfolds.get())
            n = int(self.n_pts.get())
        except ValueError:
            messagebox.showerror("Erreur", "Valeurs invalides")
            return
        
        X1, X2, X3, Y = gen_random_data(n)
        X = np.column_stack([X1, X2, X3])
        sc_obj = StandardScaler()
        X_s = sc_obj.fit_transform(X)
        bins = np.percentile(Y, [33, 66])
        y_cls = np.digitize(Y, bins)
        cv = KFold(n_splits=k, shuffle=True, random_state=42)
        
        models = {
            "Régression Log.": LogisticRegression(max_iter=500, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
            "Arbre Décision": DecisionTreeClassifier(random_state=42),
        }
        
        results = {}
        for name, m in models.items():
            scores = cross_val_score(m, X_s, y_cls, cv=cv, scoring="accuracy")
            results[name] = scores
        
        txt = f"Validation Croisée k={k}\n{'─'*34}\n"
        for name, sc_arr in results.items():
            txt += f"\n{name}:\n Moy={sc_arr.mean():.4f} Std={sc_arr.std():.4f}\n"
            txt += f" Folds: " + " | ".join(f"{s:.3f}" for s in sc_arr) + "\n"
        
        self._write_result(self.res, txt)
        self._last_data = (results, k, X_s, y_cls)
        self._draw(results, k, X_s, y_cls)
    
    def _draw(self, results, k, X_s, y_cls):
        for w in self.right.winfo_children(): 
            w.destroy()
        
        fig = Figure(figsize=(7, 5), facecolor=COLORS["bg_dark"])
        names = list(results.keys())
        means = [results[n].mean() for n in names]
        stds = [results[n].std() for n in names]
        bcolors = [self.accent, "#56E39F", "#FF6B6B"]
        
        if self._view_3d:
            ax = fig.add_subplot(111, projection='3d')
            all_scores = [results[n] for n in names]
            fold_ids = np.arange(k)
            for mi, (name, scores) in enumerate(zip(names, all_scores)):
                ax.bar3d(mi*np.ones(k) - 0.2,
                        fold_ids, np.zeros(k),
                        0.4, 0.7, scores,
                        color=bcolors[mi], alpha=0.8)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(["Log", "RF", "DT"], fontsize=7, color=COLORS["text_muted"])
            ax.set_xlabel("Modèle")
            ax.set_ylabel("Fold")
            ax.set_zlabel("Accuracy")
            ax.set_title("Performance 3D par Fold", fontsize=11)
            style_3d_ax(ax, self.accent)
            ax.set_zlim(0, 1.1)
            ax.view_init(elev=25, azim=45)
        else:
            ax = fig.add_subplot(111, facecolor=COLORS["bg_card"])
            bars = ax.bar(names, means, color=bcolors, alpha=0.85, edgecolor="none", width=0.6)
            ax.errorbar(names, means, yerr=stds, fmt="none", color="white", capsize=6, lw=2)
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                       f"{val:.3f}", ha="center", color=COLORS["text_white"], fontsize=9)
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("Exactitude Moyenne", color=COLORS["text_muted"])
            ax.set_title(f"Comparaison des Modèles ({k}-Fold CV)", color=self.accent, fontsize=13)
            ax.tick_params(colors=COLORS["text_muted"])
            for spine in ax.spines.values(): 
                spine.set_edgecolor(COLORS["border"])
        
        fig.tight_layout()
        embed_figure(self.right, fig, COLORS["bg_dark"])

class MLApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self._back_to_menu = False
        self.title(f"{APP_NAME} — Machine Learning Platform")
        self.geometry("1100x680")
        self.minsize(900, 600)
        self.configure(bg=COLORS["bg_dark"])
        self._build_header()
        self._build_tabs()

    def _back_to_start(self):
        self._back_to_menu = True
        self.destroy()

    def _build_header(self):
        hdr = tk.Frame(self, bg=COLORS["bg_panel"], height=60)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        left = tk.Frame(hdr, bg=COLORS["bg_panel"])
        left.pack(side="left", padx=20)
        tk.Label(
            left, text=APP_NAME,
            bg=COLORS["bg_panel"], fg=COLORS["text_white"],
            font=("Segoe UI", 13, "bold"),
        ).pack(anchor="w")
        tk.Label(
            left, text="AI & Machine Learning Platform",
            bg=COLORS["bg_panel"], fg=SPLASH_ACCENT,
            font=("Segoe UI", 9),
        ).pack(anchor="w")

        right = tk.Frame(hdr, bg=COLORS["bg_panel"])
        right.pack(side="right", padx=20)

        self._back_btn = tk.Button(
            right,
            text="\u25c4  Back",
            command=self._back_to_start,
            bg=COLORS["bg_card"], fg=SPLASH_ACCENT,
            font=("Segoe UI", 10, "bold"),
            relief="flat", bd=0, padx=14, pady=6,
            activebackground=COLORS["border"],
            activeforeground=SPLASH_ACCENT,
            highlightthickness=1,
            highlightbackground=SPLASH_ACCENT,
            cursor="hand2",
        )
        self._back_btn.pack(side="right", padx=(12, 0))
        self._back_btn.bind("<Enter>", lambda e: self._back_btn.config(bg=COLORS["border"]))
        self._back_btn.bind("<Leave>", lambda e: self._back_btn.config(bg=COLORS["bg_card"]))

        tk.Label(
            right, text="ETTAHIRI RAYAN — EMSI",
            bg=COLORS["bg_panel"], fg=COLORS["text_muted"],
            font=FONT_MONO,
        ).pack(side="right")
        tk.Frame(self, bg=COLORS["border"], height=1).pack(fill="x")
    
    def _build_tabs(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook", background=COLORS["bg_dark"], borderwidth=0)
        style.configure("TNotebook.Tab",
                       background=COLORS["bg_panel"],
                       foreground=COLORS["text_muted"],
                       font=("Courier New", 10, "bold"),
                       padding=[14, 8], borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", COLORS["bg_dark"])],
                 foreground=[("selected", COLORS["text_white"])])
        
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        
        tabs = [
            (" Régression", RegressionTab, COLORS["regression"]),
            (" Clustering", ClusteringTab, COLORS["clustering"]),
            (" Random Forest", ForestTab, COLORS["forest"]),
            (" Time Series", TimeSeriesTab, COLORS["timeseries"]),
            (" Neurones", NeuralTab, COLORS["neurones"]),
            (" Validation", ValidationTab, COLORS["validation"]),
        ]
        
        for label, TabClass, accent in tabs:
            frame = TabClass(nb, accent)
            nb.add(frame, text=label)

if __name__ == "__main__":
    while True:
        splash = SplashScreen()
        splash.mainloop()
        if not splash._launch:
            break
        app = MLApp()
        app.mainloop()
        if not app._back_to_menu:
            break