import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import qrcode
import webbrowser

# --- CONFIGURATION ---
BASE_URL = "https://sign2sound.vercel.app"

# --- THEME CONFIGURATION (Studio Dark) ---
THEME = {
    "BG": "#050505",       # --bg
    "SURFACE": "#0E0E0E",  # --surface
    "BORDER": "#262626",   # --border
    "TEXT": "#FFFFFF",     # --text-main
    "MUTED": "#888888",    # --text-muted
    "ACCENT": "#FF3B30",   # --accent-red
    "SUCCESS": "#22C55E"   # --focus
}

def open_browser(url):
    webbrowser.open(url)

def launch_broadcast_window(session_id):
    """
    Launches the Tkinter Broadcast Connection Window.
    """
    join_url = f"{BASE_URL}/?join={session_id}"
    
    root = tk.Tk()
    root.title("Sign2Sound Broadcast")
    
    # COMPACT SIZE
    root.geometry("320x540") 
    root.configure(bg=THEME["BG"])
    root.resizable(False, False)
    
    # Main Container
    container = tk.Frame(root, bg=THEME["BG"])
    container.pack(expand=True, fill="both", padx=20, pady=20)

    # 1. Status Indicator
    header_frame = tk.Frame(container, bg=THEME["BG"])
    header_frame.pack(fill="x", pady=(0, 15))
    
    dot_canvas = tk.Canvas(header_frame, width=8, height=8, bg=THEME["BG"], highlightthickness=0)
    dot_canvas.create_oval(0, 0, 8, 8, fill=THEME["ACCENT"], outline="")
    dot_canvas.pack(side="left", padx=(0, 6))
    
    tk.Label(
        header_frame, 
        text="LIVE BROADCAST", 
        font=("Arial", 9, "bold"),
        bg=THEME["BG"], 
        fg=THEME["ACCENT"]
    ).pack(side="left")

    # 2. The QR Card
    card_border = tk.Frame(container, bg=THEME["BORDER"], padx=1, pady=1)
    card_border.pack(fill="x", pady=5)
    
    card_surface = tk.Frame(card_border, bg=THEME["SURFACE"], padx=15, pady=25)
    card_surface.pack(fill="both", expand=True)

    # QR Code
    qr = qrcode.QRCode(box_size=6, border=2)
    qr.add_data(join_url)
    qr.make(fit=True)
    qr_img_pil = qr.make_image(fill_color="black", back_color="white")
    qr_img_pil = qr_img_pil.resize((180, 180), Image.LANCZOS)
    qr_photo = ImageTk.PhotoImage(qr_img_pil)

    qr_label = tk.Label(card_surface, image=qr_photo, bg=THEME["SURFACE"])
    qr_label.image = qr_photo 
    qr_label.pack(pady=(0, 15))

    # Session ID
    tk.Label(
        card_surface, 
        text="SESSION ID", 
        font=("Arial", 8, "bold"),
        bg=THEME["SURFACE"], 
        fg=THEME["MUTED"]
    ).pack(pady=(0, 2))
    
    id_label = tk.Label(
        card_surface, 
        text=session_id, 
        font=("Consolas", 22, "bold"),
        bg=THEME["SURFACE"], 
        fg=THEME["TEXT"]
    )
    id_label.pack(pady=(0, 8))
    
    # 3. BRANDING FOOTER (The requested change)
    footer_frame = tk.Frame(container, bg=THEME["BG"])
    footer_frame.pack(side="bottom", pady=10)

    # "Sign2Sound" Logo Text
    tk.Label(
        footer_frame, 
        text="Sign2Sound", 
        font=("Arial", 24, "bold"), # Bigger font for brand
        bg=THEME["BG"], 
        fg=THEME["TEXT"]
    ).pack()

    # "Studio" Subtitle
    tk.Label(
        footer_frame, 
        text="STUDIO BROADCAST", 
        font=("Arial", 10, "bold"),
        bg=THEME["BG"], 
        fg=THEME["MUTED"],
        justify="center"
    ).pack(pady=(2, 0))

    root.mainloop()

if __name__ == "__main__":
    launch_broadcast_window("TEST-123")