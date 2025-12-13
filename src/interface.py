import customtkinter as ctk
import sys
import io
import os
import threading
import subprocess
import platform
import ctypes
import tkinter as tk

# Imports do Matplotlib para integração com Tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

# Importa o backend
import root_locus_analyzer as backend

# --- Configuração Global do Tema ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class RootLocusApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Janela Principal ---
        self.title("Root Locus Analyzer Pro")
        self.geometry("1100x800")

        # --- Configuração de Ícone ---
        icon_path_ico = resource_path(os.path.join("assets", "icon.ico"))
        icon_path_png = resource_path(os.path.join("assets", "icon.png"))

        if os.name == 'nt':
            try:
                myappid = 'luciomario.rootlocus.tool.1.4'
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
                self.iconbitmap(icon_path_ico)
            except Exception: pass
        else:
            try:
                img = tk.Image("photo", file=icon_path_png)
                self.tk.call('wm', 'iconphoto', self._w, img)
            except Exception: pass

        # --- Layout Principal ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ================= SIDEBAR =================
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Control Systems\nAnalyzer",
                                     font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Inputs
        self.num_label = ctk.CTkLabel(self.sidebar_frame, text="Numerator Coeffs:", anchor="w")
        self.num_label.grid(row=1, column=0, padx=20, pady=(10, 0), sticky="w")
        self.num_entry = ctk.CTkEntry(self.sidebar_frame, placeholder_text="e.g., 1")
        self.num_entry.grid(row=2, column=0, padx=20, pady=(0, 10), sticky="ew")
        self.num_entry.insert(0, "1")

        self.den_label = ctk.CTkLabel(self.sidebar_frame, text="Denominator Coeffs:", anchor="w")
        self.den_label.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")
        self.den_entry = ctk.CTkEntry(self.sidebar_frame, placeholder_text="e.g., 1 2 2")
        self.den_entry.grid(row=4, column=0, padx=20, pady=(0, 10), sticky="ew")
        self.den_entry.insert(0, "1 2 2")

        # Switch PDF
        self.pdf_switch = ctk.CTkSwitch(self.sidebar_frame, text="Generate PDF Report")
        self.pdf_switch.grid(row=5, column=0, padx=20, pady=20, sticky="n")
        self.pdf_switch.select()

        # Botão Open PDF
        self.open_pdf_btn = ctk.CTkButton(self.sidebar_frame, text="OPEN PDF REPORT",
                                          command=self.open_pdf,
                                          fg_color="transparent", border_width=2,
                                          text_color=("gray10", "#DCE4EE"),
                                          state="disabled")
        self.open_pdf_btn.grid(row=6, column=0, padx=20, pady=(0, 20), sticky="s")

        # Barra de Progresso
        self.progressbar = ctk.CTkProgressBar(self.sidebar_frame, mode="indeterminate")
        self.progressbar.grid(row=7, column=0, padx=20, pady=10, sticky="ew")
        self.progressbar.set(0)

        # Botão Principal
        self.calc_btn = ctk.CTkButton(self.sidebar_frame, text="RUN ANALYSIS",
                                      command=self.start_analysis_thread,
                                      height=40, font=ctk.CTkFont(weight="bold"))
        self.calc_btn.grid(row=8, column=0, padx=20, pady=30, sticky="ew")

        # ================= MAIN AREA =================
        self.tabview = ctk.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.tabview.add("Root Locus Plot")
        self.tabview.add("Terminal Log")

        # --- Configuração Aba Gráfico ---
        self.tabview.tab("Root Locus Plot").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Root Locus Plot").grid_rowconfigure(0, weight=1)

        # Frame Container para o Matplotlib (Importante para limpar depois)
        self.plot_frame = ctk.CTkFrame(self.tabview.tab("Root Locus Plot"), fg_color="transparent")
        self.plot_frame.grid(row=0, column=0, sticky="nsew")

        # Variáveis de controle
        self.canvas = None
        self.toolbar = None

        # --- Configuração Aba Log ---
        self.tabview.tab("Terminal Log").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Terminal Log").grid_rowconfigure(0, weight=1)
        self.log_text = ctk.CTkTextbox(self.tabview.tab("Terminal Log"), width=400, font=("Consolas", 12))
        self.log_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    def log(self, message):
        self.log_text.insert("end", message)
        self.log_text.see("end")

    def start_analysis_thread(self):
        self.calc_btn.configure(state="disabled", text="Processing...")
        self.open_pdf_btn.configure(state="disabled", fg_color="transparent")
        self.progressbar.start()
        self.log_text.delete("0.0", "end")

        # Limpa o gráfico anterior se existir
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        thread = threading.Thread(target=self.run_analysis)
        thread.start()

    def run_analysis(self):
        try:
            num_str = self.num_entry.get()
            den_str = self.den_entry.get()
            num = [float(x) for x in num_str.strip().split()]
            den = [float(x) for x in den_str.strip().split()]
        except ValueError:
            self.finish_analysis(error="Invalid Input: Use space-separated numbers.")
            return

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        fig = None
        pdf_generated = False

        try:
            print(f"[System] Starting analysis for Num: {num}, Den: {den}...\n")

            # --- Backend Calls ---
            system = backend.ctrl.TransferFunction(num, den)
            poles, zeros = backend.ctrl.poles(system), backend.ctrl.zeros(system)

            asym_data = backend.analyze_asymptotes(poles, zeros)
            break_data = backend.analyze_breakaway_points(num, den)
            routh_data = backend.analyze_imaginary_crossing(num, den)
            angle_data = backend.analyze_departure_arrival_angles(poles, zeros)

            backend.display_console_summary(system, poles, zeros, asym_data, break_data, routh_data, angle_data)

            output_dir = "gui_plots"
            os.makedirs(output_dir, exist_ok=True)

            # --- A CORREÇÃO PRINCIPAL ESTÁ AQUI ---
            # Recebemos a Figura (para a GUI) e o Nome do Arquivo (para o PDF)
            fig, plot_file = backend.generate_root_locus_plot(system, poles, zeros, asym_data, output_dir=output_dir, show_only=False)

            if self.pdf_switch.get() == 1:
                angle_plot_files = []
                poles_list, zeros_list = list(poles), list(zeros)
                if angle_data['departure']:
                    for pole, _ in angle_data['departure'].items():
                        idx = poles_list.index(pole) + 1
                        fname = backend.plot_angle_calculation(pole, poles, zeros, 'departure', idx, output_dir=output_dir)
                        angle_plot_files.append(fname)
                if angle_data['arrival']:
                    for zero, _ in angle_data['arrival'].items():
                        idx = zeros_list.index(zero) + 1
                        fname = backend.plot_angle_calculation(zero, poles, zeros, 'arrival', idx, output_dir=output_dir)
                        angle_plot_files.append(fname)

                s = backend.sp.symbols('s')
                report_data = {
                    'tf_sympy': backend.sp.Poly(num, s).as_expr() / backend.sp.Poly(den, s).as_expr(),
                    'poles': poles, 'zeros': zeros,
                    # Aqui passamos apenas o caminho do arquivo (string), não a tupla!
                    'plot_filename': plot_file,
                    'asymptotes': asym_data,
                    'breakaway': break_data, 'routh_hurwitz': routh_data,
                    'angles': angle_data, 'angle_plot_files': angle_plot_files
                }
                backend.generate_report(report_data)
                print(f"\n[PDF] Report generated successfully.")
                pdf_generated = True

        except Exception as e:
            sys.stdout = old_stdout
            self.finish_analysis(error=str(e))
            if fig: plt.close(fig)
            return

        output_str = sys.stdout.getvalue()
        sys.stdout = old_stdout

        self.finish_analysis(output_str=output_str, fig=fig, pdf_ready=pdf_generated)

    def finish_analysis(self, output_str=None, fig=None, error=None, pdf_ready=False):
        self.progressbar.stop()
        self.calc_btn.configure(state="normal", text="RUN ANALYSIS")

        if error:
            self.log(f"[ERROR] {error}")
            self.tabview.set("Terminal Log")
            return

        if output_str:
            self.log(output_str)

        if fig:
            self.draw_figure(fig)
            self.tabview.set("Root Locus Plot")

        if pdf_ready:
            self.open_pdf_btn.configure(state="normal", fg_color="#2CC985", text_color="white")

    def draw_figure(self, fig):
        """ Embutir Matplotlib Figure no Tkinter Frame com Toolbar """
        # Cria o Canvas
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()

        # Barra de Ferramentas (Zoom, Pan, Save)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()

        # Empacota na interface
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def open_pdf(self):
        pdf_path = "root_locus_analysis_report.pdf"
        if not os.path.exists(pdf_path):
            messagebox.showerror("Error", "PDF not found!")
            return
        try:
            if platform.system() == 'Darwin': subprocess.call(('open', pdf_path))
            elif platform.system() == 'Windows': os.startfile(pdf_path)
            else: subprocess.call(('xdg-open', pdf_path))
        except Exception as e:
            self.log(f"\n[Error] Could not open PDF: {e}")

if __name__ == "__main__":
    app = RootLocusApp()
    app.mainloop()
