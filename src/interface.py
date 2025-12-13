import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import sys
import io
import os
import matplotlib.pyplot as plt
import control as ctrl
import sympy as sp

# Importa o seu módulo (o nome do arquivo deve ser root_locus_analyzer.py)
import root_locus_analyzer as backend

class RootLocusApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Root Locus Analyzer v1.0")
        self.root.geometry("1100x800")
        
        # Configuração de Estilo (Tema escuro básico para combinar com Arch/Hyprland)
        style = ttk.Style()
        style.theme_use('clam')
        
        # --- Layout Principal ---
        # Frame Esquerdo (Inputs e Controles)
        self.left_frame = ttk.Frame(root, padding="10")
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Frame Direito (Visualização e Logs)
        self.right_frame = ttk.Frame(root, padding="10")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Inputs ---
        ttk.Label(self.left_frame, text="Numerador (coeficientes):", font=('Arial', 10, 'bold')).pack(anchor='w', pady=(0, 5))
        self.num_entry = ttk.Entry(self.left_frame, width=30)
        self.num_entry.pack(fill=tk.X, pady=(0, 15))
        self.num_entry.insert(0, "1") # Valor default

        ttk.Label(self.left_frame, text="Denominador (coeficientes):", font=('Arial', 10, 'bold')).pack(anchor='w', pady=(0, 5))
        self.den_entry = ttk.Entry(self.left_frame, width=30)
        self.den_entry.pack(fill=tk.X, pady=(0, 15))
        self.den_entry.insert(0, "1 2 2") # Valor default

        # Opções
        self.verbose_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.left_frame, text="Gerar PDF Relatório", variable=self.verbose_var).pack(anchor='w', pady=5)

        # Botão Ação
        self.calc_btn = ttk.Button(self.left_frame, text="ANALISAR SISTEMA", command=self.run_analysis)
        self.calc_btn.pack(fill=tk.X, pady=20)

        # Log Console (Redirecionamento de stdout)
        ttk.Label(self.left_frame, text="Log de Saída:", font=('Arial', 10, 'bold')).pack(anchor='w')
        self.log_text = tk.Text(self.left_frame, height=20, width=40, font=('Consolas', 9), bg="#2E2E2E", fg="#DCDCCC")
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Área da Imagem (Plot)
        self.image_label = ttk.Label(self.right_frame, text="O gráfico aparecerá aqui")
        self.image_label.pack(fill=tk.BOTH, expand=True)

    def log(self, message):
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def run_analysis(self):
        # Limpar log
        self.log_text.delete(1.0, tk.END)
        
        # Capturar Inputs
        try:
            num_str = self.num_entry.get()
            den_str = self.den_entry.get()
            
            num = [float(x) for x in num_str.strip().split()]
            den = [float(x) for x in den_str.strip().split()]
        except ValueError:
            messagebox.showerror("Erro de Input", "Certifique-se de usar apenas números separados por espaço.")
            return

        # Redirecionar stdout para capturar os prints do seu script original
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            self.log("[Status] Iniciando análise...\n")
            
            # --- Lógica do seu Script Original ---
            # Recriando a lógica do main() do seu script, mas controlada pela GUI
            system = ctrl.TransferFunction(num, den)
            poles, zeros = ctrl.poles(system), ctrl.zeros(system)
            
            # Executa as análises
            asym_data = backend.analyze_asymptotes(poles, zeros)
            break_data = backend.analyze_breakaway_points(num, den)
            routh_data = backend.analyze_imaginary_crossing(num, den)
            angle_data = backend.analyze_departure_arrival_angles(poles, zeros)
            
            # Exibe o sumário (o output vai para o StringIO)
            backend.display_console_summary(system, poles, zeros, asym_data, break_data, routh_data, angle_data)
            
            # Gera o Plot
            output_dir = "gui_plots"
            os.makedirs(output_dir, exist_ok=True)
            
            # Gera o gráfico principal
            plot_file = backend.generate_root_locus_plot(system, poles, zeros, asym_data, output_dir=output_dir, show_only=False)
            
            # Mostra a imagem na GUI
            self.display_image(plot_file)

            # Gera PDF se solicitado
            if self.verbose_var.get():
                # Gera plots de ângulo auxiliares
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

                # Monta dados para o relatório
                s = sp.symbols('s')
                report_data = {
                    'tf_sympy': sp.Poly(num, s).as_expr() / sp.Poly(den, s).as_expr(),
                    'poles': poles, 'zeros': zeros,
                    'plot_filename': plot_file, 'asymptotes': asym_data,
                    'breakaway': break_data, 'routh_hurwitz': routh_data,
                    'angles': angle_data, 'angle_plot_files': angle_plot_files
                }
                
                backend.generate_report(report_data)
                self.log(f"\n[PDF] Relatório gerado com sucesso.")

        except Exception as e:
            sys.stdout = old_stdout
            messagebox.showerror("Erro Crítico", f"Ocorreu um erro durante a execução:\n{str(e)}")
            print(e)
            return

        # Recupera o texto capturado e coloca na GUI
        output_str = sys.stdout.getvalue()
        sys.stdout = old_stdout
        self.log(output_str)

    def display_image(self, path):
        # Carrega a imagem e redimensiona para caber na janela
        try:
            img = Image.open(path)
            
            # Lógica simples de redimensionamento mantendo aspect ratio
            display_width = self.right_frame.winfo_width()
            display_height = self.right_frame.winfo_height()
            
            if display_width < 10 or display_height < 10:
                display_width, display_height = 600, 500
                
            img.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
            
            self.photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=self.photo, text="")
        except Exception as e:
            self.log(f"\n[Erro] Não foi possível carregar a imagem: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = RootLocusApp(root)
    root.mainloop()
