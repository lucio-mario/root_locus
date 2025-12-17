import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
import sympy as sp
from pylatex import Document, Section, Math, Command, NoEscape, Subsection, Tabular, Figure, Center, Subsubsection, Itemize
from pylatex.utils import bold
import os
import tempfile
import warnings
import shutil

# Ignorar warnings do control
warnings.filterwarnings("ignore", category=FutureWarning, module='control')

def get_polynomial_input(name):
    prompt = f"Enter the coefficients for the {name} (space-separated):\n> "
    entry = input(prompt)
    return [float(num) for num in entry.strip().split()]

def analyze_asymptotes(poles, zeros):
    n, m = len(poles), len(zeros)
    q = n - m
    if q <= 0: return {'has_asymptotes': False}
    sigma = (np.sum(poles) - np.sum(zeros)) / q
    angles = [(2 * k + 1) * 180 / q for k in range(q)]
    return {'has_asymptotes': True, '#p': n, '#z': m, 'q': q, 'sum_poles': sum(p for p in poles),
            'sum_zeros': sum(z for z in zeros), 'sigma': sigma, 'angles': angles}

def analyze_breakaway_points(num, den):
    s = sp.symbols('s')
    N, D = sp.Poly(num, s).as_expr(), sp.Poly(den, s).as_expr()
    K_expr = -D / N
    dK_ds = sp.diff(K_expr, s)
    solutions = sp.solve(sp.numer(sp.simplify(dK_ds)), s)
    return {'K_expr': K_expr, 'dK_ds': dK_ds, 'solutions': solutions}

def analyze_imaginary_crossing(num, den):
    s, K = sp.symbols('s K')
    N_poly, D_poly = sp.Poly(num, s), sp.Poly(den, s)
    char_poly = D_poly + K * N_poly
    coeffs = char_poly.all_coeffs()
    if len(coeffs) < 3: return {'is_suitable': False}
    n_rows, n_cols = len(coeffs), (len(coeffs) + 1) // 2
    routh_array = sp.zeros(n_rows, n_cols)
    for i, c in enumerate(coeffs): routh_array[i % 2, i // 2] = c
    for i in range(2, n_rows):
        for j in range(n_cols - 1):
            if routh_array[i - 1, 0] == 0: continue
            num_term = routh_array[i-1,0] * routh_array[i-2,j+1] - routh_array[i-2,0] * routh_array[i-1,j+1]
            routh_array[i, j] = sp.simplify(num_term / routh_array[i-1, 0])
    s1_entry = routh_array[n_rows - 2, 0]
    k_sols = [k for k in sp.solve(s1_entry, K) if sp.re(k) >= 0 and sp.im(k) == 0]
    crossings, k_crit, aux_poly, aux_poly_sub = ([], None, None, None)
    if k_sols:
        k_crit = k_sols[0]
        s2_row = routh_array[n_rows - 3, :]
        aux_poly = s2_row[0] * s**2 + s2_row[1]
        aux_poly_sub = aux_poly.subs(K, k_crit)
        crossings = sp.solve(aux_poly_sub, s)
    return {'is_suitable': True, 'char_poly': char_poly, 'routh_array': routh_array, 's1_entry': s1_entry,
            'k_crit': k_crit, 'aux_poly': aux_poly, 'aux_poly_sub': aux_poly_sub, 'crossings': crossings}

def analyze_departure_arrival_angles(poles, zeros):
    poles = [complex(p) for p in poles]
    zeros = [complex(z) for z in zeros]
    departure_angles = {}
    arrival_angles = {}

    for p_focus in poles:
        if np.imag(p_focus) != 0:
            sum_pole_angles = 0
            sum_zero_angles = 0
            for p_other in poles:
                if p_focus != p_other:
                    sum_pole_angles += np.angle(p_focus - p_other, deg=True)
            for z in zeros:
                sum_zero_angles += np.angle(p_focus - z, deg=True)

            final_angle = (180 - (sum_pole_angles - sum_zero_angles)) % 360
            departure_angles[p_focus] = {
                'angle': final_angle,
                'sum_poles': sum_pole_angles,
                'sum_zeros': sum_zero_angles
            }

    for z_focus in zeros:
        if np.imag(z_focus) != 0:
            sum_pole_angles = 0
            sum_zero_angles = 0
            for z_other in zeros:
                if z_focus != z_other:
                    sum_zero_angles += np.angle(z_focus - z_other, deg=True)
            for p in poles:
                sum_pole_angles += np.angle(z_focus - p, deg=True)

            final_angle = (180 - (sum_zero_angles - sum_pole_angles)) % 360
            arrival_angles[z_focus] = {
                'angle': final_angle,
                'sum_poles': sum_pole_angles,
                'sum_zeros': sum_zero_angles
            }

    return {'departure': departure_angles, 'arrival': arrival_angles}

def plot_angle_calculation(focus_point, all_poles, all_zeros, point_type, index, output_dir='.'):
    fig, ax = plt.subplots(figsize=(12, 10))

    ax.scatter(np.real(all_poles), np.imag(all_poles), s=150, marker='x', c='r', linewidth=2.5, label='Other Poles')
    if all_zeros.any():
        ax.scatter(np.real(all_zeros), np.imag(all_zeros), s=150, marker='o', facecolors='none',
                   edgecolors='b', linewidths=2.5, label='Zeros')

    ax.scatter(np.real(focus_point), np.imag(focus_point), s=300, marker='*', c='g', label=f'Focus: {point_type[0].upper()}{index}')

    sum_pole_angles = 0
    sum_zero_angles = 0

    for p in all_poles:
        if p == focus_point:
            continue
        vec = focus_point - p
        angle_deg = np.angle(vec, deg=True)
        sum_pole_angles += angle_deg
        ax.arrow(np.real(p), np.imag(p), np.real(vec), np.imag(vec),
                 head_width=0.1, head_length=0.15, fc='r', ec='r', linestyle='--', length_includes_head=True)
        ax.text(np.real(p + vec / 2), np.imag(p + vec / 2) + 0.1, f'{angle_deg:.1f}°', color='red', fontsize=10)

    for z in all_zeros:
        vec = focus_point - z
        angle_deg = np.angle(vec, deg=True)
        sum_zero_angles += angle_deg
        ax.arrow(np.real(z), np.imag(z), np.real(vec), np.imag(vec),
                 head_width=0.1, head_length=0.15, fc='b', ec='b', linestyle='--', length_includes_head=True)
        ax.text(np.real(z + vec / 2), np.imag(z + vec / 2) - 0.1, f'{angle_deg:.1f}°', color='blue', fontsize=10)

    if point_type == 'departure':
        final_angle = (180 - (sum_pole_angles - sum_zero_angles)) % 360
        formula_str = f'$\\theta_{{p{index}}} = 180° - (\\sum \\theta_p - \\sum \\theta_z)$'
        result_str = f'$\\theta_{{p{index}}} = 180° - ({sum_pole_angles:.1f}° - {sum_zero_angles:.1f}°) = {final_angle:.2f}°$'
    else:
        final_angle = (180 - (sum_zero_angles - sum_pole_angles)) % 360
        formula_str = f'$\\theta_{{z{index}}} = 180° - (\\sum \\theta_z - \\sum \\theta_p)$'
        result_str = f'$\\theta_{{z{index}}} = 180° - ({sum_zero_angles:.1f}° - {sum_pole_angles:.1f}°) = {final_angle:.2f}°$'

    ax.set_title(f'Angle Calculation for {point_type.capitalize()} of {point_type[0].upper()}{index}', fontsize=16)
    fig.text(0.5, 0.02, formula_str, ha='center', fontsize=14, color='darkgreen')
    fig.text(0.5, -0.02, result_str, ha='center', fontsize=14, color='darkgreen')

    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, linestyle=':')
    ax.set_xlabel('Real Axis')
    ax.set_ylabel('Imaginary Axis')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    base_filename = f'angle_calc_{point_type}_{index}.png'
    plot_filename = os.path.join(output_dir, base_filename)
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"[Verbose Mode] Temporary angle plot saved to '{plot_filename}'")
    plt.close()
    return plot_filename.replace(os.sep, '/')

# --- NOVA IMPLEMENTAÇÃO ROBUSTA DE BUSCA DE ZETA ---
def analyze_target_zeta(system, target_zeta):
    """
    Encontra a interseção do Lugar das Raízes com a linha de amortecimento constante.
    Usa um vetor de ganho K de alta densidade para garantir que o polo DOMINANTE
    (mais próximo do eixo imaginário) seja encontrado corretamente.
    """
    # 1. Criar um vetor K denso:
    #    - Resolução fina para K pequeno (perto dos polos de malha aberta)
    #    - Escala logarítmica para K grande (para cobrir o resto do locus)
    k_dense = np.concatenate([
        np.linspace(0, 5, 5000),       # Alta precisão perto da origem
        np.logspace(0.7, 4, 5000)      # Cobertura ampla
    ])
    k_dense = np.unique(k_dense)

    # Gera locus com alta resolução
    rlist, klist = ctrl.root_locus(system, kvect=k_dense, plot=False)

    # Ângulo alvo no 2º quadrante (radianos)
    theta_rad_target = np.pi - np.arccos(target_zeta)

    roots_flat = rlist.flatten()
    gains_flat = np.repeat(klist, rlist.shape[1])

    # 2. Filtrar candidatos válidos
    #    - Semiplano superior (Im > 0)
    #    - Ganho não nulo (K > 0) para evitar o polo de malha aberta exato
    valid_mask = (np.imag(roots_flat) > 1e-6) & (gains_flat > 1e-9)

    candidates_s = roots_flat[valid_mask]
    candidates_k = gains_flat[valid_mask]

    if len(candidates_s) == 0:
        return None

    # 3. Calcular erro angular
    angles_s = np.angle(candidates_s)
    angle_errors = np.abs(angles_s - theta_rad_target)

    # 4. Encontrar pontos "na linha"
    #    Com K denso, podemos usar uma tolerância apertada (0.5 graus)
    tolerance_rad = np.deg2rad(0.5)

    on_line_indices = np.where(angle_errors < tolerance_rad)[0]

    if len(on_line_indices) == 0:
        # Fallback: pega o mais próximo absoluto se nenhum estiver na tolerância
        best_idx_local = np.argmin(angle_errors)
    else:
        # 5. Seleção de Dominância
        #    Temos um conjunto de pontos que intersectam a linha.
        #    Ordenamos pela Parte Real (Decrescente). Como são negativos,
        #    o maior valor (ex: -1 vs -5) é o mais próximo do eixo imaginário (dominante).

        possible_s = candidates_s[on_line_indices]
        sorted_indices_local = np.argsort(np.real(possible_s))[::-1]

        # Pega o primeiro da lista ordenada (o mais dominante)
        best_idx_in_subset = sorted_indices_local[0]
        best_idx_local = on_line_indices[best_idx_in_subset]

    best_s = candidates_s[best_idx_local]
    best_k = candidates_k[best_idx_local]

    actual_zeta = -np.real(best_s) / np.abs(best_s)
    theta_deg = np.degrees(np.arccos(target_zeta))

    return {
        'target_zeta': target_zeta,
        'found_k': best_k,
        'pole_location': best_s,
        'actual_zeta': actual_zeta,
        'theta_deg': theta_deg,
        'error': abs(target_zeta - actual_zeta)
    }

def generate_root_locus_plot(system, poles, zeros, asymp_data, zeta_data=None, output_dir='.', show_only=False):
    if not show_only:
        print("\n[Verbose Mode] Generating enhanced Root Locus plot...")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Usa um vetor denso também na plotagem para garantir que a curva passe pela estrela
    k_plot = np.concatenate([np.linspace(0, 10, 2000), np.logspace(1, 4, 3000)])
    k_plot = np.unique(k_plot)
    rlist, klist = ctrl.root_locus(system, kvect=k_plot, plot=False)

    if zeros.any():
        endpoints = rlist[:, -1]
        for i, endpoint in enumerate(endpoints):
            distances_to_zeros = np.abs(zeros - endpoint)
            if np.min(distances_to_zeros) < 0.5:
                closest_zero_index = np.argmin(distances_to_zeros)
                rlist[i, -1] = zeros[closest_zero_index]

    for i, branch in enumerate(rlist.T):
        label = 'Root Locus' if i == 0 else None
        ax.plot(np.real(branch), np.imag(branch), 'b-', alpha=0.8, linewidth=1.5, label=label)

        indices = [int(len(branch) * 0.4), int(len(branch) * 0.7)]
        for idx in indices:
            if idx < len(branch) - 1:
                p1, p2 = branch[idx], branch[idx+1]
                dx, dy = np.real(p2) - np.real(p1), np.imag(p2) - np.imag(p1)
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    ax.arrow(np.real(p1), np.imag(p1), dx, dy, color='k',
                             head_width=0.15, head_length=0.2, length_includes_head=True)

    ax.scatter(np.real(poles), np.imag(poles), s=150, marker='x', c='r', linewidth=2.5, label='Open-Loop Poles')
    if zeros.any():
        ax.scatter(np.real(zeros), np.imag(zeros), s=150, marker='o', facecolors='none',
                   edgecolors='b', linewidths=2.5, label='Open-Loop Zeros')

    if asymp_data['has_asymptotes']:
        sigma = asymp_data['sigma'].real
        angles_deg = asymp_data['angles']
        angles_rad = np.deg2rad(angles_deg)
        ax.scatter(sigma, 0, s=150, marker='p', c='g', label='Asymptote Centroid (σ)')

        lim_len = 100
        for i, angle in enumerate(angles_rad):
            label_to_use = 'Asymptotes' if i == 0 else None
            ax.plot([sigma, sigma + lim_len * np.cos(angle)], [0, lim_len * np.sin(angle)],
                    'g--', alpha=0.5, label=label_to_use)

    # --- ZETA LINE & DOMINANT POLE (Única linha e estrela) ---
    if zeta_data:
        s_dom = zeta_data['pole_location']

        # 1. Plota apenas o polo dominante encontrado
        ax.scatter(np.real(s_dom), np.imag(s_dom), s=250, marker='*', c='#FF00FF',
                   label=f'Dom. Pole (K={zeta_data["found_k"]:.2f})', zorder=10)

        # 2. Desenha raio da origem passando pelo polo
        angle_dom = np.angle(s_dom)
        ray_length = np.abs(s_dom) * 1.5
        if ray_length < 5: ray_length = 5

        ax.plot([0, ray_length * np.cos(angle_dom)], [0, ray_length * np.sin(angle_dom)],
                color='#FF00FF', linestyle='-.', alpha=0.8, linewidth=1.5,
                label=f'Zeta Line ($\zeta={zeta_data["target_zeta"]}$)')
    # ---------------------------------------------------------

    all_points = np.concatenate((poles, zeros))
    if zeta_data:
        all_points = np.append(all_points, zeta_data['pole_location'])

    x_min_points = np.min(np.real(all_points))
    x_max_points = np.max(np.real(all_points))
    y_min_points = np.min(np.imag(all_points))
    y_max_points = np.max(np.imag(all_points))

    x_padding = (x_max_points - x_min_points) * 0.40
    y_padding = (y_max_points - y_min_points) * 0.50

    if x_padding < 2.0: x_padding = 2.0
    if y_padding < 2.0: y_padding = 2.0

    ax.set_xlim([x_min_points - x_padding, x_max_points + x_padding])
    ax.set_ylim([y_min_points - y_padding, y_max_points + y_padding])

    ax.set_title('Root Locus Analysis', fontsize=16, fontweight='bold')
    ax.set_xlabel('Real Axis', fontsize=12)
    ax.set_ylabel('Imaginary Axis', fontsize=12)
    ax.grid(True, which='both', linestyle=':', linewidth=0.8)
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='best', fontsize='small')

    fig.tight_layout()

    if show_only:
        plt.show()
    else:
        plot_filename = os.path.join(output_dir, 'root_locus_plot.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"[Verbose Mode] Temporary plot saved to '{plot_filename}'")
        return fig, plot_filename.replace(os.sep, '/')

def display_console_summary(system, poles, zeros, asymp_data, break_data, routh_data, angle_data, zeta_data=None):
    def format_complex(c, precision=4):
        real_part = c.real
        imag_part = c.imag
        if abs(imag_part) < 1e-9:
            return f"{real_part:.{precision}f}"
        return f"{real_part:.{precision}f} {'+' if imag_part > 0 else '-'} {abs(imag_part):.{precision}f}j"

    print("\n" + "="*70)
    print("====== Quick Root Locus Analysis ======")
    print("="*70)

    print("\n[+] Open-Loop Transfer Function:")
    print(system)

    print("\n[+] Poles and Zeros:")
    poles_list = list(poles)
    zeros_list = list(zeros)
    print(f"  - Number of Poles (#p): {len(poles)}")
    for i, p in enumerate(poles):
        print(f"    - p{i+1}: {format_complex(p)}")

    print(f"\n  - Number of Zeros (#z): {len(zeros)}")
    if not zeros.any():
        print("    - No finite zeros.")
    else:
        for i, z in enumerate(zeros):
            print(f"    - z{i+1}: {format_complex(z)}")

    print(f"\n  - Number of Branches (#r): {len(poles)}")

    print("\n" + "-"*70)
    print("[+] Asymptote Analysis:")
    if not asymp_data['has_asymptotes']:
        print("  - No asymptotes needed (q <= 0).")
    else:
        print(f"  - Centroid (sigma): {asymp_data['sigma'].real:.4f}")
        print(f"  - Angles (theta):")
        for i, angle in enumerate(asymp_data['angles']):
            print(f"    - theta{i+1}: {angle:.2f}°")

    print("\n" + "-"*70)
    print("[+] Departure and Arrival Angles:")
    if not angle_data['departure']:
        print("  - No departure angles (no complex poles).")
    else:
        print("  - Departure Angles (from poles):")
        for p, angle_details in angle_data['departure'].items():
            pole_index = poles_list.index(p) + 1
            angle_value = angle_details['angle']
            print(f"    - theta_p{pole_index} (from pole {format_complex(p)}): {angle_value:.2f}°")

    if not angle_data['arrival']:
        print("\n  - No arrival angles (no complex zeros).")
    else:
        print("\n  - Arrival Angles (at zeros):")
        for z, angle_details in angle_data['arrival'].items():
            zero_index = zeros_list.index(z) + 1
            angle_value = angle_details['angle']
            print(f"    - theta_z{zero_index} (at zero {format_complex(z)}): {angle_value:.2f}°")

    print("\n" + "-"*70)
    print("[+] Breakaway / Break-in Points:")
    if not break_data['solutions']:
        print("  - No points found.")
    else:
        print("  - Candidate points (solutions for dK/ds = 0):")
        for s in break_data['solutions']:
            s_val = complex(s.evalf())
            print(f"    - s = {format_complex(s_val)}")

    print("\n" + "-"*70)
    print("[+] Imaginary Axis Crossing (Routh-Hurwitz):")
    if not routh_data['is_suitable'] or not routh_data.get('k_crit'):
        print("  - No crossing found on imaginary axis for K > 0.")
    else:
        k_crit = float(routh_data['k_crit'])
        print(f"  - Critical Gain (K_crit): {k_crit:.4f}")
        print("  - Crossing Points:")
        for s in routh_data['crossings']:
            s_val = complex(s.evalf())
            print(f"    - s = {format_complex(s_val)}")

    print("\n" + "="*70)
    print("[+] Stability Analysis (Routh-Hurwitz):")

    k_crit = routh_data.get('k_crit')

    if not routh_data['is_suitable']:
        print("  - System order < 3. Generally stable for K > 0 (check poles).")
    elif k_crit:
        k_val = float(k_crit)
        print(f"  - System is Stable for: 0 < K < {k_val:.4f}")
        print(f"  - Critical Gain (K_crit): {k_val:.4f}")
        print("  - Crossing Points (Oscillation Freq):")
        for s in routh_data['crossings']:
            s_val = complex(s.evalf())
            print(f"    - s = {format_complex(s_val)}")
    else:
        print("  - No sign change found in Routh array.")
        print("  - System is likely Stable for all K > 0 (Check if open-loop is stable).")

    if zeta_data:
        print("\n" + "-"*70)
        print(f"[+] Design Requirement (Zeta = {zeta_data['target_zeta']}):")
        print(f"  - Best Gain found: K = {zeta_data['found_k']:.4f}")
        print(f"  - Dominant Pole: s = {format_complex(zeta_data['pole_location'])}")
        print(f"  - Actual Zeta: {zeta_data['actual_zeta']:.4f}")

    print("\n" + "="*70)

def generate_report(data):
    print("[Verbose Mode] Assembling PDF report...")

    latex_options = {'documentclass': 'article', 'document_options': 'a4paper,11pt'}
    doc = Document(**latex_options)

    doc.preamble.append(Command('usepackage', 'amsmath'))
    doc.preamble.append(Command('usepackage', 'graphicx'))
    doc.preamble.append(Command('usepackage', 'geometry'))
    doc.preamble.append(Command('usepackage', 'booktabs'))
    doc.preamble.append(Command('usepackage', 'float')) # Necessário para position='H'
    doc.preamble.append(Command('title', 'Root Locus Analysis Report'))
    doc.preamble.append(Command('author', 'Automated Control System Analyzer'))
    doc.preamble.append(Command('date', NoEscape(r'\today')))
    doc.append(NoEscape(r'\maketitle'))

    def to_latex(expr, precision=4):
        if expr is None: return ""
        expr_eval = sp.N(expr)
        def clean_number(n):
            if not isinstance(n, (sp.Float, float)): return n
            val = float(n)
            rounded = round(val, precision)
            if abs(rounded - round(rounded)) < 1e-9:
                return sp.Integer(int(round(rounded)))
            return sp.Float(rounded, precision)
        if hasattr(expr_eval, '__iter__') and not isinstance(expr_eval, sp.Basic):
             return [to_latex(e, precision) for e in expr_eval]
        if hasattr(expr_eval, 'replace'):
            clean_expr = expr_eval.replace(lambda x: x.is_Float, lambda x: clean_number(x))
            return sp.latex(clean_expr)
        return sp.latex(expr_eval)

    with doc.create(Section('System Transfer Function')):
        doc.append(NoEscape(f"The analyzed open-loop transfer function is:"))
        doc.append(Math(data=f"G(s) = {to_latex(data['tf_sympy'])}", escape=False))

    with doc.create(Section('Poles and Zeros')):
        with doc.create(Itemize()) as item:
            item.add_item(NoEscape(rf"\#p (Poles): {len(data['poles'])}"))
            item.add_item(NoEscape(rf"\#z (Zeros): {len(data['zeros'])}"))
            item.add_item(NoEscape(rf"\#r (Branches): {len(data['poles'])}"))
        doc.append(Command('par'))
        doc.append(bold('System Poles:'))
        for i, p in enumerate(data['poles'], 1):
            doc.append(Math(data=f"p_{i} = {to_latex(p)}", escape=False))
        doc.append(Command('par'))
        doc.append(bold('System Zeros:'))
        if data['zeros'].any():
            for i, z in enumerate(data['zeros'], 1):
                doc.append(Math(data=f"z_{i} = {to_latex(z)}", escape=False))
        else:
            doc.append(' No finite zeros.')

    with doc.create(Section('Detailed Analysis')):
        with doc.create(Subsection('Asymptote Analysis')):
            asymp_data = data['asymptotes']
            if not asymp_data['has_asymptotes']:
                doc.append("No asymptotes are needed.")
            else:
                doc.append(NoEscape(rf"With $\#p={asymp_data['#p']}$ and $\#z={asymp_data['#z']}$, we have $q = \#p - \#z = {asymp_data['q']}$ asymptotes."))
                with doc.create(Center()):
                    doc.append(Math(data=r'\sigma_a = \frac{\sum p_i - \sum z_i}{\#p-\#z}', escape=False))
                    doc.append(NoEscape(f"$ \\rightarrow \\sigma_a = \\frac{{{to_latex(asymp_data['sum_poles'])} - ({to_latex(asymp_data['sum_zeros'])})}}{{{asymp_data['q']}}} = {to_latex(asymp_data['sigma'].real)} $"))
                doc.append(Command('par'))
                doc.append(NoEscape(r"The angles are given by $\theta_{a,k} = \frac{(2k+1)180^\circ}{q}$:"))
                for i, angle in enumerate(asymp_data['angles']):
                    doc.append(Math(data=rf"k={i}: \quad \theta_{{a,{i}}} = {angle:.2f}^\circ", escape=False))

        with doc.create(Subsection('Departure and Arrival Angles')):
            angle_data = data['angles']
            all_poles_list = list(data['poles'])
            all_zeros_list = list(data['zeros'])
            doc.append("These angles indicate the direction of the locus as it leaves a complex pole or arrives at a complex zero.")
            if angle_data['departure'] or angle_data['arrival']:
                 doc.append(NoEscape(r" (See \textbf{Appendix A} for detailed visualization plots)."))
            if angle_data['departure']:
                with doc.create(Subsubsection('Angles of Departure')):
                    for p_focus, angle_details in angle_data['departure'].items():
                        pole_index = all_poles_list.index(p_focus) + 1
                        angle = angle_details['angle']
                        doc.append(Math(data=rf"\theta_{{p{pole_index}}} = {angle:.4f}^\circ", escape=False))
            if angle_data['arrival']:
                with doc.create(Subsubsection('Angles of Arrival')):
                    for z_focus, angle_details in angle_data['arrival'].items():
                        zero_index = all_zeros_list.index(z_focus) + 1
                        angle = angle_details['angle']
                        doc.append(Math(data=rf"\theta_{{z{zero_index}}} = {angle:.4f}^\circ", escape=False))

        with doc.create(Subsection('Breakaway/Break-in Points')):
            break_data = data['breakaway']
            doc.append(NoEscape(r"These points are found by solving $\frac{dK}{ds} = 0$. This yields the following polynomial equation:"))
            numerator_poly = sp.numer(sp.simplify(break_data['dK_ds']))
            doc.append(Math(data=f"{to_latex(numerator_poly)} = 0", escape=False))
            doc.append(Command('par'))
            doc.append("The roots of this equation are the potential points:")
            for sol in break_data['solutions']:
                doc.append(Math(data=f"s = {to_latex(sol)}", escape=False))

        with doc.create(Subsection('Imaginary Axis Crossing')):
            routh_data = data['routh_hurwitz']
            if not routh_data['is_suitable']:
                 doc.append("System is 2nd order or less.")
            else:
                doc.append(NoEscape(r"The Routh-Hurwitz criterion is applied to the characteristic polynomial $1+KG(s)=0$."))
                doc.append(Math(data=f"{to_latex(routh_data['char_poly'].as_expr(), precision=4)} = 0", escape=False))
                with doc.create(Center()):
                    routh_array = routh_data['routh_array']
                    n_rows, n_cols = routh_array.shape
                    table = Tabular("c" + "c" * n_cols, booktabs=True)
                    table.add_row(["Order"] + [f"Col {j+1}" for j in range(n_cols)])
                    table.add_hline()
                    for i in range(n_rows):
                        row_data = [NoEscape(f'${to_latex(c, precision=4)}$') for c in routh_array.row(i)]
                        table.add_row([NoEscape(f"$s^{n_rows-1-i}$")] + row_data)
                    doc.append(table)

                if routh_data['k_crit']:
                    doc.append(Command('par'))
                    doc.append(NoEscape(r"The system becomes marginally stable when the $s^1$ row is zero:"))
                    doc.append(Math(data=f"{to_latex(routh_data['s1_entry'])} = 0 \\implies K_{{crit}} = {to_latex(routh_data['k_crit'])}", escape=False))
                    doc.append(Command('par'))
                    doc.append(NoEscape(r"The crossing points are found from the auxiliary polynomial ($s^2$ row) with $K = K_{crit}$:"))
                    doc.append(Math(data=f"A(s) = {to_latex(routh_data['aux_poly_sub'])} = 0", escape=False))
                    for p in routh_data['crossings']:
                        doc.append(Math(data=f"\\rightarrow s = {to_latex(p)}", escape=False))
                else:
                    doc.append(Command('par'))
                    doc.append("No positive critical gain K was found.")

        with doc.create(Subsection('Stability Analysis')):
            routh = data['routh_hurwitz']
            doc.append(NoEscape(r"\textbf{Condition:} For the system to be stable, all elements in the first column of the Routh array must be positive."))

            if routh.get('k_crit'):
                k_c = to_latex(routh['k_crit'])
                doc.append(Command('par'))
                doc.append(NoEscape(r"Analyzing the first column elements dependent on $K$:"))
                doc.append(Math(data=rf"{to_latex(routh['s1_entry'])} > 0 \implies K < {k_c}", escape=False))
                doc.append(bold("Stability Range:"))
                doc.append(Math(data=f"0 < K < {k_c}", escape=False))
                doc.append(NoEscape(r"For $K > K_{crit}$, the system becomes unstable (poles move to RHP)."))
            elif routh['is_suitable']:
                 doc.append(" The Routh array indicates no sign changes in the first column for K > 0. The system is stable for all K > 0.")

        if data.get('zeta_analysis'):
            z_data = data['zeta_analysis']
            with doc.create(Subsection(NoEscape(rf"Design Point: Damping Ratio $\zeta = {z_data['target_zeta']}$"))):
                doc.append("To achieve the desired damping ratio, we first calculate the required angle of the pole:")
                doc.append(Math(data=rf"\theta_\zeta = \cos^{{-1}}(\zeta) = \cos^{{-1}}({z_data['target_zeta']}) = {z_data['theta_deg']:.2f}^\circ", escape=False))
                doc.append("The Root Locus was searched for the intersection with the line corresponding to this angle.")
                with doc.create(Itemize()) as item:
                    item.add_item(NoEscape(rf"\textbf{{Required Gain:}} $K = {z_data['found_k']:.4f}$"))
                    item.add_item(NoEscape(rf"\textbf{{Closed-Loop Pole:}} $s = {to_latex(complex(z_data['pole_location']))}$"))
                    item.add_item(NoEscape(rf"\textbf{{Actual $\zeta$:}} {z_data['actual_zeta']:.4f}"))

    with doc.create(Section('Root Locus Plot')):
        with doc.create(Figure(position='H')) as plot:
            plot.append(Command('centering'))
            plot.append(Command('includegraphics',
                                options=NoEscape(r'width=0.85\textwidth,height=0.85\textheight,keepaspectratio'),
                                arguments=NoEscape(data['plot_filename'])))
            plot.add_caption('Complete Root Locus plot. The dominant pole is marked with a star.')

    if data.get('angle_plot_files'):
        doc.append(Command('clearpage'))
        with doc.create(Section('Appendix A: Angle Calculation Visualizations')):
            doc.append("The following plots detail the calculation for each departure and arrival angle.")
            for filename in data['angle_plot_files']:
                doc.append(Command('clearpage'))
                with doc.create(Figure(position='h!')) as fig:
                    fig.append(Command('centering'))
                    parts = os.path.basename(filename).replace('.png', '').split('_')
                    try:
                        point_type_str = "departure" if "departure" in parts else "arrival"
                        point_class_str = "pole" if "departure" in parts else "zero"
                        point_index = parts[-1]
                        caption_text = f"Visualization of the angle calculation for the {point_type_str} of {point_class_str} {point_class_str[0]}{point_index}."
                    except:
                        caption_text = "Angle calculation geometry."

                    fig.append(Command('includegraphics',
                                       options=NoEscape(r'width=0.85\textwidth,height=0.85\textheight,keepaspectratio'),
                                       arguments=NoEscape(filename)))
                    fig.add_caption(NoEscape(caption_text))

    try:
        final_pdf_filename = 'root_locus_analysis_report'
        doc.generate_pdf(final_pdf_filename, clean_tex=True, compiler='pdflatex')
        print(f"\n[Verbose Mode] Report '{final_pdf_filename}.pdf' generated successfully.")
    except Exception as e:
        print(f"\n--- LaTeX Compilation Error ---\nCould not generate PDF. Error: {e}")
        print("Please ensure you have a LaTeX distribution (like MiKTeX, TeX Live) installed and in your system's PATH.")
        raise e
