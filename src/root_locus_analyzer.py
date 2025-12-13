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

def generate_root_locus_plot(system, poles, zeros, asymp_data, output_dir='.', show_only=False):
    if not show_only:
        print("\n[Verbose Mode] Generating enhanced Root Locus plot...")

    fig, ax = plt.subplots(figsize=(12, 9))

    k_vector = np.linspace(0, 500, num=5000)
    rlist, klist = ctrl.root_locus(system, plot=False)

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
                ax.arrow(np.real(p1), np.imag(p1), dx, dy, color='k',
                         head_width=0.15, head_length=0.2, length_includes_head=True)

    ax.scatter(np.real(poles), np.imag(poles), s=150, marker='x', c='r', linewidth=2.5, label='Poles')
    if zeros.any():
        ax.scatter(np.real(zeros), np.imag(zeros), s=150, marker='o', facecolors='none',
                   edgecolors='b', linewidths=2.5, label='Zeros')

    if asymp_data['has_asymptotes']:
        sigma = asymp_data['sigma'].real
        angles_deg = asymp_data['angles']
        angles_rad = np.deg2rad(angles_deg)
        ax.scatter(sigma, 0, s=150, marker='p', c='g', label='Asymptote Centroid (σ)')
        angle_str = ", ".join([f"{angle:.1f}°" for angle in angles_deg])
        asymptote_label = f'Asymptotes ({angle_str})'
        lim = 100
        for i, angle in enumerate(angles_rad):
            label_to_use = asymptote_label if i == 0 else None
            ax.plot([sigma, sigma + lim * np.cos(angle)], [0, lim * np.sin(angle)],
                    'g--', label=label_to_use)

    all_points = np.concatenate((poles, zeros))
    x_min_points = np.min(np.real(all_points))
    x_max_points = np.max(np.real(all_points))
    y_min_points = np.min(np.imag(all_points))
    y_max_points = np.max(np.imag(all_points))

    x_padding = (x_max_points - x_min_points) * 0.30
    y_padding = (y_max_points - y_min_points) * 0.40

    if x_padding < 1.5: x_padding = 1.5
    if y_padding < 2.5: y_padding = 2.5

    ax.set_xlim([x_min_points - x_padding, x_max_points + x_padding])
    ax.set_ylim([y_min_points - y_padding, y_max_points + y_padding])

    ax.set_title('Root Locus Analysis', fontsize=18)
    ax.set_xlabel('Real Axis', fontsize=14)
    ax.set_ylabel('Imaginary Axis', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.7)
    ax.axvline(0, color='black', linewidth=0.7)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', which='major', labelsize=12)
    x_lims = ax.get_xlim()
    ax.set_xticks(np.arange(int(np.floor(x_lims[0])), int(np.ceil(x_lims[1])), 1))
    ax.legend()
    fig.tight_layout()

    if show_only:
        plt.show()
    else:
        plot_filename = os.path.join(output_dir, 'root_locus_plot.png')
        plt.savefig(plot_filename, dpi=300)
        print(f"[Verbose Mode] Temporary plot saved to '{plot_filename}'")
        return fig, plot_filename.replace(os.sep, '/')

def display_console_summary(system, poles, zeros, asymp_data, break_data, routh_data, angle_data):
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

def generate_report(data):
    print("[Verbose Mode] Assembling PDF report...")

    latex_options = {'documentclass': 'article', 'document_options': 'a4paper,11pt'}
    doc = Document(**latex_options)

    doc.preamble.append(Command('usepackage', 'amsmath'))
    doc.preamble.append(Command('usepackage', 'graphicx'))
    doc.preamble.append(Command('usepackage', 'geometry'))
    doc.preamble.append(Command('usepackage', 'booktabs'))
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
            clean_expr = expr_eval.replace(
                lambda x: x.is_Float,
                lambda x: clean_number(x)
            )
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
                        formula_str = rf"\theta_{{p{pole_index}}} = {angle:.4f}^\circ"
                        doc.append(Math(data=formula_str, escape=False))

            if angle_data['arrival']:
                with doc.create(Subsubsection('Angles of Arrival')):
                    for z_focus, angle_details in angle_data['arrival'].items():
                        zero_index = all_zeros_list.index(z_focus) + 1
                        angle = angle_details['angle']
                        formula_str = rf"\theta_{{z{zero_index}}} = {angle:.4f}^\circ"
                        doc.append(Math(data=formula_str, escape=False))

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

    with doc.create(Section('Root Locus Plot')):
        with doc.create(Figure(position='h!')) as plot:
            plot.add_image(data['plot_filename'], width=NoEscape(r'1\textwidth'))
            plot.add_caption('Complete Root Locus plot including poles, zeros, and asymptotes.')

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

                    fig.add_image(filename, width=NoEscape(r'0.75\textwidth'))
                    fig.add_caption(NoEscape(caption_text))

    try:
        final_pdf_filename = 'root_locus_analysis_report'
        doc.generate_pdf(final_pdf_filename, clean_tex=True, compiler='pdflatex')
        print(f"\n[Verbose Mode] Report '{final_pdf_filename}.pdf' generated successfully.")
    except Exception as e:
        print(f"\n--- LaTeX Compilation Error ---\nCould not generate PDF. Error: {e}")
        print("Please ensure you have a LaTeX distribution (like MiKTeX, TeX Live) installed and in your system's PATH.")
        raise e

def main():
    parser = argparse.ArgumentParser(description="Perform a complete Root Locus analysis.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Generate a detailed PDF report with all plots embedded.")
    parser.add_argument('-k', '--keep', action='store_true', help="Keep generated plot files in a 'plots' directory.")
    args = parser.parse_args()

    output_dir_name = 'plots'
    if os.path.exists(output_dir_name):
        print(f"[Info] Cleaning up previous '{output_dir_name}' directory...")
        shutil.rmtree(output_dir_name)

    pdf_filename = 'root_locus_analysis_report.pdf'
    if os.path.exists(pdf_filename):
        print(f"[Info] Cleaning up previous report file '{pdf_filename}'...")
        os.remove(pdf_filename)

    print("====== Root Locus Analysis ======")
    num = get_polynomial_input("numerator")
    den = get_polynomial_input("denominator")
    system = ctrl.TransferFunction(num, den)
    poles, zeros = ctrl.poles(system), ctrl.zeros(system)

    poles_list = list(poles)
    zeros_list = list(zeros)

    asymptotes_data = analyze_asymptotes(poles, zeros)
    breakaway_data = analyze_breakaway_points(num, den)
    routh_data = analyze_imaginary_crossing(num, den)
    angle_data = analyze_departure_arrival_angles(poles, zeros)

    if args.verbose:
        if args.keep:
            output_dir = output_dir_name
            os.makedirs(output_dir, exist_ok=True)
            print(f"\n[Verbose Mode] Plots will be saved in '{output_dir}'.")

            fig, main_plot_filename = generate_root_locus_plot(system, poles, zeros, asymptotes_data, output_dir=output_dir, show_only=False)
            plt.close(fig)

            print("\n[Verbose Mode] Generating angle calculation plots...")
            angle_plot_files = []
            if angle_data['departure']:
                for pole, details in angle_data['departure'].items():
                    pole_index = poles_list.index(pole) + 1
                    filename = plot_angle_calculation(pole, poles, zeros, 'departure', pole_index, output_dir=output_dir)
                    angle_plot_files.append(filename)
            if angle_data['arrival']:
                for zero, details in angle_data['arrival'].items():
                    zero_index = zeros_list.index(zero) + 1
                    filename = plot_angle_calculation(zero, poles, zeros, 'arrival', zero_index, output_dir=output_dir)
                    angle_plot_files.append(filename)

            s = sp.symbols('s')
            report_data = {
                'tf_sympy': sp.Poly(num, s).as_expr() / sp.Poly(den, s).as_expr(), 'poles': poles, 'zeros': zeros,
                'plot_filename': main_plot_filename, 'asymptotes': asymptotes_data, 'breakaway': breakaway_data,
                'routh_hurwitz': routh_data, 'angles': angle_data, 'angle_plot_files': angle_plot_files
            }
            generate_report(report_data)
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"\n[Verbose Mode] Using temporary directory: {temp_dir}")
                fig, main_plot_filename = generate_root_locus_plot(system, poles, zeros, asymptotes_data, output_dir=temp_dir)
                plt.close(fig)

                print("\n[Verbose Mode] Generating angle calculation plots...")
                angle_plot_files = []
                if angle_data['departure']:
                    for pole, details in angle_data['departure'].items():
                        pole_index = poles_list.index(pole) + 1
                        filename = plot_angle_calculation(pole, poles, zeros, 'departure', pole_index, output_dir=temp_dir)
                        angle_plot_files.append(filename)
                if angle_data['arrival']:
                    for zero, details in angle_data['arrival'].items():
                        zero_index = zeros_list.index(zero) + 1
                        filename = plot_angle_calculation(zero, poles, zeros, 'arrival', zero_index, output_dir=temp_dir)
                        angle_plot_files.append(filename)

                s = sp.symbols('s')
                report_data = {
                    'tf_sympy': sp.Poly(num, s).as_expr() / sp.Poly(den, s).as_expr(), 'poles': poles, 'zeros': zeros,
                    'plot_filename': main_plot_filename, 'asymptotes': asymptotes_data, 'breakaway': breakaway_data,
                    'routh_hurwitz': routh_data, 'angles': angle_data, 'angle_plot_files': angle_plot_files
                }
                generate_report(report_data)

    elif args.keep:
        output_dir = output_dir_name
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n[Keep Mode] Saving plots to '{output_dir}'...")

        fig, _ = generate_root_locus_plot(system, poles, zeros, asymptotes_data, output_dir=output_dir, show_only=False)
        plt.close(fig)

        if angle_data['departure']:
            for pole, details in angle_data['departure'].items():
                pole_index = poles_list.index(pole) + 1
                plot_angle_calculation(pole, poles, zeros, 'departure', pole_index, output_dir=output_dir)
        if angle_data['arrival']:
            for zero, details in angle_data['arrival'].items():
                zero_index = zeros_list.index(zero) + 1
                plot_angle_calculation(zero, poles, zeros, 'arrival', zero_index, output_dir=output_dir)

        display_console_summary(system, poles, zeros, asymptotes_data, breakaway_data, routh_data, angle_data)
        print(f"\nPlot files have been saved in the '{output_dir_name}' directory.")

    else:
        display_console_summary(system, poles, zeros, asymptotes_data, breakaway_data, routh_data, angle_data)
        print("\nShowing Root Locus graph in a new window...")
        generate_root_locus_plot(system, poles, zeros, asymptotes_data, show_only=True)

if __name__ == "__main__":
    main()
