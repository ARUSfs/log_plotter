import argparse
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, StringVar, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import re
import sys
import array
import math
import os



ROS_ENABLED = False 
try:
    from sensor_msgs_py import point_cloud2 
    ROS_ENABLED = True
except ImportError:
    
    pass 
except Exception:
    pass

formula_counter = 0
check_vars = {}

def read_ros_messages(input_bag: str):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=input_bag, storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )

    topic_types = reader.get_all_topics_and_types()

    def typename(topic_name):
        for topic_type in topic_types:
            if topic_type.name == topic_name:
                return topic_type.type
        raise ValueError(f"topic {topic_name} not in bag")

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        msg_type = get_message(typename(topic))
        msg = deserialize_message(data, msg_type)
        yield topic, msg, timestamp, msg_type
    del reader


def to_safe_identifier(name):
    return "_" + re.sub(r"[^a-zA-Z0-9_]", "_", name)


def get_safe_name(col_name):
    
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', col_name)
    if clean and clean[0].isdigit():
        clean = "Var" + clean
    return clean








global safe_map
global expr_entry
global name_entry
global primary_plot_config
global AGGREGATE_FUNCTIONS
global scalar_expr_entry 

formula_counter = 0
check_vars = {}
safe_map = {}

AGGREGATE_FUNCTIONS = {
    "mean": np.mean, "sum": np.sum, "std": np.std, "min": np.min, "max": np.max,
    "abs_mean": lambda x: np.mean(np.abs(x)),
}

def get_safe_name(s):
    """Limpia el nombre de variables para usarlas como claves de diccionario."""
    return s.replace('.', '_').replace('[', '_').replace(']', '_').replace('/', '_')

def get_eval_context(df_local):
    """Prepara el contexto de ejecución para las fórmulas."""
    local_vars = {get_safe_name(real_name): df_local[real_name] for safe_name, real_name in safe_map.items()}
    math_funcs = {"np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan,
                  "sqrt": np.sqrt, "abs": np.abs, "log": np.log, "exp": np.exp}
    return {"__builtins__": {}, **local_vars, **math_funcs}

def insert_text(text):
    """Inserta el nombre de la variable en el Entry del Formula Builder activo."""
    if 'expr_entry' in globals() and expr_entry.focus_get() == expr_entry:
         target = expr_entry
    elif 'scalar_expr_entry' in globals() and scalar_expr_entry.focus_get() == scalar_expr_entry:
         target = scalar_expr_entry
    else:
         return
         
    idx = target.index(tk.INSERT)
    target.insert(idx, text)
    target.focus()



def calculate_scalar(df_local, scalar_expr_entry_local, scalar_func_var_local, scalar_result_lbl_local):
    """Calcula un valor escalar (estadística de resumen) a partir de una fórmula."""
    scalar_expr = scalar_expr_entry_local.get().strip()
    agg_func_name = scalar_func_var_local.get().strip()
    
    if not scalar_expr or agg_func_name == "Select Function":
        messagebox.showwarning("Advertencia", "Debe introducir una expresión y seleccionar una función.")
        return

    try:
        context = get_eval_context(df_local)
        result_series = eval(scalar_expr, {"__builtins__": {}}, context)
        
        if not isinstance(result_series, (np.ndarray, pd.Series, list)):
            messagebox.showerror("Error", "La expresión no resultó en un vector de datos.")
            return
        
        agg_func = AGGREGATE_FUNCTIONS[agg_func_name]
        scalar_result = agg_func(result_series)
        
        result_text = f"Resultado de {agg_func_name}({scalar_expr}):\n{scalar_result:.6f}"
        scalar_result_lbl_local.config(text=result_text)

    except NameError as e:
        messagebox.showerror("Error Matemático", f"Variable o función desconocida:\n{e}")
    except Exception as e:
        messagebox.showerror("Error de Cálculo", f"No se pudo calcular el escalar.\nDetalle: {e}")



def draw_plot(df_local, fig, canvas, config):
    """
    Dibuja un único gráfico (1x1) en la figura (fig) usando la configuración (config).
    """
    
    fig.clear()
    ax = fig.add_subplot(1, 1, 1) 

    
    x_col = config['x_axis_combobox'].get()
    if x_col not in df_local.columns:
        x_data = df_local["timestamp"]; x_lbl_data = "Time (s)"
    else:
        x_data = df_local[x_col]; x_lbl_data = x_col
        
    
    variables_to_plot = [col_name for col_name, var_status in check_vars.items() 
                         if var_status.get() and col_name in df_local.columns]

    if variables_to_plot:
            
        y_col_mode = config['y_axis_mode_var'].get()
        
        if y_col_mode == "Scatter (X vs. Single Y)" and len(variables_to_plot) == 1:
            y_col = variables_to_plot[0]
            ax.plot(x_data, df_local[y_col], label=y_col, alpha=0.8, linewidth=1.5)
            y_lbl_default = y_col
        else:
            for col in variables_to_plot:
                ax.plot(x_data, df_local[col], label=col, alpha=0.8, linewidth=1.5)
            y_lbl_default = "Multiple Variables"

        ax.legend(loc='upper right', fontsize='small', framealpha=0.9)
        ax.set_aspect('auto')

        
        ax.set_title(config['title_entry'].get() if config['title_entry'].get() else "Time Series Plot")
        ax.set_xlabel(config['xlabel_entry'].get() if config['xlabel_entry'].get() else x_lbl_data)
        ax.set_ylabel(config['ylabel_entry'].get() if config['ylabel_entry'].get() else y_lbl_default)
    
    
    fig.tight_layout()
    canvas.draw()
    
def open_new_plotter_window(df_local, config):
    """
    Abre una nueva ventana Toplevel con el gráfico actual.
    """
    
    
    new_window = tk.Toplevel(config['root'])
    title_text = config['title_entry'].get() if config['title_entry'].get() else f"Gráfico Adicional - {len(config['root'].winfo_children()) - 1}"
    new_window.title(title_text)
    new_window.geometry("800x600")

    plot_frame_new = tk.Frame(new_window)
    plot_frame_new.pack(fill="both", expand=True, padx=5, pady=5)
    
    fig_new = plt.figure(figsize=(8, 6))
    canvas_new = FigureCanvasTkAgg(fig_new, master=plot_frame_new) 
    canvas_new.draw()
    
    toolbar_new = NavigationToolbar2Tk(canvas_new, plot_frame_new)
    toolbar_new.update()
    canvas_new.get_tk_widget().pack(fill="both", expand=True)

    
    draw_plot(df_local, fig_new, canvas_new, config)



def plot_variables(df, last_snapshots={}):
    """
    Inicializa la aplicación Tkinter y la interfaz de control principal.
    """
    global safe_map
    global check_vars
    global primary_plot_config 
    global expr_entry
    global name_entry
    global scalar_expr_entry
    
    
    safe_map = {get_safe_name(c): c for c in df.columns}
    check_vars = {}
    
    root = tk.Tk()
    root.title("Log Plotter - Multi-Window Interface (Simple)")

    
    main_frame = tk.Frame(root)
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    main_frame.grid_columnconfigure(0, weight=0); main_frame.grid_columnconfigure(1, weight=1)
    main_frame.grid_rowconfigure(0, weight=1)
    main_frame.grid_rowconfigure(1, weight=0) 

    
    controls_canvas = tk.Canvas(main_frame)
    scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=controls_canvas.yview)
    scrollable_frame = tk.Frame(controls_canvas)

    def adjust_scrollregion(e):
        controls_canvas.configure(scrollregion=controls_canvas.bbox("all"))
        controls_canvas.config(width=scrollable_frame.winfo_reqwidth())
    scrollable_frame.bind("<Configure>", adjust_scrollregion)
    controls_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    controls_canvas.configure(yscrollcommand=scrollbar.set)
    controls_canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
    scrollbar.grid(row=0, column=0, sticky="nse")

    
    def _on_mousewheel(event): controls_canvas.yview_scroll(int(-1 * (event.delta/120)), "units")
    def _on_linux_up(event): controls_canvas.yview_scroll(-1, "units")
    def _on_linux_down(event): controls_canvas.yview_scroll(1, "units")
    root.bind_all("<MouseWheel>", _on_mousewheel)
    root.bind_all("<Button-4>", _on_linux_up)
    root.bind_all("<Button-5>", _on_linux_down)
    
    
    plot_frame = tk.Frame(main_frame)
    plot_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
    plot_frame.grid_rowconfigure(0, weight=1); plot_frame.grid_columnconfigure(0, weight=1)

    fig = plt.figure(figsize=(10, 8))
    canvas = FigureCanvasTkAgg(fig, master=plot_frame) 
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, plot_frame)
    toolbar.update()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    
    x_axis_var = tk.StringVar(value="timestamp")
    y_axis_options = ["Time Series (Multiple Y vs. X)", "Scatter (X vs. Single Y)"]
    y_axis_mode_var = tk.StringVar(value=y_axis_options[0])
    
    
    def update_plot():
        draw_plot(df, fig, canvas, primary_plot_config)
    
    def add_expression(custom_name):
        global formula_counter
        global safe_map
        expr = expr_entry.get().strip()
        if not expr: return
        try:
            context = get_eval_context(df)
            result_series = eval(expr, {"__builtins__": {}}, context)
            display_name = custom_name.strip() if custom_name.strip() else f"Formula_{formula_counter+1}({expr})"
            df[display_name] = result_series
            safe_map[get_safe_name(display_name)] = display_name
            add_checkbox_to_group("User Formulas", display_name)
            expr_entry.delete(0, tk.END); name_entry.delete(0, tk.END)
            formula_counter += 1
            update_plot()
        except Exception as e:
            messagebox.showerror("Error Matemático", f"No se pudo calcular la serie.\nDetalle: {e}")
            
    
    groups = {}

    def add_checkbox_to_group(grp_name, col_name):
        nonlocal groups 
        
        
        if grp_name not in groups:
            gf = tk.LabelFrame(scrollable_frame, text=grp_name, bd=1, relief="solid", bg="#Ecf0f1", padx=0, pady=0)
            vf = tk.Frame(gf, bg="white")
            groups[grp_name] = {'frame': vf, 'labelframe': gf}
            
            
            btn_frame = tk.Frame(gf, bg="#Ecf0f1"); btn_frame.pack(side="top", fill="x", pady=0)
            symbol_var = tk.StringVar(value="►")
            def toggle_cmd():
                if vf.winfo_ismapped(): vf.pack_forget(); symbol_var.set("►")
                else: vf.pack(fill="x", padx=5, pady=(2, 5)); symbol_var.set("▼")
            tk.Label(btn_frame, textvariable=symbol_var, anchor="w", bg="#Ecf0f1", font=("Arial", 9, "bold")).pack(side="left", padx=5, pady=(2, 3))
            name_lbl = tk.Label(btn_frame, text=grp_name, anchor="w", relief="flat", bg="#Ecf0f1", font=("Arial", 9, "bold"))
            name_lbl.pack(side="left", fill="x", expand=True, pady=3)
            
            for widget in [name_lbl, btn_frame]: widget.bind("<Button-1>", lambda e: toggle_cmd())
            if grp_name == "User Formulas": groups[grp_name]['labelframe'].pack(fill="x", pady=1)

        vf = groups[grp_name]['frame'] 
        row_frame = tk.Frame(vf, bg="white"); row_frame.pack(fill="x")
        
        
        var = tk.BooleanVar(value=False); check_vars[col_name] = var
        safe_h = get_safe_name(col_name)

        
        tk.Button(row_frame, text="[+]", font=("Consolas", 8), bg="#eef8f8", fg="green", bd=0,
                  command=lambda h=safe_h: insert_text(h)).pack(side="left", padx=2)

        
        tk.Checkbutton(row_frame, text=col_name, variable=var, command=update_plot, bg="white", anchor="w").pack(side="left", fill="x", expand=True)

        
        
    
    math_frame = tk.LabelFrame(scrollable_frame, text="Formula Builder (Time Series)", font=("Arial", 10, "bold"), fg="#2c3e50", padx=5, pady=5)
    math_frame.pack(fill="x", pady=5)
    
    name_frame = tk.Frame(math_frame); name_frame.pack(fill="x", padx=5, pady=2)
    tk.Label(name_frame, text="Name:").pack(side="left")
    name_entry = tk.Entry(name_frame, font=("Consolas", 11), bg="#fdfeff"); name_entry.pack(side="left", fill="x", expand=True)
    expr_entry = tk.Entry(math_frame, font=("Consolas", 11), bg="#fdfeff"); expr_entry.pack(fill="x", padx=5, pady=5)
    tk.Button(math_frame, text="COMPUTE & ADD SERIES", bg="#b3dc7c", font=("Arial", 9, "bold"),
              command=lambda: add_expression(name_entry.get())).pack(fill="x", padx=5, pady=10)

    
    scalar_frame = tk.LabelFrame(scrollable_frame, text="Scalar Calculator (Summary Stat)", font=("Arial", 10, "bold"), fg="#2c3e50", padx=5, pady=5)
    scalar_frame.pack(fill="x", pady=5)
    tk.Label(scalar_frame, text="Expression (Vector):", anchor="w").pack(padx=5, pady=(5,0))
    scalar_expr_entry = tk.Entry(scalar_frame, font=("Consolas", 11), bg="#fdfeff"); scalar_expr_entry.pack(fill="x", padx=5, pady=5)
    calc_frame = tk.Frame(scalar_frame); calc_frame.pack(fill="x", padx=5, pady=5)
    tk.Label(calc_frame, text="Apply Function:").pack(side="left")
    scalar_func_var = tk.StringVar(value="Select Function")
    func_options = ["Select Function"] + sorted(AGGREGATE_FUNCTIONS.keys())
    scalar_func_combobox = ttk.Combobox(calc_frame, textvariable=scalar_func_var, values=func_options, state="readonly", width=12)
    scalar_func_combobox.pack(side="left", padx=5)
    scalar_result_lbl = tk.Label(scalar_frame, text="Resultado: N/A", anchor="w", font=("Consolas", 10, "bold"), fg="#2980b9")
    tk.Button(calc_frame, text="CALCULATE", bg="#f9c882", font=("Arial", 9, "bold"),
              command=lambda: calculate_scalar(df, scalar_expr_entry, scalar_func_var, scalar_result_lbl)).pack(side="right", fill="x", expand=True)
    scalar_result_lbl.pack(fill="x", padx=5, pady=5)

    
    fmt_frame = tk.LabelFrame(scrollable_frame, text="Appearance & Axes", padx=5, pady=5); fmt_frame.pack(fill="x", pady=5)
    f_grid = tk.Frame(fmt_frame); f_grid.pack(fill="x", padx=5)
    f_grid.grid_columnconfigure(1, weight=1)
    
    
    widgets_to_config = {}
    row_idx = 1
    for label, key in [("Title:", 'title_entry'), ("X Label:", 'xlabel_entry'), ("Y Label:", 'ylabel_entry')]:
        tk.Label(f_grid, text=label).grid(row=row_idx, column=0, sticky="w", pady=2)
        entry = tk.Entry(f_grid, width=30)
        entry.grid(row=row_idx, column=1, sticky="ew", pady=2)
        widgets_to_config[key] = entry
        row_idx += 1

    
    tk.Button(fmt_frame, text="Update Labels", command=update_plot).pack(fill="x", pady=5)
    
    
    ts_frame = tk.LabelFrame(scrollable_frame, text="Variables (Time Series)", font=("Arial", 10, "bold"), padx=5, pady=5); ts_frame.pack(fill="x", pady=5)
    xf = tk.Frame(ts_frame); xf.pack(fill="x", padx=5); xf.grid_columnconfigure(1, weight=1)

    all_cols = ["timestamp"] + sorted([col for col in df.columns if col != "timestamp"])
    
    
    tk.Label(xf, text="X-Axis:", font=("Arial", 9, "bold")).grid(row=0, column=0, sticky="w")
    x_axis_combobox = ttk.Combobox(xf, textvariable=x_axis_var, values=all_cols, state="readonly")
    x_axis_combobox.grid(row=0, column=1, sticky="ew")
    x_axis_combobox.bind("<<ComboboxSelected>>", lambda event: update_plot())
    widgets_to_config['x_axis_combobox'] = x_axis_combobox
    
   
    tk.Label(xf, text="Y-Mode:", font=("Arial", 9, "bold")).grid(row=1, column=0, sticky="w")
    y_axis_combobox = ttk.Combobox(xf, textvariable=y_axis_mode_var, values=y_axis_options, state="readonly")
    y_axis_combobox.grid(row=1, column=1, sticky="ew")
    y_axis_combobox.bind("<<ComboboxSelected>>", lambda event: update_plot())
    widgets_to_config['y_axis_mode_var'] = y_axis_mode_var 
    
    
    plot_layout_var = tk.StringVar(value="1x1 (Single Plot)") 
    widgets_to_config['layout_var'] = plot_layout_var

    
    primary_plot_config = {
        'root': root, 'df': df, 'fig': fig, 'canvas': canvas,
        **widgets_to_config
    }

    
    for col_name in sorted(df.columns):
        if col_name == "timestamp": continue
        grp = col_name.split('.')[0] if '.' in col_name else "Misc"
        add_checkbox_to_group(grp, col_name)

    for grp_name in sorted(groups.keys()):
        if grp_name != "User Formulas": groups[grp_name]['labelframe'].pack(fill="x", pady=1)
        
    
    duplicate_frame = tk.Frame(main_frame, bg="#f0f0f0")
    duplicate_frame.grid(row=1, column=1, sticky="se", pady=(5, 0))

    tk.Button(duplicate_frame, 
              text="Abrir Gráfico en Nueva Ventana", 
              bg="#2ecc71", fg="white", 
              font=("Arial", 10, "bold"),
              command=lambda: open_new_plotter_window(df, primary_plot_config)).pack(padx=5, pady=5)


    
    update_plot()
    root.mainloop()


    

def read_rosbag_mcap(file_path: str):
    rows = []
    seen_columns = set()
    
   
    last_snapshots = {} 

    print(f"Procesando archivo: {file_path}...")

    for topic, msg, timestamp, msg_type in read_ros_messages(file_path):
        ts_sec = timestamp * 1e-9
        msg_type_str = str(type(msg))

 
        if "PointCloud2" in msg_type_str:
            last_snapshots[topic] = msg
            continue

       
        is_trajectory = False

        if hasattr(msg, 'points') and isinstance(msg.points, list) and len(msg.points) > 0:
            is_trajectory = True
        elif hasattr(msg, 'poses') and isinstance(msg.poses, list) and len(msg.poses) > 0:
            is_trajectory = True
            
        if is_trajectory:
            last_snapshots[topic] = msg
            continue 

      
        row = {"timestamp": ts_sec}
        try:
            for field in msg.get_fields_and_field_types():
                if field == "header": continue
                
                value = getattr(msg, field)
                
                if isinstance(value, (int, float)):
                    col_name = f"{topic}.{field}"
                    row[col_name] = value
                    seen_columns.add(col_name)
                
                elif isinstance(value, (list, tuple, np.ndarray)) and len(value) <= 20:
                    for i, val in enumerate(value):
                        if isinstance(val, (int, float)):
                            col_name = f"{topic}.{field}[{i}]"
                            row[col_name] = val
                            seen_columns.add(col_name)
                            
            if len(row) > 1:
                rows.append(row)
                
        except Exception as e:
            pass 


    if rows:
        df = pd.DataFrame(rows)
        for col in seen_columns:
            if col not in df.columns:
                df[col] = np.nan
        df.sort_values("timestamp", inplace=True)
        df.ffill(inplace=True)
    else:
        df = pd.DataFrame()


    return df, last_snapshots





def read_can_txt_file(file_path: str):
    import os
    import re
    import pandas as pd
    import numpy as np
    import tkinter as tk
    
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    conversion_path = os.path.join(script_dir, "can_conversions.csv")

    try:
        
        can_conversions = pd.read_csv(
            conversion_path, 
            dtype={'ID': str} 
        ) 
        
        
        can_conversions['ID'] = can_conversions['ID'].str.upper().str.strip()
        can_conversions['Signed'] = can_conversions['Signed'].astype(bool) 
        
       
        can_conversions.set_index('ID', inplace=True)
        
       
        conversion_map = can_conversions.groupby(level=0).apply(lambda x: x.to_dict('records')).to_dict()

    except FileNotFoundError:
        tk.messagebox.showerror(
            "Error de Archivo CAN",
            f"No se encontró el archivo de conversiones:\n{conversion_path}\n"
        )
        return pd.DataFrame(), {} 
    except Exception as e:
        tk.messagebox.showerror(
            "Error de Carga CSV",
            f"Fallo al procesar el CSV de conversiones. Detalle:\n{e}"
        )
        return pd.DataFrame(), {}


    
    pattern = re.compile(
        
        r'^\s*\((\d+\.\d+)\)\s+' 
        
        r'[a-zA-Z0-9]+\s+' 
        
        r'([0-9a-fA-F]+)\s+' 
       
        r'\[\d+\]\s+'
        
        r'([0-9a-fA-F\s]+).*$' 
        , re.IGNORECASE
    )

    rows = []
    seen_columns = set()
    lines_processed = 0

   
    with open(file_path, "r", encoding="latin1") as f:
        for line in f:
            lines_processed += 1
            stripped_line = line.strip()
            match = pattern.match(stripped_line)
            
            if match:
                timestamp = float(match.group(1)) 
                raw_id_str = match.group(2).upper()
                data_str = match.group(3).strip() 
                
                
                can_id_key = f"0X{raw_id_str}"
                
                try:
                    data_bytes = [int(byte, 16) for byte in data_str.split()]
                except ValueError:
                    continue 
                
                
                
                
                relevant_conversions_list = conversion_map.get(can_id_key, [])

                if relevant_conversions_list:
                    
                    for can_info in relevant_conversions_list:
                        
                        bitIn = can_info["bitIn"]
                        bitFin = can_info["bitFin"]
                        
                        start_byte = bitIn
                        end_byte = bitFin + 1
                        
                        if end_byte > len(data_bytes):
                            continue
                        
                        value_bytes = data_bytes[start_byte:end_byte]

                        
                        if can_info["Signed"]:
                            raw_int = int.from_bytes(value_bytes, byteorder='little', signed=True)
                        else:
                            raw_int = int.from_bytes(value_bytes, byteorder='little', signed=False)
                            
                        
                        row = {"timestamp": timestamp} 
                        column_name = can_info["Name"]
                        
                        value = raw_int * can_info["Scale"] + can_info["Offset"]

                        row[column_name] = value
                        seen_columns.add(column_name)
                        rows.append(row)
                        
    
    print(f"Líneas procesadas: {lines_processed}, Filas decodificadas: {len(rows)}")
    if rows:
        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame()
        else:
            df.sort_values("timestamp", inplace=True)
            df.ffill(inplace=True) 
            for col in seen_columns:
                if col not in df.columns:
                    df[col] = np.nan
    else:
        df = pd.DataFrame()
        
    return df, {}





def main():
    import sys
    import tkinter as tk
    from tkinter import filedialog 
    
    file_path = None
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        root.destroy()
        
    if not file_path:
        return

    last_snapshots = {}
    
    if file_path.endswith(".txt"):
        
        df, last_snapshots = read_can_txt_file(file_path)
    else:
        
        df, last_snapshots = read_rosbag_mcap(file_path) 
        
    if not df.empty or last_snapshots:
       
        plot_variables(df, last_snapshots) 
    else:
        print("No data found.")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()