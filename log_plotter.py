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
import re
import sys
import os
import array  
import struct

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
check_geo_vars = {}
safe_map = {}
primary_plot_config = {}

AGGREGATE_FUNCTIONS = {
    "mean": np.mean, "sum": np.sum, "std": np.std, "min": np.min, "max": np.max,
    "abs_mean": lambda x: np.mean(np.abs(x)),
}



def get_safe_name(s):
    return str(s).replace('.', '_').replace('[', '_').replace(']', '_').replace('/', '_')

def to_safe_identifier(name):
    return "_" + re.sub(r"[^a-zA-Z0-9_]", "_", str(name))

def get_eval_context(df_local):
    local_vars = {get_safe_name(real_name): df_local[real_name] for safe_name, real_name in safe_map.items()}
    math_funcs = {"np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan,
                  "sqrt": np.sqrt, "abs": np.abs, "log": np.log, "exp": np.exp}
    return {"__builtins__": {}, **local_vars, **math_funcs}

def insert_text(text):
    target = None
    if 'expr_entry' in globals() and expr_entry.focus_get() == expr_entry:
         target = expr_entry
    elif 'scalar_expr_entry' in globals() and scalar_expr_entry.focus_get() == scalar_expr_entry:
         target = scalar_expr_entry
    
    if target:
        idx = target.index(tk.INSERT)
        target.insert(idx, text)
        target.focus()


def read_pcd_native(file_path):
    metadata = {}
    try:
        with open(file_path, 'rb') as f:
            header_end = False
            while not header_end:
                line = f.readline().decode('ascii').strip()
                if line.startswith('DATA'):
                    metadata['DATA'] = line.split()[1]
                    header_end = True
                    break
                parts = line.split()
                if not parts: continue
                if parts[0] == 'FIELDS': metadata['FIELDS'] = parts[1:]
                elif parts[0] == 'SIZE': metadata['SIZE'] = [int(x) for x in parts[1:]]
                elif parts[0] == 'TYPE': metadata['TYPE'] = parts[1:]

            if 'FIELDS' not in metadata or 'x' not in metadata['FIELDS']: return None
            
            idx_x, idx_y = metadata['FIELDS'].index('x'), metadata['FIELDS'].index('y')
            
            if metadata['DATA'] == 'ascii':
                data = np.loadtxt(f)
                return data[:, [idx_x, idx_y]]
            elif metadata['DATA'] == 'binary':
                dtype_list = []
                for field, size, type_char in zip(metadata['FIELDS'], metadata['SIZE'], metadata['TYPE']):
                    dt = np.float32 if size == 4 else np.float64 
                    dtype_list.append((field, dt))
                buffer = f.read()
                data = np.frombuffer(buffer, dtype=np.dtype(dtype_list))
                return np.column_stack((data['x'], data['y']))
    except Exception as e:
        print(f"Error PCD: {e}")
    return None

def read_ros_messages(input_bag: str):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=input_bag, storage_id="mcap"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
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

def read_rosbag_mcap(file_path: str):
    rows = []
    seen_columns = set()
    last_snapshots = {} 
    print(f"Procesando MCAP: {file_path}...")

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
                
                
                elif isinstance(value, (list, tuple, np.ndarray, array.array)) and len(value) <= 20:
                    for i, val in enumerate(value):
                        if isinstance(val, (int, float)):
                            
                            
                            suffix = ""
                            if "epos" in topic.lower():
                                if i == 0: suffix = "_movement_state"
                                elif i == 1: suffix = "_position"
                                elif i == 2: suffix = "_target_position"
                                elif i == 3: suffix = "_velocity"      
                                elif i == 4: suffix = "_velocity_avg"
                                elif i == 5: suffix = "_torque"

                            col_name = f"{topic}.{field}[{i}]{suffix}"
                            row[col_name] = val
                            seen_columns.add(col_name)
            
            if len(row) > 1:
                rows.append(row)
        except Exception:
            pass 

    if rows:
        df = pd.DataFrame(rows)
        df.columns = df.columns.astype(str)
        for col in seen_columns:
            if col not in df.columns:
                df[col] = np.nan
        df.sort_values("timestamp", inplace=True)
        df.ffill(inplace=True)
    else:
        df = pd.DataFrame()

    return df, last_snapshots

def read_can_txt_file(file_path: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    conversion_path = os.path.join(script_dir, "can_conversions.csv")

    
    conversion_map = {}
    try:
        can_conversions = pd.read_csv(conversion_path, dtype={'ID': str})
        can_conversions['Signed'] = can_conversions['Signed'].astype(bool)
        
        for _, row in can_conversions.iterrows():
            
            raw_csv_id = str(row['ID']).strip().upper().replace('0X', '')
            
            
            if len(raw_csv_id) == 4:
                base_id_hex = raw_csv_id[:3] 
            else:
                base_id_hex = raw_csv_id
                
            
            try:
                key_hex = str(hex(int(base_id_hex, 16)))
                
                if key_hex not in conversion_map:
                    conversion_map[key_hex] = []
                
                
                conversion_map[key_hex].append(row.to_dict())
            except ValueError:
                continue
                
    except Exception as e:
        messagebox.showerror("Error CSV", f"Error procesando CSV: {e}")
        return pd.DataFrame(), {}

    rows = []
    lines_processed = 0
    print(f"Leyendo TXT (Lógica V3.3 con soporte sub-mensajes): {file_path}")

    with open(file_path, "r", encoding="latin1") as f:
        for line in f:
            lines_processed += 1
            stripped = line.strip()
            if not stripped: continue
            
            parts = stripped.split()
            if len(parts) < 3: continue

            try:
                ts_str = parts[0].replace('(', '').replace(')', '')
                timestamp = float(ts_str)
                
                found_hex_id = None
                id_index = -1
                
                
                for i in range(1, min(5, len(parts))):
                    try:
                        val = int(parts[i], 16)
                        candidate_hex = str(hex(val)) 
                        
                        if candidate_hex in conversion_map:
                            found_hex_id = candidate_hex
                            id_index = i
                            break
                    except ValueError:
                        continue

                if found_hex_id is None: continue

                
                start_data_idx = id_index + 1
                if start_data_idx < len(parts) and parts[start_data_idx].startswith('['):
                    start_data_idx += 1
                
                data_hex_list = parts[start_data_idx:]
                data_bytes = []
                for b in data_hex_list:
                    try:
                        data_bytes.append(int(b, 16))
                    except ValueError:
                        continue 
                
                if not data_bytes: continue

                
                relevant_conversions = conversion_map[found_hex_id]
                
                for can_info in relevant_conversions:
                    start = can_info["bitIn"]
                    end = can_info["bitFin"] + 1
                    
                    if end <= len(data_bytes):
                        val_bytes = data_bytes[start:end]
                        if can_info["Signed"]:
                            raw = int.from_bytes(val_bytes, byteorder='little', signed=True)
                        else:
                            raw = int.from_bytes(val_bytes, byteorder='little', signed=False)
                            
                        phys = raw * can_info["Scale"] + can_info["Offset"]
                        col = str(can_info["Name"])
                        rows.append({"timestamp": timestamp, col: phys})
            except Exception:
                continue

    print(f"Resumen: {lines_processed} líneas, {len(rows)} datos extraídos.")

    if rows:
        full_df = pd.DataFrame(rows)
        df = full_df.groupby('timestamp', as_index=False).last()
        df.sort_values("timestamp", inplace=True)
        df.ffill(inplace=True)
        df.columns = df.columns.astype(str)
        cols_to_drop = [c for c in df.columns if c.lower() == 'nan' or c.strip() == '']
        if cols_to_drop: df.drop(columns=cols_to_drop, inplace=True)
    else:
        df = pd.DataFrame()

    return df, {}



def draw_plot(df_local, fig, canvas, config):
    fig.clear()
    ax = fig.add_subplot(1, 1, 1)

    scale_factor = 1.0
    try:
        root = config.get('root')
        if root:
            screen_width = root.winfo_screenwidth()
            scale_factor = max(1.0, (screen_width / 1920.0) * 1.2)
    except Exception:
        scale_factor = 1.0

    variables_to_plot = []

    x_col_obj = config.get('x_axis_combobox')
    x_col = x_col_obj.get() if hasattr(x_col_obj, 'get') else str(x_col_obj)

    if x_col == "index" or x_col not in df_local.columns:
        x_data = df_local.index
        x_lbl_default = "Index / Time Step"
    else:
        x_data = df_local[x_col]
        x_lbl_default = x_col

    if 'variables_to_plot' in config:
        variables_to_plot = config['variables_to_plot']
    elif 'check_vars' in globals() and check_vars:
        variables_to_plot = [col for col, var in check_vars.items() if var.get() and col in df_local.columns]

    geos_to_plot = []
    snapshots = config.get('snapshots', {})
    if 'check_geo_vars' in globals() and check_geo_vars:
        geos_to_plot = [name for name, var in check_geo_vars.items() if var.get() and name in snapshots]

    if variables_to_plot or geos_to_plot:
        y_mode_obj = config.get('y_axis_mode_var')
        y_col_mode = y_mode_obj.get() if hasattr(y_mode_obj, 'get') else "Multiple Variables"

        t_obj = config.get('title_entry')
        title_text = t_obj.get() if hasattr(t_obj, 'get') else str(t_obj)
        
        is_map = "PointCloud" in title_text or "Map" in title_text or bool(geos_to_plot)

        if variables_to_plot:
            if ("Scatter" in y_col_mode and len(variables_to_plot) == 1) or is_map:
                for col in variables_to_plot:
                    base_size = 6 if is_map else 30
                    final_size = base_size * scale_factor
                    ax.scatter(x_data, df_local[col], s=final_size, label=col, alpha=0.6)
                y_lbl_default = "Y Value"
            else:
                for col in variables_to_plot:
                    ax.plot(x_data, df_local[col], label=col, alpha=0.8, linewidth=1.5)
                y_lbl_default = "Value"
        else:
            y_lbl_default = "Y Value"

        for name in geos_to_plot:
            msg = snapshots[name]
            xs, ys = [], []
            try:
                if isinstance(msg, np.ndarray):
                    if msg.shape[1] >= 2: xs, ys = msg[:, 0], msg[:, 1]
                elif ROS_ENABLED and "PointCloud2" in str(type(msg)):
                    gen = point_cloud2.read_points(msg, field_names=['x', 'y'], skip_nans=True)
                    data = list(gen)
                    if data: xs, ys = zip(*data)
                else:
                    for field in dir(msg):
                        if field.startswith("_"): continue
                        val = getattr(msg, field)
                        if isinstance(val, (list, tuple)) and len(val) > 0:
                            if hasattr(val[0], 'x') and hasattr(val[0], 'y'):
                                xs = [p.x for p in val]; ys = [p.y for p in val]
                                break
            except Exception: pass
            
            if len(xs) > 0:
                ax.scatter(xs, ys, s=(6 * scale_factor), label=name, alpha=0.5)

        ax.legend(loc='upper right', fontsize='small', framealpha=0.9)
        ax.set_aspect('equal' if is_map else 'auto', adjustable='datalim')
        
        xl_obj = config.get('xlabel_entry')
        xlabel = xl_obj.get() if hasattr(xl_obj, 'get') else x_lbl_default
        yl_obj = config.get('ylabel_entry')
        ylabel = yl_obj.get() if hasattr(yl_obj, 'get') else y_lbl_default

        ax.set_title(title_text)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.text(0.5, 0.5, "No variables selected", ha='center', va='center', transform=ax.transAxes)

    fig.tight_layout()
    canvas.draw()

def open_new_plotter_window(df_local, config):
    new_window = tk.Toplevel(config['root'])
    t_obj = config.get('title_entry')
    title_text = t_obj.get() if hasattr(t_obj, 'get') else "Gráfico Adicional"
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

def plot_snapshot(df_local, last_snapshots, topic_name):
    if not topic_name or topic_name not in last_snapshots: return
    msg = last_snapshots[topic_name]
    title = f"Geometry Plot: {topic_name}"
    xs, ys = [], []
    is_valid = False
    msg_type_str = str(type(msg))

    try:
        if isinstance(msg, np.ndarray):
            xs, ys = msg[:, 0], msg[:, 1]
            is_valid = True
        elif "PointCloud2" in msg_type_str:
            if ROS_ENABLED:
                gen = point_cloud2.read_points(msg, field_names=['x', 'y'], skip_nans=True)
                data = list(gen)
                if data:
                    xs, ys = zip(*data)
                    is_valid = True
        else:
            for field in dir(msg):
                if field.startswith("_"): continue
                value = getattr(msg, field)
                if isinstance(value, (list, tuple)) and len(value) > 0:
                    if hasattr(value[0], 'x') and hasattr(value[0], 'y'):
                        xs = [p.x for p in value]
                        ys = [p.y for p in value]
                        is_valid = True
                        break 
    except Exception as e:
        messagebox.showerror("Error", f"Fallo al leer geometría: {e}")
        return

    if is_valid:
        new_window = tk.Toplevel()
        new_window.title(title)
        new_window.geometry("800x600")
        fig_geo = plt.figure(figsize=(8, 6))
        ax_geo = fig_geo.add_subplot(111)
        canvas_geo = FigureCanvasTkAgg(fig_geo, master=new_window)
        canvas_geo.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(canvas_geo, new_window)

        if "PointCloud" in msg_type_str:
            ax_geo.scatter(xs, ys, s=15, alpha=0.6)
            ax_geo.set_aspect('equal', adjustable='datalim')
        else:
            ax_geo.plot(xs, ys, 'o-', markersize=3)
            ax_geo.set_aspect('auto')
        
        ax_geo.set_title(title)
        ax_geo.grid(True)
        canvas_geo.draw()
    else:
        messagebox.showinfo("Info", "No se encontraron datos geométricos.")

def calculate_scalar(df_local, entry, func_var, lbl):
    try:
        expr = entry.get().strip()
        func = func_var.get()
        if not expr or func not in AGGREGATE_FUNCTIONS: return
        ctx = get_eval_context(df_local)
        res = eval(expr, {"__builtins__": {}}, ctx)
        val = AGGREGATE_FUNCTIONS[func](res)
        lbl.config(text=f"Res: {val:.6f}")
    except Exception as e:
        messagebox.showerror("Error", str(e))



def plot_variables(df, last_snapshots={}):
    global safe_map, check_vars, primary_plot_config, expr_entry, name_entry, scalar_expr_entry, scrollable_frame
    
    
    df.columns = df.columns.astype(str)
    cols_to_drop = [c for c in df.columns if c.lower() == 'nan' or c.strip() == '']
    if cols_to_drop: df.drop(columns=cols_to_drop, inplace=True)

    safe_map = {get_safe_name(c): c for c in df.columns}
    check_vars = {}
    
    root = tk.Tk()
    root.title("Graphic.cArus")

    
    main_frame = tk.Frame(root)
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    main_frame.grid_columnconfigure(0, weight=0, minsize=450) 
    main_frame.grid_columnconfigure(1, weight=1)
    main_frame.grid_rowconfigure(0, weight=1)

    
    left_container = tk.Frame(main_frame, bg="#f0f0f0", bd=1, relief="sunken")
    left_container.grid(row=0, column=0, sticky="nsew")

    
    scrollbar = tk.Scrollbar(left_container, orient="vertical")
    scrollbar.pack(side="right", fill="y")

    
    controls_canvas = tk.Canvas(left_container, bg="#f0f0f0", yscrollcommand=scrollbar.set)
    controls_canvas.pack(side="left", fill="both", expand=True)
    
    scrollbar.config(command=controls_canvas.yview)

    
    scrollable_frame = tk.Frame(controls_canvas, bg="#f0f0f0")
    
    def adjust_scrollregion(e):
        controls_canvas.configure(scrollregion=controls_canvas.bbox("all"))
        controls_canvas.itemconfig(frame_window_id, width=e.width)

    scrollable_frame.bind("<Configure>", lambda e: controls_canvas.configure(scrollregion=controls_canvas.bbox("all")))
    frame_window_id = controls_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    controls_canvas.bind("<Configure>", adjust_scrollregion)

    
    root.bind_all("<MouseWheel>", lambda e: controls_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
    root.bind_all("<Button-4>", lambda e: controls_canvas.yview_scroll(-1, "units"))
    root.bind_all("<Button-5>", lambda e: controls_canvas.yview_scroll(1, "units"))

    
    plot_frame = tk.Frame(main_frame)
    plot_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
    plot_frame.grid_rowconfigure(0, weight=1); plot_frame.grid_columnconfigure(0, weight=1)
    fig = plt.figure(figsize=(10, 8))
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    NavigationToolbar2Tk(canvas, plot_frame)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    x_axis_var = tk.StringVar(value="timestamp")
    y_axis_mode_var = tk.StringVar(value="Time Series (Multiple Y vs. X)")
    widgets_to_config = {'layout_var': tk.StringVar(value="1x1")}

    def update_plot():
        draw_plot(df, fig, canvas, primary_plot_config)

    
    groups = {}
    
    def add_checkbox_to_group(grp_name, col_name):
        if grp_name not in groups:
            gf = tk.LabelFrame(scrollable_frame, text="", bd=1, relief="solid", bg="#Ecf0f1")
            gf.pack(fill="x", pady=1, padx=2)
            vf = tk.Frame(gf, bg="white")
            
            btn_frame = tk.Frame(gf, bg="#Ecf0f1", height=25)
            btn_frame.pack(fill="x", side="top")
            btn_frame.pack_propagate(False) 
            
            symbol_var = tk.StringVar(value="►")
            def toggle(v_frame=vf, s_var=symbol_var):
                if v_frame.winfo_ismapped():
                    v_frame.pack_forget(); s_var.set("►")
                else:
                    v_frame.pack(fill="x", padx=5, pady=2); s_var.set("▼")
            
            lbl_arrow = tk.Label(btn_frame, textvariable=symbol_var, bg="#Ecf0f1", font=("Arial", 10, "bold"), width=3)
            lbl_arrow.pack(side="left")
            lbl_name = tk.Label(btn_frame, text=grp_name, bg="#Ecf0f1", font=("Arial", 9, "bold"), anchor="w")
            lbl_name.pack(side="left", fill="x", expand=True)
            
            for w in [btn_frame, lbl_arrow, lbl_name]: w.bind("<Button-1>", lambda e: toggle())
            groups[grp_name] = {'content': vf, 'header': gf, 'toggle_func': toggle}
            if grp_name == "User Formulas": toggle()

        vf = groups[grp_name]['content']
        row = tk.Frame(vf, bg="white"); row.pack(fill="x", pady=1)
        var = tk.BooleanVar(value=False); check_vars[col_name] = var
        safe_h = get_safe_name(col_name)
        
        tk.Button(row, text="[+]", command=lambda: insert_text(safe_h), 
                  font=("Consolas", 7), bd=0, bg="#eef8f8", fg="green", width=3).pack(side="left")
        tk.Checkbutton(row, text=col_name, variable=var, command=update_plot, 
                       bg="white", anchor="w").pack(side="left", fill="x", expand=True)
        
        return groups[grp_name]['toggle_func']

    
    mf = tk.LabelFrame(scrollable_frame, text="Formula Builder", font=("Arial",10,"bold"), bg="#dfe6e9", pady=5)
    mf.pack(fill="x", padx=5, pady=5)
    
    fr1 = tk.Frame(mf, bg="#dfe6e9"); fr1.pack(fill="x", padx=5)
    tk.Label(fr1, text="Nombre:", bg="#dfe6e9").pack(side="left")
    name_entry = tk.Entry(fr1); name_entry.pack(side="left", fill="x", expand=True)
    
    tk.Label(mf, text="Expr (ej: /vel * 2):", bg="#dfe6e9", anchor="w").pack(fill="x", padx=5)
    expr_entry = tk.Entry(mf); expr_entry.pack(fill="x", padx=5)
    
    def add_expr():
        global formula_counter
        try:
            e = expr_entry.get(); n = name_entry.get().strip() or f"Expr_{formula_counter}"
            res = eval(e, {"__builtins__":{}}, get_eval_context(df))
            df[n] = res; safe_map[get_safe_name(n)] = n
            toggle_fn = add_checkbox_to_group("User Formulas", n)
            check_vars[n].set(True)
            formula_counter += 1
            update_plot()
        except Exception as x: messagebox.showerror("Error", str(x))

    tk.Button(mf, text="AÑADIR VARIABLE", command=add_expr, bg="#b3dc7c").pack(fill="x", padx=5, pady=5)

    
    if last_snapshots:
        sf = tk.LabelFrame(scrollable_frame, text="Geometría / Mapas", font=("Arial",10,"bold"), bg="#dfe6e9", pady=5)
        sf.pack(fill="x", padx=5, pady=5)
        s_opts = sorted(last_snapshots.keys())
        s_var = tk.StringVar(value=s_opts[0])
        ttk.Combobox(sf, textvariable=s_var, values=s_opts, state="readonly").pack(fill="x", padx=5, pady=2)
        tk.Button(sf, text="PLOTEAR", command=lambda: plot_snapshot(df, last_snapshots, s_var.get()), 
                  bg="#3498db", fg="white", font=("Arial", 9, "bold")).pack(fill="x", padx=5, pady=5)

    
    scf = tk.LabelFrame(scrollable_frame, text="Calculadora", font=("Arial",10,"bold"), bg="#dfe6e9", pady=5)
    scf.pack(fill="x", padx=5, pady=5)
    scalar_expr_entry = tk.Entry(scf); scalar_expr_entry.pack(fill="x", padx=5)
    fr2 = tk.Frame(scf, bg="#dfe6e9"); fr2.pack(fill="x", padx=5, pady=2)
    s_func = tk.StringVar(value="mean")
    ttk.Combobox(fr2, textvariable=s_func, values=list(AGGREGATE_FUNCTIONS.keys()), state="readonly", width=10).pack(side="left")
    tk.Button(fr2, text="CALC", command=lambda: calculate_scalar(df, scalar_expr_entry, s_func, s_res), bg="#f9c882").pack(side="left", padx=5)
    s_res = tk.Label(scf, text="Res: -", bg="#dfe6e9"); s_res.pack(fill="x")

    
    cf = tk.LabelFrame(scrollable_frame, text="Configuración Gráfica", font=("Arial",10,"bold"), bg="#dfe6e9", pady=5)
    cf.pack(fill="x", padx=5, pady=5)
    for l, k in [("Titulo", 'title_entry'), ("Eje X", 'xlabel_entry'), ("Eje Y", 'ylabel_entry')]:
        fr = tk.Frame(cf, bg="#dfe6e9"); fr.pack(fill="x", padx=5, pady=1)
        tk.Label(fr, text=l, width=8, anchor="w", bg="#dfe6e9").pack(side="left")
        e = tk.Entry(fr); e.pack(side="left", fill="x", expand=True); widgets_to_config[k] = e
    tk.Button(cf, text="Actualizar Etiquetas", command=update_plot).pack(fill="x", padx=5, pady=2)

    vf = tk.LabelFrame(scrollable_frame, text="Ejes", font=("Arial",10,"bold"), bg="#dfe6e9", pady=5)
    vf.pack(fill="x", padx=5, pady=5)
    tk.Label(vf, text="Variable Eje X:", bg="#dfe6e9").pack(anchor="w", padx=5)
    xc = ttk.Combobox(vf, textvariable=x_axis_var, values=["timestamp"]+sorted([c for c in df.columns if c!="timestamp"]), state="readonly")
    xc.pack(fill="x", padx=5); xc.bind("<<ComboboxSelected>>", lambda e: update_plot())
    widgets_to_config['x_axis_combobox'] = xc
    tk.Label(vf, text="Tipo de Gráfico:", bg="#dfe6e9").pack(anchor="w", padx=5)
    yc = ttk.Combobox(vf, textvariable=y_axis_mode_var, values=["Time Series", "Scatter"], state="readonly")
    yc.pack(fill="x", padx=5); yc.bind("<<ComboboxSelected>>", lambda e: update_plot())
    widgets_to_config['y_axis_mode_var'] = y_axis_mode_var

    primary_plot_config = {'root': root, 'df': df, 'fig': fig, 'canvas': canvas, 'snapshots': last_snapshots, **widgets_to_config}

    
    for col_raw in sorted(df.columns):
        col_name = str(col_raw)
        if col_name == "timestamp": continue
        grp = col_name.split('.')[0] if '.' in col_name else "Misc"
        add_checkbox_to_group(grp, col_name)

    if last_snapshots:
        for name in sorted(last_snapshots.keys()):
            grp_name = "Geometry / Mapas"
            if grp_name not in groups:
                pass 

    def add_geo_checkbox(name):
        grp_name = "Geometry / Mapas"
        if grp_name not in groups:
            gf = tk.LabelFrame(scrollable_frame, text="", bd=1, relief="solid", bg="#Ecf0f1")
            gf.pack(fill="x", pady=1, padx=2)
            vf = tk.Frame(gf, bg="white")
            btn_frame = tk.Frame(gf, bg="#Ecf0f1", height=25)
            btn_frame.pack(fill="x", side="top"); btn_frame.pack_propagate(False)
            symbol_var = tk.StringVar(value="►")
            def toggle(v_frame=vf, s_var=symbol_var):
                if v_frame.winfo_ismapped(): v_frame.pack_forget(); s_var.set("►")
                else: v_frame.pack(fill="x", padx=5, pady=2); s_var.set("▼")
            lbl_arrow = tk.Label(btn_frame, textvariable=symbol_var, bg="#Ecf0f1", font=("Arial",10,"bold"), width=3)
            lbl_arrow.pack(side="left")
            tk.Label(btn_frame, text=grp_name, bg="#Ecf0f1", font=("Arial",9,"bold"), anchor="w").pack(side="left", fill="x", expand=True)
            for w in [btn_frame, lbl_arrow]: w.bind("<Button-1>", lambda e: toggle())
            groups[grp_name] = {'content': vf, 'header': gf, 'toggle_func': toggle}
            toggle() 

        vf = groups[grp_name]['content']
        row = tk.Frame(vf, bg="white"); row.pack(fill="x", pady=1)
        var = tk.BooleanVar(value=False)
        check_geo_vars[name] = var 
        
        tk.Checkbutton(row, text=name, variable=var, command=update_plot, 
                       bg="white", anchor="w", fg="blue").pack(side="left", fill="x", expand=True)

    
    for name in sorted(last_snapshots.keys()):
        add_geo_checkbox(name)

    bottom_btn_frame = tk.Frame(main_frame)
    bottom_btn_frame.grid(row=1, column=1, sticky="se", pady=10, padx=10)

    def import_pcd_callback():
        file_path = filedialog.askopenfilename(filetypes=[("PCD Files", "*.pcd")])
        if not file_path: return
        points = read_pcd_native(file_path)
        if points is not None:
            name = os.path.basename(file_path)
            last_snapshots[name] = points
            add_geo_checkbox(name)
            check_geo_vars[name].set(True)
            update_plot()
        else:
            messagebox.showerror("Error", "No se pudo leer el PCD.")

    tk.Button(bottom_btn_frame, text="Importar PCD", command=import_pcd_callback, 
              bg="#9b59b6", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=5)

    tk.Button(bottom_btn_frame, text="Abrir en Ventana Nueva", 
              command=lambda: open_new_plotter_window(df, primary_plot_config), 
              bg="#2ecc71", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=5)
    
    update_plot()
    root.mainloop()

def main():
    file_path = None
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        root = tk.Tk(); root.withdraw()
        file_path = filedialog.askopenfilename()
        root.destroy()
    
    if not file_path: return

    last_snapshots = {}
    if file_path.endswith(".txt"):
        df, last_snapshots = read_can_txt_file(file_path)
    else:
        df, last_snapshots = read_rosbag_mcap(file_path)

    if not df.empty or last_snapshots:
        plot_variables(df, last_snapshots)
    else:
        print("No se encontraron datos.")

if __name__ == "__main__":
    main()