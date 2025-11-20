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



def plot_variables(df, last_snapshots={}):
    global formula_counter
    global check_vars
    global safe_map
    global expr_entry
    global name_entry
    
    
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    check_vars = {}
    safe_map = {get_safe_name(c): c for c in df.columns}

    root = tk.Tk()
    root.title("Log Plotter - Stable Wide Layout")

    main_frame = tk.Frame(root)
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)

    main_frame.grid_columnconfigure(0, weight=0)
    main_frame.grid_columnconfigure(1, weight=1)
    main_frame.grid_rowconfigure(0, weight=1)

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

    
    def _on_mousewheel(event):
        controls_canvas.yview_scroll(int(-1 * (event.delta/120)), "units")
        
    def _on_linux_up(event):
        controls_canvas.yview_scroll(-1, "units")
        
    def _on_linux_down(event):
        controls_canvas.yview_scroll(1, "units")

    root.bind_all("<MouseWheel>", _on_mousewheel)
    root.bind_all("<Button-4>", _on_linux_up)
    root.bind_all("<Button-5>", _on_linux_down)
    
    
    plot_frame = tk.Frame(main_frame)
    plot_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
    
    plot_frame.grid_rowconfigure(0, weight=1)
    plot_frame.grid_columnconfigure(0, weight=1)

    fig, ax = plt.subplots()
    
    
    canvas = FigureCanvasTkAgg(fig, master=plot_frame) 
    canvas.draw()
    
    toolbar = NavigationToolbar2Tk(canvas, plot_frame)
    toolbar.update()
    
    canvas.get_tk_widget().pack(fill="both", expand=True)

    def update_plot():
        ax.clear()
        x_col = x_axis_combobox.get()
        
        if x_col in df.columns:
            x_data = df[x_col]; x_lbl = x_col
        else:
            x_data = df["timestamp"]; x_lbl = "Time (s)"
            
        has_plots = False
        for col, var in check_vars.items():
            if var.get() and col in df.columns:
                ax.plot(x_data, df[col], label=col, alpha=0.8, linewidth=1.5)
                has_plots = True
                
        if has_plots:
            ax.legend(loc='upper right', fontsize='small', framealpha=0.9)
            ax.set_aspect('auto')
            ax.set_xlabel(x_lbl)
            if title_entry.get(): ax.set_title(title_entry.get())
            if ylabel_entry.get(): ax.set_ylabel(ylabel_entry.get())
            
        canvas.draw()

    def plot_snapshot():
        global ROS_ENABLED, point_cloud2

        if not ROS_ENABLED:
            tk.messagebox.showinfo("Función Desactivada", "Esta función requiere la librería 'sensor_msgs.msg.point_cloud2'.")
            return
            
        topic = snapshot_selector.get()
        if topic not in last_snapshots:
            return
            
        msg = last_snapshots[topic]
        ax.clear()
        plotted = False
        
        try:
            msg_type_str = str(type(msg))

           
            if "PointCloud2" in msg_type_str and ROS_ENABLED:
                gen = point_cloud2.read_points(msg, field_names=("x", "y"), skip_nans=True)
                points = list(gen)
                if points:
                    x = [p[0] for p in points]; y = [p[1] for p in points]
                    ax.scatter(x, y, s=1.5, c='black', alpha=0.6, label=f'Map: {topic}')
                    plotted = True
            
            
            elif hasattr(msg, 'points') and hasattr(msg, 'y') or hasattr(msg, 'poses'):
                plist = getattr(msg, 'points', []) or getattr(msg, 'poses', [])
                x, y = [], []
                
                if len(plist) > 0:
                    p0 = plist[0]
                    
                    if hasattr(p0, 'x') and hasattr(p0, 'y'):
                        x = [p.x for p in plist]; y = [p.y for p in plist]
                    
                    elif hasattr(p0, 'pose') and hasattr(p0.pose, 'position'):
                        x = [p.pose.position.x for p in plist]; y = [p.pose.position.y for p in plist]
                   
                    elif isinstance(p0, (list, tuple)) and len(p0) >= 2:
                         x = [p[0] for p in plist]; y = [p[1] for p in plist]
                
                if x and y:
                    ax.plot(x, y, marker='o', markersize=3, label=f'Trj: {topic}')
                    ax.plot(x[0], y[0], 'go', ax.plot(x[-1], y[-1], 'rx'))
                    plotted = True

            if plotted:
                ax.set_aspect('equal', 'box'); ax.legend(); ax.grid(True); canvas.draw()

        except Exception as e:
            tk.messagebox.showerror("Error Snapshot", f"Ocurrió un error al procesar el mensaje:\n{e}")
            print(f"Error Snapshot: {e}")

    def insert_text(text):
        idx = expr_entry.index(tk.INSERT)
        expr_entry.insert(idx, text)
        expr_entry.focus()
    
    def add_expression(custom_name):
        global formula_counter
        global safe_map
        global name_entry 

        expr = expr_entry.get().strip()
        if not expr:
            return

        local_vars = {safe_name: df[real_name] for safe_name, real_name in safe_map.items()}
        math_funcs = {"np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan,
                      "sqrt": np.sqrt, "abs": np.abs, "log": np.log, "exp": np.exp}

        try:
            context = {"__builtins__": {}, **local_vars, **math_funcs}
            result_series = eval(expr, {"__builtins__": {}}, context)
            
            if custom_name.strip():
                display_name = custom_name.strip()
            else:
                formula_counter += 1
                display_name = f"Formula_{formula_counter}({expr})"

            df[display_name] = result_series
            safe_map[get_safe_name(display_name)] = display_name
            
            
            add_checkbox_to_group("User Formulas", display_name)
            
            
            expr_entry.delete(0, tk.END)
            name_entry.delete(0, tk.END)
            
            update_plot()

        except NameError as e:
            tk.messagebox.showerror("Error Matemático", f"Variable o función desconocida:\n{e}")
        except Exception as e:
            tk.messagebox.showerror("Error Matemático", f"No se pudo calcular.\nDetalle: {e}")

    
    def add_checkbox_to_group(grp_name, col_name):
        nonlocal groups 

        
        if grp_name not in groups or not isinstance(groups[grp_name], dict):
            
            gf = tk.LabelFrame(scrollable_frame, text=grp_name, bd=1, relief="solid", bg="#Ecf0f1", padx=0, pady=0)
            
            
            vf = tk.Frame(gf, bg="white")
            
            
            groups[grp_name] = {'frame': vf, 'labelframe': gf}
            
            
            btn_frame = tk.Frame(gf, bg="#Ecf0f1")
            btn_frame.pack(side="top", fill="x", pady=0)
            
            
            symbol_var = tk.StringVar(value="►")
            
            def toggle_cmd():
                if vf.winfo_ismapped():
                    vf.pack_forget()
                    symbol_var.set("►")
                else:
                    vf.pack(fill="x", padx=5, pady=(2, 5))
                    symbol_var.set("▼")
            
            
            symbol_lbl = tk.Label(btn_frame, textvariable=symbol_var, anchor="w", bg="#Ecf0f1", font=("Arial", 9, "bold"))
            symbol_lbl.pack(side="left", padx=5, pady=(2, 3))
            symbol_lbl.bind("<Button-1>", lambda e: toggle_cmd())

            
            name_lbl = tk.Label(btn_frame, text=grp_name, anchor="w", relief="flat", bg="#Ecf0f1", font=("Arial", 9, "bold"))
            name_lbl.pack(side="left", fill="x", expand=True, pady=3)
            name_lbl.bind("<Button-1>", lambda e: toggle_cmd())
            
            
            def on_enter(event): btn_frame.config(cursor="hand2")
            def on_leave(event): btn_frame.config(cursor="")
            btn_frame.bind("<Enter>", on_enter)
            btn_frame.bind("<Leave>", on_leave)
            
            
            btn_frame.bind("<Button-1>", lambda e: toggle_cmd())
            
            
            if grp_name == "User Formulas":
                
                groups[grp_name]['labelframe'].pack(fill="x", pady=1)

        
        vf = groups[grp_name]['frame'] 
        
        row_frame = tk.Frame(vf, bg="white")
        row_frame.pack(fill="x")
        
        var = tk.BooleanVar(value=False)
        check_vars[col_name] = var
        
        safe_h = get_safe_name(col_name)

        
        insert_btn = tk.Button(row_frame, text="[+]", font=("Consolas", 8),
                                bg="#eef8f8", fg="green", bd=0,
                                command=lambda h=safe_h: insert_text(h))
        insert_btn.pack(side="left", padx=2)

        
        chk = tk.Checkbutton(row_frame, text=col_name, variable=var, command=update_plot, bg="white", anchor="w")
        chk.pack(side="left", fill="x", expand=True)

        check_vars[col_name] = var
        
   
    if last_snapshots:
        snap_frame = tk.LabelFrame(scrollable_frame, text="Map & Trajectory", font=("Arial", 10, "bold"), fg="#2c3e50", padx=5, pady=5)
        snap_frame.pack(fill="x", pady=5)
        
        global ROS_ENABLED
        if ROS_ENABLED:
            snapshot_topics = list(last_snapshots.keys())
            snapshot_selector = ttk.Combobox(snap_frame, values=snapshot_topics, state="readonly")
            snapshot_selector.pack(fill="x", pady=5)
            
            if snapshot_topics:
                snapshot_selector.set(snapshot_topics[0])
                tk.Button(snap_frame, text="PLOT GEOMETRY", bg="#Ecf0f1", command=plot_snapshot).pack(fill="x", pady=5)
            else:
                tk.Label(snap_frame, text="Función de Mapa Deshabilitada", fg="#e74c3c").pack(padx=5)
        else:
            tk.Label(snap_frame, text="Función de Mapa Deshabilitada", fg="#e74c3c").pack(padx=5)



    math_frame = tk.LabelFrame(scrollable_frame, text="Formula Builder", font=("Arial", 10, "bold"), fg="#2c3e50", padx=5, pady=5)
    math_frame.pack(fill="x", pady=5)

    tk.Label(math_frame, text="1. Select variables below using [+].", anchor="w").pack(padx=5)
    tk.Label(math_frame, text="2. Add operators here: *, /, +, -, sin(), cos(), etc.", anchor="w").pack(padx=5)

   
    name_frame = tk.Frame(math_frame)
    name_frame.pack(fill="x", padx=5, pady=2)
    tk.Label(name_frame, text="Name:").pack(side="left")
    name_entry = tk.Entry(name_frame, font=("Consolas", 11), bg="#fdfeff")
    name_entry.pack(side="left", fill="x", expand=True)

   
    expr_entry = tk.Entry(math_frame, font=("Consolas", 11), bg="#fdfeff")
    expr_entry.pack(fill="x", padx=5, pady=5)
    
    
    tk.Button(math_frame, text="COMPUTE & PLOT", bg="#b3dc7c", font=("Arial", 9, "bold"),
              command=lambda: add_expression(name_entry.get())).pack(fill="x", padx=5, pady=10)


    
    fmt_frame = tk.LabelFrame(scrollable_frame, text="Appearance", padx=5, pady=5)
    fmt_frame.pack(fill="x", pady=5)
    
    f_grid = tk.Frame(fmt_frame)
    f_grid.pack(fill="x", padx=5)
    
   
    tk.Label(f_grid, text="Title:").grid(row=0, column=0, sticky="w")
    title_entry = tk.Entry(f_grid, width=30)
    title_entry.grid(row=0, column=1, sticky="ew")
    
    
    tk.Label(f_grid, text="Y Label:").grid(row=1, column=0, sticky="w")
    ylabel_entry = tk.Entry(f_grid, width=30)
    ylabel_entry.grid(row=1, column=1, sticky="ew")
    
    f_grid.grid_columnconfigure(1, weight=1)
    
    tk.Button(fmt_frame, text="Update Labels", command=update_plot).pack(fill="x", pady=2)
    
    
    ts_frame = tk.LabelFrame(scrollable_frame, text="Variables (Time Series)", font=("Arial", 10, "bold"), padx=5, pady=5)
    ts_frame.pack(fill="x", pady=5)

    xf = tk.Frame(ts_frame)
    xf.pack(fill="x", padx=5)

    tk.Label(xf, text="X-Axis:", font=("Arial", 9, "bold")).pack(side="left")
    
    
    x_axis_var = tk.StringVar(value="timestamp")
    x_axis_options = ["timestamp"] + [col for col in df.columns if col != "timestamp"]
    x_axis_combobox = ttk.Combobox(xf, textvariable=x_axis_var, values=x_axis_options, state="readonly")
    x_axis_combobox.pack(side="left", fill="x", expand=True)
    
    x_axis_combobox.bind("<<ComboboxSelected>>", lambda event: update_plot())

    
    groups = {}

    for col_name in sorted(df.columns):
        if col_name == "timestamp":
            continue

        if '.' in col_name:
            grp = col_name.split('.')[0]
        else:
            grp = "Misc"

        add_checkbox_to_group(grp, col_name)

    
    for grp_name in sorted(groups.keys()):
        
        if grp_name != "User Formulas":
            groups[grp_name]['labelframe'].pack(fill="x", pady=1)

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