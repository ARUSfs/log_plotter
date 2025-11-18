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
from sensor_msgs_py import point_cloud2


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


def plot_variables(df, last_snapshots={}):
    root = tk.Tk()
    root.title("Log Plotter - ROS2 & CAN")
    root.geometry("1200x800")
    
    main_frame = tk.Frame(root)
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    
    controls_canvas = tk.Canvas(main_frame, width=350)
    scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=controls_canvas.yview)
    scrollable_frame = tk.Frame(controls_canvas)
    
    scrollable_frame.bind("<Configure>", lambda e: controls_canvas.configure(scrollregion=controls_canvas.bbox("all")))
    controls_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    controls_canvas.configure(yscrollcommand=scrollbar.set)
    
    controls_canvas.pack(side="left", fill="y")
    scrollbar.pack(side="left", fill="y")
    
    
    plot_frame = tk.Frame(main_frame)
    plot_frame.pack(side="right", fill="both", expand=True)
    
    fig, ax = plt.subplots()
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    toolbar = NavigationToolbar2Tk(canvas, plot_frame)
    toolbar.update()
    
    check_vars = {} 
    
    
    def update_plot():
        ax.clear()
        
        x_col = x_axis_var.get()
        if x_col not in df.columns:
            x_data = df["timestamp"]
            x_lbl = "Time (s)"
        else:
            x_data = df[x_col]
            x_lbl = x_col

        has_plots = False
        for col, var in check_vars.items():
            if var.get():
                has_plots = True
                ax.plot(x_data, df[col], label=col)
        
        if has_plots:
            ax.set_xlabel(x_lbl)
            ax.legend()
            ax.grid(True)
            ax.set_aspect('auto') 
        
        canvas.draw()

    
    def plot_snapshot():
        topic = snapshot_selector.get()
        if topic not in last_snapshots: return
            
        msg = last_snapshots[topic]
        ax.clear()
        plotted = False
        
        try:
            
            if "PointCloud2" in str(type(msg)):
                points = list(point_cloud2.read_points(msg, field_names=("x", "y"), skip_nans=True))
                if points:
                    x = [p[0] for p in points]
                    y = [p[1] for p in points]
                    ax.scatter(x, y, s=2, c='black', label=f"Map: {topic}")
                    plotted = True

            
            elif hasattr(msg, 'points') or hasattr(msg, 'poses'):
                plist = getattr(msg, 'points', getattr(msg, 'poses', []))
                x, y = [], []
                if len(plist) > 0:
                    p0 = plist[0]
                    
                    if hasattr(p0, 'x') and hasattr(p0, 'y'):
                        x = [p.x for p in plist]
                        y = [p.y for p in plist]
                    elif hasattr(p0, 'pose'):
                        if hasattr(p0.pose, 'position'):
                            x = [p.pose.position.x for p in plist]
                            y = [p.pose.position.y for p in plist]
                    elif isinstance(p0, (list, tuple)) and len(p0) >= 2:
                        x = [p[0] for p in plist]
                        y = [p[1] for p in plist]

                if x and y:
                    ax.plot(x, y, '-o', markersize=3, label=f"Traj: {topic}")
                    ax.plot(x[0], y[0], 'go', label="Start") 
                    ax.plot(x[-1], y[-1], 'rx', label="End") 
                    plotted = True

            if plotted:
                ax.set_title(f"Snapshot: {topic}")
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m)")
                ax.legend()
                ax.grid(True)
                ax.set_aspect('equal', 'box') 
                canvas.draw()
                
        except Exception as e:
            print(f"Error plot snapshot: {e}")

    
    if last_snapshots:
        snap_frame = tk.LabelFrame(scrollable_frame, text="Map & Trajectory Viewer", font=("Bold"), fg="blue")
        snap_frame.pack(fill="x", padx=5, pady=5)
        
        snapshot_selector = ttk.Combobox(snap_frame, values=list(last_snapshots.keys()), state="readonly")
        snapshot_selector.pack(fill="x", padx=5)
        if list(last_snapshots.keys()): snapshot_selector.current(0)
            
        tk.Button(snap_frame, text="PLOT MAP / CIRCUIT", bg="#ddd", command=plot_snapshot).pack(fill="x", pady=5)

    
    ts_frame = tk.LabelFrame(scrollable_frame, text="Time Series")
    ts_frame.pack(fill="x", padx=5, pady=10)
    
    x_axis_var = tk.StringVar(value="timestamp")
    ttk.Combobox(ts_frame, textvariable=x_axis_var, values=["timestamp"] + list(df.columns)).pack(fill="x", padx=5)

   
    grouped = {}
    for c in df.columns:
        if c == "timestamp": continue
        grp = c.split('.')[0] if '.' in c else "Misc"
        if grp not in grouped: grouped[grp] = []
        grouped[grp].append(c)

    for grp, cols in grouped.items():
        gf = tk.Frame(ts_frame, bd=1, relief="solid")
        gf.pack(fill="x", pady=1)
        vf = tk.Frame(gf)
        
        def toggle(f=vf, b=None, t=grp):
            if f.winfo_viewable():
                f.pack_forget()
                if b: b.config(text=f"▶ {t}")
            else:
                f.pack(fill="x", padx=10)
                if b: b.config(text=f"▼ {t}")

        btn = tk.Button(gf, text=f"▶ {grp}", anchor="w", relief="flat")
        btn.config(command=lambda f=vf, b=btn, t=grp: toggle(f, b, t))
        btn.pack(fill="x")
        
        for c in cols:
            var = tk.BooleanVar()
            tk.Checkbutton(vf, text=c, variable=var, command=update_plot).pack(anchor="w")
            check_vars[c] = var

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
    pattern = re.compile(
        r"\(([\d.]+)\)\s+can\d+\s+([0-9A-Fa-f]+)\s+\[\d+\]\s+((?:[0-9A-Fa-f]{2}\s+)+)"
    )

    can_conversions = pd.read_csv("/home/alvaro/log_plotter/can_conversions.csv", index_col=0)

    print(can_conversions)

    rows = []
    seen_columns = set()

    with open(file_path, "r", encoding="latin1") as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                timestamp = float(match.group(1))
                can_id = str(hex(int(match.group(2), 16)))
                data_str = match.group(3).strip()
                data_bytes = [int(byte, 16) for byte in data_str.split()]

                if can_id in can_conversions.index:
                    bitIn = can_conversions["bitIn"][can_id]
                    bitFin = can_conversions["bitFin"][can_id]
                    value_bytes = [data_bytes[i] for i in range(bitIn, bitFin + 1) if i < len(data_bytes)]
                    if can_conversions["Signed"][can_id] == "False":
                        raw_int = int.from_bytes(value_bytes, byteorder="little", signed=False)
                    else:
                        raw_int = int.from_bytes(value_bytes, byteorder="little", signed=True)
                    row = {"timestamp": timestamp}
                    column_name = can_conversions["Name"][can_id] + " ("+can_id+")"
                    row[column_name] = raw_int*can_conversions["Scale"][can_id] + can_conversions["Offset"][can_id]
                    seen_columns.add(column_name)
                    if len(row) > 1:
                        rows.append(row)

                for i in range(1,4):
                    subid = can_id + str(i)

                    if subid in can_conversions.index:
                        bitIn = can_conversions["bitIn"][subid]
                        bitFin = can_conversions["bitFin"][subid]
                        value_bytes = [data_bytes[i] for i in range(bitIn, bitFin + 1) if i < len(data_bytes)]
                        if can_conversions["Signed"][subid] == "False":
                            raw_int = int.from_bytes(value_bytes, byteorder="little", signed=False)
                        else:
                            raw_int = int.from_bytes(value_bytes, byteorder="little", signed=True)
                        row = {"timestamp": timestamp}
                        column_name = can_conversions["Name"][subid] + " ("+can_id+")"
                        row[column_name] = raw_int*can_conversions["Scale"][subid] + can_conversions["Offset"][subid]
                        seen_columns.add(column_name)
                        if len(row) > 1:
                            rows.append(row)
                    
                
    if rows:
        df = pd.DataFrame(rows)

        for col in seen_columns:
            if col not in df.columns:
                df[col] = np.nan

        df.sort_values("timestamp", inplace=True)
        df.ffill(inplace=True)


    return df


def main():
    import sys

    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        root.destroy()

    if not file_path: return

    last_snapshots = {} 
    
    if file_path.endswith(".txt"):
        df = read_can_txt_file(file_path)
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