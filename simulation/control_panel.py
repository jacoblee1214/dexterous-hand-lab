"""Small Tk control panels that enqueue commands for the MuJoCo thread."""

from __future__ import annotations

from queue import Empty, Queue
import threading


class _TkPanel:
    def __init__(self, title: str) -> None:
        self.title = title
        self.commands: Queue = Queue()
        self.states: Queue = Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_safe, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()

    def publish(self, state: dict) -> None:
        while self.states.full():
            try:
                self.states.get_nowait()
            except Empty:
                break
        self.states.put_nowait(state)

    def _run_safe(self) -> None:
        try:
            self._run()
        except Exception as exc:  # GUI failure must not kill the physics viewer.
            print(f"Control panel unavailable: {exc}")

    def _run(self) -> None:
        raise NotImplementedError


class HandControlPanel(_TkPanel):
    """Basic five-finger and advanced 20-actuator command panel."""

    def __init__(
        self,
        joint_details: list[dict],
        finger_names: list[str],
        reconstruction_controls: bool = False,
        sensor_details: list[dict] | None = None,
    ) -> None:
        super().__init__(
            "MuJoCo Hand + Reconstruction Control"
            if reconstruction_controls
            else "MuJoCo Hand Actuator Control"
        )
        self.joint_details = joint_details
        self.finger_names = finger_names
        self.reconstruction_controls = reconstruction_controls
        self.sensor_details = sensor_details or []

    def _run(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        root.title(self.title)
        root.geometry("1320x980" if self.reconstruction_controls else "520x820")
        root.protocol("WM_DELETE_WINDOW", self.close)

        control_parent = root
        telemetry_parent = root
        if self.reconstruction_controls:
            notebook = ttk.Notebook(root)
            notebook.pack(fill="both", expand=True)
            control_parent = ttk.Frame(notebook)
            telemetry_parent = ttk.Frame(notebook)
            notebook.add(control_parent, text="Grasp + Joint Control")
            notebook.add(telemetry_parent, text="Live Sensor Telemetry")

        basic = ttk.LabelFrame(control_parent, text="Basic actuator control")
        basic.pack(fill="x", padx=8, pady=8)
        for label, command in (
            ("Open Hand", "open"),
            ("Close Hand", "close"),
            ("Reset Hand", "reset"),
        ):
            ttk.Button(
                basic,
                text=label,
                command=lambda value=command: self.commands.put(("hand", value)),
            ).pack(side="left", expand=True, fill="x", padx=3, pady=5)

        fingers = ttk.LabelFrame(control_parent, text="Finger flexion (actuator targets)")
        fingers.pack(fill="x", padx=8, pady=4)
        for finger in self.finger_names:
            row = ttk.Frame(fingers)
            row.pack(fill="x", padx=5, pady=2)
            ttk.Label(row, text=finger, width=14).pack(side="left")
            scale = tk.Scale(
                row,
                from_=0.0,
                to=1.0,
                resolution=0.01,
                orient="horizontal",
                length=330,
                command=lambda value, name=finger: self.commands.put(
                    ("finger", name, float(value))
                ),
            )
            scale.pack(side="left", fill="x", expand=True)

        reconstruction_status = None
        telemetry_items = {}
        selected_sensor_detail = None
        selected_grasp_label = None
        if self.reconstruction_controls:
            reconstruction = ttk.LabelFrame(
                control_parent, text="Tactile reconstruction display"
            )
            reconstruction.pack(fill="x", padx=8, pady=4)
            buttons = (
                ("Sensor surfaces (V)", "V", 0, 0),
                ("Active sensors (X)", "X", 0, 1),
                ("Current points (E)", "E", 0, 2),
                ("Point history (M)", "M", 1, 0),
                ("Ground truth (G)", "G", 1, 1),
                ("Accumulate/Pause (A)", "A", 1, 2),
                ("Color mode (T)", "T", 2, 0),
                ("Save run (L)", "L", 2, 1),
                ("Clear + pause (Z)", "Z", 2, 2),
                ("Fit sphere (U)", "U", 3, 0),
                ("Reset fit (Q)", "Q", 3, 1),
                ("Estimated sphere (W)", "W", 3, 2),
                ("Manual / Auto (0)", "0", 4, 0),
                ("30 mm (6)", "6", 5, 0),
                ("40 mm (7)", "7", 5, 1),
                ("50 mm (8)", "8", 5, 2),
                ("60 mm (9)", "9", 6, 1),
            )
            for label, key, row_index, column_index in buttons:
                ttk.Button(
                    reconstruction,
                    text=label,
                    command=lambda value=key: self.commands.put(
                        ("viewer_key", ord(value))
                    ),
                ).grid(
                    row=row_index,
                    column=column_index,
                    columnspan=1,
                    padx=3,
                    pady=3,
                    sticky="ew",
                )
            for column_index in range(3):
                reconstruction.columnconfigure(column_index, weight=1)
            reconstruction_status = ttk.Label(
                reconstruction,
                text="Waiting for viewer state…",
                justify="left",
            )
            reconstruction_status.grid(
                row=7, column=0, columnspan=3, padx=5, pady=4, sticky="w"
            )

            grasp = ttk.LabelFrame(control_parent, text="Repeatable sphere grasp (actuator targets only)")
            grasp.pack(fill="x", padx=8, pady=4)
            ttk.Button(
                grasp,
                text="Sphere Grasp",
                command=lambda: self.commands.put(("sphere_grasp",)),
            ).grid(row=0, column=0, padx=3, pady=3, sticky="ew")
            for stage in range(1, 5):
                ttk.Button(
                    grasp,
                    text=f"Stage {stage}",
                    command=lambda value=stage: self.commands.put(("grasp_stage", value)),
                ).grid(row=0, column=stage, padx=3, pady=3, sticky="ew")
            preset_name = tk.StringVar(value="sphere_30mm")
            ttk.Label(grasp, text="Preset name").grid(row=1, column=0, padx=3, pady=3)
            ttk.Entry(grasp, textvariable=preset_name, width=24).grid(
                row=1, column=1, columnspan=2, padx=3, pady=3, sticky="ew"
            )
            ttk.Button(
                grasp,
                text="Save Current Grasp Pose",
                command=lambda: self.commands.put(("save_grasp", preset_name.get())),
            ).grid(row=1, column=3, padx=3, pady=3, sticky="ew")
            ttk.Button(
                grasp,
                text="Load Named Preset",
                command=lambda: self.commands.put(("load_grasp", preset_name.get())),
            ).grid(row=1, column=4, padx=3, pady=3, sticky="ew")
            ttk.Button(
                grasp,
                text="Reload YAML",
                command=lambda: self.commands.put(("reload_grasps",)),
            ).grid(row=2, column=4, padx=3, pady=3, sticky="ew")
            selected_grasp_label = ttk.Label(grasp, text="Preset: sphere_30mm")
            selected_grasp_label.grid(row=2, column=0, columnspan=4, padx=4, pady=3, sticky="w")
            for column_index in range(5):
                grasp.columnconfigure(column_index, weight=1)

            telemetry_frame = ttk.LabelFrame(telemetry_parent, text="Live physical sensor telemetry — all 18 channels")
            telemetry_frame.pack(fill="both", padx=8, pady=4)
            columns = (
                "sensor", "finger", "parent", "state", "scalar", "force",
                "pressure", "indentation",
            )
            telemetry_tree = ttk.Treeview(
                telemetry_frame, columns=columns, show="headings", height=9
            )
            headings = {
                "sensor": "Sensor ID",
                "finger": "Finger",
                "parent": "Parent Link",
                "state": "Active",
                "scalar": "Scalar (N)",
                "force": "Normal Force (N)",
                "pressure": "Pressure (Pa)",
                "indentation": "Indentation (m)",
            }
            widths = {
                "sensor": 185, "finger": 70, "parent": 80, "state": 62,
                "scalar": 82, "force": 110, "pressure": 95, "indentation": 105,
            }
            for column in columns:
                telemetry_tree.heading(column, text=headings[column])
                telemetry_tree.column(column, width=widths[column], anchor="center")
            for sensor in self.sensor_details:
                sensor_id = sensor["sensor_id"]
                telemetry_items[sensor_id] = telemetry_tree.insert(
                    "", "end", values=(
                        sensor_id, sensor["finger_id"], sensor["parent_link"],
                        "INACTIVE", "0", "0", "0", "0",
                    )
                )
            telemetry_scroll = ttk.Scrollbar(
                telemetry_frame, orient="vertical", command=telemetry_tree.yview
            )
            telemetry_tree.configure(yscrollcommand=telemetry_scroll.set)
            telemetry_tree.pack(side="left", fill="both", expand=True)
            telemetry_scroll.pack(side="right", fill="y")
            telemetry_tree.bind(
                "<<TreeviewSelect>>",
                lambda _event: (
                    self.commands.put(
                        ("select_sensor", telemetry_tree.item(telemetry_tree.selection()[0], "values")[0])
                    )
                    if telemetry_tree.selection()
                    else None
                ),
            )
            selected_sensor_detail = ttk.Label(
                telemetry_parent,
                text="Select a telemetry row for world pose and estimated contact.",
                justify="left",
                font=("TkFixedFont", 9),
            )
            selected_sensor_detail.pack(fill="x", padx=12, pady=2)

            camera = ttk.LabelFrame(telemetry_parent, text="3D view")
            camera.pack(fill="x", padx=8, pady=4)
            camera_buttons = (
                ("↶ Left", "orbit_left", 0, 0),
                ("↑ Up", "orbit_up", 0, 1),
                ("Right ↷", "orbit_right", 0, 2),
                ("Zoom +", "zoom_in", 1, 0),
                ("↓ Down", "orbit_down", 1, 1),
                ("Zoom −", "zoom_out", 1, 2),
                ("Reset view", "reset", 2, 0),
            )
            for label, action, row_index, column_index in camera_buttons:
                ttk.Button(
                    camera,
                    text=label,
                    command=lambda value=action: self.commands.put(
                        ("camera", value)
                    ),
                ).grid(
                    row=row_index,
                    column=column_index,
                    columnspan=3 if action == "reset" else 1,
                    padx=3,
                    pady=3,
                    sticky="ew",
                )
            for column_index in range(3):
                camera.columnconfigure(column_index, weight=1)
            ttk.Label(
                camera,
                text="Viewer mouse: left-drag orbit · right-drag pan · wheel zoom",
            ).grid(row=3, column=0, columnspan=3, padx=4, pady=(2, 5))

        advanced = ttk.LabelFrame(
            control_parent,
            text=(
                "Grasp tuning — all 20 joint actuator targets (saved by Save Current Grasp Pose)"
                if self.reconstruction_controls
                else "Advanced joint control — all 20 position actuators"
            ),
        )
        advanced.pack(fill="both", expand=True, padx=8, pady=8)
        canvas = tk.Canvas(advanced, highlightthickness=0)
        scrollbar = ttk.Scrollbar(advanced, orient="vertical", command=canvas.yview)
        content = ttk.Frame(canvas)
        content.bind(
            "<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        current_labels = {}
        target_scales = {}
        scale_callbacks = {}
        for details in self.joint_details:
            name = details["name"]
            row = ttk.Frame(content)
            row.pack(fill="x", padx=4, pady=2)
            ttk.Label(row, text=name, width=12).pack(side="left")
            scale = tk.Scale(
                row,
                from_=details["lower"],
                to=details["upper"],
                resolution=0.01,
                orient="horizontal",
                length=260,
            )
            scale.set(details["target"])
            # Installing the callback after set() prevents Tk from injecting a
            # synthetic actuator command while the panel is being constructed.
            callback = lambda value, joint=name: self.commands.put(
                ("joint", joint, float(value))
            )
            scale.configure(command=callback)
            scale.pack(side="left")
            label = ttk.Label(row, text=f"{details['angle']:+.3f}", width=8)
            label.pack(side="left")
            current_labels[name] = label
            target_scales[name] = scale
            scale_callbacks[name] = callback
            ttk.Label(
                content,
                text=f"range [{details['lower']:+.3f}, {details['upper']:+.3f}] rad",
            ).pack(anchor="w", padx=18)

        def refresh() -> None:
            try:
                while True:
                    state = self.states.get_nowait()
                    for name, angle in state.get("joint_angles", {}).items():
                        if name in current_labels:
                            current_labels[name].configure(text=f"{angle:+.3f}")
                    for name, target in state.get("joint_targets", {}).items():
                        scale = target_scales.get(name)
                        if scale is not None and abs(float(scale.get()) - target) > 0.004:
                            # Synchronize preset/staged actuator targets without
                            # generating a new command back into the simulator.
                            scale.configure(command="")
                            scale.set(target)
                            scale.configure(command=scale_callbacks[name])
                    reconstruction_state = state.get("reconstruction")
                    if reconstruction_status is not None and reconstruction_state:
                        reconstruction_status.configure(
                            text=(
                                f"Physical Sensor Channels: {reconstruction_state['physical_sensor_count']}  |  "
                                f"Currently Active: {reconstruction_state['currently_active_count']}  |  "
                                f"Unique Active: {reconstruction_state['unique_active_sensor_count']}  |  "
                                f"Unique Fingers in Contact: {reconstruction_state['active_finger_count']}\n"
                                f"Accumulated Estimated Points: {reconstruction_state['point_count']}  |  "
                                f"Accepted Points: {reconstruction_state['point_count']}  |  "
                                f"Rejected Duplicate Points: {reconstruction_state['duplicate_count']}\n"
                                f"Contributing Sensors/Fingers: "
                                f"{reconstruction_state['unique_contributing_sensor_count']}/"
                                f"{reconstruction_state['unique_contributing_finger_count']}  |  "
                                f"Spatial Spread: {reconstruction_state['spatial_spread_m']:.5f} m\n"
                                f"Sphere Reconstruction Status: {reconstruction_state['fit_status']}  |  "
                                f"Fit Mode: {reconstruction_state['fit_mode']}  |  "
                                f"Radius: {reconstruction_state['sphere_radius_mm']:.0f} mm\n"
                                f"Display S/A/C/H/E/GT: {reconstruction_state['surfaces']}/"
                                f"{reconstruction_state['active']}/{reconstruction_state['current']}/"
                                f"{reconstruction_state['history']}/{reconstruction_state['estimated_sphere']}/"
                                f"{reconstruction_state['ground_truth']}  |  "
                                f"Accumulation: {reconstruction_state['accumulation']}"
                            )
                        )
                        if selected_grasp_label is not None:
                            selected_grasp_label.configure(
                                text=(
                                    f"Preset: {reconstruction_state['grasp_preset']}  |  "
                                    f"stage={reconstruction_state['grasp_stage']}/4  |  "
                                    f"{'RUNNING' if reconstruction_state['grasp_running'] else 'IDLE'}"
                                )
                            )
                        telemetry_by_id = {
                            row["sensor_id"]: row
                            for row in reconstruction_state.get("sensor_telemetry", [])
                        }
                        for sensor_id, row in telemetry_by_id.items():
                            item = telemetry_items.get(sensor_id)
                            if item:
                                telemetry_tree.item(
                                    item,
                                    values=(
                                        sensor_id,
                                        row["finger_id"],
                                        row["parent_link"],
                                        "ACTIVE" if row["active"] else "inactive",
                                        f"{row['scalar_value_n']:.5f}",
                                        f"{row['normal_force_n']:.5f}",
                                        f"{row['pressure_pa']:.1f}",
                                        f"{row['indentation_m']:.7f}",
                                    ),
                                    tags=("active",) if row["active"] else (),
                                )
                        telemetry_tree.tag_configure("active", background="#ffb0a8")
                        selected_id = reconstruction_state.get("selected_sensor_id")
                        selected = telemetry_by_id.get(selected_id)
                        if selected_sensor_detail is not None and selected:
                            selected_sensor_detail.configure(
                                text=(
                                    f"Selected: {selected_id}\n"
                                    f"Position world/base: {selected['position_xyz']}\n"
                                    f"Sensing direction world/base: {selected['sensing_direction_xyz']}\n"
                                    f"Estimated 3D contact: "
                                    f"{selected['estimated_contact_xyz'] if selected['estimated_contact_xyz'] is not None else 'none'}"
                                )
                            )
            except Empty:
                pass
            if self._stop.is_set():
                root.destroy()
                return
            root.after(80, refresh)

        root.after(80, refresh)
        root.mainloop()


class CalibrationControlPanel(_TkPanel):
    """Manual surface placement/orientation/size editor."""

    def __init__(self, sensor_names: list[str], parent_links: list[str]) -> None:
        super().__init__("Sensor Mount Calibration")
        self.sensor_names = sensor_names
        self.parent_links = parent_links

    def _run(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        root.title(self.title)
        root.geometry("580x790")
        root.protocol("WM_DELETE_WINDOW", self.close)

        select = ttk.LabelFrame(root, text="Selection")
        select.pack(fill="x", padx=8, pady=8)
        ttk.Label(select, text="Link").grid(row=0, column=0, padx=4, pady=4)
        link_value = tk.StringVar(value=self.parent_links[0])
        link_box = ttk.Combobox(
            select, textvariable=link_value, values=self.parent_links, state="readonly", width=34
        )
        link_box.grid(row=0, column=1, padx=4, pady=4)
        link_box.bind(
            "<<ComboboxSelected>>",
            lambda _event: self.commands.put(("select_link", link_value.get())),
        )
        ttk.Label(select, text="Sensor").grid(row=1, column=0, padx=4, pady=4)
        sensor_value = tk.StringVar(value=self.sensor_names[0])
        sensor_box = ttk.Combobox(
            select, textvariable=sensor_value, values=self.sensor_names, state="readonly", width=34
        )
        sensor_box.grid(row=1, column=1, padx=4, pady=4)
        sensor_box.bind(
            "<<ComboboxSelected>>",
            lambda _event: self.commands.put(("select_sensor", sensor_value.get())),
        )

        camera = ttk.LabelFrame(root, text="3D hand view (mouse control is also enabled)")
        camera.pack(fill="x", padx=8, pady=5)
        camera_buttons = (
            ("↶ Left", "orbit_left", 0, 0),
            ("Right ↷", "orbit_right", 0, 2),
            ("↑ Up", "orbit_up", 0, 1),
            ("↓ Down", "orbit_down", 1, 1),
            ("Zoom +", "zoom_in", 1, 0),
            ("Zoom −", "zoom_out", 1, 2),
            ("Focus selected sensor", "focus_selected", 2, 0),
            ("Reset view", "reset", 2, 2),
        )
        for label, action, row, column in camera_buttons:
            button = ttk.Button(
                camera,
                text=label,
                command=lambda value=action: self.commands.put(("camera", value)),
            )
            button.grid(
                row=row,
                column=column,
                padx=3,
                pady=3,
                sticky="ew",
                columnspan=2 if action == "focus_selected" else 1,
            )
        for column in range(3):
            camera.columnconfigure(column, weight=1)
        ttk.Label(
            camera,
            text="Viewer mouse: left-drag orbit · right-drag pan · wheel zoom",
        ).grid(row=3, column=0, columnspan=3, padx=4, pady=(2, 5))

        center = ttk.LabelFrame(root, text="Place center in parent-link coordinates (0.5 mm)")
        center.pack(fill="x", padx=8, pady=5)
        for index, axis in enumerate("XYZ"):
            ttk.Label(center, text=axis, width=4).grid(row=index, column=0, padx=4, pady=3)
            ttk.Button(
                center,
                text="−",
                command=lambda i=index: self.commands.put(("move_center", i, -0.0005)),
            ).grid(row=index, column=1, padx=3)
            ttk.Button(
                center,
                text="+",
                command=lambda i=index: self.commands.put(("move_center", i, 0.0005)),
            ).grid(row=index, column=2, padx=3)

        orientation = ttk.LabelFrame(root, text="Adjust surface orientation in link frame (2°)")
        orientation.pack(fill="x", padx=8, pady=5)
        for index, axis in enumerate("XYZ"):
            ttk.Label(orientation, text=f"Rotate {axis}", width=12).grid(
                row=index, column=0, padx=4, pady=3
            )
            ttk.Button(
                orientation,
                text="−2°",
                command=lambda i=index: self.commands.put(("rotate", i, -2.0)),
            ).grid(row=index, column=1, padx=3)
            ttk.Button(
                orientation,
                text="+2°",
                command=lambda i=index: self.commands.put(("rotate", i, 2.0)),
            ).grid(row=index, column=2, padx=3)

        dimensions = ttk.LabelFrame(root, text="Finite sensing-region size (0.5 mm)")
        dimensions.pack(fill="x", padx=8, pady=5)
        for index, dimension in enumerate(("width", "height")):
            ttk.Label(dimensions, text=dimension.title(), width=12).grid(
                row=index, column=0, padx=4, pady=3
            )
            ttk.Button(
                dimensions,
                text="−",
                command=lambda name=dimension: self.commands.put(("resize", name, -0.0005)),
            ).grid(row=index, column=1, padx=3)
            ttk.Button(
                dimensions,
                text="+",
                command=lambda name=dimension: self.commands.put(("resize", name, 0.0005)),
            ).grid(row=index, column=2, padx=3)

        actions = ttk.Frame(root)
        actions.pack(fill="x", padx=8, pady=8)
        ttk.Button(
            actions, text="Flip Normal", command=lambda: self.commands.put(("flip_normal",))
        ).pack(side="left", expand=True, fill="x", padx=3)
        ttk.Button(
            actions, text="Save sensor_config.yaml", command=lambda: self.commands.put(("save",))
        ).pack(side="left", expand=True, fill="x", padx=3)

        status = ttk.Label(root, text="", justify="left", font=("TkFixedFont", 9))
        status.pack(fill="both", expand=True, padx=10, pady=8)

        def refresh() -> None:
            try:
                while True:
                    state = self.states.get_nowait()
                    sensor_value.set(state["sensor_id"])
                    link_value.set(state["parent_link"])
                    status.configure(
                        text=(
                            f"type: {state['sensor_type']}\n"
                            f"center_xyz: {state['center_xyz']}\n"
                            f"surface_u_axis: {state['surface_u_axis']}\n"
                            f"surface_v_axis: {state['surface_v_axis']}\n"
                            f"outward_normal: {state['outward_normal']}\n"
                            f"width: {state['width']:.6f} m\n"
                            f"height: {state['height']:.6f} m\n\n"
                            "Changes are live in the cyan overlay. Save explicitly to persist."
                        )
                    )
            except Empty:
                pass
            if self._stop.is_set():
                root.destroy()
                return
            root.after(80, refresh)

        root.after(80, refresh)
        root.mainloop()
