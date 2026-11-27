import math
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.simulated_annealing import (
    create_bins,
    validate_parameters,
    classify_bins,
    get_pickup_bins,
    simulated_annealing,
)


BG = "#eef3f1"
WHITE = "#ffffff"
GREEN = "#236b59"
DARK_GREEN = "#173e35"
GRAY = "#64748b"
ORANGE = "#b45309"


class AnnealingTab(tk.Frame):

    def __init__(self, parent, model, on_route_planned=None):
        super().__init__(parent, bg=BG)

        self.model = model

        self.on_route_planned = on_route_planned
        self.bins = []

        self.editor = None

        self.last_result = None

        self.pickup_count = tk.StringVar(value="--")
        self.best_distance = tk.StringVar(value="--")
        self.iterations = tk.StringVar(value="--")
        self.route_text = tk.StringVar(value="--")

        self.bin_count = tk.StringVar(value="10")
        self.initial_temp = tk.StringVar(value="60")
        self.cooling_rate = tk.StringVar(value="0.997")

        self.status = tk.StringVar(value="Click 'Create Bins' to start")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        self.create_header()
        self.create_settings()
        self.create_table()
        self.create_result()

    def create_header(self):

        frame = tk.Frame(self, bg=DARK_GREEN, padx=8, pady=5)
        frame.grid(row=0, column=0, sticky="ew", padx=8, pady=4)

        tk.Label(
            frame,
            text="Campus Waste Collection",
            font=("Segoe UI", 16, "bold"),
            bg=DARK_GREEN,
            fg=WHITE
        ).pack(anchor="w")

        tk.Label(
            frame,
            text="Route Planning with Simulated Annealing",
            font=("Segoe UI", 10),
            bg=DARK_GREEN,
            fg="#b5d4ca"
        ).pack(anchor="w")

    def create_settings(self):

        frame = tk.Frame(self, bg=WHITE, padx=10, pady=6)
        frame.grid(row=1, column=0, sticky="ew", padx=8, pady=3)

        tk.Label(
            frame,
            text="01. Bins and Settings",
            font=("Segoe UI", 10, "bold"),
            bg=WHITE,
            fg=DARK_GREEN
        ).grid(row=0, column=0, columnspan=7, sticky="w", pady=(0, 4))

        tk.Label(frame, text="Bin Count", bg=WHITE).grid(row=1, column=0, sticky="w", padx=5)
        ttk.Entry(frame, textvariable=self.bin_count, width=8).grid(row=2, column=0, sticky="w", padx=5)

        tk.Button(
            frame,
            text="Create Bins",
            command=self.on_create_bins,
            bg=GREEN, fg=WHITE,
            font=("Segoe UI", 10, "bold"),
            relief="flat", cursor="hand2", width=14
        ).grid(row=2, column=1, padx=(5, 25))

        tk.Label(frame, text="Initial Temperature", bg=WHITE).grid(row=1, column=2, sticky="w", padx=5)
        ttk.Entry(frame, textvariable=self.initial_temp, width=10).grid(row=2, column=2, sticky="w", padx=5)

        tk.Label(frame, text="Cooling Rate", bg=WHITE).grid(row=1, column=3, sticky="w", padx=5)
        ttk.Entry(frame, textvariable=self.cooling_rate, width=10).grid(row=2, column=3, sticky="w", padx=5)

        tk.Button(
            frame,
            text="Plan Route",
            command=self.on_plan_route,
            bg=ORANGE, fg=WHITE,
            font=("Segoe UI", 10, "bold"),
            relief="flat", cursor="hand2", width=14
        ).grid(row=2, column=4, padx=(25, 5))

        tk.Label(
            frame,
            textvariable=self.status,
            bg=WHITE,
            fg=GRAY
        ).grid(row=3, column=0, columnspan=7, sticky="w", padx=5, pady=(6, 0))

    def create_table(self):

        frame = tk.Frame(self, bg=WHITE, padx=10, pady=6)
        frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=3)

        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        tk.Label(
            frame,
            text="02. Bins",
            font=("Segoe UI", 10, "bold"),
            bg=WHITE,
            fg=DARK_GREEN
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))

        self.columns = ("id", "x", "y", "x1", "x2", "x3", "prediction")
        headings = {
            "id": "Bin ID",
            "x": "X (m)",
            "y": "Y (m)",
            "x1": "Current Fill (X1)",
            "x2": "Fill Change (X2)",
            "x3": "Time Since (X3)",
            "prediction": "Prediction",
        }

        self.table = ttk.Treeview(frame, columns=self.columns, show="headings", height=8)

        for col in self.columns:
            self.table.heading(col, text=headings[col])
            self.table.column(col, width=110, anchor="center")

        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)

        self.table.grid(row=1, column=0, sticky="nsew")
        scroll.grid(row=1, column=1, sticky="ns")

        self.table.tag_configure("pickup", foreground=ORANGE)
        self.table.tag_configure("no_pickup", foreground=GREEN)

        self.table.bind("<Double-1>", self.on_double_click)

    def refresh_table(self):
        for row in self.table.get_children():
            self.table.delete(row)

        for index in range(len(self.bins)):
            b = self.bins[index]

            if b["prediction"] == 1:
                prediction_text = "Pickup (+1)"
                tag = "pickup"
            elif b["prediction"] == -1:
                prediction_text = "No Pickup (-1)"
                tag = "no_pickup"
            else:
                prediction_text = "--"
                tag = ""

            self.table.insert(
                "", "end", iid=str(index), tags=(tag,),
                values=(b["id"], b["x"], b["y"], b["x1"], b["x2"], b["x3"], prediction_text)
            )

    def create_result(self):

        frame = tk.Frame(self, bg=WHITE, padx=10, pady=6)
        frame.grid(row=3, column=0, sticky="ew", padx=8, pady=(3, 8))

        for i in range(3):
            frame.columnconfigure(i, weight=1)

        tk.Label(
            frame,
            text="03. Route Result",
            font=("Segoe UI", 10, "bold"),
            bg=WHITE,
            fg=DARK_GREEN
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))

        cards = [
            ("Pickup Bins", self.pickup_count),
            ("Best Distance", self.best_distance),
            ("Iterations", self.iterations),
        ]

        for i in range(len(cards)):
            name, variable = cards[i]

            card = tk.Frame(frame, bg=BG, padx=5, pady=2)
            card.grid(row=1, column=i, sticky="nsew", padx=3, pady=3)

            tk.Label(card, text=name, bg=BG, fg=GRAY).pack()
            tk.Label(
                card,
                textvariable=variable,
                font=("Segoe UI", 14, "bold"),
                bg=BG,
                fg=GREEN
            ).pack(pady=1)

        tk.Label(frame, text="Best Route", bg=WHITE, fg=GRAY).grid(
            row=2, column=0, columnspan=3, sticky="w", padx=3, pady=(4, 0)
        )
        tk.Label(
            frame,
            textvariable=self.route_text,
            bg=WHITE,
            fg=DARK_GREEN,
            font=("Segoe UI", 10, "bold"),
            wraplength=900,
            justify="left"
        ).grid(row=3, column=0, columnspan=3, sticky="w", padx=3)

    def on_double_click(self, event):

        if self.table.identify("region", event.x, event.y) != "cell":
            return

        row_id = self.table.identify_row(event.y)
        col_id = self.table.identify_column(event.x)
        col_name = self.columns[int(col_id[1:]) - 1]

        if col_name not in ("x1", "x2", "x3"):
            return

        self.close_editor()

        x, y, width, height = self.table.bbox(row_id, col_id)
        self.editor = ttk.Entry(self.table, justify="center")
        self.editor.place(x=x, y=y, width=width, height=height)

        index = int(row_id)
        self.editor.insert(0, str(self.bins[index][col_name]))
        self.editor.select_range(0, "end")
        self.editor.focus()

        self.editor.bind("<Return>", lambda e: self.save_edit(index, col_name))
        self.editor.bind("<FocusOut>", lambda e: self.save_edit(index, col_name))
        self.editor.bind("<Escape>", lambda e: self.close_editor())

    def save_edit(self, index, col_name):
        if self.editor is None:
            return

        text = self.editor.get().strip()
        self.close_editor()

        try:
            value = float(text)
        except ValueError:
            messagebox.showerror("Invalid Value", "'%s' is not a number. The old value was kept." % text)
            return
        if not math.isfinite(value):
            messagebox.showerror("Invalid Value", "Please enter a finite number.")
            return

        b = self.bins[index]
        new_inputs = [b["x1"], b["x2"], b["x3"]]
        new_inputs[["x1", "x2", "x3"].index(col_name)] = value
        try:
            self.model.validate_inputs(new_inputs)
        except ValueError as e:
            messagebox.showerror("Invalid Value", str(e) + "\nThe old value was kept.")
            return

        b[col_name] = value
        b["prediction"] = None
        self.refresh_table()
        self.clear_result()
        self.status.set("%s: %s changed to %g" % (b["id"], col_name, value))

    def close_editor(self):
        if self.editor is not None:
            editor = self.editor
            self.editor = None
            editor.destroy()

    def on_create_bins(self):
        self.close_editor()
        try:
            self.bins = create_bins(self.bin_count.get())
        except ValueError as e:
            self.status.set("Bins were not created")
            messagebox.showerror("Invalid Bin Count", str(e))
            return

        self.refresh_table()
        self.clear_result()
        self.status.set("%d bins created" % len(self.bins))

    def clear_result(self):
        self.last_result = None
        self.pickup_count.set("--")
        self.best_distance.set("--")
        self.iterations.set("--")
        self.route_text.set("--")

        self.notify_visualization()

    def notify_visualization(self):
        if self.on_route_planned is not None:
            self.on_route_planned(self.bins, self.last_result)

    def on_plan_route(self):
        self.close_editor()

        try:
            initial_temp, cooling_rate = validate_parameters(
                self.initial_temp.get(), self.cooling_rate.get()
            )
        except ValueError as e:
            self.status.set("Please fix the settings")
            messagebox.showerror("Invalid Setting", str(e))
            return

        if len(self.bins) == 0:
            self.status.set("No bins yet")
            messagebox.showwarning("No Bins", "Click 'Create Bins' first.")
            return

        try:
            classify_bins(self.bins, self.model)
        except ValueError as e:
            self.status.set("Route was not planned")
            messagebox.showerror("Perceptron", str(e))
            return

        self.refresh_table()

        pickup_bins = get_pickup_bins(self.bins)

        self.status.set("Planning route...")
        self.update_idletasks()

        result = simulated_annealing(pickup_bins, initial_temp, cooling_rate)
        self.last_result = result

        self.pickup_count.set("%d of %d" % (len(pickup_bins), len(self.bins)))
        self.best_distance.set("%.1f m" % result["best_distance"])
        self.iterations.set(str(result["iterations"]))

        names = [b[0] for b in result["route_with_depot"]]
        self.route_text.set(" \u2192 ".join(names))

        if len(pickup_bins) == 0:
            self.status.set("No bin needs pickup - the truck stays at the depot")
        else:
            self.status.set("Route planned through %d pickup bin(s)" % len(pickup_bins))

        self.notify_visualization()


if __name__ == "__main__":

    from algorithms.perceptron import Perceptron

    model = Perceptron()
    model.train_and_test()

    root = tk.Tk()
    root.title("Campus Waste Collection")
    root.geometry("1000x680")
    root.configure(bg=BG)

    def print_result(bins, result):
        if result is None:
            print("[visualization] clear the plots (%d bins on the map)" % len(bins))
            return
        print("[visualization] bins:", len(bins),
              "| route:", [b[0] for b in result["route_with_depot"]],
              "| best: %.1f m" % result["best_distance"],
              "| history points:", len(result["current_history"]))

    tab = AnnealingTab(root, model, on_route_planned=print_result)
    tab.pack(fill="both", expand=True)

    root.mainloop()
