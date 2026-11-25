import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import math
import pandas as pd

from algorithms.perceptron import Perceptron, train_file, test_file


# Colors
BG = "#eef3f1"
WHITE = "#ffffff"
GREEN = "#236b59"
DARK_GREEN = "#173e35"
GRAY = "#64748b"
ORANGE = "#b45309"


class PerceptronTab(tk.Frame):

    def __init__(self, parent, model):
        super().__init__(parent, bg=BG)

        self.model = model

        # Dataset paths
        self.train_path = str(train_file)
        self.test_path = str(test_file)

        # Training settings
        self.Learning_rate = tk.StringVar(value=str(model.Learning_rate))
        self.Epochs = tk.StringVar(value=str(model.Epochs))
        self.Patience = tk.StringVar(value=str(model.Patience))
        self.seed = tk.StringVar(value=str(model.seed))

        # Evaluation results
        self.training_accuracy = tk.StringVar(value="--")
        self.test_accuracy = tk.StringVar(value="--")
        self.missed_pickups = tk.StringVar(value="--")
        self.false_alerts = tk.StringVar(value="--")
        self.completed_epochs = tk.StringVar(value="--")

        # Prediction results
        self.prediction = tk.StringVar(value="--")
        self.fill_status = tk.StringVar(value="--")
        self.status = tk.StringVar(value="Ready to train")

        # Input fields
        self.inputs = []

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        self.create_header()
        self.create_settings()
        self.create_results()
        self.create_prediction()


    def create_header(self):

        frame = tk.Frame(self, bg=DARK_GREEN, padx=8, pady=5)
        frame.grid(row=0, column=0, columnspan=2,
                   sticky="ew", padx=8, pady=4)

        tk.Label(
            frame,
            text="Campus Waste Collection",
            font=("Segoe UI", 16, "bold"),
            bg=DARK_GREEN,
            fg=WHITE
        ).pack(anchor="w")

        tk.Label(
            frame,
            text="Perceptron Classification",
            font=("Segoe UI", 10),
            bg=DARK_GREEN,
            fg="#b5d4ca"
        ).pack(anchor="w")


    def create_settings(self):

        frame = tk.Frame(self, bg=WHITE, padx=10, pady=6)
        frame.grid(row=1, column=0, sticky="nsew",
                   padx=(8, 4), pady=3)

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        tk.Label(
            frame,
            text="01. Model Settings",
            font=("Segoe UI", 10, "bold"),
            bg=WHITE,
            fg=DARK_GREEN
        ).grid(row=0, column=0, columnspan=2,
               sticky="w", pady=(0, 4))

        tk.Label(
            frame, text="Training Dataset", bg=WHITE
        ).grid(row=1, column=0, sticky="w")

        self.train_name = tk.StringVar(
            value=Path(self.train_path).name
        )

        tk.Label(
            frame,
            textvariable=self.train_name,
            bg=WHITE,
            fg=GRAY,
            wraplength=260
        ).grid(row=2, column=0, sticky="w")

        ttk.Button(
            frame,
            text="Browse",
            command=self.select_train_file
        ).grid(row=2, column=1, sticky="e")

        tk.Label(
            frame, text="Testing Dataset", bg=WHITE
        ).grid(row=3, column=0, sticky="w", pady=(4, 0))

        self.test_name = tk.StringVar(
            value=Path(self.test_path).name
        )

        tk.Label(
            frame,
            textvariable=self.test_name,
            bg=WHITE,
            fg=GRAY,
            wraplength=260
        ).grid(row=4, column=0, sticky="w")

        ttk.Button(
            frame,
            text="Browse",
            command=self.select_test_file
        ).grid(row=4, column=1, sticky="e")

        ttk.Separator(frame).grid(
            row=5, column=0, columnspan=2,
            sticky="ew", pady=4
        )

        settings = [
            ("Learning Rate", self.Learning_rate),
            ("Maximum Epochs", self.Epochs),
            ("Patience", self.Patience),
            ("Random Seed", self.seed)
        ]

        for i in range(len(settings)):

            name, variable = settings[i]

            row = 6 + (i // 2) * 2
            column = i % 2

            tk.Label(
                frame, text=name, bg=WHITE
            ).grid(row=row, column=column,
                   sticky="w", padx=5, pady=3)

            ttk.Entry(
                frame,
                textvariable=variable,
                width=15
            ).grid(row=row + 1, column=column,
                   sticky="ew", padx=5, pady=3)

        self.train_button = tk.Button(
            frame,
            text="Train and Test",
            command=self.train_and_test,
            bg=GREEN,
            fg=WHITE,
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            cursor="hand2",
            pady=2,
            width=20
        )

        self.train_button.grid(
            row=10, column=0, columnspan=2,
            sticky="ew",pady=(5, 2)
        )

        tk.Label(
            frame,
            textvariable=self.status,
            bg=WHITE,
            fg=GRAY
        ).grid(row=11, column=0, columnspan=2, pady=1)


    def create_results(self):

        frame = tk.Frame(self, bg=WHITE, padx=10, pady=6)
        frame.grid(row=1, column=1, sticky="nsew",
                   padx=(4, 8), pady=3)

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        tk.Label(
            frame,
            text="02. Model Performance",
            font=("Segoe UI", 10, "bold"),
            bg=WHITE,
            fg=DARK_GREEN
        ).grid(row=0, column=0, columnspan=2,
               sticky="w", pady=(0, 4))

        tk.Label(
            frame,
            text="Test Accuracy",
            font=("Segoe UI", 10),
            bg=WHITE,
            fg=GRAY
        ).grid(row=1, column=0, columnspan=2)

        tk.Label(
            frame,
            textvariable=self.test_accuracy,
            font=("Segoe UI", 21, "bold"),
            bg=WHITE,
            fg=GREEN
        ).grid(row=2, column=0, columnspan=2)

        self.accuracy_bar = ttk.Progressbar(
            frame,
            maximum=100,
            mode="determinate"
        )

        self.accuracy_bar.grid(
            row=3, column=0, columnspan=2,
            sticky="ew", pady=4
        )

        results = [
            ("Training Accuracy", self.training_accuracy),
            ("Completed Epochs", self.completed_epochs),
            ("Missed Pickups", self.missed_pickups),
            ("False Pickup Alerts", self.false_alerts)
        ]

        for i in range(len(results)):

            name, variable = results[i]

            card = tk.Frame(
                frame, bg=BG, padx=5, pady=2
            )

            card.grid(
                row=4 + i // 2,
                column=i % 2,
                sticky="nsew",
                padx=3, pady=3
            )

            tk.Label(
                card, text=name,
                bg=BG, fg=GRAY
            ).pack()

            color = GREEN

            if i >= 2:
                color = ORANGE

            tk.Label(
                card,
                textvariable=variable,
                font=("Segoe UI", 15, "bold"),
                bg=BG,
                fg=color
            ).pack(pady=1)

    def create_prediction(self):

      frame = tk.Frame(self, bg=WHITE, padx=10, pady=6)

      frame.grid(
        row=2, column=0, columnspan=2,
        sticky="ew", padx=8, pady=4
    )

      tk.Label(
        frame,
        text="03. Predict Waste Bin",
        font=("Segoe UI", 10, "bold"),
        bg=WHITE,
        fg=DARK_GREEN
    ).grid(row=0, column=0, columnspan=3,
           sticky="w", pady=(0, 4))

      names = [
        "Current Fill (X1)",
        "Recent Fill Change (X2)",
        "Time Since Collection (X3)"
    ]

      for i in range(3):

        frame.columnconfigure(i, weight=1)

        tk.Label(
            frame,
            text=names[i],
            bg=WHITE
        ).grid(row=1, column=i, sticky="w", padx=5)

        entry = ttk.Entry(frame, width=18)

        entry.grid(
            row=2, column=i, sticky="ew",
            padx=5, pady=2
        )

        self.inputs.append(entry)

      ttk.Separator(frame, orient="horizontal").grid(
        row=3, column=0, columnspan=3,
        sticky="ew", padx=5, pady=5
    )

      tk.Button(
        frame,
        text="Predict",
        command=self.predict_bin,
        bg=GREEN,
        fg=WHITE,
        font=("Segoe UI", 10, "bold"),
        relief="flat",
        cursor="hand2",
        pady=2,
        width=16
    ).grid(
        row=4, column=0, rowspan=3, pady=3
    )

      tk.Label(
        frame, text="Current Fill Status",
        bg=WHITE, fg=GRAY
    ).grid(row=4, column=1, pady=1)

      tk.Label(
        frame,
        textvariable=self.fill_status,
        font=("Segoe UI", 15, "bold"),
        bg=WHITE,
        fg=GREEN
    ).grid(row=5, column=1)

      self.fill_bar = ttk.Progressbar(
        frame, maximum=100
    )

      self.fill_bar.grid(
        row=6, column=1,
        sticky="ew", padx=10, pady=2
    )

      tk.Label(
        frame, text="Prediction",
        bg=WHITE, fg=GRAY
    ).grid(row=4, column=2)

      self.prediction_label = tk.Label(
        frame,
        textvariable=self.prediction,
        font=("Segoe UI", 15, "bold"),
        bg=WHITE,
        fg=DARK_GREEN
    )

      self.prediction_label.grid(
        row=5, column=2
    )        

    def select_train_file(self):

        path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv")]
        )

        if path:
            self.train_path = path
            self.train_name.set(Path(path).name)


    def select_test_file(self):

        path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv")]
        )

        if path:
            self.test_path = path
            self.test_name.set(Path(path).name)


    def train_and_test(self):

        try:

            learning_rate = float(self.Learning_rate.get())
            epochs = int(self.Epochs.get())
            patience = int(self.Patience.get())

            seed_text = self.seed.get().strip()

            if seed_text == "":
                seed = None
            else:
                seed = int(seed_text)

            if not math.isfinite(learning_rate) or learning_rate <= 0:
                raise ValueError("Learning rate must be positive")

            if epochs <= 0 or patience <= 0:
                raise ValueError("Epochs and patience must be positive")

            if seed is not None and not 0 <= seed < 2**32:
                raise ValueError("Invalid random seed")

            if Path(self.train_path).resolve() == Path(self.test_path).resolve():
                raise ValueError("Use different training and testing files")

            train_columns = list(
                pd.read_csv(self.train_path, nrows=0).columns
            )

            test_columns = list(
                pd.read_csv(self.test_path, nrows=0).columns
            )

            expected_columns = [
                "x1_current_fill",
                "x2_recent_fill_change",
                "x3_time_since_collection",
                "pickup_label"
            ]

            if [c.lower() for c in train_columns] != expected_columns:
                raise ValueError("Invalid training dataset columns")

            if train_columns != test_columns:
                raise ValueError("Training and testing columns must match")

            self.model.Learning_rate = learning_rate
            self.model.Epochs = epochs
            self.model.Patience = patience
            self.model.seed = seed

            self.status.set("Training...")
            self.train_button.config(state="disabled")
            self.update_idletasks()

            results = self.model.train_and_test(
                self.train_path,
                self.test_path
            )

            accuracy = results["accuracy"]
            missed = results["missed_pickup_alerts"]

            self.test_accuracy.set(f"{accuracy:.2f}%")

            self.training_accuracy.set(
                f"{self.model.training_accuracy:.2f}%"
            )

            self.missed_pickups.set(str(missed))

            self.false_alerts.set(
                str(results["false_pickup_alerts"])
            )

            completed = len(self.model.training_history)

            self.completed_epochs.set(f"{completed} / {epochs}")

            self.accuracy_bar["value"] = accuracy

            self.prediction.set("--")
            self.fill_status.set("--")
            self.fill_bar["value"] = 0

            self.status.set("Training completed successfully")

        except Exception as e:

            self.status.set("Training or testing failed")
            messagebox.showerror("Error", str(e))

        finally:

            self.train_button.config(state="normal")


    def predict_bin(self):

        try:

            if not self.model.is_trained:
                raise ValueError("Train the model first")

            inputs = []

            for entry in self.inputs:

                value = float(entry.get())

                if not math.isfinite(value):
                    raise ValueError("Invalid input value")

                inputs.append(value)

            if not 0 <= inputs[0] <= 1:
                raise ValueError("Current fill must be between 0 and 1")

            output = self.model.predict(inputs)

            if output == 1:

                self.prediction.set("Pickup (+1)")
                self.prediction_label.config(fg=ORANGE)

            else:

                self.prediction.set("No Pickup (-1)")
                self.prediction_label.config(fg=GREEN)

            fill_percentage = inputs[0] * 100

            self.fill_status.set(f"{fill_percentage:.1f}%")
            self.fill_bar["value"] = fill_percentage

        except Exception as e:

            messagebox.showerror("Prediction Error", str(e))


if __name__ == "__main__":

    root = tk.Tk()

    root.title("Campus Waste Collection")
    root.geometry("1000x580")
    root.configure(bg=BG)

    model = Perceptron()

    tab = PerceptronTab(root, model)
    tab.pack(fill="both", expand=True)

    root.mainloop()
