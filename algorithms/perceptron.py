from numpy import random
import pandas as pd
from pathlib import Path
import math

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

train_file = DATA_DIR / "waste_bin_perceptron_train (2).csv"
test_file = DATA_DIR / "waste_bin_perceptron_test (1).csv"


def load_data(file_path):
    df = pd.read_csv(file_path)

    if len(df.columns) != 4:
        raise ValueError("The dataset must contain 4 columns")

    if df.columns[3] != "pickup_label":
        raise ValueError("The last column must be pickup_label")

    original_count = len(df)
    if original_count == 0:
        raise ValueError("The dataset is empty")

    df = df.apply(pd.to_numeric, errors="coerce")

    df = df.replace([float("inf"), float("-inf")], float("nan"))

    df = df.dropna()

    df = df[df["pickup_label"].isin([1, -1])]

    if len(df) == 0:
        raise ValueError("No valid data remaining")

    removed_count = original_count - len(df)

    print("Removed rows:", removed_count)
    print("Valid rows:", len(df))

    x = df.to_numpy(dtype=float)
    return x


class Perceptron:

    def __init__(self, Learning_rate=0.01, Epochs=50, Patience=20, seed=42):

        self.Learning_rate = Learning_rate
        self.Epochs = Epochs
        self.Patience = Patience
        self.seed = seed

        self.weights = []
        self.Bias = 0
        self.is_trained = False
        self.training_history = []
        self.training_accuracy = 0


    def initialize_weights(self):

        random.seed(self.seed)

        self.weights = [random.uniform(-1, 1),
                        random.uniform(-1, 1),
                        random.uniform(-1, 1)]

        self.Bias = random.uniform(-1, 1)

    def validate_inputs(self, inputs):
        if len(inputs) != 3:
            raise ValueError("Three inputs are required")

        for value in inputs:
           if not math.isfinite(value):
             raise ValueError("Invalid numeric value")
 
        if not 0 <= inputs[0] <= 1:
           raise ValueError("Current fill must be between 0 and 1")

        if not -1 <= inputs[1] <= 1:
           raise ValueError("Recent fill change must be between -1 and 1")

        if not 0 <= inputs[2] <= 1:
           raise ValueError("Time since collection must be between 0 and 1")

    def predict(self, inputs):

        if not self.is_trained:
            raise ValueError("Train the model first")

        self.validate_inputs(inputs)
        return self.calculate_output(inputs)


    def calculate_output(self, inputs):

        weighted_sum = 0

        for i in range(len(inputs)):
            weighted_sum += inputs[i] * self.weights[i]

        weighted_sum += self.Bias

        if weighted_sum >= 0:
            return 1
        else:
            return -1


    def train(self, file_path=train_file):

        if self.Learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        if self.Epochs <= 0:
            raise ValueError("Epochs must be positive")

        if self.Patience <= 0:
            raise ValueError("Patience must be positive")

        x = load_data(file_path)

        self.is_trained = False
        self.training_history = []
        self.training_accuracy = 0

        self.initialize_weights()

        best_weights = self.weights.copy()
        best_bias = self.Bias
        best_mistakes = len(x) + 1

        no_improvement = 0

        for epoch in range(self.Epochs):

            random.shuffle(x)

            mistake_count = 0

            for i in range(len(x)):

                inputs = x[i, 0:3]
                output = self.calculate_output(inputs)
                error = x[i, 3] - output

                if error != 0:

                    for j in range(len(self.weights)):
                        self.weights[j] += self.Learning_rate * error * inputs[j]

                    self.Bias += self.Learning_rate * error

                    mistake_count += 1

            training_mistakes = 0

            for i in range(len(x)):

                output = self.calculate_output(x[i, 0:3])

                if output != x[i, 3]:
                    training_mistakes += 1

            self.training_history.append(training_mistakes)

            print("Epoch:", epoch + 1,
                  "Mistakes:", training_mistakes)

            if training_mistakes < best_mistakes:

                best_mistakes = training_mistakes
                best_weights = self.weights.copy()
                best_bias = self.Bias

                no_improvement = 0

            else:
                no_improvement += 1

            if training_mistakes == 0:
                break

            if no_improvement >= self.Patience:
                print("Early stopping")
                break

        self.weights = best_weights.copy()
        self.Bias = best_bias

        self.is_trained = True

        self.training_accuracy = (len(x) - best_mistakes) / len(x) * 100

        return self.training_accuracy


    def test(self, file_path=test_file):

        if not self.is_trained:
            raise ValueError("Train the model first")

        x = load_data(file_path)

        number_of_correct_predictions = 0
        number_of_incorrect_predictions = 0

        false_pickup_alerts = 0
        missed_pickup_alerts = 0

        for i in range(len(x)):

            inputs = x[i, 0:3]
            output = self.predict(inputs)

            if output == x[i, 3]:
                number_of_correct_predictions += 1

            else:

                number_of_incorrect_predictions += 1

                if output == 1:
                    false_pickup_alerts += 1
                else:
                    missed_pickup_alerts += 1

        accuracy = number_of_correct_predictions / len(x) * 100

        results = {
            "accuracy": accuracy,
            "false_pickup_alerts": false_pickup_alerts,
            "missed_pickup_alerts": missed_pickup_alerts
        }

        return results


    def train_and_test(self, train_path=train_file, test_path=test_file):

        self.train(train_path)

        results = self.test(test_path)

        return results


if __name__ == "__main__":

    model = Perceptron()

    results = model.train_and_test()

    print("Training Accuracy:", model.training_accuracy, "%")
    print("Test Accuracy:", results["accuracy"], "%")
    print("False pickup alerts:", results["false_pickup_alerts"])
    print("Missed pickups:", results["missed_pickup_alerts"])

    print("Completed Epochs:", len(model.training_history))
    print("Final weights:", [round(float(w), 6) for w in model.weights])
    print("Final Bias:", round(float(model.Bias), 6))
