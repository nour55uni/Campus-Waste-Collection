import math
import random

DEPOT = ("Depot", 500.0, 500.0)


def euclidean_distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx ** 2 + dy ** 2)


def route_distance(route, depot=DEPOT):
    if len(route) == 0:
        return 0.0

    depot_xy = (depot[1], depot[2])
    total = 0.0

    total += euclidean_distance(depot_xy, (route[0][1], route[0][2]))

    for i in range(len(route) - 1):
        a = (route[i][1], route[i][2])
        b = (route[i + 1][1], route[i + 1][2])
        total += euclidean_distance(a, b)

    total += euclidean_distance((route[-1][1], route[-1][2]), depot_xy)

    return total


def validate_parameters(initial_temp, cooling_rate):
    try:
        initial_temp = float(initial_temp)
    except (TypeError, ValueError):
        raise ValueError("Initial Temperature must be a number.")

    try:
        cooling_rate = float(cooling_rate)
    except (TypeError, ValueError):
        raise ValueError("Cooling Rate must be a number.")

    if not math.isfinite(initial_temp) or not math.isfinite(cooling_rate):
        raise ValueError("Please enter finite numbers.")

    if initial_temp <= 0:
        raise ValueError("Initial Temperature must be greater than 0.")

    if not (0 < cooling_rate < 1):
        raise ValueError("Cooling Rate must be between 0 and 1 (not 0 or 1).")

    return initial_temp, cooling_rate


def swap_neighbor(route, i, j):
    new_route = list(route)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route


def two_opt_neighbor(route, i, j):
    new_route = list(route)
    new_route[i:j + 1] = reversed(new_route[i:j + 1])
    return new_route


def get_neighbor(route):
    if len(route) < 2:
        return list(route)

    i, j = sorted(random.sample(range(len(route)), 2))

    if random.random() < 0.8:
        return two_opt_neighbor(route, i, j)
    else:
        return swap_neighbor(route, i, j)


def acceptance_probability(delta, temperature):
    if delta < 0:
        return 1.0
    return math.exp(-delta / temperature)


def accept(delta, temperature):
    return random.random() < acceptance_probability(delta, temperature)


MIN_TEMP = 0.001

MAX_ITERATIONS = 100000


def simulated_annealing(pickup_bins, initial_temp=60, cooling_rate=0.997,depot=DEPOT, seed=None):
    initial_temp, cooling_rate = validate_parameters(initial_temp, cooling_rate)

    if seed is not None:
        random.seed(seed)
    if len(pickup_bins) <= 1:
        route = list(pickup_bins)
        distance = route_distance(route, depot)
        return {
            "best_route": route,
            "route_with_depot": [depot] + route + [depot],
            "best_distance": distance,
            "current_history": [distance],
            "best_history": [distance],
            "iterations": 0,
        }
    current_route = list(pickup_bins)
    random.shuffle(current_route)

    current_distance = route_distance(current_route, depot)
    best_route = list(current_route)
    best_distance = current_distance

    current_history = [current_distance]
    best_history = [best_distance]

    temperature = initial_temp
    iteration = 0

    while temperature > MIN_TEMP and iteration < MAX_ITERATIONS:
        iteration += 1

        new_route = get_neighbor(current_route)
        new_distance = route_distance(new_route, depot)

        delta = new_distance - current_distance

        if accept(delta, temperature):
            current_route = new_route
            current_distance = new_distance

        if current_distance < best_distance:
            best_route = list(current_route)
            best_distance = current_distance

        current_history.append(current_distance)
        best_history.append(best_distance)

        temperature = temperature * cooling_rate

    return {
        "best_route": best_route,
        "route_with_depot": [depot] + best_route + [depot],
        "best_distance": best_distance,
        "current_history": current_history,
        "best_history": best_history,
        "iterations": iteration,
    }


MAP_SIZE = 1000.0
MARGIN = 30.0


def random_fill(rng):
    if rng.random() < 0.5:
        return round(rng.uniform(0.85, 1.0), 2)
    return round(rng.uniform(0.0, 0.85), 2)


def create_bins(count, seed=None):
    try:
        count = int(count)
    except (TypeError, ValueError):
        raise ValueError("Bin Count must be a whole number.")
    if count < 1:
        raise ValueError("Bin Count must be at least 1.")

    rng = random.Random(seed)
    bins = []
    for k in range(count):
        bins.append({
            "id": "Bin %d" % (k + 1),
            "x": round(rng.uniform(MARGIN, MAP_SIZE - MARGIN), 1),
            "y": round(rng.uniform(MARGIN, MAP_SIZE - MARGIN), 1),
            "x1": random_fill(rng),
            "x2": round(rng.uniform(-0.1, 0.3), 2) + 0.0,
            "x3": round(rng.uniform(0.0, 0.6), 2),
            "prediction": None,
        })
    return bins


def classify_bins(bins, model):
    if not model.is_trained:
        raise ValueError("Train the Perceptron first (Perceptron tab).")

    for b in bins:
        b["prediction"] = model.predict([b["x1"], b["x2"], b["x3"]])
    return bins


def get_pickup_bins(bins):
    pickup = []
    for b in bins:
        if b["prediction"] == 1:
            pickup.append((b["id"], b["x"], b["y"]))
    return pickup


if __name__ == "__main__":
    import time

    passed = 0
    failed = 0

    def check(name, condition):
        global passed, failed
        if condition:
            passed += 1
            print("PASS |", name)
        else:
            failed += 1
            print("FAIL |", name)

    def close(a, b):
        return abs(a - b) < 0.01

    check("distance (0,0)-(3,4) = 5", close(euclidean_distance((0, 0), (3, 4)), 5))
    check("distance (100,100)-(400,500) = 500", close(euclidean_distance((100, 100), (400, 500)), 500))

    bin_a = ("Bin 1", 500.0, 800.0)
    bin_b = ("Bin 2", 800.0, 800.0)
    check("route [] = 0", route_distance([]) == 0)
    check("route [one bin] = 600", close(route_distance([bin_a]), 600))
    check("route [a, b] = 1024.26", close(route_distance([bin_a, bin_b]), 1024.26))
    check("reversed route has the same length",
          close(route_distance([bin_a, bin_b]), route_distance([bin_b, bin_a])))

    check("valid settings accepted", validate_parameters("60", "0.997") == (60.0, 0.997))
    for temp, rate in [("0", "0.997"), ("-5", "0.997"), ("60", "1"), ("60", "0"),
                       ("60", "1.5"), ("abc", "0.997"), ("60", ""), ("inf", "0.997")]:
        try:
            validate_parameters(temp, rate)
            check("rejects temp=%r rate=%r" % (temp, rate), False)
        except ValueError:
            check("rejects temp=%r rate=%r" % (temp, rate), True)

    letters = ["A", "B", "C", "D", "E"]
    check("swap i=1 j=4", swap_neighbor(letters, 1, 4) == ["A", "E", "C", "D", "B"])
    check("2-opt i=1 j=4", two_opt_neighbor(letters, 1, 4) == ["A", "E", "D", "C", "B"])
    check("original route not changed", letters == ["A", "B", "C", "D", "E"])
    random.seed(1)
    all_ok = all(sorted(get_neighbor(letters)) == sorted(letters) for k in range(1000))
    check("1000 random neighbours keep the same bins", all_ok)

    check("better route always accepted", acceptance_probability(-50, 60) == 1.0)
    check("P(T=60, d=10) = 0.846", close(acceptance_probability(10, 60), 0.846))
    check("P(T=1, d=10) almost 0", acceptance_probability(10, 1) < 0.001)

    square = [
        ("Bin 1", 200.0, 200.0), ("Bin 2", 500.0, 200.0), ("Bin 3", 800.0, 200.0),
        ("Bin 4", 800.0, 500.0), ("Bin 5", 800.0, 800.0), ("Bin 6", 500.0, 800.0),
        ("Bin 7", 200.0, 800.0), ("Bin 8", 200.0, 500.0),
    ]
    result = simulated_annealing(square, seed=1)
    check("square: best distance = 2824.26 (the true optimum)", close(result["best_distance"], 2824.26))
    check("square: 3662 iterations", result["iterations"] == 3662)
    check("history lengths = iterations + 1",
          len(result["current_history"]) == len(result["best_history"]) == result["iterations"] + 1)
    check("best distance never goes up",
          all(result["best_history"][k + 1] <= result["best_history"][k]
              for k in range(len(result["best_history"]) - 1)))
    check("best is never worse than the start", result["best_distance"] <= result["current_history"][0])
    check("best_distance matches the returned route",
          close(route_distance(result["best_route"]), result["best_distance"]))
    route = result["route_with_depot"]
    check("route starts and ends at the depot", route[0] == DEPOT and route[-1] == DEPOT)
    check("every bin visited exactly once", sorted(result["best_route"]) == sorted(square))
    check("input list not changed", square[0][0] == "Bin 1" and square[-1][0] == "Bin 8")

    r0 = simulated_annealing([])
    check("0 bins: distance 0, 0 iterations", r0["best_distance"] == 0 and r0["iterations"] == 0)
    check("0 bins: route Depot -> Depot", r0["route_with_depot"] == [DEPOT, DEPOT])
    r1 = simulated_annealing([bin_a])
    check("1 bin: distance 600, 0 iterations", close(r1["best_distance"], 600) and r1["iterations"] == 0)
    try:
        simulated_annealing([], initial_temp=0)
        check("0 bins with bad temperature still raises an error", False)
    except ValueError:
        check("0 bins with bad temperature still raises an error", True)

    rng = random.Random(3)
    big = [("Bin %d" % k, rng.uniform(30, 970), rng.uniform(30, 970)) for k in range(50)]
    start = time.time()
    rb = simulated_annealing(big, seed=1)
    seconds = time.time() - start
    print("INFO | 50 bins: start %.0f m -> best %.0f m in %.2f s"
          % (rb["current_history"][0], rb["best_distance"], seconds))
    check("50 bins finish in under 5 seconds", seconds < 5)

    bins = create_bins(10, seed=1)
    check("create_bins makes 10 bins", len(bins) == 10)
    check("ids are Bin 1 .. Bin 10", [b["id"] for b in bins] == ["Bin %d" % k for k in range(1, 11)])
    check("all locations inside the map",
          all(0 <= b["x"] <= MAP_SIZE and 0 <= b["y"] <= MAP_SIZE for b in bins))
    check("all readings inside the Perceptron ranges",
          all(0 <= b["x1"] <= 1 and -1 <= b["x2"] <= 1 and 0 <= b["x3"] <= 1 for b in bins))
    check("no prediction before classify", all(b["prediction"] is None for b in bins))
    check("same seed -> same bins", create_bins(5, seed=7) == create_bins(5, seed=7))
    for bad in ["abc", "", "0", "-3"]:
        try:
            create_bins(bad)
            check("create_bins rejects %r" % bad, False)
        except ValueError:
            check("create_bins rejects %r" % bad, True)

    class FakeModel:
        is_trained = True

        def predict(self, inputs):
            return 1 if inputs[0] >= 0.8 else -1

    classify_bins(bins, FakeModel())
    check("every bin got +1 or -1", all(b["prediction"] in (1, -1) for b in bins))
    check("fake model rule applied",
          all(b["prediction"] == (1 if b["x1"] >= 0.8 else -1) for b in bins))
    pickup = get_pickup_bins(bins)
    check("only Pickup bins selected",
          len(pickup) == sum(1 for b in bins if b["prediction"] == 1))
    check("pickup bins are (name, x, y)",
          all(len(p) == 3 and isinstance(p[0], str) for p in pickup))

    class UntrainedModel:
        is_trained = False

    try:
        classify_bins(bins, UntrainedModel())
        check("untrained model is refused", False)
    except ValueError:
        check("untrained model is refused", True)

    import io
    import os
    import sys
    import contextlib

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from algorithms.perceptron import Perceptron

    model = Perceptron()
    with contextlib.redirect_stdout(io.StringIO()):
        model.train_and_test()

    bins = create_bins(10, seed=1)
    classify_bins(bins, model)
    pickup = get_pickup_bins(bins)
    full = simulated_annealing(pickup, seed=1)
    print("INFO | real Perceptron:", len(pickup), "of 10 bins need pickup")
    print("INFO | route:", " -> ".join(b[0] for b in full["route_with_depot"]),
          "| %.2f m" % full["best_distance"])
    check("real model: every pickup bin is in the route",
  sorted(full["best_route"]) == sorted(pickup))
    print()
    print("Passed:", passed, "| Failed:", failed)
