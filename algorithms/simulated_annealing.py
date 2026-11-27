import math
import random

DEPOT = ("Depot", 500.0, 500.0)


def euclidean_distance(p1, p2):
    """Straight-line distance in metres between two points (x, y)."""
    dx = p2[0] - p1[0]          # difference in x
    dy = p2[1] - p1[1]          # difference in y
    return math.sqrt(dx ** 2 + dy ** 2)

def route_distance(route, depot=DEPOT):
    """
    Total length (m) of the closed route:
        Depot -> route[0] -> route[1] -> ... -> route[-1] -> Depot
    route: list of bins, each bin is (name, x, y). The depot is NOT inside it.
    """
    # No bins -> the truck does not move
    if len(route) == 0:
        return 0.0

    depot_xy = (depot[1], depot[2])
    total = 0.0

    # 1) from the depot to the first bin
    total += euclidean_distance(depot_xy, (route[0][1], route[0][2]))

    # 2) between every two consecutive bins
    for i in range(len(route) - 1):
        a = (route[i][1], route[i][2])
        b = (route[i + 1][1], route[i + 1][2])
        total += euclidean_distance(a, b)

    # 3) from the last bin back to the depot
    total += euclidean_distance((route[-1][1], route[-1][2]), depot_xy)

    return total


def validate_parameters(initial_temp, cooling_rate):
    """
    Check the SA settings. Raise ValueError with a clear message if wrong.
        initial_temp  must be a number > 0
        cooling_rate  must be a number between 0 and 1 (exclusive)
    Returns the two values as floats.
    """
    # 1) Convert to numbers (the GUI gives us text from the entry boxes)
    try:
        initial_temp = float(initial_temp)
    except (TypeError, ValueError):
        raise ValueError("Initial Temperature must be a number.")

    try:
        cooling_rate = float(cooling_rate)
    except (TypeError, ValueError):
        raise ValueError("Cooling Rate must be a number.")

    # 2) Reject inf and nan (float("inf") is accepted by float() but not valid here)
    if not math.isfinite(initial_temp) or not math.isfinite(cooling_rate):
        raise ValueError("Please enter finite numbers.")

    # 3) Check the allowed ranges
    if initial_temp <= 0:
        raise ValueError("Initial Temperature must be greater than 0.")

    if not (0 < cooling_rate < 1):
        raise ValueError("Cooling Rate must be between 0 and 1 (not 0 or 1).")

    return initial_temp, cooling_rate


def swap_neighbor(route, i, j):
    """Return a NEW route where the bins at positions i and j are swapped."""
    new_route = list(route)                      # copy -> the original stays unchanged
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route


def two_opt_neighbor(route, i, j):
    """Return a NEW route where the part from position i to j is reversed."""
    new_route = list(route)                      # copy -> the original stays unchanged
    new_route[i:j + 1] = reversed(new_route[i:j + 1])
    return new_route


def get_neighbor(route):
    """
    Make a random neighbour of the route:
      - choose two different positions i < j
      - 80% of the time use 2-opt (reverse), 20% use swap
    The depot is not inside the route, so it never moves.
    """
    # With 0 or 1 bins there is nothing to change
    if len(route) < 2:
        return list(route)

    # pick two DIFFERENT positions and sort them so that i < j
    i, j = sorted(random.sample(range(len(route)), 2))

    if random.random() < 0.8:
        return two_opt_neighbor(route, i, j)
    else:
        return swap_neighbor(route, i, j)

def acceptance_probability(delta, temperature):
    """
    Probability of moving to the new route (Metropolis criterion).
        delta = new_distance - current_distance
        better route (delta < 0)        -> 1.0 (always accept)
        worse or equal (delta >= 0)     -> e^(-delta / T)
    """
    if delta < 0:
        return 1.0
    return math.exp(-delta / temperature)


def accept(delta, temperature):
    """Decide randomly: True = move to the new route, False = stay."""
    return random.random() < acceptance_probability(delta, temperature)


# The loop stops when the temperature drops below this value
MIN_TEMP = 0.001

# Safety limit: with a cooling rate very close to 1 (e.g. 0.99999)
# the loop would need over a million iterations
MAX_ITERATIONS = 100000


def simulated_annealing(pickup_bins, initial_temp=60, cooling_rate=0.997,depot=DEPOT, seed=None):
    """
    Find a short closed route: Depot -> all pickup bins -> Depot.
    pickup_bins: list of bins (name, x, y), ONLY the bins predicted as Pickup.
    seed: optional number to get the same result every run (for testing).

    Returns a dictionary:
        best_route        -> the bins in the best order (without the depot)
        route_with_depot  -> [depot] + best_route + [depot]  (ready for drawing)
        best_distance     -> length of the best route in metres
        current_history   -> current distance at every iteration
        best_history      -> best distance so far at every iteration
        iterations        -> how many iterations were done
    """
    # 1) Check the settings (raises ValueError if wrong)
    initial_temp, cooling_rate = validate_parameters(initial_temp, cooling_rate)

    if seed is not None:
        random.seed(seed)
    # Special cases: 0 or 1 pickup bins -> only ONE possible route,
    # so there is nothing to optimise. Return it directly.
    #   0 bins -> Depot -> Depot              (distance 0)
    #   1 bin  -> Depot -> Bin -> Depot       (there and back)
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
    # 2) Start with a random route
    current_route = list(pickup_bins)
    random.shuffle(current_route)

    # 3) Its length is both the current and the best so far
    current_distance = route_distance(current_route, depot)
    best_route = list(current_route)
    best_distance = current_distance

    # Lists for the convergence plot (iteration 0 = the starting route)
    current_history = [current_distance]
    best_history = [best_distance]

    temperature = initial_temp
    iteration = 0

    # 4) Main loop: run until the system "cools down"
    while temperature > MIN_TEMP and iteration < MAX_ITERATIONS:
        iteration += 1

        # a) make a neighbour route and measure it
        new_route = get_neighbor(current_route)
        new_distance = route_distance(new_route, depot)

        # b) how much worse (+) or better (-) is it?
        delta = new_distance - current_distance

        # c) accept it or stay where we are
        if accept(delta, temperature):
            current_route = new_route
            current_distance = new_distance

        # d) remember the best route ever found
        if current_distance < best_distance:
            best_route = list(current_route)
            best_distance = current_distance

        # e) save values for the plot
        current_history.append(current_distance)
        best_history.append(best_distance)

        # f) cool down
        temperature = temperature * cooling_rate

    # 5) Return everything Member 3 needs for the GUI and the plots
    return {
        "best_route": best_route,
        "route_with_depot": [depot] + best_route + [depot],
        "best_distance": best_distance,
        "current_history": current_history,
        "best_history": best_history,
        "iterations": iteration,
    }

# ======================================================================
# Step 10: create bins and classify them with the trained Perceptron
# ======================================================================

MAP_SIZE = 1000.0       # the campus map is 1000 m x 1000 m
MARGIN = 30.0           # keep bins a little away from the map edges


def random_fill(rng):
    """
    Random current fill (x1) between 0 and 1.
    In the training data almost only (nearly) full bins are labelled Pickup,
    so if x1 were completely random, most demo runs would have 0 or 1 pickup bins.
    To get a useful route, about half of the bins are made nearly full (0.85 .. 1.0).
    """
    if rng.random() < 0.5:
        return round(rng.uniform(0.85, 1.0), 2)
    return round(rng.uniform(0.0, 0.85), 2)


def create_bins(count, seed=None):
    """
    Create `count` bins, each one is a dictionary:
        id          -> "Bin 1", "Bin 2", ...
        x, y        -> random location on the map (metres)
        x1, x2, x3  -> random readings, inside the ranges the Perceptron accepts
        prediction  -> None for now (filled later by classify_bins)
    """
    # The bin count must be a whole number from 1 upwards
    try:
        count = int(count)
    except (TypeError, ValueError):
        raise ValueError("Bin Count must be a whole number.")
    if count < 1:
        raise ValueError("Bin Count must be at least 1.")

    rng = random.Random(seed)       # own generator -> does not disturb other code
    bins = []
    for k in range(count):
        bins.append({
            "id": "Bin %d" % (k + 1),
            "x": round(rng.uniform(MARGIN, MAP_SIZE - MARGIN), 1),
            "y": round(rng.uniform(MARGIN, MAP_SIZE - MARGIN), 1),
            "x1": random_fill(rng),                     # current fill       0 .. 1
            "x2": round(rng.uniform(-0.1, 0.3), 2) + 0.0,  # recent fill change (most data: -0.04 .. 0.25)
            # "+ 0.0" turns -0.0 into 0.0
            "x3": round(rng.uniform(0.0, 0.6), 2),     # time since collection (most data: 0.06 .. 0.52)
            "prediction": None,
        })
    return bins


def classify_bins(bins, model):
    """
    Use the trained Perceptron (from Part 1) to predict every bin.
    Saves +1 (Pickup) or -1 (No pickup) in bin["prediction"].
    """
    if not model.is_trained:
        raise ValueError("Train the Perceptron first (Perceptron tab).")

    for b in bins:
        b["prediction"] = model.predict([b["x1"], b["x2"], b["x3"]])
    return bins


def get_pickup_bins(bins):
    """
    Keep ONLY the bins predicted as Pickup (+1) and convert them
    to the (name, x, y) form that simulated_annealing() expects.
    """
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
        """Print PASS/FAIL for one test and count the results."""
        global passed, failed
        if condition:
            passed += 1
            print("PASS |", name)
        else:
            failed += 1
            print("FAIL |", name)


    def close(a, b):
        """True if two float numbers are equal up to 0.01."""
        return abs(a - b) < 0.01


    # ---------- Step 2: distance ----------
    check("distance (0,0)-(3,4) = 5", close(euclidean_distance((0, 0), (3, 4)), 5))
    check("distance (100,100)-(400,500) = 500", close(euclidean_distance((100, 100), (400, 500)), 500))

    # ---------- Step 3: route length ----------
    bin_a = ("Bin 1", 500.0, 800.0)
    bin_b = ("Bin 2", 800.0, 800.0)
    check("route [] = 0", route_distance([]) == 0)
    check("route [one bin] = 600", close(route_distance([bin_a]), 600))
    check("route [a, b] = 1024.26", close(route_distance([bin_a, bin_b]), 1024.26))
    check("reversed route has the same length",
          close(route_distance([bin_a, bin_b]), route_distance([bin_b, bin_a])))

    # ---------- Step 4: validation ----------
    check("valid settings accepted", validate_parameters("60", "0.997") == (60.0, 0.997))
    for temp, rate in [("0", "0.997"), ("-5", "0.997"), ("60", "1"), ("60", "0"),
                       ("60", "1.5"), ("abc", "0.997"), ("60", ""), ("inf", "0.997")]:
        try:
            validate_parameters(temp, rate)
            check("rejects temp=%r rate=%r" % (temp, rate), False)
        except ValueError:
            check("rejects temp=%r rate=%r" % (temp, rate), True)

    # ---------- Step 5: neighbours ----------
    letters = ["A", "B", "C", "D", "E"]
    check("swap i=1 j=4", swap_neighbor(letters, 1, 4) == ["A", "E", "C", "D", "B"])
    check("2-opt i=1 j=4", two_opt_neighbor(letters, 1, 4) == ["A", "E", "D", "C", "B"])
    check("original route not changed", letters == ["A", "B", "C", "D", "E"])
    random.seed(1)
    all_ok = all(sorted(get_neighbor(letters)) == sorted(letters) for k in range(1000))
    check("1000 random neighbours keep the same bins", all_ok)

    # ---------- Step 6: acceptance ----------
    check("better route always accepted", acceptance_probability(-50, 60) == 1.0)
    check("P(T=60, d=10) = 0.846", close(acceptance_probability(10, 60), 0.846))
    check("P(T=1, d=10) almost 0", acceptance_probability(10, 1) < 0.001)

    # ---------- Step 7: full algorithm ----------
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

    # ---------- Step 8: special cases ----------
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

    # ---------- Speed: 50 bins ----------
    rng = random.Random(3)
    big = [("Bin %d" % k, rng.uniform(30, 970), rng.uniform(30, 970)) for k in range(50)]
    start = time.time()
    rb = simulated_annealing(big, seed=1)
    seconds = time.time() - start
    print("INFO | 50 bins: start %.0f m -> best %.0f m in %.2f s"
          % (rb["current_history"][0], rb["best_distance"], seconds))
    check("50 bins finish in under 5 seconds", seconds < 5)

    # ---------- Step 10: create and classify bins ----------
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
        """A pretend Perceptron: Pickup when the bin is at least 80% full."""
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

    # ---------- Step 10: the REAL Perceptron from Part 1 ----------
    import io
    import os
    import sys
    import contextlib

    # Let Python find the "algorithms" folder when this file is run directly:
    # add the project folder (one level above this file) to the search path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from algorithms.perceptron import Perceptron

    model = Perceptron()
    with contextlib.redirect_stdout(io.StringIO()):    # hide the epoch prints
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
