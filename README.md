# Campus Waste Collection

## Project Overview

A Python application that helps a campus waste collection team decide which bins need pickup and find a short collection route using Perceptron and Simulated Annealing.

## Technologies

- Python
- Pandas — Data handling.
- Tkinter — Graphical user interface.
- Matplotlib — Visualization and graphs.
- Git & GitHub — Collaboration.

## Project Structure

- `data/` — Provided CSV datasets.
- `algorithms/perceptron.py` — Model training, testing, and prediction.
- `algorithms/simulated_annealing.py` — Distance calculations and route optimization.
- `gui/perceptron_tab.py` — Perceptron interface.
- `gui/annealing_tab.py` — Simulated Annealing interface.
- `gui/visualization.py` — Route visualization and graphs.
- `__init__.py` — Python package initialization.
- `main.py` — Main application and tab integration.
- `requirements.txt` — Required libraries.
- `.gitignore` — Files excluded from Git tracking.

## Getting Started

1. Make sure Python and Git are installed.

2. Clone the repository:

   `git clone https://github.com/nour55uni/Campus-Waste-Collection.git`

3. Open the project folder in VS Code:

   `cd Campus-Waste-Collection`

   `code .`

4. Install the required libraries:

   `python -m pip install -r requirements.txt`

   You can use a virtual environment if you prefer.

5. Run the application once the implementation is ready:

   `python main.py`

## Team Responsibilities

### Member 1 — Perceptron & GUI

**Assigned files:**

- `algorithms/perceptron.py`
- `gui/perceptron_tab.py`

**Algorithm:**

- [x] Read the provided CSV datasets.
- [x] Implement Perceptron from scratch using three weights and a bias.
- [x] Train the model using the training dataset.
- [x] Evaluate the final model using the test dataset.
- [x] Calculate test accuracy.
- [x] Calculate missed pickups and false pickup alerts.
- [x] Implement prediction using three input readings.
- [x] Make the trained model available to the Simulated Annealing component.

**GUI:**

- [x] Create the Perceptron tab.
- [x] Add the Train and Test button.
- [x] Display evaluation results.
- [x] Add three input fields and a Predict button.
- [x] Display Pickup (+1) or No Pickup (-1).
- [x] Display the current fill status.
- [x] Validate inputs and test functionality.

### Member 2 — Simulated Annealing & GUI

**Assigned files:**

- `algorithms/simulated_annealing.py`
- `gui/annealing_tab.py`

**Algorithm:**

- [ ] Create bins with IDs, locations, and three editable readings.
- [ ] Use the trained Perceptron to classify each bin.
- [ ] Select only bins predicted as Pickup.
- [ ] Calculate distances and total route cost.
- [ ] Implement Simulated Annealing from scratch.
- [ ] Implement neighbor generation, acceptance probability, and cooling.
- [ ] Find the best route starting and ending at the depot.
- [ ] Track current distance and best distance at each iteration.
- [ ] Provide the route and distance history to Member 3.

**GUI:**

- [ ] Create the Simulated Annealing tab.
- [ ] Add the bin count input and Create Bins button.
- [ ] Display bins with editable readings.
- [ ] Display each bin's prediction.
- [ ] Add temperature and cooling rate inputs.
- [ ] Add the Plan Route button.
- [ ] Validate inputs and test functionality.

**Default settings:**

- Number of bins: 10
- Initial temperature: 60 (greater than 0)
- Cooling rate: 0.997 (between 0 and 1, exclusive)

### Member 3 — Visualization & GUI Integration

**Assigned files:**

- `gui/visualization.py`
- `main.py`

**Visualization:**

- [ ] Plot bin locations and the depot.
- [ ] Distinguish Pickup and No Pickup bins using colors.
- [ ] Display bin IDs.
- [ ] Draw the best collection route, including the return to the depot.
- [ ] Plot current distance against iteration number.
- [ ] Plot best distance so far against iteration number.
- [ ] Display distances in meters.
- [ ] Embed the plots into the GUI.

**GUI Integration & Improvements:**

- [ ] Create the main application window.
- [ ] Integrate the Perceptron and Simulated Annealing tabs.
- [ ] Connect the visualization with the route planning results.
- [ ] Improve the overall layout, colors, spacing, and consistency.
- [ ] Test the complete application.
- [ ] Fix remaining GUI and integration issues with the team.

## Git Workflow

Each member works on a separate branch:

- Member 1: `feature/perceptron`
- Member 2: `feature/annealing`
- Member 3: `feature/visualization`

**Before starting:**

`git switch main`

`git pull origin main`

`git switch -c feature/your-branch`

**Save and upload your work:**

`git add .`

`git commit -m "Describe your changes"`

`git push -u origin feature/your-branch`

**When your part is ready, merge it into main:**

`git switch main`

`git pull origin main`

`git merge feature/your-branch`

`git push origin main`

## Important Note

As mentioned by the course instructor:

- Implement both algorithms from scratch.
- Do not use built-in implementations of Perceptron or Simulated Annealing.

I think we can use supporting libraries such as Pandas, NumPy,
Tkinter, and Matplotlib, as long as we implement the algorithms
ourselves.

## Team Notes

- Work mainly on your assigned files.
- Let each other know when a component is ready.
- Agree on shared function inputs and outputs.
- Pull the latest changes before merging.
- Test your part before merging it into main.
- Keep the code simple, readable, and easy to understand.
- Feel free to ask each other for help or suggestions.
- Do not upload virtual environments or temporary files.
- Test the complete application together before submission.

