#1 DeepMind Eye Care AI
class DeepMindEyeCareAI:
    def __init__(self):
        self.data = []
        self.accuracy = 0.0

    def collect_data(self):
        self.data = ["scan1", "scan2", "scan3"]
        print("Data collected.")

    def train_model(self):
        if self.data:
            self.accuracy = 94
            print("Model trained successfully with 94% accuracy.")
        else:
            print("No data available to train.")

    def diagnose(self, scan):
        if self.accuracy > 0:
            print(f"Diagnosis for {scan}: Diabetic Retinopathy")
        else:
            print("Model not trained.")

# Usage
ai = DeepMindEyeCareAI()
ai.collect_data()
ai.train_model()
ai.diagnose("scan1")

#1b Waymo Autonomous Vehicles
class WaymoSafetyInitiative:
    def __init__(self):
        self.is_trained = False
        self.transparency_reports = []
        self.public_events = []

    def train_system(self):
        print("Training autonomous vehicle system...")
        self.is_trained = True
        print("System trained successfully.")

    def generate_report(self, incidents, decision_logic):
        report = {"Incidents": incidents, "Logic": decision_logic}
        self.transparency_reports.append(report)
        print("Transparency report created.")

    def organize_event(self, name, location):
        self.public_events.append({"Event": name, "Location": location})
        print(f"Organized public event: {name} at {location}.")

    def simulate_crash_scenario(self, scenario, outcome):
        print(f"Simulating crash scenario: {scenario}")
        print(f"Outcome: {outcome}")

    def results(self):
        print("\nResults:")
        print("Transparency Reports:", self.transparency_reports)
        print("Public Events:", self.public_events)

# Usage
waymo = WaymoSafetyInitiative()
waymo.train_system()
waymo.generate_report(["Collision with pole"], "Prioritize human safety")
waymo.organize_event("Safety Demo", "San Francisco")
waymo.simulate_crash_scenario("Pedestrian crossing", "Emergency stop executed")
waymo.results()

#1c Autonomous Drones in Defense
class AutonomousDroneProgram:
    def __init__(self):
        self.drones = []
        self.efficiency_improved = False
        self.ethics_compliance = True

    def develop_drones(self, num_drones):
        self.drones = [{"id": i, "status": "Operational"} for i in range(1, num_drones + 1)]
        print(f"{num_drones} drones developed for surveillance.")

    def promote_ethical_use(self):
        self.ethics_compliance = True
        print("Ethical standards for human oversight promoted.")

    def improve_efficiency(self):
        self.efficiency_improved = True
        print("Efficiency improved by reducing human involvement.")

    def assess_risks(self):
        print("Risk Assessment: Addressed concerns over unintended escalation.")

    def results(self):
        print("\nProgram Results:")
        print(f"Drones Developed: {len(self.drones)}")
        print(f"Ethical Compliance: {self.ethics_compliance}")
        print(f"Efficiency Improved: {self.efficiency_improved}")

# Usage
drones = AutonomousDroneProgram()
drones.develop_drones(5)
drones.promote_ethical_use()
drones.improve_efficiency()
drones.assess_risks()
drones.results()

#2 Experiment 2: Exploratory Data Analysis on Linear Regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate sample data
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.randn(100, 1)

# Train regression model
model = LinearRegression()
model.fit(X, y)

# Visualize
plt.scatter(X, y, label="Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.legend()
plt.show()

# Output
print(f"Slope: {model.coef_[0][0]}, Intercept: {model.intercept_[0]}")

#3 Regression Model With and Without Bias
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate synthetic data
X = np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# With bias
model_with_bias = LinearRegression(fit_intercept=True)
model_with_bias.fit(X, y)

# Without bias
model_without_bias = LinearRegression(fit_intercept=False)
model_without_bias.fit(X, y)

print(f"With Bias: Slope={model_with_bias.coef_[0][0]}, Intercept={model_with_bias.intercept_[0]}")
print(f"Without Bias: Slope={model_without_bias.coef_[0][0]}")

#4 Perceptron Classification
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X, y = data.data, (data.target == 0).astype(int)  # Binary classification

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Perceptron with bias
model_with_bias = Perceptron()
model_with_bias.fit(X_train, y_train)

# Train Perceptron without bias
model_without_bias = Perceptron(fit_intercept=False)
model_without_bias.fit(X_train, y_train)

# Accuracy
print("With Bias:", accuracy_score(y_test, model_with_bias.predict(X_test)))
print("Without Bias:", accuracy_score(y_test, model_without_bias.predict(X_test)))

#5 Ontology for Ethics in Healthcare
from rdflib import Graph, Namespace, URIRef

# Create RDF graph
g = Graph()
ns = Namespace("http://example.org/healthcare/")

# Define ontology elements
g.add((URIRef(ns.Patient), URIRef(ns.hasDiagnosis), URIRef(ns.Diagnosis)))
g.add((URIRef(ns.Patient), URIRef(ns.hasMedication), URIRef(ns.Medication)))

# Save
g.serialize("ontology.rdf", format="xml")
print("Ontology saved as ontology.rdf")


#6 Optimization in AI Affecting Ethics
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset
X = np.array([[25, 0], [45, 1], [35, 0], [50, 1]])
y = np.array([1, 1, 0, 1])  # Approved or denied

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Train logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))


