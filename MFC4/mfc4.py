import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score
)

# =====================================================
# 1. LOAD DATA
# =====================================================
df = pd.read_csv("Traffic.csv")

# Total vehicles
df["Total"] = (
    df["BikeCount"] +
    df["CarCount"] +
    df["BusCount"] +
    df["TruckCount"]
)

# =====================================================
# 2. ENCODING
# =====================================================
day_encoder = LabelEncoder()
df["Day_encoded"] = day_encoder.fit_transform(df["Day of the week"])

traffic_encoder = LabelEncoder()
df["Traffic_encoded"] = traffic_encoder.fit_transform(df["Traffic Situation"])

df["Hour"] = pd.to_datetime(df["Time"], format="%I:%M:%S %p").dt.hour

# =====================================================
# 3. TOEPLITZ MATRIX
# =====================================================
def toeplitz_matrix(signal, k=3):
    n = len(signal)
    rows = []

    for i in range(k-1, n):   
        row = []
        for j in range(k):
            row.append(signal[i - j])
        rows.append(row)

    return np.array(rows)

toe_car = toeplitz_matrix(df["CarCount"].values)
toe_bike = toeplitz_matrix(df["BikeCount"].values)
toe_bus = toeplitz_matrix(df["BusCount"].values)
toe_truck = toeplitz_matrix(df["TruckCount"].values)

toe_features = np.hstack([toe_car, toe_bike, toe_bus, toe_truck])

# =====================================================
# 4. FFT FEATURES
# =====================================================
def fft_features(signal, k=3):
    fft_vals = np.abs(np.fft.fft(signal))
    return fft_vals[1:k+1]   

fft_car = fft_features(df["CarCount"].values)
fft_bike = fft_features(df["BikeCount"].values)
fft_bus = fft_features(df["BusCount"].values)
fft_truck = fft_features(df["TruckCount"].values)

fft_all = np.hstack([fft_car, fft_bike, fft_bus, fft_truck])
fft_all = np.tile(fft_all, (len(toe_car), 1))  

# =====================================================
# 5. FINAL FEATURE MATRIX 
# =====================================================

kernel_length = 3
valid_start = kernel_length - 1

base_features = df[[
    "Hour", "Day_encoded",
    "BikeCount", "CarCount", "BusCount", "TruckCount"
]].values[valid_start:]

X = np.hstack([base_features, toe_features, fft_all])
X_bike = np.delete(X, 2, axis=1)# Predicting BikeCount → remove BikeCount column
X_car = np.delete(X, 3, axis=1)# Predicting CarCount → remove CarCount column
X_bus = np.delete(X, 4, axis=1)# Predicting BusCount → remove BusCount column
X_truck = np.delete(X, 5, axis=1)# Predicting TruckCount → remove TruckCount

# =====================================================
# 6. TARGETS 
# =====================================================

y_total = df["Total"].values[valid_start:]
y_traffic = df["Traffic_encoded"].values[valid_start:]

y_bike = df["BikeCount"].values[valid_start:]
y_car = df["CarCount"].values[valid_start:]
y_bus = df["BusCount"].values[valid_start:]
y_truck = df["TruckCount"].values[valid_start:]



# =====================================================
# 7. TRAIN–TEST SPLIT
# =====================================================

train_idx, test_idx = train_test_split(
    np.arange(len(X)),  
    test_size=0.2,
    random_state=42
)

X_train, X_test = X[train_idx], X[test_idx]

y_total_train, y_total_test = y_total[train_idx], y_total[test_idx]
y_tr_train, y_tr_test = y_traffic[train_idx], y_traffic[test_idx]#low,medium, high

X_bike_train, X_bike_test   = X_bike[train_idx], X_bike[test_idx]
X_car_train, X_car_test     = X_car[train_idx], X_car[test_idx]
X_bus_train, X_bus_test     = X_bus[train_idx], X_bus[test_idx]
X_truck_train, X_truck_test = X_truck[train_idx], X_truck[test_idx]

y_bike_tr, y_bike_te = y_bike[train_idx], y_bike[test_idx]
y_car_tr, y_car_te = y_car[train_idx], y_car[test_idx]
y_bus_tr, y_bus_te = y_bus[train_idx], y_bus[test_idx]
y_truck_tr, y_truck_te = y_truck[train_idx], y_truck[test_idx]


# =====================================================
# 8. DECISION TREE 
# =====================================================
class DecisionTree:
    def __init__(self, max_depth=10, min_samples=5, task="regression"):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.task = task

    def fit(self, X, y, depth=0):
        if depth >= self.max_depth or len(y) < self.min_samples or len(set(y)) == 1:
            self.value = self.leaf_value(y)
            return

        feat, thresh = self.best_split(X, y)
        if feat is None:
            self.value = self.leaf_value(y)
            return

        self.feature = feat
        self.threshold = thresh

        left_idx = X[:, feat] <= thresh
        right_idx = X[:, feat] > thresh

        self.left = DecisionTree(self.max_depth, self.min_samples, self.task)
        self.right = DecisionTree(self.max_depth, self.min_samples, self.task)

        self.left.fit(X[left_idx], y[left_idx], depth + 1)
        self.right.fit(X[right_idx], y[right_idx], depth + 1)

    def best_split(self, X, y):
        best_score = float("inf")
        best_f, best_t = None, None

        features = np.random.choice(X.shape[1], int(np.sqrt(X.shape[1])), replace=False)

        for f in features:
            for t in np.unique(X[:, f]):
                left = y[X[:, f] <= t]
                right = y[X[:, f] > t]
                if len(left) == 0 or len(right) == 0:
                    continue
                score = self.impurity(left, right)
                if score < best_score:
                    best_score, best_f, best_t = score, f, t
        return best_f, best_t

    def impurity(self, left, right):
        if self.task == "regression":
            return len(left)*np.var(left) + len(right)*np.var(right)
        else:
            return len(left)*self.gini(left) + len(right)*self.gini(right)

    def gini(self, y):
        counts = Counter(y)
        return 1 - sum((c/len(y))**2 for c in counts.values())

    def leaf_value(self, y):
        return np.mean(y) if self.task == "regression" else Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self.predict_row(row) for row in X])

    def predict_row(self, row):
        if hasattr(self, "value"):
            return self.value
        if row[self.feature] <= self.threshold:
            return self.left.predict_row(row)
        else:
            return self.right.predict_row(row)

# =====================================================
# 9. RANDOM FOREST (FROM SCRATCH)
# =====================================================
class RandomForest:
    def __init__(self, n_trees=15, task="regression"):
        self.n_trees = n_trees
        self.task = task
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            idx = np.random.choice(len(X), len(X), replace=True)
            tree = DecisionTree(task=self.task)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        if self.task == "regression":
            return np.mean(preds, axis=0)
        else:
            return np.array([Counter(preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])])

# =====================================================
# 10. TRAIN MODELS
# =====================================================
rf_total = RandomForest(task="regression")
rf_traffic = RandomForest(task="classification")

rf_car = RandomForest(task="regression")
rf_bike = RandomForest(task="regression")
rf_bus = RandomForest(task="regression")
rf_truck = RandomForest(task="regression")

rf_total.fit(X_train, y_total_train)
rf_traffic.fit(X_train, y_tr_train)

rf_car.fit(X_car_train, y_car_tr)
rf_bike.fit(X_bike_train, y_bike_tr)
rf_bus.fit(X_bus_train, y_bus_tr)
rf_truck.fit(X_truck_train, y_truck_tr)

# =====================================================
# 11. PREDICTIONS
# =====================================================
total_pred = rf_total.predict(X_test)
traffic_pred = rf_traffic.predict(X_test)

car_pred = rf_car.predict(X_car_test)
bike_pred = rf_bike.predict(X_bike_test)
bus_pred = rf_bus.predict(X_bus_test)
truck_pred = rf_truck.predict(X_truck_test)

# =====================================================
# 12. METRICS
# =====================================================
print("\n--- Regression (Total Vehicles) ---")
print("MAE :", mean_absolute_error(y_total_test, total_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_total_test, total_pred)))

print("\n--- Classification (Traffic) ---")
print("Accuracy :", accuracy_score(y_tr_test, traffic_pred))
print("Precision:", precision_score(y_tr_test, traffic_pred, average="weighted"))
print("Recall   :", recall_score(y_tr_test, traffic_pred, average="weighted"))
print("F1-score :", f1_score(y_tr_test, traffic_pred, average="weighted"))

# =====================================================
# 13. LAST 10 RESULTS
# =====================================================
results = pd.DataFrame({
    "Actual_Bike": y_bike_te[-10:], "Pred_Bike": bike_pred[-10:].astype(int),
    "Actual_Car": y_car_te[-10:], "Pred_Car": car_pred[-10:].astype(int),
    "Actual_Bus": y_bus_te[-10:], "Pred_Bus": bus_pred[-10:].astype(int),
    "Actual_Truck": y_truck_te[-10:], "Pred_Truck": truck_pred[-10:].astype(int),
    "Actual_Total": y_total_test[-10:], "Pred_Total": total_pred[-10:].astype(int),
    "Actual_Traffic": traffic_encoder.inverse_transform(y_tr_test[-10:]),
    "Pred_Traffic": traffic_encoder.inverse_transform(traffic_pred[-10:])
})

print("\n--- Last 10 Predictions ---")
print(results)

# ============================================
# 14. FUTURE 10 PREDICTIONS 
# ============================================

future_steps = 10

# last Toeplitz values in order t, t-1, t-2
car_hist = list(df["CarCount"].values[-3:])[::-1]
bike_hist = list(df["BikeCount"].values[-3:])[::-1]
bus_hist = list(df["BusCount"].values[-3:])[::-1]
truck_hist = list(df["TruckCount"].values[-3:])[::-1]

last_hour = df["Hour"].iloc[-1]
day_encoded = df["Day_encoded"].iloc[-1]

future_results = []

for i in range(future_steps):

    
    last_hour = (last_hour + 0.25) % 24

    future_X = np.array([
        last_hour, day_encoded,
        *car_hist,
        *bike_hist,
        *bus_hist,
        *truck_hist,
        *fft_all[0]
    ]).reshape(1, -1)

    # remove target columns like before
    future_X_bike = np.delete(future_X, 2, axis=1)
    future_X_car = np.delete(future_X, 3, axis=1)
    future_X_bus = np.delete(future_X, 4, axis=1)
    future_X_truck = np.delete(future_X, 5, axis=1)

    # predictions
    car = rf_car.predict(future_X_car)[0]
    bike = rf_bike.predict(future_X_bike)[0]
    bus = rf_bus.predict(future_X_bus)[0]
    truck = rf_truck.predict(future_X_truck)[0]

    total = rf_total.predict(future_X)[0]
    traffic = rf_traffic.predict(future_X)[0]

    future_results.append([
        int(car), int(bike), int(bus), int(truck),
        int(total),
        traffic_encoder.inverse_transform([traffic])[0]
    ])

   
    car_hist = [car] + car_hist[:-1]
    bike_hist = [bike] + bike_hist[:-1]
    bus_hist = [bus] + bus_hist[:-1]
    truck_hist = [truck] + truck_hist[:-1]

future_df = pd.DataFrame(
    future_results,
    columns=["Car","Bike","Bus","Truck","Total","Traffic"]
)
print("Future Value:")
print(future_df)