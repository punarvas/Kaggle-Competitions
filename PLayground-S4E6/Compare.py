import pandas as pd

df = pd.read_csv("data/val.csv")
print(df.shape)


# Select all predictors
x = df.iloc[:, 1:-1].values
y = df.iloc[:, -1]

# transforming class names
categories = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
y = y.map(lambda x: categories[x])
y = y.values

def reverse_map(value: int):
    for key, val in categories.items():
        if val == value:
            return key
    return -1

print(x.shape)
print(y.shape)


# Following 90-10 train-test split with shuffle for a relatively new problem
from sklearn.model_selection import train_test_split

# Build a validation set 10% of the size of train dataset
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.1, shuffle=True, random_state=42)


y_train = y_train.reshape(-1, 1)

import numpy as np
train = np.hstack((X_val, y_val))

cols = df.columns[1:]
df_train = pd.DataFrame(data=train, columns=cols)
df_train.to_csv("train.csv", index=False)


y_val = y_val.reshape(-1, 1)
val = np.hstack((X_val, y_val))
val_cols = df.columns[1:]
df_val = pd.DataFrame(data=val, columns=val_cols)
df_val.to_csv("val.csv", index=False)


df_val.head()
df_val.shape

import sys
print(sys.executable)
print(sys.path)

#Check if the device is using cuda or cpu resources
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

#SVM Classifier
from tensorflow.keras.models import Sequential
# Train a SVM classifier
from sklearn.svm import SVC

#model = SVC(kernel="rbf")
model = SVC(kernel="poly", degree = 17)
#model = SVC(kernel="sigmoid")
#model = SVC(kernel="linear")
model.fit(X_train, y_train)

train_predictions = model.predict(X_train)
val_predictions = model.predict(X_val)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_train, train_predictions)
print(f"Accuracy score (SVM train): {round(score * 100, 3)}")

score = accuracy_score(y_val, val_predictions)
print(f"Accuracy score (SVM validation): {round(score * 100, 3)}")


# Train a XGBoost classifier
from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X_train, y_train)

train_predictions = model.predict(X_train)
val_predictions = model.predict(X_val)

from sklearn.metrics import accuracy_score

score = accuracy_score(y_train, train_predictions)
print(f"Accuracy score (XGB train): {round(score * 100, 3)}")
score = accuracy_score(y_val, val_predictions)
print(f"Accuracy score (XGB validation): {round(score * 100, 3)}")

importance = model.feature_importances_
imp = pd.DataFrame(data = {"Feature":df.columns[1:-1], "Importance":importance*100})
imp = imp.sort_values(by="Importance", ascending=False).reset_index()
imp["index"] = imp["index"] + 1
imp


# Following 90-10 train-test split with shuffle for a relatively new problem
from sklearn.model_selection import train_test_split

# Build a validation set 10% of the size of train dataset
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.1, shuffle=True, random_state=42)





# Decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#model = DecisionTreeClassifier()
model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
model.fit(X_train, y_train)

train_predictions = model.predict(X_train)
val_predictions = model.predict(X_val)

from sklearn.metrics import accuracy_score

score = accuracy_score(y_train, train_predictions)
print(f"Accuracy score (DTC train): {round(score * 100, 3)}")

score = accuracy_score(y_val, val_predictions)
print(f"Accuracy score (DTC validation): {round(score * 100, 3)}")

index = 28
print(df.iloc[:, index].value_counts())
df.iloc[:, index].hist()



# Predictions on test dataset
test_df = pd.read_csv("/kaggle/input/playground-series-s4e6/test.csv")
test_df.head()

X_test = test_df.iloc[:, cols].values

test_predictions = model.predict(X_test)
test_predictions = list(map(lambda x: reverse_map(x), test_predictions))


# Making a submission
submission = pd.read_csv("/kaggle/input/playground-series-s4e6/sample_submission.csv")
submission['Target'] = test_predictions
submission.to_csv('/kaggle/working/submission_rf.csv',index=False)

submission.head()