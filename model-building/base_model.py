import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

Data = r"D:/Flight-Fare-prediction/Data/Data_Train.xlsx"
data = pd.read_excel(Data)
print(data.head())

print(data.columns)

def create_day_and_month(data):
    data["Journey_day"] = pd.to_datetime(data.Date_of_Journey, format="%d/%m/%Y").dt.day
    data["Journey_month"] = pd.to_datetime(data["Date_of_Journey"], format="%d/%m/%Y").dt.month
    data.drop(["Date_of_Journey"], axis=1, inplace=True)


create_day_and_month(data)


def create_dep_hour_and_min(data):
    # Extracting Hours
    data["Dep_hour"] = pd.to_datetime(data["Dep_Time"]).dt.hour

    # Extracting Minutes
    data["Dep_min"] = pd.to_datetime(data["Dep_Time"]).dt.minute

    # Now we can drop Dep_Time as it is of no use
    data.drop(["Dep_Time"], axis=1, inplace=True)


create_dep_hour_and_min(data)


def create_arrival_hour_and_min(data):
    # Extracting Hours
    data["Arrival_hour"] = pd.to_datetime(data.Arrival_Time).dt.hour

    # Extracting Minutes
    data["Arrival_min"] = pd.to_datetime(data.Arrival_Time).dt.minute

    # Now we can drop Arrival_Time as it is of no use
    data.drop(["Arrival_Time"], axis=1, inplace=True)


create_arrival_hour_and_min(data)


def create_duration_hours_and_min(data):
    # Assigning and converting Duration column into list
    duration = list(data["Duration"])

    for i in range(len(duration)):
        if len(duration[i].split()) != 2:  # Check if duration contains only hour or mins
            if "h" in duration[i]:
                duration[i] = duration[i].strip() + " 0m"  # Adds 0 minute
            else:
                duration[i] = "0h " + duration[i]  # Adds 0 hour

    duration_hours = []
    duration_mins = []
    for i in range(len(duration)):
        duration_hours.append(int(duration[i].split(sep="h")[0]))  # Extract hours from duration
        duration_mins.append(int(duration[i].split(sep="m")[0].split()[-1]))  # Extracts only minutes from duration

    data["Duration_hours"] = duration_hours
    data["Duration_mins"] = duration_mins
    data.drop(["Duration"], axis=1, inplace=True)


create_duration_hours_and_min(data)

def create_no_of_stops(data):
    data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace=True)


create_no_of_stops(data)

sns.catplot(y = "Price", x = "Airline", data = data.sort_values("Price", ascending = False), kind="boxen", height = 6, aspect = 3)
plt.show()
Airline = data[["Airline"]]
Airline = pd.get_dummies(Airline, drop_first= True)

sns.catplot(y = "Price", x = "Source", data = data.sort_values("Price", ascending = False), kind="boxen", height = 4, aspect = 3)
plt.show()
Source = data[["Source"]]
Source = pd.get_dummies(Source, drop_first= True)
Destination = data[["Destination"]]
Destination = pd.get_dummies(Destination, drop_first = True)


data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)
data_train = pd.concat([data, Airline, Source, Destination], axis = 1)
print(data_train.columns)

# Feature scaling the model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X = data_train.drop("Price", axis=1)
y = data_train["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

