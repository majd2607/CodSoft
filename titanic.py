import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the Titanic dataset
file_path = 'C:/Users/majd/Desktop/codsoftDS/Titanic-Dataset.csv'
titanic_data = pd.read_csv(file_path)

# Data preprocessing
# Fill missing values for age with the median
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)

# Fill missing values for Embarked with the most frequent value (mode)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# Drop the 'Cabin' column due to many missing values
titanic_data.drop(columns=['Cabin'], inplace=True)

# Convert categorical variables to numerical variables
titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked'], drop_first=True)

# Select features and target variable
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
X = titanic_data[features]
y = titanic_data['Survived']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display the results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Function to predict a passenger's survival
def predict_survival(pclass, age, sibsp, parch, fare, sex_male, embarked_q, embarked_s):
    # Create a DataFrame for the passenger's data
    passenger = pd.DataFrame({
        'Pclass': [pclass],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Sex_male': [sex_male],
        'Embarked_Q': [embarked_q],
        'Embarked_S': [embarked_s]
    })
    # Make the prediction
    prediction = model.predict(passenger)
    return 'Survived' if prediction[0] == 1 else 'Not Survived'

# Function to get valid input from the user
def get_valid_input(prompt, type_func, condition_func):
    while True:
        try:
            value = type_func(input(prompt))
            if condition_func(value):
                return value
            else:
                print("Invalid value. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a valid value.")

# Ask the user for the passenger's information
print("Please enter the passenger's information:")
pclass = get_valid_input("Class (1, 2, 3): ", int, lambda x: x in [1, 2, 3])
age = get_valid_input("Age: ", float, lambda x: x > 0)
sibsp = get_valid_input("Number of siblings/spouses aboard: ", int, lambda x: x >= 0)
parch = get_valid_input("Number of parents/children aboard: ", int, lambda x: x >= 0)
fare = get_valid_input("Fare paid for the ticket: ", float, lambda x: x >= 0)
sex = get_valid_input("Sex (male/female): ", str, lambda x: x.lower() in ['male', 'female'])
sex_male = 1 if sex.lower() == 'male' else 0
embarked = get_valid_input("Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton): ", str, lambda x: x.lower() in ['c', 'q', 's'])
embarked_q = 1 if embarked.lower() == 'q' else 0
embarked_s = 1 if embarked.lower() == 's' else 0

# Make the prediction for the given passenger
result = predict_survival(pclass, age, sibsp, parch, fare, sex_male, embarked_q, embarked_s)
print(f"The passenger will: {result}")
