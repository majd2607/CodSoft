import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

file_path = 'C:/Users/majd/Desktop/codsoftDS/Titanic-Dataset.csv'
titanic_data = pd.read_csv(file_path)

titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
titanic_data.drop(columns=['Cabin'], inplace=True)
titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked'], drop_first=True)

features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
X = titanic_data[features]
y = titanic_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

def predict_survival(pclass, age, sibsp, parch, fare, sex_male, embarked_q, embarked_s):
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
    prediction = model.predict(passenger)
    return 'Survived' if prediction[0] == 1 else 'Not Survived'

print("Please enter the passenger's information:")
pclass = int(input("Class (1, 2, 3): "))
age = float(input("Age: "))
sibsp = int(input("Number of siblings/spouses aboard: "))
parch = int(input("Number of parents/children aboard: "))
fare = float(input("Fare paid for the ticket: "))
sex = input("Sex (male/female): ")
sex_male = 1 if sex.lower() == 'male' else 0
embarked = input("Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton): ")
embarked_q = 1 if embarked.lower() == 'q' else 0
embarked_s = 1 if embarked.lower() == 's' else 0

result = predict_survival(pclass, age, sibsp, parch, fare, sex_male, embarked_q, embarked_s)
print(f"The passenger will: {result}")
