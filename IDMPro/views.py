from django.shortcuts import render
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from .models import ModelResult 

def view_index(request):
    return render(request, 'index.html')

def model_results(request):
  
    # Replace with the path to your CSV file
    data_path = "C:/Users/wasee/Desktop/IDM/IDM/vehicles.csv"

    try:
        # Read the CSV data
        df = pd.read_csv(data_path)

        # Drop unnecessary columns
        df = df.drop(['url', 'region', 'size', 'region_url', 'image_url', 'county',
                      'lat', 'long', 'state', 'description', 'VIN', 'posting_date'], axis=1)

        # Preprocess categorical variables
        le = LabelEncoder()
        for col in ['manufacturer', 'model', 'condition', 'fuel', 'title_status',
                   'transmission', 'drive', 'type', 'paint_color']:
            df[col] = le.fit_transform(df[col])

        # Impute missing values (consider more sophisticated methods if needed)
        imputer = SimpleImputer(strategy='most_frequent')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        # Handle outliers (consider more sophisticated methods if needed)
        df['price'] = df['price'].where(df['price'] > 0, df['price'].median())

        # Separate features and target variable
        X = df.drop('manufacturer', axis=1)
        y = df['manufacturer']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # ---- Model Training and Evaluation ----

        # Logistic Regression
        model_lr = LogisticRegression()
        model_lr.fit(X_train, y_train)
        y_pred_lr = model_lr.predict(X_test)
        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        precision_lr = precision_score(y_test, y_pred_lr, average='weighted')
        recall_lr = recall_score(y_test, y_pred_lr, average='weighted')
        f1_lr = f1_score(y_test, y_pred_lr, average='weighted')

        # Decision Tree
        model_dt = DecisionTreeClassifier()
        model_dt.fit(X_train, y_train)
        y_pred_dt = model_dt.predict(X_test)
        accuracy_dt = accuracy_score(y_test, y_pred_dt)
        precision_dt = precision_score(y_test, y_pred_dt, average='weighted')
        recall_dt = recall_score(y_test, y_pred_dt, average='weighted')
        f1_dt = f1_score(y_test, y_pred_dt, average='weighted')

        # Naive Bayes
        model_nb = GaussianNB()
        model_nb.fit(X_train, y_train)
        y_pred_nb = model_nb.predict(X_test)
        accuracy_nb = accuracy_score(y_test, y_pred_nb)
        precision_nb = precision_score(y_test, y_pred_nb, average='weighted')
        recall_nb = recall_score(y_test, y_pred_nb, average='weighted')
        f1_nb = f1_score(y_test, y_pred_nb, average='weighted')

        # ---- Prepare Results for Template ----

        results = [
            {
                "Model": "Logistic Regression",
                "Accuracy": accuracy_lr,
                "Precision": precision_lr,
                "Recall": recall_lr,
                "F1 Score": f1_lr
            },
            {
                "Model": "Decision Tree",
                "Accuracy": accuracy_dt,
                "Precision": precision_dt,
                "Recall": recall_dt,
                "F1 Score": f1_dt
                
            },
            {
                "Model": "Naive Bayes",
                "Accuracy": accuracy_nb,
                "Precision": precision_nb,
                "Recall": recall_nb,
                "F1 Score": f1_nb
            }
                  ]
        context = {'results': results, }
        return render(request, 'results.html', context)
    except:
        return render(request, 'index.html')
         
        