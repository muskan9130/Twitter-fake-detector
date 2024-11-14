import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from collections import OrderedDict

filename = None
ErrorrateMeans = []
AccuracyMeans = []

def browse_file():
    global filename
    filename = filedialog.askopenfilename()
    print("Selected file:", filename)

def show_prediction_result(label, input_window):
    account_type = "Fake" if label == 1 else "Not Fake"
    tk.Label(input_window, text=f"Account Type: {account_type}").grid(row=24)

def create_input_frame():
    input_window = tk.Toplevel(root)
    input_window.geometry("400x500")
    input_window.resizable(0, 0)

    input_entries = {}
    labels = [
        "UserID", "No Of Abuse Report", "Rejected Friend Requests",
        "No Of Friend Requests Not Accepted", "No Of Friends", "No Of Followers",
        "No Of Likes To Unknown Account", "No Of Comments Per Day"
    ]

    for i, label_text in enumerate(labels):
        tk.Label(input_window, text=f"Enter {label_text}").grid(row=i*2)
        entry = tk.Entry(input_window)
        entry.grid(row=i*2 + 1, column=0)
        input_entries[label_text] = entry

    def get_input_data():
        return pd.DataFrame(
            OrderedDict(
                {key: [entry.get()] for key, entry in input_entries.items()}
            )
        )

    def predict_with_model(model, test_data):
        if filename:
            df = pd.read_csv(filename)
            train_data, test_data_values = split_data(df, test_data)
            model.fit(train_data['features'], train_data['labels'])
            prediction = model.predict(test_data_values)
            print(f"Predicted Class: {prediction[0]}")
            show_prediction_result(prediction[0], input_window)

    def split_data(df, test_data):
        msk = np.random.rand(len(df)) < 0.7
        train = df[msk]
        return {
            'features': train.values[:, 0:7],
            'labels': train.values[:, 8].astype('int')
        }, test_data.values[:, 0:7]

    def run_naive_bayes():
        input_data = get_input_data()
        predict_with_model(MultinomialNB(), input_data)

    def run_linear_svc():
        input_data = get_input_data()
        predict_with_model(LinearSVC(), input_data)

    def run_knn():
        input_data = get_input_data()
        predict_with_model(KNeighborsClassifier(n_neighbors=3), input_data)

    tk.Button(input_window, text="Naive Bayes", command=run_naive_bayes).place(relx=0.025, rely=0.8)
    tk.Button(input_window, text="Linear SVC", command=run_linear_svc).place(relx=0.31, rely=0.8)
    tk.Button(input_window, text="KNN", command=run_knn).place(relx=0.7, rely=0.8)

def run_classifier(model_name):
    if filename:
        global AccuracyMeans, ErrorrateMeans
        df = pd.read_csv(filename)
        msk = np.random.rand(len(df)) < 0.7
        train = df[msk]
        test = df[~msk]

        features, labels = train.values[:, 0:7], train.values[:, 8].astype('int')
        testing_data, testing_data_labels = test.values[:, 0:7], test.values[:, 8]

        if model_name == 'Naive Bayes':
            model = MultinomialNB()
        elif model_name == 'Linear SVC':
            model = LinearSVC()
        else:
            model = KNeighborsClassifier(n_neighbors=3)

        model.fit(features, labels)
        predictions = model.predict(testing_data)

        accuracy = accuracy_score(testing_data_labels, predictions) * 100
        error_rate = 100 - accuracy
        AccuracyMeans.append(accuracy)
        ErrorrateMeans.append(error_rate)
        precision = precision_score(testing_data_labels, predictions) * 100
        recall = recall_score(testing_data_labels, predictions) * 100

        print(f"{model_name}:\n")
        print("Confusion Matrix:\n", confusion_matrix(testing_data_labels, predictions))
        print(f"Accuracy: {accuracy}%, Error Rate: {error_rate}%")
        print(f"Precision: {precision}%, Recall: {recall}%\n")

        labels = ['Error Rate', 'Accuracy']
        sizes = [error_rate, accuracy]
        plt.pie(sizes, explode=(0, 0.1), labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        plt.title(f'{model_name} Algorithm')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

def compare_models():
    models = ['Naive Bayes', 'Linear SVC', 'KNN']
    for model in models:
        run_classifier(model)

    plt.bar(['Naive Bayes', 'Linear SVC', 'KNN'], AccuracyMeans, width=0.35, label='Accuracy')
    plt.bar(['Naive Bayes', 'Linear SVC', 'KNN'], ErrorrateMeans, width=0.35, bottom=AccuracyMeans, label='Error Rate')
    plt.ylabel('Scores')
    plt.title('Performance by Classifiers')
    plt.legend()
    plt.show()

root = tk.Tk()
root.title("Twitter Fake Account Detector")
root.geometry("600x500")
root.resizable(0, 0)

tk.Label(root, text="Fake Account Detector", fg="dark violet", bg="light blue",
         width=400, height=2, font="Helvetica 35 bold italic").pack()

browse_button = tk.Button(root, text="Browse File", command=browse_file, bg="thistle",
                          font="Helvetica 15 bold italic")
browse_button.pack(pady=10)

input_window_button = tk.Button(root, text="Create Input Window", command=create_input_frame,
                                bg="light green", font="Helvetica 15 bold italic")
input_window_button.pack(pady=10)

compare_button = tk.Button(root, text="Compare Models", command=compare_models, bg="light coral",
                           font="Helvetica 15 bold italic")
compare_button.pack(pady=10)

root.mainloop()
