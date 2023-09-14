import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(
    n_samples=1000,
    n_features=20, 
    n_informative=15, 
    n_classes=5,
    random_state=999 
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

precision_micro = precision_score(y_test, y_pred, average='micro')
recall_micro = recall_score(y_test, y_pred, average='micro')
f1_micro = f1_score(y_test, y_pred, average='micro')

print("Metric in micro :")
print(f"Micro precision : {precision_micro:.2f}")
print(f"Micro recall : {recall_micro:.2f}")
print(f"Micro f1 : {f1_micro:.2f}")

classification_report_micro = classification_report(y_test, y_pred, target_names=[str(i) for i in range(5)], output_dict=True)
print("\nMicro classification report :")
print(classification_report_micro)
