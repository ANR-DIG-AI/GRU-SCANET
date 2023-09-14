import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression 

X, y = make_classification(
    n_samples=1000, # 1000 observations 
    n_features=20, # 5 total features
    n_informative=15, # 3 'useful' features
    n_classes=2, # binary target/label 
    random_state=999 # if you want the same results as mine
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création d'un classificateur (dans cet exemple, nous utilisons une régression logistique)
classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(X_train, y_train)

# Prédictions sur les données de test
y_pred = classifier.predict(X_test)

# Calcul des métriques en micro
precision_micro = precision_score(y_test, y_pred, average='micro')
recall_micro = recall_score(y_test, y_pred, average='micro')
f1_micro = f1_score(y_test, y_pred, average='micro')

# Affichage des résultats
print("Métriques en micro :")
print(f"Précision en micro : {precision_micro:.2f}")
print(f"Rappel en micro : {recall_micro:.2f}")
print(f"F1-score en micro : {f1_micro:.2f}")

# Rapport de classification en micro
classification_report_micro = classification_report(y_test, y_pred, target_names=[str(i) for i in range(2)], output_dict=True)
print("\nRapport de classification en micro :")
print(classification_report_micro)
