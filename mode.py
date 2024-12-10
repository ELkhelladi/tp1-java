import pandas as pd
from imblearn.over_sampling import SMOTE

# Charger les données dans un DataFrame
data = pd.read_csv("votre_fichier.csv")

# Divisez votre DataFrame en caractéristiques (X) et labels (y)
X = data.drop(columns=['emotion'])
y = data['emotion']

# Appliquer SMOTE pour générer des exemples supplémentaires pour les classes minoritaires
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Combiner X et y équilibrés dans un DataFrame
balanced_data = pd.DataFrame(X_res, columns=X.columns)
balanced_data['emotion'] = y_res

# Supprimer les doublons exacts
balanced_data = balanced_data.drop_duplicates()

# Sauvegarder les données équilibrées sans doublons
balanced_data.to_csv("data_équilibrée_sans_doublons.csv", index=False)
