# %%
# Loading library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
from sklearn.preprocessing import OrdinalEncoder

# %%
# Load CSV dataset
data = pd.read_csv('dataset_buys _comp.csv')

# %%
# Tampilkan info dasar
print(data.info())
data.head()

# %%
# Cek deskripsi statistik
print(data.describe().T)

# %%
data.apply(lambda x: x.unique())

# %%
# Encoding
age_categories = ['Muda', 'Paruh Baya', 'Tua']
income_categories = ['Rendah', 'Sedang', 'Tinggi']

ordinal_encoder = OrdinalEncoder(categories=[age_categories, income_categories])
encoded_features = ordinal_encoder.fit_transform(data[['Age', 'Income']])

data['Age'] = encoded_features[:, 0]
data['Income'] = encoded_features[:, 1]

data['Student'] = data['Student'].map({'Ya': 1, 'Tidak': 0})
data['Credit_Rating'] = data['Credit_Rating'].map({'Baik': 1, 'Buruk': 0})
data.info()

# %%
data.head()

# %%
# Pairplot untuk visualisasi awal
sns.pairplot(data, hue='Buys_Computer', palette='Set2')

# %%
# Split fitur dan target
X = data.drop('Buys_Computer', axis=1)
y = data['Buys_Computer']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)


# %%
# Train Decision Tree
model = DecisionTreeClassifier(random_state=10)
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# %%
# Classification report
print(classification_report(y_test, y_pred))

# %%
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7,7))

sns.set(font_scale=1.4)
sns.heatmap(cm, ax=ax, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16})
plt.xlabel('Predicted', fontsize=18)
plt.ylabel('Actual', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

# %%
# Visualize the decision tree
fig, ax = plt.subplots(figsize=(25, 20))
tree.plot_tree(model, feature_names=X.columns, class_names=["Tidak Layak Kredit", "Layak Kredit"], filled=True)
plt.show()


# %%
# Example of creating a single data data point as a dictionary
import pandas as pd

# Data input
age = 'Muda'
income = 'Sedang'
student = 'Tidak'
credit_rating = 'Baik'

# Buat dictionary untuk data input
data_input = {
    'Age': [age],
    'Income': [income],
    'Student': [student],
    'Credit_Rating': [credit_rating]
}

# Konversi ke DataFrame
input_df = pd.DataFrame(data_input)

# Encode fitur kategorikal
encoded_ordinal = ordinal_encoder.transform(input_df[['Age', 'Income']])

student_map = {'Ya': 1, 'Tidak': 0}
credit_rating_map = {'Baik': 1, 'Buruk': 0}

# Buat DataFrame baru dengan fitur yang sudah di-encode
encoded_df = pd.DataFrame({
    'Age': encoded_ordinal[:, 0],
    'Income': encoded_ordinal[:, 1],
    'Student': [student_map[student]],
    'Credit_Rating': [credit_rating_map[credit_rating]]
})

# Lakukan prediksi
prediction = model.predict(encoded_df)[0]
probabilities = model.predict_proba(encoded_df)[0]

result = "Layak Kredit" if prediction == 1 else "Tidak Layak Kredit"
predicted_prob = probabilities[prediction]

# Menampilkan hasil
print(f"Data input: {data_input}")
print(f"Hasil Prediksi: {result}")
print(f"Probabilitas: {predicted_prob:.2f}")
print("-" * 50)


# %%
