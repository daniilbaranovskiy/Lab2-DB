import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# Крок 1. Завантаження даних
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

print(dataset.head())

# Крок 2. Візуалізація даних
# Діаграма розмаху
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.show()

# Гістограма розподілу атрибутів
dataset.hist()
plt.show()

# Матриця діаграм розсіювання
scatter_matrix(dataset)
plt.show()

# Крок 3. Створення навчального та тестового наборів
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)

# Крок 4. Класифікація (побудова моделі)
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Крок 6. Отримання прогнозу
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Крок 7. Оцінка якості моделі (оцінюємо точність на контрольній вибірці)
print("Точність на контрольній вибірці: {:.2f}%".format(accuracy_score(Y_validation, predictions) * 100))
print("Матриця помилок:\n", confusion_matrix(Y_validation, predictions))
print("Звіт про класифікацію:\n", classification_report(Y_validation, predictions))

# Крок 8. Отримання прогнозу для нових даних
X_new = np.array([[5.0, 2.9, 1.0, 0.2]])
print("Форма масиву X_new: {}".format(X_new.shape))

prediction = model.predict(X_new)
predicted_class = dataset['class'][dataset['class'] == prediction[0]].index[0]
predicted_class_name = dataset['class'][predicted_class]
print("Прогноз: {}".format(predicted_class_name))
