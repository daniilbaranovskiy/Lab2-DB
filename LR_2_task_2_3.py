import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Вхідний файл, який містить дані
input_file = 'income_data.txt'

# Читання даних
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue

        data = line[:-1].split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

# Перетворення на масив numpy
X = np.array(X)

# Перетворення рядкових даних на числові
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Кодування міток класу
label_encoder_y = preprocessing.LabelEncoder()
y_encoded = label_encoder_y.fit_transform(y)

# Розділення даних на тренувальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=5)

# Створення SVМ-класифікатора
classifier = OneVsOneClassifier(SVC(kernel='sigmoid', max_iter=6000))

# Навчання класифікатора
classifier.fit(X_train, y_train)

# Обчислення F-міри для SVМ-класифікатора
f1 = cross_val_score(classifier, X, y_encoded, scoring='f1_weighted', cv=3)
print("F1 score: " + str(round(100 * f1.mean(), 2)) + "%")

# Передбачення результату для тестової точки даних
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family', 'White',
              'Male', '0', '0', '40', 'United-States']

# Кодування тестової точки даних
input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        try:
            input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]])[0])
        except (ValueError, IndexError):
            input_data_encoded[i] = -1
    count += 1
input_data_encoded = np.array(input_data_encoded)

# Використання класифікатора для кодованої точки даних та виведення результату
predicted_class = classifier.predict(input_data_encoded.reshape(1, -1))
print(label_encoder[-1].inverse_transform(predicted_class)[0])

# Передбачення класифікатора на тестовому наборі
y_pred = classifier.predict(X_test)

# Розрахунок акуратності
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: " + str(round(100 * accuracy, 2)) + "%")

# Розрахунок повноти
recall = recall_score(y_test, y_pred, average='weighted')
print("Recall: " + str(round(100 * recall, 2)) + "%")

# Розрахунок точності
precision = precision_score(y_test, y_pred, average='weighted')
print("Precision: " + str(round(100 * precision, 2)) + "%")
