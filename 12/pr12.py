# 1. Импорт необходимых библиотек
import pandas as pd                     # для работы с табличными данными
import numpy as np                      # для математических операций
from sklearn.neighbors import LocalOutlierFactor  # алгоритм LOF
from sklearn.ensemble import IsolationForest     # алгоритм Isolation Forest
from sklearn.metrics import classification_report, confusion_matrix  # метрики качества

# 2. Загрузка данных из файла banks.txt
# Разделитель — запятая, файл в кодировкsе UTF-8
df = pd.read_csv('banks.csv', delimiter=',', encoding='utf-8')

# 3. Предобработка данных
# Первый столбец — название банка (не участвует в обучении), остальные — числовые признаки
# Удаляем строки с некорректными значениями (если есть)
X = df.iloc[:, 1:5].values  # берём колонки Assents, OwnCapital, IndFunds, NBSLoans
# В исходном файле есть строка с отрицательным капиталом ("Траст") — оставляем как есть, это реальные данные

# 4. Часть 1: LOF в обычном режиме (novelty=False) с параметрами по умолчанию
lof_default = LocalOutlierFactor(novelty=False)
# Обучение и предсказание (fit_predict сразу возвращает метки: 1 — норма, -1 — выброс)
y_pred_default = lof_default.fit_predict(X)

# 5. Подсчёт числа аномалий в обычном режиме
n_anomalies_default = sum(y_pred_default == -1)
print(f"1. Число аномалий (LOF, novelty=False): {n_anomalies_default}")

# 6. Отфильтруем найденные аномалии (оставим только нормальные точки)
X_filtered = X[y_pred_default == 1]

# 7. Часть 2: LOF в режиме обнаружения новизны (novelty=True)
# Обучаем на отфильтрованных (чистых) данных
lof_novelty = LocalOutlierFactor(novelty=True)
lof_novelty.fit(X_filtered)

# 8. Прогоняем через эту модель ранее отфильтрованные аномалии (те, что были удалены)
anomalies_removed = X[y_pred_default == -1]
y_pred_novelty_on_anomalies = lof_novelty.predict(anomalies_removed)

# 9. Оценка числа совпадений: сколько из удалённых аномалий снова признаны аномалиями
coincidences = sum(y_pred_novelty_on_anomalies == -1)
print(f"3-4. Число совпадений (аномалии повторно найдены novelty=True): {coincidences} из {len(anomalies_removed)}")

# 10. Часть 5: Isolation Forest
# contamination='auto' — автоматическая оценка доли выбросов (по умолчанию)
iso_forest = IsolationForest(contamination='auto', random_state=42)
y_pred_iso = iso_forest.fit_predict(X)

n_anomalies_iso = sum(y_pred_iso == -1)
print(f"5. Число аномалий (Isolation Forest): {n_anomalies_iso}")

# 11. Часть 6: Метрики качества для задач обнаружения аномалий
# Поскольку у нас нет истинных меток, используем пример с синтетическими метками:
# Допустим, что истинные аномалии — те, что нашёл LOF (для демонстрации метрик)
true_labels = (y_pred_default == -1).astype(int)   # 1 — аномалия
pred_labels = (y_pred_iso == -1).astype(int)       # 1 — аномалия

print("\n6. Метрики качества (пример: LOF default как истина, Isolation Forest как предсказание):")
print(confusion_matrix(true_labels, pred_labels))
print(classification_report(true_labels, pred_labels, target_names=['норма', 'аномалия']))

# Дополнительное пояснение метрик
print("\n--- Краткое описание метрик в задачах Anomaly Detection ---")
print("1. Precision (Точность): TP / (TP + FP) — доля правильно найденных аномалий среди всех предсказанных аномалий.")
print("2. Recall (Полнота): TP / (TP + FN) — доля найденных истинных аномалий от их общего числа.")
print("3. F1-score: гармоническое среднее точности и полноты.")
print("4. ROC-AUC: способность модели различать классы при разных порогах.")
print("5. Confusion Matrix: показывает TP, TN, FP, FN.")
print("   - TP: аномалию предсказали как аномалию")
print("   - FN: аномалию предсказали как норму (пропуск)")
print("   - FP: норму предсказали как аномалию (ложная тревога)")
print("   - TN: норму предсказали как норму")