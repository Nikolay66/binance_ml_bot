
from pathlib import Path
import json

import joblib
import lightgbm as lgb
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "training_dataset.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "lightgbm_model.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"

FEATURES = [
    "rsi_14",
    "atr_14",
    "atr_pct",
    "atr_dynamic",
    "ema_distance",
    "ema_trend",
    "bb_width",
    "bb_position",
    "return_1",
    "return_3",
    "return_5",
    "volume_ratio",
]


def load_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Не найден датасет: {DATA_FILE}\n"
            "Сначала запусти python prepare_data.py"
        )

    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    required = FEATURES + ["target", "timestamp"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"В датасете отсутствуют столбцы: {missing}")

    df = df.dropna(subset=FEATURES + ["target"]).copy()
    df["target"] = df["target"].astype(int)

    if not set(df["target"].unique()).issubset({0, 1}):
        raise ValueError("Target должен содержать только классы 0 и 1.")

    return df


def evaluate_model(df):
    X = df[FEATURES]
    y = df["target"]

    if len(df) < 100:
        raise ValueError("Для временной проверки недостаточно данных.")

    # Последний участок оставляем для финальной проверки.
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]

    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "В обучающем или тестовом участке отсутствует один из классов."
        )

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=5,
        min_child_samples=30,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1,
    )

    # Внутренняя временная кросс-валидация на обучающем участке.
    tscv = TimeSeriesSplit(n_splits=3)
    fold_scores = []

    print("\n========== TIME SERIES SPLIT ==========")

    for fold, (train_idx, val_idx) in enumerate(
        tscv.split(X_train), start=1
    ):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]

        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        if (
            y_fold_train.nunique() < 2
            or y_fold_val.nunique() < 2
        ):
            print(f"Fold {fold}: пропущен — представлен не каждый класс.")
            continue

        fold_model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=5,
            min_child_samples=30,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            verbosity=-1,
        )

        fold_model.fit(X_fold_train, y_fold_train)

        predictions = fold_model.predict(X_fold_val)

        score = balanced_accuracy_score(
            y_fold_val, predictions
        )

        fold_scores.append(score)

        print(
            f"Fold {fold}: "
            f"Balanced Accuracy = {score:.4f}"
        )

    # Финальная оценка на более позднем отрезке.
    model.fit(X_train, y_train)

    # Обычный прогноз: 0 = SHORT, 1 = LONG.
    predictions = model.predict(X_test)

    # Вероятности каждого класса.
    probabilities = model.predict_proba(X_test)

    # Вероятность класса LONG (1).
    long_probabilities = probabilities[:, 1]

    # Уверенность модели в выбранном направлении.
    confidence = probabilities.max(axis=1)

    # Проверка калибровки вероятностей модели
    print("\n========== КАЛИБРОВКА УВЕРЕННОСТИ ==========")

    confidence_bins = [
        (0.50, 0.55),
        (0.55, 0.60),
        (0.60, 0.65),
        (0.65, 0.70),
        (0.70, 1.01),
    ]

    correct = (predictions == y_test.to_numpy())

    for lower, upper in confidence_bins:
        mask = (
            (confidence >= lower)
            & (confidence < upper)
        )

        count = int(mask.sum())

        if count == 0:
            print(
                f"{lower:.0%}–{min(upper, 1.0):.0%}: "
                "нет прогнозов"
            )
            continue

        actual_accuracy = correct[mask].mean()
        mean_confidence = confidence[mask].mean()

        print(
            f"{lower:.0%}–{min(upper, 1.0):.0%} | "
            f"Прогнозов: {count} | "
            f"Средняя уверенность: {mean_confidence:.2%} | "
            f"Фактическая точность: {actual_accuracy:.2%}"
        )

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(
        y_test, predictions
    )

    # Эксперимент: сравнение 12 признаков с 9 признаками.
    print("\n========== СРАВНЕНИЕ ПРИЗНАКОВ ==========")

    reduced_features = [
        feature for feature in FEATURES
        if feature not in ["return_1", "return_3", "return_5"]
    ]

    experiment_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=5,
        min_child_samples=30,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1,
    )

    experiment_model.fit(
        X_train[reduced_features],
        y_train,
    )

    experiment_predictions = experiment_model.predict(
        X_test[reduced_features]
    )

    experiment_accuracy = accuracy_score(
        y_test,
        experiment_predictions,
    )

    experiment_balanced_accuracy = balanced_accuracy_score(
        y_test,
        experiment_predictions,
    )

    print(f"Модель A — все признаки ({len(FEATURES)}):")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Balanced Accuracy: {balanced_accuracy:.4f}")

    print(f"Модель B — без трёх return-признаков ({len(reduced_features)}):")
    print(f"  Accuracy: {experiment_accuracy:.4f}")
    print(
        f"  Balanced Accuracy: "
        f"{experiment_balanced_accuracy:.4f}"
    )

    # Важность признаков модели
    print("\n📊 ВАЖНОСТЬ ПРИЗНАКОВ")

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        by="importance",
        ascending=False,
    )

    for _, row in importance.iterrows():
        print(
            f"{row['feature']:<20} "
            f"{int(row['importance'])}"
        )

    print("\n========== ФИНАЛЬНЫЙ ТЕСТ ==========")
    print(f"Обучающих строк: {len(X_train)}")
    print(f"Тестовых строк: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

    if fold_scores:
        print(
            "Средняя CV Balanced Accuracy: "
            f"{sum(fold_scores) / len(fold_scores):.4f}"
        )

    print("\n========== АНАЛИЗ УВЕРЕННОСТИ ==========")

    # Проверяем несколько порогов уверенности.
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]

    for threshold in thresholds:
        mask = confidence >= threshold

        count = int(mask.sum())

        if count == 0:
            print(
                f"Уверенность >= {threshold:.0%}: "
                "сигналов нет"
            )
            continue

        selected_predictions = predictions[mask]
        selected_actual = y_test.iloc[mask]

        score = balanced_accuracy_score(
            selected_actual,
            selected_predictions,
        )

        accuracy_selected = accuracy_score(
            selected_actual,
            selected_predictions,
        )

        print(
            f"Уверенность >= {threshold:.0%} | "
            f"Сигналов: {count} из {len(X_test)} | "
            f"Accuracy: {accuracy_selected:.2%} | "
            f"Balanced Accuracy: {score:.2%}"
        )

    # Отдельно проверяем прогнозы LONG и SHORT.
    print("\nСредняя вероятность LONG:")
    print(f"{long_probabilities.mean():.2%}")

    print("Средняя уверенность выбранного направления:")
    print(f"{confidence.mean():.2%}")

    print("\nОтчёт по классам:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["SHORT", "LONG"],
            zero_division=0,
        )
    )

    # Сохраняем модель и метрики.
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
        },
        MODEL_FILE,
    )

    metrics = {
        "rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "cv_balanced_accuracy": (
            sum(fold_scores) / len(fold_scores)
            if fold_scores else None
        ),
        "features": FEATURES,
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=4)

    print(f"\nМодель сохранена: {MODEL_FILE}")
    print(f"Метрики сохранены: {METRICS_FILE}")

    return model


def main():
    df = load_dataset()
    print(f"Загружено строк для обучения: {len(df)}")

    evaluate_model(df)


if __name__ == "__main__":
    main()

