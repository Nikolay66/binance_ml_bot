
from pathlib import Path

import pandas as pd
import lightgbm as lgb

from sklearn.metrics import accuracy_score, balanced_accuracy_score


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "btc_usdt_5m.csv"
OUTPUT_DIR = BASE_DIR / "data"

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

TRAIN_RATIO = 0.80
FORECAST_CANDLES = 5

CONFIDENCE_LEVELS = [0.65, 0.70]

ATR_STOP = 1.0
ATR_TAKE_PROFIT = 2.0
FEE_RATE = 0.0005


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Не найден файл: {DATA_FILE}. "
            "Сначала запусти python prepare_data.py"
        )

    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    required = FEATURES + [
        "timestamp", "open", "high", "low", "close", "atr_14"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"В данных отсутствуют столбцы: {missing}")

    df = df.dropna(subset=FEATURES + ["atr_14"]).copy()
    df = df.reset_index(drop=True)

    # Та же логика целевой переменной, что и в prepare_data.py.
    future_return = (
        df["close"].shift(-FORECAST_CANDLES) / df["close"] - 1
    )
    threshold = 0.25 * df["atr_14"] / df["close"]

    df["target"] = pd.NA
    df.loc[future_return > threshold, "target"] = 1
    df.loc[future_return < -threshold, "target"] = 0

    return df


def create_model():
    return lgb.LGBMClassifier(
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


def run_backtest(df, model, confidence_level):
    test = df.copy().reset_index(drop=True)

    probabilities = model.predict_proba(test[FEATURES])
    class_positions = {
        int(value): index
        for index, value in enumerate(model.classes_)
    }

    test["p_short"] = probabilities[:, class_positions[0]]
    test["p_long"] = probabilities[:, class_positions[1]]
    test["confidence"] = probabilities.max(axis=1)

    trades = []
    position = None

    # Используем закрытую свечу для сигнала,
    # следующую свечу — для входа.
    for i in range(len(test) - 1):
        candle = test.iloc[i]
        next_candle = test.iloc[i + 1]

        if position is not None:
            stop_hit = False
            target_hit = False
            exit_price = None
            exit_reason = None

            if position["side"] == "LONG":
                stop_hit = candle["low"] <= position["stop"]
                target_hit = candle["high"] >= position["target"]
            else:
                stop_hit = candle["high"] >= position["stop"]
                target_hit = candle["low"] <= position["target"]

            # Если в одной свече достигнуты обе цены,
            # используем консервативный сценарий: стоп первым.
            if stop_hit:
                exit_price = position["stop"]
                exit_reason = "SL"
            elif target_hit:
                exit_price = position["target"]
                exit_reason = "TP"

            if exit_price is not None:
                if position["side"] == "LONG":
                    gross_return = (
                        exit_price - position["entry"]
                    ) / position["entry"]
                else:
                    gross_return = (
                        position["entry"] - exit_price
                    ) / position["entry"]

                fees = FEE_RATE * (
                    1 + exit_price / position["entry"]
                )

                net_return = gross_return - fees
                risk_fraction = (
                    abs(position["entry"] - position["stop"])
                    / position["entry"]
                )

                trades.append({
                    "side": position["side"],
                    "entry_time": position["entry_time"],
                    "exit_time": candle["timestamp"],
                    "entry": position["entry"],
                    "exit": exit_price,
                    "reason": exit_reason,
                    "gross_return_pct": gross_return * 100,
                    "net_return_pct": net_return * 100,
                    "result_r": net_return / risk_fraction,
                    "risk_fraction": risk_fraction,
                })

                position = None
                continue

        if position is not None:
            continue

        if candle["confidence"] < confidence_level:
            continue

        side = (
            "LONG"
            if candle["p_long"] > candle["p_short"]
            else "SHORT"
        )

        entry = float(next_candle["open"])
        atr = float(candle["atr_14"])

        if pd.isna(atr) or atr <= 0 or entry <= 0:
            continue

        if side == "LONG":
            stop = entry - ATR_STOP * atr
            target = entry + ATR_TAKE_PROFIT * atr
        else:
            stop = entry + ATR_STOP * atr
            target = entry - ATR_TAKE_PROFIT * atr

        position = {
            "side": side,
            "entry_time": next_candle["timestamp"],
            "entry": entry,
            "stop": stop,
            "target": target,
        }

    # Закрываем оставшуюся позицию по последней доступной цене.
    if position is not None:
        last = test.iloc[-1]
        exit_price = float(last["close"])

        if position["side"] == "LONG":
            gross_return = (
                exit_price - position["entry"]
            ) / position["entry"]
        else:
            gross_return = (
                position["entry"] - exit_price
            ) / position["entry"]

        fees = FEE_RATE * (
            1 + exit_price / position["entry"]
        )

        net_return = gross_return - fees
        risk_fraction = (
            abs(position["entry"] - position["stop"])
            / position["entry"]
        )

        trades.append({
            "side": position["side"],
            "entry_time": position["entry_time"],
            "exit_time": last["timestamp"],
            "entry": position["entry"],
            "exit": exit_price,
            "reason": "END",
            "gross_return_pct": gross_return * 100,
            "net_return_pct": net_return * 100,
            "result_r": net_return / risk_fraction,
            "risk_fraction": risk_fraction,
        })

    results = pd.DataFrame(trades)

    if results.empty:
        print("Нет сделок для анализа.")
        return

    # Разделяем результат сделки до и после комиссий
    results["fees_r"] = (
        FEE_RATE * (1 + results["exit"] / results["entry"])
    ) / results["risk_fraction"]

    results["gross_r"] = results["result_r"] + results["fees_r"]

    print("\n📊 ПОДРОБНЫЙ АНАЛИЗ БЭКТЕСТА")
    print("-" * 45)

    print(f"Количество сделок: {len(results)}")
    print(f"Результат до комиссий: {results['gross_r'].sum():.2f}R")
    print(f"Комиссии всего:       {results['fees_r'].sum():.2f}R")
    print(f"Результат после комиссий: {results['result_r'].sum():.2f}R")
    print(f"Средняя комиссия на сделку: {results['fees_r'].mean():.3f}R")

    winning = results[results["result_r"] > 0]
    losing = results[results["result_r"] < 0]

    print(f"\nПрибыльных сделок: {len(winning)}")
    print(f"Убыточных сделок: {len(losing)}")

    if not winning.empty:
        print(f"Средняя прибыльная сделка: {winning['result_r'].mean():.3f}R")

    if not losing.empty:
        print(f"Средняя убыточная сделка: {losing['result_r'].mean():.3f}R")

    print("\nПо направлениям:")

    for side in ["LONG", "SHORT"]:
        subset = results[results["side"] == side]

        if subset.empty:
            continue

        win_rate = (subset["result_r"] > 0).mean() * 100

        print(
            f"{side}: сделок {len(subset)}, "
            f"Win Rate {win_rate:.2f}%, "
            f"результат {subset['result_r'].sum():.2f}R"
        )

    print("\nПо причинам закрытия:")

    for reason in ["TP", "SL", "END"]:
        subset = results[results["reason"] == reason]

        if not subset.empty:
            print(
                f"{reason}: {len(subset)} сделок, "
                f"результат {subset['result_r'].sum():.2f}R"
            )

    print(
        f"\n========== INDEPENDENT BACKTEST "
        f"{confidence_level:.0%} =========="
    )
    print(f"Завершённых сделок: {len(results)}")

    if results.empty:
        print("Сделок нет.")
        return

    wins = results["net_return_pct"] > 0
    win_rate = wins.mean() * 100

    total_r = results["result_r"].sum()
    average_r = results["result_r"].mean()

    gross_profit = results.loc[
        results["result_r"] > 0, "result_r"
    ].sum()
    gross_loss = abs(results.loc[
        results["result_r"] < 0, "result_r"
    ].sum())

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0 else float("inf")
    )

    equity = results["result_r"].cumsum()
    drawdown = equity - equity.cummax()

    print(f"Win Rate после комиссий: {win_rate:.2f}%")
    print(f"Суммарный результат: {total_r:.2f}R")
    print(f"Средний результат: {average_r:.3f}R")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Максимальная просадка: {drawdown.min():.2f}R")
    print(f"LONG: {(results['side'] == 'LONG').sum()}")
    print(f"SHORT: {(results['side'] == 'SHORT').sum()}")
    print(f"TP: {(results['reason'] == 'TP').sum()}")
    print(f"SL: {(results['reason'] == 'SL').sum()}")
    print(f"Закрытие в конце теста: {(results['reason'] == 'END').sum()}")

    output_file = OUTPUT_DIR / (
        f"independent_backtest_{int(confidence_level * 100)}.csv"
    )
    results.to_csv(output_file, index=False)
    print(f"Сделки сохранены: {output_file}")


def main():
    df = load_data()

    split_index = int(len(df) * TRAIN_RATIO)

    # Не допускаем, чтобы метки обучающих строк использовали
    # цены из тестового периода.
    train_end = split_index - FORECAST_CANDLES

    train = df.iloc[:train_end].dropna(subset=["target"]).copy()
    test = df.iloc[split_index:-1].copy()

    if train.empty or test.empty:
        raise ValueError("Недостаточно данных для независимого теста.")

    train["target"] = train["target"].astype(int)

    print(f"Всего строк: {len(df)}")
    print(f"Обучающих строк: {len(train)}")
    print(f"Тестовых строк: {len(test)}")
    print(f"Начало теста: {test['timestamp'].iloc[0]}")
    print(f"Конец теста: {test['timestamp'].iloc[-1]}")

    model = create_model()
    model.fit(train[FEATURES], train["target"])

    test_labels = test.dropna(subset=["target"]).copy()
    if not test_labels.empty:
        predictions = model.predict(test_labels[FEATURES])
        print(
            "Тестовая Balanced Accuracy: "
            f"{balanced_accuracy_score(test_labels['target'].astype(int), predictions):.4f}"
        )
        print(
            "Тестовая Accuracy: "
            f"{accuracy_score(test_labels['target'].astype(int), predictions):.4f}"
        )

        # Проверка точности при разных уровнях уверенности
        print("\n📊 ТОЧНОСТЬ ПО УРОВНЯМ УВЕРЕННОСТИ")

        probabilities = model.predict_proba(test[FEATURES])
        p_short = probabilities[:, 0]
        p_long = probabilities[:, 1]

        confidence = probabilities.max(axis=1)
        predictions = probabilities.argmax(axis=1)

        valid_mask = test["target"].notna().to_numpy()
        y_true_all = test["target"].to_numpy()

        for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
            mask = (confidence >= threshold) & valid_mask

            count = int(mask.sum())

            if count == 0:
                print(f"{threshold:.0%}: нет прогнозов")
                continue

            accuracy = accuracy_score(
                y_true_all[mask].astype(int),
                predictions[mask]
            )

            balanced = balanced_accuracy_score(
                y_true_all[mask].astype(int),
                predictions[mask]
            )

            print(
                f"Уверенность >= {threshold:.0%}: "
                f"{count} прогнозов, "
                f"Accuracy {accuracy:.2%}, "
                f"Balanced Accuracy {balanced:.2%}"
            )

    for confidence_level in CONFIDENCE_LEVELS:
        run_backtest(test, model, confidence_level)


if __name__ == "__main__":
    main()
