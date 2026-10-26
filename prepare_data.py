import time
import ccxt
import pandas as pd
import numpy as np

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from pathlib import Path


# Настройки проекта
SYMBOL = "BTC/USDT:USDT"
TIMEFRAME = "5m"
CANDLE_LIMIT = 20000

# Папка для данных
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_DIR / "btc_usdt_5m.csv"
NEW_DATA_FILE = DATA_DIR / "btc_usdt_5m_new.csv"


def create_exchange():
    """Подключение к публичному API Binance Futures."""

    exchange = ccxt.binanceusdm({
        "enableRateLimit": True,
        "timeout": 20000,
    })

    return exchange


def fetch_new_candles(exchange):
    if not OUTPUT_FILE.exists():
        raise FileNotFoundError(
            f"Не найден основной файл: {OUTPUT_FILE}"
        )

    if NEW_DATA_FILE.exists():
        old_df = pd.read_csv(NEW_DATA_FILE)
        print("Продолжаем загрузку накопленных данных...")
    else:
        old_df = pd.read_csv(OUTPUT_FILE)
        print("Начинаем загрузку после основного датасета...")

    old_df["timestamp"] = pd.to_datetime(
        old_df["timestamp"], utc=True
    )

    last_timestamp = old_df["timestamp"].max()

    timeframe_ms = exchange.parse_timeframe(TIMEFRAME) * 1000

    since = int(last_timestamp.timestamp() * 1000) + timeframe_ms

    now_ms = exchange.milliseconds()
    last_closed_open = (now_ms // timeframe_ms) * timeframe_ms

    candles = []

    print(f"Последняя старая свеча: {last_timestamp}")
    print("Загружаем новые закрытые свечи...")

    while since < last_closed_open:
        batch = exchange.fetch_ohlcv(
            SYMBOL,
            timeframe=TIMEFRAME,
            since=since,
            limit=1000,
        )

        if not batch:
            break

        valid_batch = [
            row for row in batch
            if since <= row[0] < last_closed_open
        ]

        if not valid_batch:
            break

        candles.extend(valid_batch)

        next_since = max(row[0] for row in valid_batch) + timeframe_ms

        if next_since <= since:
            break

        since = next_since

        time.sleep(exchange.rateLimit / 1000)

    if not candles:
        print("Новых закрытых свечей пока нет.")
        return

    new_df = pd.DataFrame(
        candles,
        columns=[
            "timestamp", "open", "high",
            "low", "close", "volume",
        ],
    )

    new_df["timestamp"] = pd.to_datetime(
        new_df["timestamp"], unit="ms", utc=True
    )

    new_df = (
        new_df.drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if NEW_DATA_FILE.exists():
        saved_df = pd.read_csv(NEW_DATA_FILE)

        saved_df["timestamp"] = pd.to_datetime(
            saved_df["timestamp"], utc=True
        )

        new_df = pd.concat(
            [saved_df, new_df],
            ignore_index=True,
        )

    new_df = (
        new_df.drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    new_df.to_csv(NEW_DATA_FILE, index=False)

    print(f"Всего накоплено новых свечей: {len(new_df)}")
    print(f"Файл сохранён: {NEW_DATA_FILE}")


def fetch_ohlcv(exchange):
    print(
        f"Загружаем {CANDLE_LIMIT} свечей {SYMBOL}, таймфрейм {TIMEFRAME}...")

    timeframe_ms = exchange.parse_timeframe(TIMEFRAME) * 1000

    now_ms = exchange.milliseconds()
    last_closed_open = (now_ms // timeframe_ms) * timeframe_ms

    since = last_closed_open - CANDLE_LIMIT * timeframe_ms

    candles = []
    seen = set()

    while since < last_closed_open and len(candles) < CANDLE_LIMIT:
        remaining = CANDLE_LIMIT - len(candles)
        limit = min(1000, remaining)

        batch = exchange.fetch_ohlcv(
            SYMBOL,
            timeframe=TIMEFRAME,
            since=since,
            limit=limit,
        )

        if not batch:
            break

        valid_batch = [
            row for row in batch
            if since <= row[0] < last_closed_open
            and row[0] not in seen
        ]

        if not valid_batch:
            latest_timestamp = max(row[0] for row in batch)

            if latest_timestamp >= last_closed_open:
                break

            if latest_timestamp < since:
                break

            since = latest_timestamp + timeframe_ms
            continue

        for row in valid_batch:
            seen.add(row[0])
            candles.append(row)

        latest_timestamp = max(row[0] for row in valid_batch)
        next_since = latest_timestamp + timeframe_ms

        if next_since <= since:
            break

        since = next_since

        print(
            f"\rЗагружено: {len(candles)}/{CANDLE_LIMIT}",
            end="",
            flush=True,
        )

        time.sleep(exchange.rateLimit / 1000)

    print()

    if not candles:
        raise RuntimeError("Binance не вернула исторические свечи.")

    df = pd.DataFrame(
        candles,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ],
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    df = (
        df.drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(f"Получено свечей: {len(df)}")

    return df


def add_features(df):
    """Расчёт технических индикаторов для ML."""

    df = df.copy()

    # RSI (14)
    df["rsi_14"] = RSIIndicator(
        close=df["close"],
        window=14,
    ).rsi()

    # ATR (14)
    df["atr_14"] = AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
    ).average_true_range()

    # ATR в процентах от цены
    df["atr_pct"] = df["atr_14"] / df["close"] * 100

    # Динамический ATR относительно среднего за 100 свечей
    atr_average = df["atr_14"].rolling(100).mean()

    df["atr_dynamic"] = (
        df["atr_14"] / atr_average.replace(0, np.nan)
    )

    # EMA 50 и EMA 200
    df["ema_50"] = EMAIndicator(
        close=df["close"],
        window=50,
    ).ema_indicator()

    df["ema_200"] = EMAIndicator(
        close=df["close"],
        window=200,
    ).ema_indicator()

    # Положение цены относительно EMA
    df["ema_distance"] = (
        df["close"] / df["ema_50"] - 1
    )

    # Разница между EMA 50 и EMA 200
    df["ema_trend"] = (
        df["ema_50"] / df["ema_200"] - 1
    )

    # Полосы Боллинджера (20, 2)
    bb = BollingerBands(
        close=df["close"],
        window=20,
        window_dev=2,
    )

    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    df["bb_width"] = (
        (df["bb_upper"] - df["bb_lower"])
        / df["close"]
    )

    df["bb_position"] = (
        (df["close"] - df["bb_lower"])
        / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    )

    # Изменение цены за 1, 3 и 5 свечей
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    # Отношение объёма к среднему за 20 свечей
    volume_average = df["volume"].rolling(20).mean()

    df["volume_ratio"] = (
        df["volume"] / volume_average.replace(0, np.nan)
    )

    # Убираем бесконечные значения
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def add_target(df):
    """Создаёт метку направления цены через 5 свечей."""

    df = df.copy()

    forecast_candles = 5
    atr_threshold = 0.25

    # Будущая доходность через 5 свечей
    future_return = (
        df["close"].shift(-forecast_candles)
        / df["close"] - 1
    )

    # Порог движения, рассчитанный по текущему ATR
    threshold = (
        atr_threshold * df["atr_14"] / df["close"]
    )

    # 1 — рост, 0 — снижение, NaN — слабое движение
    df["target"] = np.select(
        [
            future_return > threshold,
            future_return < -threshold,
        ],
        [1, 0],
        default=np.nan,
    )

    return df


def create_training_dataset(df):
    """Очищает данные и сохраняет датасет для LightGBM."""

    feature_columns = [
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

    # Оставляем признаки и целевую переменную.
    dataset = df[["timestamp"] + feature_columns + ["target"]].copy()

    # Удаляем строки с пропущенными и бесконечными значениями.
    dataset = dataset.replace([np.inf, -np.inf], np.nan)

    dataset = dataset.dropna(
        subset=feature_columns + ["target"]
    )

    dataset["target"] = dataset["target"].astype(int)

    # Сохраняем временной порядок.
    dataset = dataset.sort_values("timestamp").reset_index(drop=True)

    output_file = DATA_DIR / "training_dataset.csv"
    dataset.to_csv(output_file, index=False)

    print("\n========== ОБУЧАЮЩИЙ ДАТАСЕТ ==========")
    print(f"Файл: {output_file}")
    print(f"Количество строк: {len(dataset)}")
    print(f"Количество признаков: {len(feature_columns)}")

    print("\nРаспределение классов:")
    print(dataset["target"].value_counts().sort_index())

    print("\nПервые 5 строк:")
    print(dataset.head().to_string(index=False))

    return dataset


def main():
    exchange = create_exchange()

    try:
        exchange.load_markets()

        df = fetch_ohlcv(exchange)
        df = add_features(df)
        df = add_target(df)
        create_training_dataset(df)
        df.to_csv(OUTPUT_FILE, index=False)

        print("\nЗагрузка завершена.")
        print(f"Файл сохранён: {OUTPUT_FILE}")

        print("\nПоследние 5 свечей с индикаторами:")

        columns_to_show = [
            "timestamp",
            "close",
            "rsi_14",
            "atr_14",
            "atr_dynamic",
            "ema_50",
            "ema_200",
            "bb_width",
            "volume_ratio",
            "target",
        ]

        print(
            df[columns_to_show]
            .tail()
            .to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )

        print("\nРаспределение целевой переменной:")
        print(df["target"].value_counts(dropna=False).sort_index())

        # Последние 5 строк не имеют известного будущего результата.
        print("\nСтрок без будущей метки:", df["target"].isna().sum())

    except Exception as error:
        print(f"\nОшибка: {error}")

    finally:
        exchange.close()


if __name__ == "__main__":
    main()
