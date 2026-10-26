
import time

from prepare_data import create_exchange, fetch_new_candles


INTERVAL_SECONDS = 300  # 5 минут


def main():
    exchange = create_exchange()

    try:
        exchange.load_markets()

        print("Автоматическое накопление свечей запущено.")
        print("Проверка новых свечей: каждые 5 минут.")
        print("Для остановки нажми Ctrl+C.\n")

        while True:
            try:
                fetch_new_candles(exchange)

            except Exception as error:
                print(f"Ошибка загрузки: {error}")

            print("\nСледующая проверка через 5 минут.")

            time.sleep(INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nНакопление данных остановлено.")

    finally:
        exchange.close()


if __name__ == "__main__":
    main()
