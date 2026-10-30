
import csv
import json
import threading
import time

import websocket

OUT = "data_2.csv"

dictt = {}
lock = threading.Lock()
stop_flag = threading.Event()


def on_message(ws, message):
    data = json.loads(message)
    vid = data.get("id")
    if vid is not None:
        with lock:
            dictt[vid] = message


def on_error(ws, error):
    # websocket-client перехватывает Ctrl+C и передаёт его сюда как ошибку
    # с пустым текстом, поэтому без этой проверки программу не остановить.
    if isinstance(error, KeyboardInterrupt):
        print("Ctrl+C: останавливаюсь...")
        stop_flag.set()
        return
    print("Error:", error)


def on_close(ws, close_status_code, close_msg):
    print("Connection closed:", close_status_code, close_msg)


def on_open(ws):
    print("Connection opened")


def flush():
    with lock:
        msgs_to_save = list(dictt.values())
        dictt.clear()
    if msgs_to_save:
        with open(OUT, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for msg in msgs_to_save:
                writer.writerow([msg])
        print(time.strftime("%H:%M:%S"), "записано строк:", len(msgs_to_save))


def save():
    while not stop_flag.is_set():
        time.sleep(30)
        flush()


csv_thread = threading.Thread(target=save)
csv_thread.daemon = True
csv_thread.start()

try:
    while not stop_flag.is_set():
        try:
            ws1 = websocket.WebSocketApp(
                "wss://ut.kb.gov.spb.ru/pub/",
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            ws1.run_forever(ping_timeout=10, ping_interval=20)
        except Exception as e:
            print("Exception during connection:", e)

        if not stop_flag.is_set():
            time.sleep(5)
except KeyboardInterrupt:
    stop_flag.set()

flush()
print("Остановлено, данные сохранены в", OUT)
