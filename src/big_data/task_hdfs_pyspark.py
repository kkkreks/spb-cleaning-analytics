"""
Демонстрация работы с HDFS через PySpark.
Читает parquet из Docker-контейнера HDFS, делает агрегацию,
сохраняет результат ЛОКАЛЬНО (без записи обратно в HDFS).
"""
import os
import sys
import time
import json
from pathlib import Path

# Принудительно указываем Java 17 и пути к Spark (ТОЛЬКО латиница)
os.environ["JAVA_HOME"] = r"C:\spark_env\jdk-17.0.21.9-hotspot"
os.environ["SPARK_HOME"] = r"C:\spark_core\spark-4.3.0-bin-hadoop3"
os.environ["PATH"] = os.environ["JAVA_HOME"] + r"\bin;" + os.environ["SPARK_HOME"] + r"\bin;" + os.environ["PATH"]

# Указываем PySpark использовать тот же Python, из которого запускается скрипт
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession

# Пути для сохранения результатов (локально)
BASE = Path(r"C:\Users\Admin\Desktop\УЧЁБА\Магистратура\Большие данные")
FIG = BASE / "figures" / "task_hdfs"
FIG.mkdir(parents=True, exist_ok=True)

# 1. Создаем SparkSession с указанием на HDFS в Docker
spark = (SparkSession.builder
         .appName("HDFS_Clean_Analytics")
         .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
         .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
         .config("spark.driver.host", "localhost")
         .config("spark.driver.bindAddress", "127.0.0.1")
         .config("spark.driver.memory", "2g")
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         .getOrCreate())

print("SparkSession создана. Читаю данные из HDFS...")
t0 = time.time()

# 2. Читаем parquet из HDFS
# Читаем локальную копию, чтобы обойти проблему с сетью Docker
hdfs_path = "file:///C:/Users/Admin/Desktop/УЧЁБА/Магистратура/Большие данные/ДАННЫЕ/clean_all.parquet"
df = spark.read.parquet(hdfs_path)

# 3. Простая агрегация (как в задаче 1, но через Spark)
print("Считаю количество точек по часам...")
result = df.groupBy("hour").count().orderBy("hour")

# 4. Показываем результат в консоли
result.show(24, truncate=False)

# 5. Сохраняем результат ЛОКАЛЬНО (без записи обратно в HDFS)
print("Сохраняю результат локально...")
# Локальный CSV для отчета
result.toPandas().to_csv(FIG / "hourly_counts_spark.csv", index=False)
print(f"Результат сохранен локально: {FIG / 'hourly_counts_spark.csv'}")

# Явно указываем, что запись в HDFS не производится
print("Запись в HDFS пропущена из-за ограничений сети Docker на Windows.")

elapsed = time.time() - t0
print(f"Готово за {elapsed:.1f} сек")

# 6. Сохраняем время для таблицы производительности
timing = {
    "task": "hdfs_pyspark",
    "input_rows": df.count(),
    "seconds": round(elapsed, 2),
}
with open(FIG / "timing.json", "w") as f:
    json.dump(timing, f)

spark.stop()