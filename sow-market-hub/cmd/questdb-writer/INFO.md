## 🚀 Почему QuestDB?

**QuestDB** - это высокопроизводительная база данных временных рядов, специально оптимизированная для финансовых данных и real-time аналитики.

### Преимущества для криптовалютных данных:

- ⚡ **Экстремальная скорость записи** - до 4M+ записей в секунду
- 📊 **Нативная поддержка временных рядов** - оптимизирована для time-series данных
- 💾 **Эффективное сжатие** - экономия до 90% дискового пространства
- 🔍 **Быстрые аналитические запросы** - LATEST BY, время-базированные агрегации
- 🐘 **PostgreSQL совместимость** - знакомый SQL синтаксис
- 🏗️ **Простота развертывания** - минимальная настройка

## 📈 Производительность

| Метрика | QuestDB | TimescaleDB | InfluxDB |
| --- | --- | --- | --- |
| **Запись (записей/сек)** | 4,000,000+ | 1,000,000 | 500,000 |
| **Потребление RAM** | Очень низкое | Среднее | Высокое |
| **Размер на диске** | Минимальный | Средний | Большой |
| **Скорость запросов** | Очень быстро | Быстро | Медленно |

## 🏗️ Архитектура данных

Real-time Flow:
Bybit WebSocket → NATS → Dragonfly (Cache) → QuestDB (Long-term)
↓
WebSocket → Frontend

```

### Структура таблицы market_data:

```sql
CREATE TABLE market_data (
    timestamp TIMESTAMP,    -- Время сделки
    symbol SYMBOL,         -- Торговая пара (BTCUSDT, ETHUSDT)
    market SYMBOL,         -- Тип рынка (SPOT, FUTURES)
    price DOUBLE,          -- Цена
    volume DOUBLE,         -- Объем
    high24h DOUBLE,        -- Максимум за 24ч
    low24h DOUBLE,         -- Минимум за 24ч
    change24h DOUBLE       -- Изменение за 24ч %
) TIMESTAMP(timestamp) PARTITION BY DAY;

```

## 🔧 Развертывание

### Docker Compose

```yaml
services:
  questdb:
    image: questdb/questdb:latest
    ports:
      - "9000:9000"    # Web Console
      - "8812:8812"    # Postgres wire protocol
      - "9009:9009"    # InfluxDB line protocol
    volumes:
      - questdb_data:/var/lib/questdb
    environment:
      - QDB_CAIRO_SQL_COPY_WORK_ROOT=/tmp
      - QDB_PG_ENABLED=true
      - QDB_HTTP_ENABLED=true

```

### Запуск

```bash
# Запуск QuestDB
docker-compose up questdb -d

# Запуск QuestDB Writer
go run cmd/questdb-writer/main.go

```

## 📊 Web Console

QuestDB поставляется с встроенной веб-консолью для мониторинга и выполнения запросов:

- **URL:** [http://localhost:9000](http://localhost:9000/)
- **Функции:**
    - SQL редактор с автодополнением
    - Визуализация данных в реальном времени
    - Мониторинг производительности
    - Управление таблицами и индексами

## 🔍 Примеры запросов

### Получить последние цены всех активов

```sql
SELECT timestamp, symbol, market, price
FROM market_data
LATEST BY symbol, market;

```

### Статистика за последние 24 часа

```sql
SELECT
    symbol,
    market,
    min(price) as min_price,
    max(price) as max_price,
    avg(price) as avg_price,
    first(price) as open_price,
    last(price) as close_price,
    count() as data_points
FROM market_data
WHERE timestamp >= dateadd('h', -24, now())
GROUP BY symbol, market;

```

### Почасовые OHLC данные

```sql
SELECT
    symbol,
    market,
    sample_by(1h, timestamp) as hour,
    first(price) as open,
    max(price) as high,
    min(price) as low,
    last(price) as close,
    sum(volume) as volume
FROM market_data
WHERE symbol = 'BTCUSDT'
AND timestamp >= dateadd('d', -7, now())
SAMPLE BY 1h ALIGN TO CALENDAR;

```

### Топ изменений за день

```sql
WITH price_changes AS (
    SELECT
        symbol,
        market,
        first(price) as open_price,
        last(price) as close_price,
        ((last(price) - first(price)) / first(price)) * 100 as change_pct
    FROM market_data
    WHERE timestamp >= dateadd('d', -1, now())
    GROUP BY symbol, market
)
SELECT * FROM price_changes
ORDER BY change_pct DESC
LIMIT 10;

```

## 🎯 Оптимизации

### Автоматические оптимизации QuestDB:

1. **Колоночное хранение** - эффективное сжатие и быстрые агрегации
2. **Партиционирование по времени** - автоматическое разделение данных по дням
3. **SYMBOL индексы** - мгновенный поиск по торговым парам
4. **Векторизованные вычисления** - использование SIMD инструкций
5. **Zero-copy операции** - минимальное потребление памяти

### Рекомендации по производительности:

```sql
-- Используйте LATEST BY для получения последних значений
SELECT * FROM market_data LATEST BY symbol, market;

-- Используйте SAMPLE BY для агрегации по времени
SELECT avg(price) FROM market_data SAMPLE BY 1m;

-- Фильтруйте по времени для лучшей производительности
WHERE timestamp >= dateadd('h', -1, now())

```

## 📡 API Endpoints

Встроенное REST API для доступа к данным:

```bash
# Последние цены всех активов
GET /api/prices/latest

# Исторические данные конкретного актива
GET /api/prices/BTCUSDT/SPOT/history?from=2024-01-01T00:00:00Z&to=2024-01-02T00:00:00Z

# Статистика по активу
GET /api/prices/BTCUSDT/SPOT/stats

```

## 🔧 Мониторинг

### Ключевые метрики для мониторинга:

- **Скорость записи** - записей в секунду
- **Использование диска** - рост размера данных
- **Время выполнения запросов** - производительность аналитики
- **Количество подключений** - нагрузка на систему

### Проверка здоровья системы:

```sql
-- Проверка последних записей
SELECT count() FROM market_data
WHERE timestamp >= dateadd('m', -1, now());

-- Размер таблицы
SELECT pg_size_pretty(pg_total_relation_size('market_data'));

-- Статистика по партициям
SHOW PARTITIONS FROM market_data;

```

## 🚀 Масштабирование

### Горизонтальное масштабирование:

- **Кластеризация** - распределение данных по узлам
- **Репликация** - резервное копирование данных
- **Шардинг** - разделение по торговым парам или времени

### Вертикальное масштабирование:

- **SSD диски** - для максимальной скорости записи
- **Больше RAM** - для кеширования индексов
- **Многоядерные процессоры** - для параллельных запросов

## 🔐 Безопасность

- **Аутентификация** - базовая HTTP авторизация
- **Шифрование** - TLS для сетевых соединений
- **Резервное копирование** - регулярные снапшоты данных
- **Мониторинг доступа** - логирование всех операций

## 📚 Дополнительные ресурсы

- [Официальная документация QuestDB](https://questdb.io/docs/)
- [SQL Reference](https://questdb.io/docs/reference/sql/)
- [Performance Guide](https://questdb.io/docs/operations/performance/)
- [Community Discord](https://discord.gg/bwvFJHj)

```

Эта документация покрывает все основные аспекты использования QuestDB в твоем проекте! 🚀

```