# SOW Market Hub Documentation

## 📋 Обзор проекта

**SOW Market Hub** - это высокопроизводительная система для получения, обработки и распространения рыночных данных криптовалют в реальном времени с биржи Bybit.

### 🏗️ Архитектура

```
Bybit WebSocket → Market Listener → NATS Message Broker → Multiple Consumers
                                                        ├── Dashboard (Live UI)
                                                        ├── Subscriber (Logs)
                                                        └── Dragonfly (Storage)

```

## 📁 Структура проекта

```
sow-market-hub/
├── api/                          # API endpoints (будущее расширение)
├── cmd/                          # Исполняемые файлы
│   ├── dashboard/
│   │   └── main.go              # Консольный дашборд
│   ├── dragonfly/
│   │   └── main.go              # Consumer для Dragonfly
│   ├── full/
│   │   └── main.go              # Полная система
│   ├── publisher/
│   │   └── main.go              # NATS Publisher
│   └── subscriber/
│       └── main.go              # NATS Subscriber
├── market/                       # Основная бизнес-логика
│   ├── dashboard.go             # Консольный UI
│   ├── dragonfly_consumer.go    # Dragonfly интеграция
│   ├── market.go                # Основной market listener
│   ├── nats_publisher.go        # NATS publisher
│   ├── subscriber.go            # NATS subscriber
│   └── types.go                 # Общие типы данных
├── config.json                  # Конфигурация
├── go.mod                       # Go модули
└── go.sum                       # Зависимости

```

## 🚀 Компоненты системы

### 1. Market Listener (`market/market.go`)

**Назначение:** Подключается к Bybit WebSocket API и получает данные в реальном времени.

**Функциональность:**

- Подключение к Bybit V5 API
- Получение данных по SPOT и FUTURES рынкам
- Retry логика при обрывах соединения
- Graceful shutdown
- Поддержка multiple data handlers

**Поддерживаемые символы:**

- BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT (настраивается в config.json)

### 2. NATS Publisher (`market/nats_publisher.go`)

**Назначение:** Публикует рыночные данные в NATS message broker.

**Топики:**

- `quotes.spot.{symbol}` - спот цены
- `quotes.futures.{symbol}` - фьючерсы

**Особенности:**

- Автореконнект к NATS
- Thread-safe операции
- Structured logging

### 3. Console Dashboard (`market/dashboard.go`)

**Назначение:** Live отображение котировок в консоли.

**Функции:**

- Real-time обновление цен
- Цветовая индикация трендов (↗↘→)
- Статус подключения (🟢🟡🔴)
- Статистика активных соединений

### 4. NATS Subscriber (`market/subscriber.go`)

**Назначение:** Подписывается на NATS топики и выводит данные.

**Возможности:**

- Подписка на все символы
- Подписка на конкретный символ
- Wildcard подписки
- Graceful shutdown

### 5. Dragonfly Consumer (`market/dragonfly_consumer.go`)

**Назначение:** Сохраняет исторические данные в Dragonfly/Redis.

**Структуры данных:**

- `latest:{market}:{symbol}` - последние цены
- `history:{market}:{symbol}` - история (1000 записей)
- `timeseries:{market}:{symbol}:{date}` - временные ряды
- `stats:{market}:{symbol}:*` - статистика

## ⚙️ Конфигурация

### config.json

```json
{
  "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"],
  "reconnect_timeout_seconds": 5,
  "max_retries": 10,
  "log_level": "info"
}

```

### Переменные окружения

- `NATS_URL` - URL NATS сервера (по умолчанию: nats://localhost:4222)
- `DRAGONFLY_URL` - URL Dragonfly сервера (по умолчанию: localhost:6379)

## 🏃‍♂️ Запуск системы

### Предварительные требования

```bash
# Установка NATS Server
docker run -p 4222:4222 nats:latest

# Установка Dragonfly
docker run -p 6379:6379 docker.dragonflydb.io/dragonflydb/dragonfly

```

### Установка зависимостей

```bash
go mod tidy

```

### Варианты запуска

### 1. Полная система (одна консоль)

```bash
go run cmd/full/main.go

```

### 2. Отдельные компоненты (разные консоли)

**Терминал 1 - Publisher:**

```bash
go run cmd/publisher/main.go

```

**Терминал 2 - Dashboard:**

```bash
go run cmd/dashboard/main.go

```

**Терминал 3 - Subscriber:**

```bash
go run cmd/subscriber/main.go

```

**Терминал 4 - Dragonfly Consumer:**

```bash
go run cmd/dragonfly/main.go

```

## 📊 Мониторинг и логирование

### Логи

- **JSON формат** для structured logging
- **Уровни:** debug, info, warn, error
- **Поля:** timestamp, level, message, symbol, market, price

### Метрики

- Количество обработанных сообщений
- Статус подключений
- Время последнего обновления
- Ошибки подключения

## 🔧 API Reference

### MarketDataHandler Interface

```go
type MarketDataHandler interface {
    OnSpotPriceUpdate(symbol, price string)
    OnFuturesPriceUpdate(symbol, price string)
    Start()
    Stop()
}

```

### QuoteData Structure

```go
type QuoteData struct {
    Symbol    string    `json:"symbol"`
    Price     float64   `json:"price"`
    Market    string    `json:"market"`    // "SPOT" или "FUTURES"
    Timestamp time.Time `json:"timestamp"`
}

```

## 🚨 Обработка ошибок

### Retry механизмы

- **WebSocket reconnect:** автоматический с экспоненциальным backoff
- **NATS reconnect:** встроенный механизм с настраиваемыми параметрами
- **Dragonfly reconnect:** автоматический retry при потере соединения

### Graceful Shutdown

Все компоненты поддерживают корректное завершение по сигналам:

- `SIGINT` (Ctrl+C)
- `SIGTERM`

## 📈 Производительность

### Пропускная способность

- **NATS:** до 1M+ сообщений/сек
- **WebSocket:** real-time данные от Bybit
- **Dragonfly:** высокая скорость записи в память

### Оптимизации

- Connection pooling
- Batch операции для Dragonfly
- Efficient JSON marshaling/unmarshaling
- Minimal memory allocations

## 🔒 Безопасность

### Подключения

- Проверка SSL сертификатов
- Timeout настройки
- Rate limiting (планируется)

### Данные

- Валидация входящих данных
- Санитизация символов
- Error handling для некорректных данных

## 🛠️ Разработка

### Добавление новых символов

1. Обновить `config.json`
2. Перезапустить систему

### Добавление новых consumer'ов

1. Реализовать `MarketDataHandler` интерфейс
2. Добавить в `CombinedHandler`
3. Создать отдельный main.go в cmd/

### Тестирование

```bash
go test ./...

```

## 📋 TODO / Roadmap

### Ближайшие планы

- [ ]  Unit тесты
- [ ]  Integration тесты
- [ ]  Prometheus метрики
- [ ]  REST API для исторических данных
- [ ]  WebSocket API для real-time данных

### Долгосрочные планы

- [ ]  Kubernetes deployment
- [ ]  Horizontal scaling
- [ ]  Machine learning интеграция
- [ ]  Alert system
- [ ]  Web UI dashboard

## 🤝 Contributing

1. Fork проект
2. Создай feature branch (`git checkout -b feature/amazing-feature`)
3. Commit изменения (`git commit -m 'Add amazing feature'`)
4. Push в branch (`git push origin feature/amazing-feature`)
5. Открой Pull Request

## 📄 License

Этот проект лицензирован под MIT License - см. [LICENSE](https://www.notion.so/opencrypto/LICENSE) файл для деталей.

## 📞 Поддержка

Для вопросов и поддержки:

- Создай Issue в репозитории
- Свяжись с командой разработки

---

**SOW Market Hub** - надежное решение для работы с криптовалютными данными в реальном времени! 🚀