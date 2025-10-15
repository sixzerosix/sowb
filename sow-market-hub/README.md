# SOW Market Hub Documentation

## 📋 Обзор проекта

**SOW Market Hub** - это высокопроизводительная система для получения, обработки и распространения рыночных данных криптовалют в реальном времени с биржи Bybit с долгосрочным хранением в QuestDB.

### 🏗️ Архитектура

Bybit WebSocket → Market Listener → NATS Message Broker → Multiple Consumers
├── Dashboard (Live UI)
├── Subscriber (Logs)
├── Dragonfly (Cache)
└── QuestDB (Long-term Storage)
↓
REST API ← Web Dashboard

### 🗃️ Многоуровневое хранение данных

Real-time:    NATS (сообщения) → Dragonfly (кеш, последние данные)
Long-term:    Dragonfly (батчи) → QuestDB (исторические данные)
Analytics:    QuestDB → REST API → Dashboards/Reports

## 📁 Структура проекта

```markdown
sow-market-hub/
├── api/                          # REST API endpoints
│   └── questdb_api.go           # QuestDB API handlers
├── cmd/                          # Исполняемые файлы
│   ├── dashboard/
│   │   └── main.go              # Консольный дашборд
│   ├── dragonfly/
│   │   └── main.go              # Consumer для Dragonfly
│   ├── questdb-writer/
│   │   └── main.go              # QuestDB Writer
│   ├── websocket-server/
│   │   └── main.go              # WebSocket сервер для фронтенда
│   ├── full/
│   │   └── main.go              # Полная система
│   ├── publisher/
│   │   └── main.go              # NATS Publisher
│   └── subscriber/
│       └── main.go              # NATS Subscriber
├── market/                       # Основная бизнес-логика
│   ├── dashboard.go             # Консольный UI
│   ├── dragonfly_consumer.go    # Dragonfly интеграция
│   ├── questdb_writer.go        # QuestDB интеграция
│   ├── market.go                # Основной market listener
│   ├── nats_publisher.go        # NATS publisher
│   ├── subscriber.go            # NATS subscriber
│   └── types.go                 # Общие типы данных
├── frontend/                     # Next.js веб-интерфейс
│   ├── components/
│   │   └── VirtualizedMarketTable.tsx
│   ├── hooks/
│   │   └── useWebSocket.ts
│   ├── store/
│   │   └── marketStore.ts
│   └── types/
│       └── market.ts
├── config.json                  # Конфигурация
├── docker-compose.yml           # Docker окружение
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

- Настраивается в config.json (по умолчанию: BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT и др.)

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

**Назначение:** Кеширует данные в Dragonfly/Redis для быстрого доступа.

**Структуры данных:**

- `latest:{market}:{symbol}` - последние цены
- `history:{market}:{symbol}` - история (1000 записей)
- `timeseries:{market}:{symbol}:{date}` - временные ряды
- `stats:{market}:{symbol}:*` - статистика

### 6. QuestDB Writer (`market/questdb_writer.go`)

**Назначение:** Долгосрочное хранение исторических данных в QuestDB.

**Функции:**

- Batch запись данных из Dragonfly в QuestDB
- Оптимизированные временные запросы
- Автоматическое партиционирование по дням
- REST API для аналитики

### 7. WebSocket Server (`cmd/websocket-server/main.go`)

**Назначение:** Мост между NATS и веб-интерфейсом.

**Функции:**

- Real-time трансляция данных в браузер
- Батчинг и оптимизация сообщений
- CORS поддержка
- Автореконнект

### 8. Web Dashboard (Next.js Frontend)

**Назначение:** Современный веб-интерфейс для мониторинга рынка.

**Функции:**

- Виртуализированная таблица для производительности
- Real-time обновления через WebSocket
- Поиск и фильтрация символов
- Адаптивный дизайн
- Zustand для управления состоянием

## 🗄️ QuestDB Integration

### Почему QuestDB?

- ⚡ **4M+ записей/сек** - экстремальная производительность
- 📊 **Нативная поддержка временных рядов** - LATEST BY, SAMPLE BY
- 💾 **Эффективное сжатие** - экономия до 90% места
- 🔍 **Быстрые аналитические запросы** - оптимизированы для финансовых данных
- 🐘 **PostgreSQL совместимость** - знакомый SQL

### Структура данных в QuestDB:

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

### Примеры запросов:

```sql
-- Последние цены всех активов
SELECT * FROM market_data LATEST BY symbol, market;

-- Статистика за 24 часа
SELECT symbol, market, min(price), max(price), avg(price)
FROM market_data
WHERE timestamp >= dateadd('h', -24, now())
GROUP BY symbol, market;

-- Почасовые OHLC данные
SELECT symbol, sample_by(1h, timestamp) as hour,
       first(price) as open, max(price) as high,
       min(price) as low, last(price) as close
FROM market_data
WHERE symbol = 'BTCUSDT' AND timestamp >= dateadd('d', -7, now())
SAMPLE BY 1h;

```

## ⚙️ Конфигурация

### config.json

```json
{
  "symbols": [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "TRXUSDT", "TONUSDT", "AVAXUSDT",
    "LINKUSDT", "DOTUSDT", "MATICUSDT", "ICPUSDT", "NEARUSDT"
  ],
  "reconnect_timeout_seconds": 5,
  "max_retries": 10,
  "log_level": "info"
}
```

### docker-compose.yml

```yaml
version: '3.8'
services:
  nats:
    image: nats:latest
    ports:
      - "4222:4222"

  dragonfly:
    image: docker.dragonflydb.io/dragonflydb/dragonfly
    ports:
      - "6379:6379"

  questdb:
    image: questdb/questdb:latest
    ports:
      - "9000:9000"    # Web Console
      - "8812:8812"    # Postgres wire protocol
    volumes:
      - questdb_data:/var/lib/questdb

volumes:
  questdb_data:

```

### Переменные окружения

- `NATS_URL` - URL NATS сервера (по умолчанию: nats://localhost:4222)
- `DRAGONFLY_URL` - URL Dragonfly сервера (по умолчанию: localhost:6379)
- `QUESTDB_URL` - URL QuestDB (по умолчанию: postgres://admin:quest@localhost:8812/qdb)

## 🏃‍♂️ Запуск системы

### Быстрый старт

```bash
# 1. Запуск инфраструктуры
docker-compose up -d

# 2. Установка зависимостей Go
go mod tidy

# 3. Установка зависимостей Frontend
cd frontend && npm install

# 4. Запуск всех сервисов
make start-all
```

### Пошаговый запуск

**Терминал 1 - Publisher:**

```go
go run cmd/publisher/main.go
```

**Терминал 2 - Dragonfly Consumer:**

```go
go run cmd/dragonfly/main.go
```

**Терминал 3 - QuestDB Writer:**

```go
go run cmd/questdb-writer/main.go
```

**Терминал 4 - WebSocket Server:**

```go
go run cmd/websocket-server/main.go
```

**Терминал 5 - Frontend:**

```bash
cd frontend && npm run dev
```

### Makefile команды

```bash
make start-infra     # Запуск Docker сервисов
make start-backend   # Запуск Go сервисов
make start-frontend  # Запуск Next.js
make start-all       # Запуск всего
make stop-all        # Остановка всего
make logs           # Просмотр логов
```

## 📊 Мониторинг и веб-интерфейсы

### QuestDB Web Console

- **URL:** [http://localhost:9000](http://localhost:9000/)
- **Функции:** SQL запросы, визуализация данных, мониторинг

### Market Dashboard (Next.js)

- **URL:** http://localhost:3000/market
- **Функции:** Real-time таблица котировок, поиск, статистика

### REST API

- **Base URL:** http://localhost:8080/api
- **Endpoints:**
    - `GET /prices/latest` - последние цены
    - `GET /prices/{symbol}/{market}/history` - исторические данные
    - `GET /prices/{symbol}/{market}/stats` - статистика

## 📈 Производительность

### Пропускная способность

- **NATS:** до 1M+ сообщений/сек
- **QuestDB:** до 4M+ записей/сек
- **WebSocket:** real-time данные от Bybit
- **Dragonfly:** высокая скорость кеширования

### Оптимизации

- Виртуализация таблиц в веб-интерфейсе
- Батчинг обновлений (100ms интервалы)
- Буферизация WebSocket сообщений
- Партиционирование QuestDB по дням
- Zustand для эффективного state management

### Архитектурные решения

- **Многоуровневое хранение:** Hot (Dragonfly) + Cold (QuestDB)
- **Микросервисная архитектура:** независимые компоненты
- **Event-driven:** NATS для асинхронной обработки
- **Horizontal scaling:** легко масштабируется

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

### REST API Endpoints

```go
// Последние цены всех активов
GET /api/prices/latest

// Исторические данные
GET /api/prices/BTCUSDT/SPOT/history?from=2024-01-01T00:00:00Z&to=2024-01-02T00:00:00Z

// Статистика по активу
GET /api/prices/BTCUSDT/SPOT/stats

// Health check
GET /health
```

### WebSocket API

```tsx
// Подключение к WebSocket
const ws = new WebSocket('ws://localhost:8080/ws');

// Подписка на символы
ws.send(JSON.stringify({
  type: 'subscribe',
  symbols: ['BTCUSDT', 'ETHUSDT']
}));

// Получение данных
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // { symbol: "BTCUSDT", price: 45000, market: "SPOT", timestamp: "..." }
};
```

## 🚨 Обработка ошибок

### Retry механизмы

- **WebSocket reconnect:** экспоненциальный backoff
- **NATS reconnect:** встроенная устойчивость
- **QuestDB reconnect:** автоматический retry
- **Frontend reconnect:** автоматическое переподключение

### Graceful Shutdown

Все компоненты поддерживают корректное завершение:

- `SIGINT` (Ctrl+C)
- `SIGTERM`
- Сохранение буферизованных данных
- Закрытие всех подключений

## 🔒 Безопасность

### Подключения

- TLS для внешних подключений
- Валидация WebSocket origin
- Rate limiting на API endpoints
- Timeout настройки для всех соединений

### Данные

- Валидация входящих данных
- Санитизация символов
- Error handling для некорректных данных
- Логирование всех операций

## 🛠️ Разработка

### Добавление новых символов

1. Обновить `config.json`
2. Перезапустить систему
3. Новые символы автоматически подхватятся

### Добавление новых consumer'ов

1. Реализовать `MarketDataHandler` интерфейс
2. Добавить в `CombinedHandler`
3. Создать отдельный main.go в cmd/

### Тестирование

```bash
go test ./...                    # Backend тесты
cd frontend && npm test          # Frontend тесты
make integration-test           # Интеграционные тесты
```

## 📋 TODO / Roadmap

### Ближайшие планы

- [ ]  Unit тесты для всех компонентов
- [ ]  Integration тесты
- [ ]  Prometheus метрики
- [ ]  Grafana дашборды
- [ ]  Alert система
- [ ]  Rate limiting для API

### Долгосрочные планы

- [ ]  Kubernetes deployment
- [ ]  Horizontal scaling
- [ ]  Machine learning для прогнозов
- [ ]  Mobile приложение
- [ ]  Telegram бот для уведомлений
- [ ]  Поддержка других бирж

### Оптимизации

- [ ]  Кеширование API запросов
- [ ]  CDN для статических файлов
- [ ]  Database connection pooling
- [ ]  Compressed WebSocket messages
- [ ]  Server-side rendering

## 🤝 Contributing

1. Fork проект
2. Создай feature branch (`git checkout -b feature/amazing-feature`)
3. Commit изменения (`git commit -m 'Add amazing feature'`)
4. Push в branch (`git push origin feature/amazing-feature`)
5. Открой Pull Request

### Стандарты кода

- Go: следуй `go fmt` и `golint`
- TypeScript: используй ESLint и Prettier
- Commit messages: следуй Conventional Commits
- Документация: обновляй README при изменениях API

## 📄 License

Этот проект лицензирован под MIT License - см. [LICENSE](https://www.notion.so/opencrypto/LICENSE) файл для деталей.

## 📞 Поддержка

Для вопросов и поддержки:

- Создай Issue в репозитории
- Свяжись с командой разработки
- Присоединяйся к Discord сообществу

## 🏆 Благодарности

- [Bybit API](https://bybit-exchange.github.io/docs/) - источник рыночных данных
- [NATS](https://nats.io/) - message broker
- [QuestDB](https://questdb.io/) - time-series database
- [Dragonfly](https://dragonflydb.io/) - in-memory datastore
- [Next.js](https://nextjs.org/) - React framework
- [Zustand](https://github.com/pmndrs/zustand) - state management

---

**SOW Market Hub** - профессиональное решение для работы с криптовалютными данными в реальном времени! 🚀