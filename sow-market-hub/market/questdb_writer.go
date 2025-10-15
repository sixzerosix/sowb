package market

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"time"

	_ "github.com/lib/pq" // PostgreSQL driver для QuestDB
	"github.com/redis/go-redis/v9"
)

// QuestDBWriter пишет данные из Dragonfly в QuestDB
type QuestDBWriter struct {
	redisClient *redis.Client
	db          *sql.DB
	ctx         context.Context
	isRunning   bool
}

// NewQuestDBWriter создает новый writer
func NewQuestDBWriter(dragonflyURL, questdbURL string) (*QuestDBWriter, error) {
	// Подключение к Dragonfly
	redisClient := redis.NewClient(&redis.Options{
		Addr: dragonflyURL,
	})

	// Подключение к QuestDB через Postgres wire protocol
	if questdbURL == "" {
		questdbURL = "postgres://admin:quest@localhost:8812/qdb?sslmode=disable"
	}

	db, err := sql.Open("postgres", questdbURL)
	if err != nil {
		return nil, fmt.Errorf("ошибка подключения к QuestDB: %w", err)
	}

	// Проверяем подключения
	ctx := context.Background()
	if err := redisClient.Ping(ctx).Result(); err != nil {
		return nil, fmt.Errorf("ошибка подключения к Dragonfly: %w", err)
	}

	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("ошибка подключения к QuestDB: %w", err)
	}

	return &QuestDBWriter{
		redisClient: redisClient,
		db:          db,
		ctx:         ctx,
	}, nil
}

// CreateTables создает таблицы в QuestDB
func (qw *QuestDBWriter) CreateTables() error {
	// QuestDB оптимизирован для временных рядов
	createTableSQL := `
	CREATE TABLE IF NOT EXISTS market_data (
		timestamp TIMESTAMP,
		symbol SYMBOL,
		market SYMBOL,
		price DOUBLE,
		volume DOUBLE,
		high24h DOUBLE,
		low24h DOUBLE,
		change24h DOUBLE
	) TIMESTAMP(timestamp) PARTITION BY DAY;
	
	-- Создаем индексы для быстрых запросов
	-- QuestDB автоматически индексирует SYMBOL колонки
	`

	_, err := qw.db.Exec(createTableSQL)
	return err
}

// Start запускает периодическое сохранение
func (qw *QuestDBWriter) Start(batchInterval time.Duration) {
	qw.isRunning = true
	log.Println("🗄️ QuestDB Writer запущен")

	// Создаем таблицы
	if err := qw.CreateTables(); err != nil {
		log.Printf("❌ Ошибка создания таблиц: %v", err)
		return
	}

	go func() {
		ticker := time.NewTicker(batchInterval)
		defer ticker.Stop()

		for qw.isRunning {
			select {
			case <-ticker.C:
				qw.processBatch()
			}
		}
	}()
}

// Stop останавливает writer
func (qw *QuestDBWriter) Stop() {
	qw.isRunning = false
	if qw.db != nil {
		qw.db.Close()
	}
	if qw.redisClient != nil {
		qw.redisClient.Close()
	}
	log.Println("🛑 QuestDB Writer остановлен")
}

// processBatch обрабатывает пачку данных
func (qw *QuestDBWriter) processBatch() {
	symbols := []string{"BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"}
	markets := []string{"SPOT", "FUTURES"}

	for _, symbol := range symbols {
		for _, market := range markets {
			// Получаем данные из истории Dragonfly
			key := fmt.Sprintf("history:%s:%s", market, symbol)

			// Берем последние 100 записей и удаляем их из Redis
			data, err := qw.redisClient.LRange(qw.ctx, key, 0, 99).Result()
			if err != nil {
				log.Printf("❌ Ошибка чтения из Dragonfly: %v", err)
				continue
			}

			if len(data) == 0 {
				continue
			}

			// Сохраняем в QuestDB
			saved := qw.saveBatchToQuestDB(symbol, market, data)

			if saved > 0 {
				// Удаляем обработанные записи из Redis
				qw.redisClient.LTrim(qw.ctx, key, int64(saved), -1)
				log.Printf("💾 Сохранено %d записей для %s:%s в QuestDB", saved, market, symbol)
			}
		}
	}
}

// saveBatchToQuestDB сохраняет пачку в QuestDB используя batch insert
func (qw *QuestDBWriter) saveBatchToQuestDB(symbol, market string, data []string) int {
	if len(data) == 0 {
		return 0
	}

	// QuestDB поддерживает эффективные batch inserts
	var values []string
	var args []interface{}
	argIndex := 1

	for _, item := range data {
		var record struct {
			Price     float64 `json:"price"`
			Timestamp int64   `json:"timestamp"`
		}

		if err := json.Unmarshal([]byte(item), &record); err != nil {
			log.Printf("❌ Ошибка парсинга JSON: %v", err)
			continue
		}

		// QuestDB timestamp format
		timestamp := time.Unix(record.Timestamp, 0).UTC()

		values = append(values, fmt.Sprintf("($%d, $%d, $%d, $%d, $%d, $%d, $%d, $%d)",
			argIndex, argIndex+1, argIndex+2, argIndex+3, argIndex+4, argIndex+5, argIndex+6, argIndex+7))

		args = append(args,
			timestamp,    // timestamp
			symbol,       // symbol
			market,       // market
			record.Price, // price
			0.0,          // volume (заглушка)
			record.Price, // high24h
			record.Price, // low24h
			0.0,          // change24h (заглушка)
		)

		argIndex += 8
	}

	if len(values) == 0 {
		return 0
	}

	// Batch insert - очень эффективно в QuestDB
	query := fmt.Sprintf(`
		INSERT INTO market_data (timestamp, symbol, market, price, volume, high24h, low24h, change24h) 
		VALUES %s`, strings.Join(values, ","))

	_, err := qw.db.Exec(query, args...)
	if err != nil {
		log.Printf("❌ Ошибка batch insert: %v", err)
		return 0
	}

	return len(values)
}

// GetHistoricalData получает исторические данные из QuestDB
func (qw *QuestDBWriter) GetHistoricalData(symbol, market string, from, to time.Time) ([]MarketDataPoint, error) {
	// QuestDB оптимизирован для временных запросов
	query := `
		SELECT timestamp, symbol, market, price, volume, high24h, low24h, change24h
		FROM market_data 
		WHERE symbol = $1 AND market = $2 
		AND timestamp BETWEEN $3 AND $4
		ORDER BY timestamp DESC
		LIMIT 10000
	`

	rows, err := qw.db.Query(query, symbol, market, from, to)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []MarketDataPoint
	for rows.Next() {
		var point MarketDataPoint
		var volume, high24h, low24h, change24h sql.NullFloat64

		err := rows.Scan(&point.Time, &point.Symbol, &point.Market, &point.Price,
			&volume, &high24h, &low24h, &change24h)
		if err != nil {
			continue
		}

		// Заполняем дополнительные поля если есть
		if volume.Valid {
			point.Volume = volume.Float64
		}
		if high24h.Valid {
			point.High24h = high24h.Float64
		}
		if low24h.Valid {
			point.Low24h = low24h.Float64
		}
		if change24h.Valid {
			point.Change24h = change24h.Float64
		}

		results = append(results, point)
	}

	return results, nil
}

// GetLatestPrices получает последние цены для всех символов
func (qw *QuestDBWriter) GetLatestPrices() (map[string]MarketDataPoint, error) {
	// QuestDB LATEST BY - очень эффективная операция
	query := `
		SELECT timestamp, symbol, market, price, volume, high24h, low24h, change24h
		FROM market_data 
		LATEST BY symbol, market
	`

	rows, err := qw.db.Query(query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	results := make(map[string]MarketDataPoint)
	for rows.Next() {
		var point MarketDataPoint
		var volume, high24h, low24h, change24h sql.NullFloat64

		err := rows.Scan(&point.Time, &point.Symbol, &point.Market, &point.Price,
			&volume, &high24h, &low24h, &change24h)
		if err != nil {
			continue
		}

		key := fmt.Sprintf("%s_%s", point.Symbol, point.Market)
		results[key] = point
	}

	return results, nil
}

// GetPriceStats получает статистику по ценам
func (qw *QuestDBWriter) GetPriceStats(symbol, market string, interval string) (*PriceStats, error) {
	// QuestDB поддерживает эффективные аггрегации
	query := `
		SELECT 
			min(price) as min_price,
			max(price) as max_price,
			avg(price) as avg_price,
			first(price) as first_price,
			last(price) as last_price,
			count() as data_points
		FROM market_data 
		WHERE symbol = $1 AND market = $2 
		AND timestamp >= dateadd('h', -24, now())
	`

	var stats PriceStats
	err := qw.db.QueryRow(query, symbol, market).Scan(
		&stats.MinPrice, &stats.MaxPrice, &stats.AvgPrice,
		&stats.FirstPrice, &stats.LastPrice, &stats.DataPoints)

	if err != nil {
		return nil, err
	}

	// Вычисляем изменение за период
	if stats.FirstPrice > 0 {
		stats.Change24h = ((stats.LastPrice - stats.FirstPrice) / stats.FirstPrice) * 100
	}

	return &stats, nil
}

// MarketDataPoint расширенная структура
type MarketDataPoint struct {
	Time      time.Time `json:"time"`
	Symbol    string    `json:"symbol"`
	Market    string    `json:"market"`
	Price     float64   `json:"price"`
	Volume    float64   `json:"volume,omitempty"`
	High24h   float64   `json:"high24h,omitempty"`
	Low24h    float64   `json:"low24h,omitempty"`
	Change24h float64   `json:"change24h,omitempty"`
}

// PriceStats статистика по ценам
type PriceStats struct {
	MinPrice   float64 `json:"min_price"`
	MaxPrice   float64 `json:"max_price"`
	AvgPrice   float64 `json:"avg_price"`
	FirstPrice float64 `json:"first_price"`
	LastPrice  float64 `json:"last_price"`
	Change24h  float64 `json:"change24h"`
	DataPoints int64   `json:"data_points"`
}
