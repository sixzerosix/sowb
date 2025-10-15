package market

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"time"

	_ "github.com/lib/pq" // PostgreSQL driver
	"github.com/redis/go-redis/v9"
)

// TimescaleWriter пишет данные из Dragonfly в TimescaleDB
type TimescaleWriter struct {
	redisClient *redis.Client
	db          *sql.DB
	ctx         context.Context
	isRunning   bool
}

// NewTimescaleWriter создает новый writer
func NewTimescaleWriter(dragonflyURL, timescaleURL string) (*TimescaleWriter, error) {
	// Подключение к Dragonfly
	redisClient := redis.NewClient(&redis.Options{
		Addr: dragonflyURL,
	})

	// Подключение к TimescaleDB
	db, err := sql.Open("postgres", timescaleURL)
	if err != nil {
		return nil, fmt.Errorf("ошибка подключения к TimescaleDB: %w", err)
	}

	// Проверяем подключения
	ctx := context.Background()
	if _, err := redisClient.Ping(ctx).Result(); err != nil {
		return nil, fmt.Errorf("ошибка подключения к Dragonfly: %w", err)
	}

	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("ошибка подключения к TimescaleDB: %w", err)
	}

	return &TimescaleWriter{
		redisClient: redisClient,
		db:          db,
		ctx:         ctx,
	}, nil
}

// CreateTables создает таблицы в TimescaleDB
func (tw *TimescaleWriter) CreateTables() error {
	// Создаем основную таблицу для котировок
	createTableSQL := `
	CREATE TABLE IF NOT EXISTS market_data (
		time        TIMESTAMPTZ NOT NULL,
		symbol      TEXT NOT NULL,
		market      TEXT NOT NULL,
		price       DECIMAL(20,8) NOT NULL,
		volume      DECIMAL(20,8),
		high24h     DECIMAL(20,8),
		low24h      DECIMAL(20,8),
		change24h   DECIMAL(20,8),
		created_at  TIMESTAMPTZ DEFAULT NOW()
	);

	-- Создаем hypertable (если еще не создана)
	SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);

	-- Создаем индексы для быстрых запросов
	CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data (symbol, time DESC);
	CREATE INDEX IF NOT EXISTS idx_market_data_market_time ON market_data (market, time DESC);
	`

	_, err := tw.db.Exec(createTableSQL)
	return err
}

// Start запускает периодическое сохранение
func (tw *TimescaleWriter) Start(batchInterval time.Duration) {
	tw.isRunning = true
	log.Println("🗄️ TimescaleDB Writer запущен")

	// Создаем таблицы
	if err := tw.CreateTables(); err != nil {
		log.Printf("❌ Ошибка создания таблиц: %v", err)
		return
	}

	go func() {
		ticker := time.NewTicker(batchInterval)
		defer ticker.Stop()

		for tw.isRunning {
			select {
			case <-ticker.C:
				tw.processBatch()
			}
		}
	}()
}

// Stop останавливает writer
func (tw *TimescaleWriter) Stop() {
	tw.isRunning = false
	if tw.db != nil {
		tw.db.Close()
	}
	if tw.redisClient != nil {
		tw.redisClient.Close()
	}
	log.Println("🛑 TimescaleDB Writer остановлен")
}

// processBatch обрабатывает пачку данных
func (tw *TimescaleWriter) processBatch() {
	symbols := []string{"BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"}
	markets := []string{"SPOT", "FUTURES"}

	for _, symbol := range symbols {
		for _, market := range markets {
			// Получаем данные из истории Dragonfly
			key := fmt.Sprintf("history:%s:%s", market, symbol)

			// Берем последние 100 записей и удаляем их из Redis
			data, err := tw.redisClient.LRange(tw.ctx, key, 0, 99).Result()
			if err != nil {
				log.Printf("❌ Ошибка чтения из Dragonfly: %v", err)
				continue
			}

			if len(data) == 0 {
				continue // Нет новых данных
			}

			// Сохраняем в TimescaleDB
			saved := tw.saveBatchToTimescale(symbol, market, data)

			if saved > 0 {
				// Удаляем обработанные записи из Redis
				tw.redisClient.LTrim(tw.ctx, key, int64(saved), -1)
				log.Printf("💾 Сохранено %d записей для %s:%s в TimescaleDB", saved, market, symbol)
			}
		}
	}
}

// saveBatchToTimescale сохраняет пачку в TimescaleDB
func (tw *TimescaleWriter) saveBatchToTimescale(symbol, market string, data []string) int {
	if len(data) == 0 {
		return 0
	}

	// Начинаем транзакцию
	tx, err := tw.db.Begin()
	if err != nil {
		log.Printf("❌ Ошибка начала транзакции: %v", err)
		return 0
	}
	defer tx.Rollback()

	// Подготавливаем statement
	stmt, err := tx.Prepare(`
		INSERT INTO market_data (time, symbol, market, price, volume, high24h, low24h, change24h) 
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
	`)
	if err != nil {
		log.Printf("❌ Ошибка подготовки statement: %v", err)
		return 0
	}
	defer stmt.Close()

	saved := 0
	for _, item := range data {
		// Парсим JSON
		var record MarketDataPoint
		if err := json.Unmarshal([]byte(item), &record); err != nil {
			log.Printf("❌ Ошибка парсинга JSON: %v", err)
			continue
		}

		// Вставляем запись
		_, err := stmt.Exec(
			record.Time, record.Symbol, record.Market, record.Price,
			record.Volume, record.High24h, record.Low24h, record.Change24h)
		if err != nil {
			log.Printf("❌ Ошибка вставки записи: %v", err)
			continue
		}

		saved++
	}

	// Коммитим транзакцию
	if err := tx.Commit(); err != nil {
		log.Printf("❌ Ошибка коммита транзакции: %v", err)
		return 0
	}

	return saved
}

// GetHistoricalData получает исторические данные из TimescaleDB
func (tw *TimescaleWriter) GetHistoricalData(symbol, market string, from, to time.Time) ([]MarketDataPoint, error) {
	query := `
		SELECT time, symbol, market, price, volume, high24h, low24h, change24h 
		FROM market_data 
		WHERE symbol = $1 AND market = $2 AND time BETWEEN $3 AND $4
		ORDER BY time DESC
	`

	rows, err := tw.db.Query(query, symbol, market, from, to)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []MarketDataPoint
	for rows.Next() {
		var point MarketDataPoint
		err := rows.Scan(
			&point.Time, &point.Symbol, &point.Market, &point.Price,
			&point.Volume, &point.High24h, &point.Low24h, &point.Change24h)
		if err != nil {
			continue
		}
		results = append(results, point)
	}

	return results, nil
}


