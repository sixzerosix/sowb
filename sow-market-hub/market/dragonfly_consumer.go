package market

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/redis/go-redis/v9"
)

// DragonflyConsumer сохраняет котировки в Dragonfly
type DragonflyConsumer struct {
	nc          *nats.Conn
	redisClient *redis.Client
	isRunning   bool
	mu          sync.Mutex
	ctx         context.Context
}

// NewDragonflyConsumer создает новый Dragonfly consumer
func NewDragonflyConsumer(natsURL, dragonflyURL string) (*DragonflyConsumer, error) {
	// Подключение к NATS
	if natsURL == "" {
		natsURL = nats.DefaultURL
	}

	nc, err := nats.Connect(natsURL,
		nats.MaxReconnects(-1),
		nats.ReconnectWait(time.Second*5),
	)
	if err != nil {
		return nil, fmt.Errorf("ошибка подключения к NATS: %w", err)
	}

	// Подключение к Dragonfly
	if dragonflyURL == "" {
		dragonflyURL = "localhost:6379" // стандартный порт Dragonfly
	}

	redisClient := redis.NewClient(&redis.Options{
		Addr:     dragonflyURL,
		Password: "", // без пароля по умолчанию
		DB:       0,  // база данных 0
	})

	// Проверяем подключение
	ctx := context.Background()
	_, err = redisClient.Ping(ctx).Result()
	if err != nil {
		return nil, fmt.Errorf("ошибка подключения к Dragonfly: %w", err)
	}

	return &DragonflyConsumer{
		nc:          nc,
		redisClient: redisClient,
		ctx:         ctx,
	}, nil
}

// Start запускает consumer
func (dc *DragonflyConsumer) Start() {
	dc.mu.Lock()
	defer dc.mu.Unlock()

	dc.isRunning = true

	symbols := []string{"btcusdt", "ethusdt", "bnbusdt", "solusdt"}
	markets := []string{"spot", "futures"}

	log.Println("💾 Dragonfly Consumer запущен")

	for _, symbol := range symbols {
		for _, market := range markets {
			subject := fmt.Sprintf("quotes.%s.%s", market, symbol)

			_, err := dc.nc.Subscribe(subject, func(m *nats.Msg) {
				dc.handleQuote(m)
			})

			if err != nil {
				log.Printf("❌ Ошибка подписки на %s: %v", subject, err)
			} else {
				log.Printf("✅ Подписались на %s для сохранения в Dragonfly", subject)
			}
		}
	}
}

// Stop останавливает consumer
func (dc *DragonflyConsumer) Stop() {
	dc.mu.Lock()
	defer dc.mu.Unlock()

	dc.isRunning = false
	if dc.nc != nil {
		dc.nc.Close()
	}
	if dc.redisClient != nil {
		dc.redisClient.Close()
	}
	log.Println("🛑 Dragonfly Consumer остановлен")
}

// handleQuote обрабатывает и сохраняет котировки
func (dc *DragonflyConsumer) handleQuote(m *nats.Msg) {
	var quote QuoteData
	err := json.Unmarshal(m.Data, &quote)
	if err != nil {
		log.Printf("❌ Ошибка десериализации: %v", err)
		return
	}

	// Сохраняем в разных форматах для разных целей
	dc.saveLatestPrice(quote)    // Последняя цена
	dc.saveHistoricalData(quote) // Исторические данные
	dc.saveToTimeSeries(quote)   // Временные ряды
	dc.updateStatistics(quote)   // Статистика

	log.Printf("💾 Сохранено в Dragonfly: [%s] %s = %.4f",
		quote.Market, quote.Symbol, quote.Price)
}

// saveLatestPrice сохраняет последнюю цену (для быстрого доступа)
func (dc *DragonflyConsumer) saveLatestPrice(quote QuoteData) {
	key := fmt.Sprintf("latest:%s:%s", quote.Market, quote.Symbol)

	data := map[string]interface{}{
		"price":     quote.Price,
		"timestamp": quote.Timestamp.Unix(),
	}

	err := dc.redisClient.HMSet(dc.ctx, key, data).Err()
	if err != nil {
		log.Printf("❌ Ошибка сохранения последней цены: %v", err)
	}
}

// saveHistoricalData сохраняет в список для истории
func (dc *DragonflyConsumer) saveHistoricalData(quote QuoteData) {
	key := fmt.Sprintf("history:%s:%s", quote.Market, quote.Symbol)

	data := map[string]interface{}{
		"price":     quote.Price,
		"timestamp": quote.Timestamp.Unix(),
	}

	jsonData, _ := json.Marshal(data)

	// Добавляем в список (последние 1000 записей)
	pipe := dc.redisClient.Pipeline()
	pipe.LPush(dc.ctx, key, jsonData)
	pipe.LTrim(dc.ctx, key, 0, 999) // оставляем только последние 1000
	_, err := pipe.Exec(dc.ctx)

	if err != nil {
		log.Printf("❌ Ошибка сохранения истории: %v", err)
	}
}

// saveToTimeSeries сохраняет во временные ряды (по дням)
func (dc *DragonflyConsumer) saveToTimeSeries(quote QuoteData) {
	// Ключ по дате: timeseries:SPOT:BTCUSDT:2024-10-15
	dateKey := quote.Timestamp.Format("2006-01-02")
	key := fmt.Sprintf("timeseries:%s:%s:%s", quote.Market, quote.Symbol, dateKey)

	// Timestamp как score для sorted set
	score := float64(quote.Timestamp.Unix())
	member := fmt.Sprintf("%.4f", quote.Price)

	err := dc.redisClient.ZAdd(dc.ctx, key, redis.Z{
		Score:  score,
		Member: member,
	}).Err()

	if err != nil {
		log.Printf("❌ Ошибка сохранения временного ряда: %v", err)
	}
}

// updateStatistics обновляет статистику
func (dc *DragonflyConsumer) updateStatistics(quote QuoteData) {
	statsKey := fmt.Sprintf("stats:%s:%s", quote.Market, quote.Symbol)

	pipe := dc.redisClient.Pipeline()

	// Счетчик сообщений
	pipe.Incr(dc.ctx, statsKey+":count")

	// Последнее обновление
	pipe.Set(dc.ctx, statsKey+":last_update", quote.Timestamp.Unix(), 0)

	// Минимальная и максимальная цена за день
	dateKey := quote.Timestamp.Format("2006-01-02")
	pipe.ZAdd(dc.ctx, statsKey+":daily_min:"+dateKey, redis.Z{Score: quote.Price, Member: "min"})
	pipe.ZAdd(dc.ctx, statsKey+":daily_max:"+dateKey, redis.Z{Score: quote.Price, Member: "max"})

	_, err := pipe.Exec(dc.ctx)
	if err != nil {
		log.Printf("❌ Ошибка обновления статистики: %v", err)
	}
}

// StartDragonflyConsumer публичная функция для запуска
func StartDragonflyConsumer(natsURL, dragonflyURL string) (*DragonflyConsumer, error) {
	consumer, err := NewDragonflyConsumer(natsURL, dragonflyURL)
	if err != nil {
		return nil, err
	}

	consumer.Start()
	return consumer, nil
}
