package market

import (
	"encoding/json"
	"fmt"
	"log"
	"strconv"
	"strings"
	"sync" // Импорт мьютекса
	"time"

	"github.com/nats-io/nats.go"
)

// NATSPublisher структура для публикации в NATS
type NATSPublisher struct {
	nc        *nats.Conn
	isRunning bool
	mu        sync.Mutex // Добавляем мьютекс
}

// NewNATSPublisher создает новый NATS publisher
func NewNATSPublisher(natsURL string) (MarketDataHandler, error) { // Возвращаем интерфейс
	if natsURL == "" {
		natsURL = nats.DefaultURL // "nats://localhost:4222"
	}

	nc, err := nats.Connect(natsURL,
		nats.MaxReconnects(-1),
		nats.ReconnectWait(time.Second*5),
		nats.DisconnectErrHandler(func(nc *nats.Conn, err error) {
			log.Printf("NATS disconnected: %v. Reconnecting...", err)
		}),
		nats.ReconnectHandler(func(nc *nats.Conn) {
			log.Println("NATS reconnected!")
		}),
		nats.ClosedHandler(func(nc *nats.Conn) {
			if nc.LastError() != nil {
				log.Fatalf("NATS connection closed: %v", nc.LastError())
			}
			log.Println("NATS connection closed gracefully.")
		}),
	)
	if err != nil {
		return nil, fmt.Errorf("ошибка подключения к NATS: %w", err)
	}

	return &NATSPublisher{
		nc: nc,
	}, nil
}

// Start запускает publisher (реализация интерфейса MarketDataHandler)
func (np *NATSPublisher) Start() {
	np.mu.Lock()
	defer np.mu.Unlock()
	np.isRunning = true
	log.Println("🚀 NATS Publisher запущен")
}

// Stop останавливает publisher (реализация интерфейса MarketDataHandler)
func (np *NATSPublisher) Stop() {
	np.mu.Lock()
	defer np.mu.Unlock()
	np.isRunning = false
	if np.nc != nil {
		np.nc.Close()
	}
	log.Println("🛑 NATS Publisher остановлен")
}

// OnSpotPriceUpdate публикует спот цену (реализация интерфейса MarketDataHandler)
func (np *NATSPublisher) OnSpotPriceUpdate(symbol, price string) {
	np.mu.Lock() // Защищаем чтение isRunning
	if !np.isRunning {
		np.mu.Unlock()
		return
	}
	np.mu.Unlock()

	priceFloat, err := strconv.ParseFloat(price, 64)
	if err != nil {
		log.Printf("Ошибка парсинга цены %s: %v", price, err)
		return
	}

	quote := QuoteData{
		Symbol:    symbol,
		Price:     priceFloat,
		Market:    "SPOT",
		Timestamp: time.Now(),
	}

	np.publishQuote(quote, "spot")
}

// OnFuturesPriceUpdate публикует фьючерсную цену (реализация интерфейса MarketDataHandler)
func (np *NATSPublisher) OnFuturesPriceUpdate(symbol, price string) {
	np.mu.Lock() // Защищаем чтение isRunning
	if !np.isRunning {
		np.mu.Unlock()
		return
	}
	np.mu.Unlock()

	priceFloat, err := strconv.ParseFloat(price, 64)
	if err != nil {
		log.Printf("Ошибка парсинга цены %s: %v", price, err)
		return
	}

	quote := QuoteData{
		Symbol:    symbol,
		Price:     priceFloat,
		Market:    "FUTURES",
		Timestamp: time.Now(),
	}

	np.publishQuote(quote, "futures")
}

// publishQuote публикует котировку в NATS
func (np *NATSPublisher) publishQuote(quote QuoteData, marketType string) {
	data, err := json.Marshal(quote)
	if err != nil {
		log.Printf("Ошибка при сериализации JSON: %v", err)
		return
	}

	subject := fmt.Sprintf("quotes.%s.%s", marketType, strings.ToLower(quote.Symbol))

	err = np.nc.Publish(subject, data)
	if err != nil {
		log.Printf("Ошибка при публикации в %s: %v", subject, err)
	} else {
		// Используем log.Printf для консистентности
		log.Printf("📡 Опубликовано в NATS [%s]: %s = %.4f\n", subject, quote.Symbol, quote.Price)
	}
}
