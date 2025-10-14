package market

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/nats-io/nats.go"
)

// MarketSubscriber структура для подписчика
type MarketSubscriber struct {
	nc        *nats.Conn
	isRunning bool
	mu        sync.Mutex
}

// StartSubscriber создает и запускает subscriber (публичная функция)
func StartSubscriber(natsURL string) (*MarketSubscriber, error) {
	if natsURL == "" {
		natsURL = nats.DefaultURL
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

	subscriber := &MarketSubscriber{
		nc: nc,
	}

	return subscriber, nil
}

// Start запускает подписки
func (ms *MarketSubscriber) Start() {
	ms.mu.Lock()
	defer ms.mu.Unlock()

	ms.isRunning = true

	symbols := []string{"btcusdt", "ethusdt", "bnbusdt", "solusdt"}
	markets := []string{"spot", "futures"}

	log.Println("📡 NATS Subscriber запущен")

	for _, symbol := range symbols {
		for _, market := range markets {
			subject := fmt.Sprintf("quotes.%s.%s", market, symbol)

			_, err := ms.nc.Subscribe(subject, func(m *nats.Msg) {
				ms.handleQuote(m)
			})

			if err != nil {
				log.Printf("❌ Ошибка подписки на %s: %v", subject, err)
			} else {
				log.Printf("✅ Подписались на %s", subject)
			}
		}
	}
}

// Stop останавливает subscriber
func (ms *MarketSubscriber) Stop() {
	ms.mu.Lock()
	defer ms.mu.Unlock()

	ms.isRunning = false
	if ms.nc != nil {
		ms.nc.Close()
	}
	log.Println("🛑 NATS Subscriber остановлен")
}

// handleQuote обрабатывает полученные котировки
func (ms *MarketSubscriber) handleQuote(m *nats.Msg) {
	var quote QuoteData
	err := json.Unmarshal(m.Data, &quote)
	if err != nil {
		log.Printf("❌ Ошибка десериализации: %v", err)
		return
	}

	marketIcon := "📈"
	if quote.Market == "FUTURES" {
		marketIcon = "🔮"
	}

	log.Printf("%s [%s] %s: %.4f @ %s",
		marketIcon,
		quote.Market,
		quote.Symbol,
		quote.Price,
		quote.Timestamp.Format("15:04:05.000"),
	)
}

// ИСПРАВЬ ЭТИ ФУНКЦИИ - добавь правильную handleQuote
// handleQuote для отдельных функций (не метод структуры)
func handleQuote(m *nats.Msg, marketType string) {
	var quote QuoteData
	err := json.Unmarshal(m.Data, &quote)
	if err != nil {
		log.Printf("❌ Ошибка при десериализации JSON: %v\n", err)
		return
	}

	marketIcon := "📈"
	if quote.Market == "FUTURES" {
		marketIcon = "🔮"
	}

	fmt.Printf("%s [%s] %s: %.4f @ %s (от %s)\n",
		marketIcon,
		quote.Market,
		quote.Symbol,
		quote.Price,
		quote.Timestamp.Format("15:04:05.000"),
		m.Subject,
	)
}

// SubscribeToSymbol подписывается только на конкретный символ
func SubscribeToSymbol(symbol string) {
	nc, err := nats.Connect(nats.DefaultURL)
	if err != nil {
		log.Fatal(err)
	}
	defer nc.Close()

	// Подписка на конкретный символ (и спот, и фьючерсы)
	spotSubject := fmt.Sprintf("quotes.spot.%s", symbol)
	futuresSubject := fmt.Sprintf("quotes.futures.%s", symbol)

	nc.Subscribe(spotSubject, func(m *nats.Msg) {
		handleQuote(m, "SPOT")
	})

	nc.Subscribe(futuresSubject, func(m *nats.Msg) {
		handleQuote(m, "FUTURES")
	})

	fmt.Printf("📡 Подписались на %s (спот и фьючерсы)\n", symbol)

	// Graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	<-sigChan
}

// SubscribeToAll подписывается на все котировки используя wildcard
func SubscribeToAll() {
	nc, err := nats.Connect(nats.DefaultURL)
	if err != nil {
		log.Fatal(err)
	}
	defer nc.Close()

	// Wildcard подписка на все котировки
	_, err = nc.Subscribe("quotes.*.*", func(m *nats.Msg) {
		handleQuote(m, "ALL")
	})

	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("📡 Подписались на ВСЕ котировки (quotes.*.*)")

	// Graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	<-sigChan
}
