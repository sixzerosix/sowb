package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/nats-io/nats.go"
)

type EnhancedWebSocketServer struct {
	clients          map[*websocket.Conn]*EnhancedClient
	clientsMux       sync.RWMutex
	natsConn         *nats.Conn
	upgrader         websocket.Upgrader
	subscriptions    map[string]*nats.Subscription
	subscriptionsMux sync.RWMutex
}

type EnhancedClient struct {
	conn *websocket.Conn

	// Мьютекс для записи в WebSocket соединение
	writeMux sync.Mutex

	// Фильтры клиента
	symbols    map[string]bool
	markets    map[string]bool
	dataTypes  map[string]bool
	filtersMux sync.RWMutex // Мьютекс для доступа к maps: symbols, markets, dataTypes

	// Настройки throttle
	throttle    time.Duration
	lastSent    map[string]time.Time
	lastSentMux sync.Mutex // Мьютекс для доступа к lastSent
}

type ClientMessage struct {
	Type    string                 `json:"type"`
	Action  string                 `json:"action,omitempty"` // subscribe/unsubscribe
	Filters map[string]interface{} `json:"filters,omitempty"`
}

type ServerMessage struct {
	Type      string      `json:"type"`
	Symbol    string      `json:"symbol,omitempty"`
	Market    string      `json:"market,omitempty"`
	DataType  string      `json:"data_type,omitempty"`
	Data      interface{} `json:"data"`
	Timestamp time.Time   `json:"timestamp"`
}

func NewEnhancedWebSocketServer(natsURL string) (*EnhancedWebSocketServer, error) {
	nc, err := nats.Connect(natsURL,
		nats.MaxReconnects(-1),
		nats.ReconnectWait(time.Second*5),
		nats.DisconnectErrHandler(func(nc *nats.Conn, err error) {
			log.Printf("NATS disconnected: %v", err)
		}),
		nats.ReconnectHandler(func(nc *nats.Conn) {
			log.Println("NATS reconnected!")
		}),
	)
	if err != nil {
		return nil, fmt.Errorf("ошибка подключения к NATS: %w", err)
	}

	ews := &EnhancedWebSocketServer{
		clients:       make(map[*websocket.Conn]*EnhancedClient),
		natsConn:      nc,
		subscriptions: make(map[string]*nats.Subscription),
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				return true
			},
		},
	}

	// Auto-subscribe to common publisher patterns so server receives messages even before clients
	ews.autoSubscribeDefaults()

	return ews, nil
}

// autoSubscribeDefaults subscribes to wildcard patterns used by the publisher
func (ews *EnhancedWebSocketServer) autoSubscribeDefaults() {
	patterns := []string{
		"quotes.*.*",
		"trades.*.*",
		"orderbook.*.*",
		"klines.*.*.*",
		"analytics.*.*",
	}

	for _, pat := range patterns {
		ews.subscriptionsMux.Lock()
		if _, exists := ews.subscriptions[pat]; exists {
			ews.subscriptionsMux.Unlock()
			continue
		}

		sub, err := ews.natsConn.Subscribe(pat, func(m *nats.Msg) {
			parts := strings.Split(m.Subject, ".")
			if len(parts) < 3 {
				log.Printf("⚠️ Unexpected subject format: %s", m.Subject)
				return
			}

			var dataType, market, symbol string
			if parts[0] == "klines" {
				// klines.interval.market.symbol
				if len(parts) >= 4 {
					dataType = "klines"
					market = parts[2]
					symbol = parts[3]
				} else {
					log.Printf("⚠️ Unexpected klines subject: %s", m.Subject)
					return
				}
			} else {
				dataType = parts[0]
				market = parts[1]
				symbol = parts[2]
			}

			ews.handleNATSMessage(m, dataType, symbol, market)
		})

		if err != nil {
			log.Printf("❌ Error auto-subscribing to %s: %v", pat, err)
			ews.subscriptionsMux.Unlock()
			continue
		}

		ews.subscriptions[pat] = sub
		ews.subscriptionsMux.Unlock()
		log.Printf("📡 Auto-subscribed to NATS pattern: %s", pat)
	}
}

func (ews *EnhancedWebSocketServer) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := ews.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("❌ WebSocket upgrade error: %v", err)
		return
	}
	defer conn.Close()

	// Создаем расширенного клиента
	client := &EnhancedClient{
		conn:      conn,
		symbols:   make(map[string]bool),
		markets:   make(map[string]bool),
		dataTypes: make(map[string]bool),
		throttle:  100 * time.Millisecond, // Дефолтный throttle
		lastSent:  make(map[string]time.Time),
	}

	// Добавляем клиента
	ews.clientsMux.Lock()
	ews.clients[conn] = client
	ews.clientsMux.Unlock()

	log.Printf("✅ New Enhanced WebSocket client connected. Total: %d", len(ews.clients))

	// Удаляем клиента при отключении
	defer func() {
		ews.clientsMux.Lock()
		delete(ews.clients, conn)
		ews.clientsMux.Unlock()
		log.Printf("❌ Enhanced WebSocket client disconnected. Total: %d", len(ews.clients))
	}()

	// Отправляем приветственное сообщение
	ews.sendWelcomeMessage(client)

	// Обрабатываем сообщения от клиента
	for {
		var msg ClientMessage
		err := conn.ReadJSON(&msg)
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("❌ WebSocket error: %v", err)
			}
			break
		}

		ews.handleClientMessage(client, msg)
	}
}

func (ews *EnhancedWebSocketServer) sendWelcomeMessage(client *EnhancedClient) {
	welcome := ServerMessage{
		Type: "welcome",
		Data: map[string]interface{}{
			"message":              "Connected to Enhanced Market Data Stream",
			"available_data_types": []string{"quotes", "trades", "orderbook", "klines", "analytics"},
			"available_markets":    []string{"spot", "futures"},
			"throttle_ms":          int(client.throttle.Milliseconds()),
		},
		Timestamp: time.Now(),
	}

	client.writeMux.Lock()
	client.conn.WriteJSON(welcome)
	client.writeMux.Unlock()
}

func (ews *EnhancedWebSocketServer) handleClientMessage(client *EnhancedClient, msg ClientMessage) {
	switch msg.Type {
	case "subscribe":
		ews.handleSubscribe(client, msg)
	case "unsubscribe":
		ews.handleUnsubscribe(client, msg)
	case "configure":
		ews.handleConfigure(client, msg)
	default:
		log.Printf("❌ Unknown message type: %s", msg.Type)
	}
}

func (ews *EnhancedWebSocketServer) handleSubscribe(client *EnhancedClient, msg ClientMessage) {
	client.filtersMux.Lock() // Защищаем доступ к фильтрам
	defer client.filtersMux.Unlock()

	// Парсим фильтры
	if symbols, ok := msg.Filters["symbols"].([]interface{}); ok {
		for _, sym := range symbols {
			if symbol, ok := sym.(string); ok {
				client.symbols[strings.ToUpper(symbol)] = true
			}
		}
	}

	if markets, ok := msg.Filters["markets"].([]interface{}); ok {
		for _, mkt := range markets {
			if market, ok := mkt.(string); ok {
				client.markets[strings.ToUpper(market)] = true
			}
		}
	}

	if dataTypes, ok := msg.Filters["data_types"].([]interface{}); ok {
		for _, dt := range dataTypes {
			if dataType, ok := dt.(string); ok {
				client.dataTypes[strings.ToLower(dataType)] = true
			}
		}
	}

	// Создаем NATS подписки
	ews.ensureNATSSubscriptions(client)

	// Отправляем подтверждение
	response := ServerMessage{
		Type: "subscription_confirmed",
		Data: map[string]interface{}{
			"symbols":    getKeys(client.symbols),
			"markets":    getKeys(client.markets),
			"data_types": getKeys(client.dataTypes),
		},
		Timestamp: time.Now(),
	}

	client.writeMux.Lock()
	client.conn.WriteJSON(response)
	client.writeMux.Unlock()
}

func (ews *EnhancedWebSocketServer) handleUnsubscribe(client *EnhancedClient, msg ClientMessage) {
	client.filtersMux.Lock() // Защищаем доступ к фильтрам
	defer client.filtersMux.Unlock()

	// Реализация отписки
	if symbols, ok := msg.Filters["symbols"].([]interface{}); ok {
		for _, sym := range symbols {
			if symbol, ok := sym.(string); ok {
				delete(client.symbols, strings.ToUpper(symbol))
			}
		}
	}

	// Отправляем подтверждение
	response := ServerMessage{
		Type:      "unsubscription_confirmed",
		Data:      msg.Filters,
		Timestamp: time.Now(),
	}

	client.writeMux.Lock()
	client.conn.WriteJSON(response)
	client.writeMux.Unlock()
}

func (ews *EnhancedWebSocketServer) handleConfigure(client *EnhancedClient, msg ClientMessage) {
	client.filtersMux.Lock() // Защищаем доступ к фильтрам
	defer client.filtersMux.Unlock()

	// Настройка throttle
	if throttleMs, ok := msg.Filters["throttle_ms"].(float64); ok {
		client.throttle = time.Duration(throttleMs) * time.Millisecond
	}

	response := ServerMessage{
		Type: "configuration_updated",
		Data: map[string]interface{}{
			"throttle_ms": int(client.throttle.Milliseconds()),
		},
		Timestamp: time.Now(),
	}

	client.writeMux.Lock()
	client.conn.WriteJSON(response)
	client.writeMux.Unlock()
}

func (ews *EnhancedWebSocketServer) ensureNATSSubscriptions(client *EnhancedClient) {
	client.filtersMux.RLock() // Читаем фильтры клиента
	defer client.filtersMux.RUnlock()

	// Подписываемся на все необходимые NATS топики
	for dataType := range client.dataTypes {
		for symbol := range client.symbols {
			for market := range client.markets {
				subject := fmt.Sprintf("%s.%s.%s",
					dataType,
					strings.ToLower(market),
					strings.ToLower(symbol))

				ews.subscriptionsMux.Lock() // Защищаем map ews.subscriptions
				if _, exists := ews.subscriptions[subject]; !exists {
					sub, err := ews.natsConn.Subscribe(subject, func(m *nats.Msg) {
						ews.handleNATSMessage(m, dataType, symbol, market)
					})

					if err != nil {
						log.Printf("❌ Error subscribing to %s: %v", subject, err)
					} else {
						ews.subscriptions[subject] = sub
						log.Printf("📡 Subscribed to NATS subject: %s", subject)
					}
				}
				ews.subscriptionsMux.Unlock()
			}
		}
	}
}

func (ews *EnhancedWebSocketServer) handleNATSMessage(m *nats.Msg, dataType, symbol, market string) {
	// Парсим сообщение от NATS
	var data interface{}
	if err := json.Unmarshal(m.Data, &data); err != nil {
		log.Printf("❌ Error parsing NATS message: %v", err)
		return
	}

	// Создаем сообщение для WebSocket клиентов
	message := ServerMessage{
		Type:      "data",
		Symbol:    symbol,
		Market:    market,
		DataType:  dataType,
		Data:      data,
		Timestamp: time.Now(),
	}

	// Отправляем заинтересованным клиентам
	ews.broadcastToInterestedClients(message)
}

func (ews *EnhancedWebSocketServer) broadcastToInterestedClients(message ServerMessage) {
	ews.clientsMux.RLock()
	clients := make([]*EnhancedClient, 0, len(ews.clients))
	for _, client := range ews.clients {
		clients = append(clients, client)
	}
	ews.clientsMux.RUnlock()

	var wg sync.WaitGroup
	for _, client := range clients {
		// Проверяем интерес клиента
		client.filtersMux.RLock() // Защищаем чтение фильтров
		interested := ews.isClientInterested(client, message)
		client.filtersMux.RUnlock()

		if !interested {
			continue
		}

		// Проверяем throttle
		key := fmt.Sprintf("%s_%s_%s", message.Symbol, message.Market, message.DataType)

		client.lastSentMux.Lock() // Защищаем lastSent при чтении
		if time.Since(client.lastSent[key]) < client.throttle {
			client.lastSentMux.Unlock()
			continue
		}
		client.lastSentMux.Unlock()

		wg.Add(1)
		go func(c *EnhancedClient) {
			defer wg.Done()

			c.writeMux.Lock()
			err := c.conn.WriteJSON(message)
			c.writeMux.Unlock()

			if err != nil {
				// Удаляем проблемного клиента
				ews.clientsMux.Lock()
				delete(ews.clients, c.conn)
				ews.clientsMux.Unlock()
				c.conn.Close()
			} else {
				c.lastSentMux.Lock() // Защищаем lastSent при записи
				c.lastSent[key] = time.Now()
				c.lastSentMux.Unlock()
			}
		}(client)
	}
	wg.Wait()
}

func (ews *EnhancedWebSocketServer) isClientInterested(client *EnhancedClient, message ServerMessage) bool {
	// client.filtersMux.RLock() // <--- ЭТОТ RLock НЕ НУЖЕН ЗДЕСЬ, ОН УЖЕ ВЫШЕ
	// defer client.filtersMux.RUnlock()

	// Проверяем символ
	if len(client.symbols) > 0 && !client.symbols[strings.ToUpper(message.Symbol)] {
		return false
	}

	// Проверяем рынок
	if len(client.markets) > 0 && !client.markets[strings.ToUpper(message.Market)] {
		return false
	}

	// Проверяем тип данных
	if len(client.dataTypes) > 0 && !client.dataTypes[strings.ToLower(message.DataType)] {
		return false
	}

	return true
}

func (ews *EnhancedWebSocketServer) Start(port string) {
	// WebSocket endpoint
	http.HandleFunc("/ws", ews.handleWebSocket)

	// Health check
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":        "ok",
			"clients":       len(ews.clients),
			"subscriptions": len(ews.subscriptions),
			"nats_status":   ews.natsConn.Status().String(),
		})
	})

	// Stats endpoint
	http.HandleFunc("/stats", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Content-Type", "application/json")

		stats := map[string]interface{}{
			"clients":       len(ews.clients),
			"subscriptions": len(ews.subscriptions),
			"uptime":        time.Since(time.Now()).String(),
		}

		json.NewEncoder(w).Encode(stats)
	})

	log.Printf("🚀 Enhanced WebSocket server starting on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

// Вспомогательные функции
func getKeys(m map[string]bool) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

func main() {
	natsURL := os.Getenv("NATS_URL")
	if natsURL == "" {
		natsURL = "nats://31.207.77.179:4222"
	}

	server, err := NewEnhancedWebSocketServer(natsURL)
	if err != nil {
		log.Fatalf("❌ Failed to create Enhanced WebSocket server: %v", err)
	}
	defer server.natsConn.Close()

	server.Start("8080")
}
