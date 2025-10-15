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

// Конфигурация
type Config struct {
	Server struct {
		Port        string   `json:"port"`
		CORSOrigins []string `json:"cors_origins"`
	} `json:"server"`
	NATS struct {
		URL                  string `json:"url"`
		MaxReconnects        int    `json:"max_reconnects"`
		ReconnectWaitSeconds int    `json:"reconnect_wait_seconds"`
	} `json:"nats"`
	Symbols []string `json:"symbols"`
	Markets []string `json:"markets"`
}

type WebSocketServer struct {
	config           *Config
	clients          map[*websocket.Conn]*Client
	clientsMux       sync.RWMutex
	natsConn         *nats.Conn
	upgrader         websocket.Upgrader
	subscriptions    map[string]*nats.Subscription
	subscriptionsMux sync.RWMutex
}

type Client struct {
	conn       *websocket.Conn
	symbols    map[string]bool
	symbolsMux sync.RWMutex
	writeMux   sync.Mutex // ДОБАВЛЯЕМ МЬЮТЕКС ДЛЯ ЗАПИСИ
}

type ClientMessage struct {
	Type    string   `json:"type"`
	Symbols []string `json:"symbols,omitempty"`
}

type MarketDataMessage struct {
	Symbol    string    `json:"symbol"`
	Price     float64   `json:"price"`
	Market    string    `json:"market"`
	Timestamp time.Time `json:"timestamp"`
}

type NATSQuoteData struct {
	Symbol    string    `json:"symbol"`
	Price     float64   `json:"price"`
	Market    string    `json:"market"`
	Timestamp time.Time `json:"timestamp"`
}

// loadConfig загружает конфигурацию
func loadConfig(configPath string) (*Config, error) {
	if configPath == "" {
		configPath = "config.json"
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("ошибка чтения конфига: %w", err)
	}

	var config Config
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("ошибка парсинга конфига: %w", err)
	}

	// Валидация конфига
	if config.Server.Port == "" {
		config.Server.Port = "8080"
	}
	if config.NATS.URL == "" {
		config.NATS.URL = "nats://31.207.77.179:4222"
	}
	if len(config.Symbols) == 0 {
		return nil, fmt.Errorf("список символов не может быть пустым")
	}
	if len(config.Markets) == 0 {
		config.Markets = []string{"spot", "futures"}
	}

	return &config, nil
}

func NewWebSocketServer(config *Config) (*WebSocketServer, error) {
	// Подключение к NATS с настройками из конфига
	opts := []nats.Option{
		nats.MaxReconnects(config.NATS.MaxReconnects),
		nats.ReconnectWait(time.Duration(config.NATS.ReconnectWaitSeconds) * time.Second),
		nats.DisconnectErrHandler(func(nc *nats.Conn, err error) {
			log.Printf("NATS disconnected: %v", err)
		}),
		nats.ReconnectHandler(func(nc *nats.Conn) {
			log.Println("NATS reconnected!")
		}),
	}

	nc, err := nats.Connect(config.NATS.URL, opts...)
	if err != nil {
		return nil, fmt.Errorf("ошибка подключения к NATS: %w", err)
	}

	return &WebSocketServer{
		config:        config,
		clients:       make(map[*websocket.Conn]*Client),
		natsConn:      nc,
		subscriptions: make(map[string]*nats.Subscription),
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				// Проверяем CORS origins из конфига
				origin := r.Header.Get("Origin")
				for _, allowedOrigin := range config.Server.CORSOrigins {
					if allowedOrigin == "*" || allowedOrigin == origin {
						return true
					}
				}
				return len(config.Server.CORSOrigins) == 0
			},
		},
	}, nil
}

func (ws *WebSocketServer) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := ws.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("❌ WebSocket upgrade error: %v", err)
		return
	}
	defer conn.Close()

	// Создаем клиента с мьютексом для записи
	client := &Client{
		conn:    conn,
		symbols: make(map[string]bool),
	}

	// Добавляем клиента
	ws.clientsMux.Lock()
	ws.clients[conn] = client
	ws.clientsMux.Unlock()

	log.Printf("✅ New WebSocket client connected. Total: %d", len(ws.clients))

	// Удаляем клиента при отключении
	defer func() {
		ws.clientsMux.Lock()
		delete(ws.clients, conn)
		ws.clientsMux.Unlock()
		log.Printf("❌ WebSocket client disconnected. Total: %d", len(ws.clients))
	}()

	// Автоматически подписываемся на все символы из конфига
	ws.subscribeClientToSymbols(client, ws.config.Symbols)

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

		// Обрабатываем дополнительные подписки
		if msg.Type == "subscribe" && len(msg.Symbols) > 0 {
			ws.subscribeClientToSymbols(client, msg.Symbols)
		}
	}
}

func (ws *WebSocketServer) subscribeClientToSymbols(client *Client, symbols []string) {
	client.symbolsMux.Lock()
	for _, symbol := range symbols {
		client.symbols[strings.ToUpper(symbol)] = true
	}
	client.symbolsMux.Unlock()

	// Подписываемся на NATS топики (если еще не подписаны)
	for _, symbol := range symbols {
		ws.ensureNATSSubscription(symbol)
	}
}

func (ws *WebSocketServer) ensureNATSSubscription(symbol string) {
	symbolLower := strings.ToLower(symbol)

	// Подписываемся на все рынки из конфига
	for _, market := range ws.config.Markets {
		subject := fmt.Sprintf("quotes.%s.%s", market, symbolLower)

		ws.subscriptionsMux.Lock()
		if _, exists := ws.subscriptions[subject]; !exists {
			sub, err := ws.natsConn.Subscribe(subject, func(m *nats.Msg) {
				ws.handleNATSMessage(m, subject)
			})

			if err != nil {
				log.Printf("❌ Error subscribing to %s: %v", subject, err)
			} else {
				ws.subscriptions[subject] = sub
				log.Printf("📡 Subscribed to NATS subject: %s", subject)
			}
		}
		ws.subscriptionsMux.Unlock()
	}
}

func (ws *WebSocketServer) handleNATSMessage(m *nats.Msg, subject string) {
	// Парсим сообщение от NATS
	var natsData NATSQuoteData
	if err := json.Unmarshal(m.Data, &natsData); err != nil {
		log.Printf("❌ Error parsing NATS message from %s: %v", subject, err)
		return
	}

	// Создаем сообщение для WebSocket клиентов
	wsMessage := MarketDataMessage{
		Symbol:    natsData.Symbol,
		Price:     natsData.Price,
		Market:    natsData.Market,
		Timestamp: natsData.Timestamp,
	}

	// Отправляем заинтересованным клиентам (БЕЗ ЛОГИРОВАНИЯ ДЛЯ ПРОИЗВОДИТЕЛЬНОСТИ)
	ws.broadcastToInterestedClients(wsMessage)
}

// ИСПРАВЛЕННАЯ ФУНКЦИЯ С THREAD-SAFE ЗАПИСЬЮ
func (ws *WebSocketServer) broadcastToInterestedClients(message MarketDataMessage) {
	ws.clientsMux.RLock()
	clients := make([]*Client, 0, len(ws.clients))
	for _, client := range ws.clients {
		clients = append(clients, client)
	}
	ws.clientsMux.RUnlock()

	// Отправляем сообщения параллельно, но безопасно
	var wg sync.WaitGroup
	for _, client := range clients {
		// Проверяем, интересует ли клиента этот символ
		client.symbolsMux.RLock()
		interested := client.symbols[strings.ToUpper(message.Symbol)]
		client.symbolsMux.RUnlock()

		if interested {
			wg.Add(1)
			go func(c *Client) {
				defer wg.Done()

				// ИСПОЛЬЗУЕМ МЬЮТЕКС ДЛЯ БЕЗОПАСНОЙ ЗАПИСИ
				c.writeMux.Lock()
				err := c.conn.WriteJSON(message)
				c.writeMux.Unlock()

				if err != nil {
					// Удаляем проблемного клиента
					ws.clientsMux.Lock()
					delete(ws.clients, c.conn)
					ws.clientsMux.Unlock()
					c.conn.Close()
				}
			}(client)
		}
	}
	wg.Wait()
}

func (ws *WebSocketServer) Start() {
	// CORS middleware
	http.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		// Устанавливаем CORS заголовки
		if len(ws.config.Server.CORSOrigins) > 0 {
			origin := r.Header.Get("Origin")
			for _, allowedOrigin := range ws.config.Server.CORSOrigins {
				if allowedOrigin == "*" || allowedOrigin == origin {
					w.Header().Set("Access-Control-Allow-Origin", allowedOrigin)
					break
				}
			}
		} else {
			w.Header().Set("Access-Control-Allow-Origin", "*")
		}

		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		ws.handleWebSocket(w, r)
	})

	// Health check endpoint
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":        "ok",
			"clients":       len(ws.clients),
			"subscriptions": len(ws.subscriptions),
			"nats_status":   ws.natsConn.Status().String(),
			"symbols":       ws.config.Symbols,
			"markets":       ws.config.Markets,
		})
	})

	// Config endpoint для отладки
	http.HandleFunc("/config", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(ws.config)
	})

	log.Printf("🚀 WebSocket server starting on port %s", ws.config.Server.Port)
	log.Printf("📡 NATS connection: %s (status: %s)", ws.config.NATS.URL, ws.natsConn.Status().String())
	log.Printf("💰 Tracking %d symbols across %d markets", len(ws.config.Symbols), len(ws.config.Markets))

	log.Fatal(http.ListenAndServe(":"+ws.config.Server.Port, nil))
}

func main() {
	// Загружаем конфигурацию
	configPath := "config.json"
	if len(os.Args) > 1 {
		configPath = os.Args[1]
	}

	config, err := loadConfig(configPath)
	if err != nil {
		log.Fatalf("❌ Failed to load config: %v", err)
	}

	log.Printf("📋 Loaded config from %s", configPath)

	// Создаем сервер
	server, err := NewWebSocketServer(config)
	if err != nil {
		log.Fatalf("❌ Failed to create WebSocket server: %v", err)
	}
	defer server.natsConn.Close()

	// Запускаем сервер
	server.Start()
}
