package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/redis/go-redis/v9"
)

type DragonflyWebSocketServer struct {
	clients      map[*websocket.Conn]bool
	clientsMux   sync.RWMutex
	redisClient  *redis.Client
	upgrader     websocket.Upgrader
	ctx          context.Context
	subscribers  map[string]*redis.PubSub
	subsMux      sync.RWMutex
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

func NewDragonflyWebSocketServer(dragonflyURL string) (*DragonflyWebSocketServer, error) {
	// Подключение к Dragonfly
	rdb := redis.NewClient(&redis.Options{
		Addr: dragonflyURL,
	})

	ctx := context.Background()
	_, err := rdb.Ping(ctx).Result()
	if err != nil {
		return nil, err
	}

	return &DragonflyWebSocketServer{
		clients:     make(map[*websocket.Conn]bool),
		redisClient: rdb,
		ctx:         ctx,
		subscribers: make(map[string]*redis.PubSub),
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				return true
			},
		},
	}, nil
}

func (ws *DragonflyWebSocketServer) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := ws.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("❌ WebSocket upgrade error: %v", err)
		return
	}
	defer conn.Close()

	// Добавляем клиента
	ws.clientsMux.Lock()
	ws.clients[conn] = true
	ws.clientsMux.Unlock()

	log.Printf("✅ New WebSocket client connected. Total: %d", len(ws.clients))

	// Удаляем клиента при отключении
	defer func() {
		ws.clientsMux.Lock()
		delete(ws.clients, conn)
		ws.clientsMux.Unlock()
		log.Printf("❌ WebSocket client disconnected. Total: %d", len(ws.clients))
	}()

	// Отправляем текущие данные новому клиенту
	ws.sendCurrentData(conn)

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

		if msg.Type == "subscribe" && len(msg.Symbols) > 0 {
			ws.subscribeToSymbols(msg.Symbols)
		}
	}
}

func (ws *DragonflyWebSocketServer) sendCurrentData(conn *websocket.Conn) {
	// Получаем все последние цены из Dragonfly
	symbols := []string{
		"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT",
		// ... добавь остальные символы
	}
	
	for _, symbol := range symbols {
		for _, market := range []string{"SPOT", "FUTURES"} {
			key := fmt.Sprintf("latest:%s:%s", market, symbol)
			
			data, err := ws.redisClient.HGetAll(ws.ctx, key).Result()
			if err != nil || len(data) == 0 {
				continue
			}

			price, _ := strconv.ParseFloat(data["price"], 64)
			timestamp, _ := strconv.ParseInt(data["timestamp"], 10, 64)

			message := MarketDataMessage{
				Symbol:    symbol,
				Price:     price,
				Market:    market,
				Timestamp: time.Unix(timestamp, 0),
			}

			conn.WriteJSON(message)
		}
	}
}

func (ws *DragonflyWebSocketServer) subscribeToSymbols(symbols []string) {
	for _, symbol := range symbols {
		// Подписываемся на обновления для каждого символа
		pattern := fmt.Sprintf("updates:%s:*", strings.ToUpper(symbol))
		
		ws.subsMux.Lock()
		if _, exists := ws.subscribers[pattern]; !exists {
			pubsub := ws.redisClient.PSubscribe(ws.ctx, pattern)
			ws.subscribers[pattern] = pubsub
			
			go func(ps *redis.PubSub, sym string) {
				defer ps.Close()
				
				ch := ps.Channel()
				for msg := range ch {
					ws.handleRedisMessage(msg, sym)
				}
			}(pubsub, symbol)
			
			log.Printf("📡 Subscribed to updates for %s", symbol)
		}
		ws.subsMux.Unlock()
	}
}

func (ws *DragonflyWebSocketServer) handleRedisMessage(msg *redis.Message, symbol string) {
	// Парсим сообщение от Redis/Dragonfly
	var updateData struct {
		Symbol    string    `json:"symbol"`
		Price     float64   `json:"price"`
		Market    string    `json:"market"`
		Timestamp time.Time `json:"timestamp"`
	}

	if err := json.Unmarshal([]byte(msg.Payload), &updateData); err != nil {
		log.Printf("❌ Error parsing Redis message: %v", err)
		return
	}

	// Отправляем всем подключенным клиентам
	ws.broadcastToClients(updateData)
}

func (ws *DragonflyWebSocketServer) broadcastToClients(message MarketDataMessage) {
	ws.clientsMux.RLock()
	defer ws.clientsMux.RUnlock()

	for client := range ws.clients {
		err := client.WriteJSON(message)
		if err != nil {
			log.Printf("❌ Error sending to client: %v", err)
			client.Close()
			delete(ws.clients, client)
		}
	}
}

func (ws *DragonflyWebSocketServer) startPeriodicUpdates() {
	ticker := time.NewTicker(1 * time.Second)
	go func() {
		for range ticker.C {
			// Периодически отправляем обновления из Dragonfly
			ws.sendUpdatesFromDragonfly()
		}
	}()
}

func (ws *DragonflyWebSocketServer) sendUpdatesFromDragonfly() {
	symbols := []string{"BTCUSDT", "ETHUSDT", "SOLUSDT"} // добавь остальные
	
	for _, symbol := range symbols {
		for _, market := range []string{"SPOT", "FUTURES"} {
			key := fmt.Sprintf("latest:%s:%s", market, symbol)
			
			data, err := ws.redisClient.HGetAll(ws.ctx, key).Result()
			if err != nil || len(data) == 0 {
				continue
			}

			price, _ := strconv.ParseFloat(data["price"], 64)
			timestamp, _ := strconv.ParseInt(data["timestamp"], 10, 64)

			message := MarketDataMessage{
				Symbol:    symbol,
				Price:     price,
				Market:    market,
				Timestamp: time.Unix(timestamp, 0),
			}

			ws.broadcastToClients(message)
		}
	}
}

func (ws *DragonflyWebSocketServer) Start(port string) {
	// Запускаем периодические обновления
	ws.startPeriodicUpdates()
	
	http.HandleFunc("/ws", ws.handleWebSocket)
	
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":  "ok",
			"clients": len(ws.clients),
		})
	})

	log.Printf("🚀 Dragonfly WebSocket server starting on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

func main() {
	server, err := NewDragonflyWebSocketServer("localhost:6379")
	if err != nil {
		log.Fatalf("❌ Failed to create WebSocket server: %v", err)
	}
	defer server.redisClient.Close()

	server.Start("8080")
}
