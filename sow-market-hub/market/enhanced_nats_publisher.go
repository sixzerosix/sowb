package market

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"

	"github.com/nats-io/nats.go"
)

// EnhancedNATSPublisher - расширенный NATS publisher
type EnhancedNATSPublisher struct {
	nc        *nats.Conn
	isRunning bool
	mu        sync.Mutex

	// Статистика
	messagesSent map[string]int64
	statsMux     sync.RWMutex
}

// EnhancedQuoteData - расширенная структура для NATS
type EnhancedQuoteData struct {
	// Базовые данные
	Symbol    string    `json:"symbol"`
	Price     float64   `json:"price"`
	Market    string    `json:"market"`
	Timestamp time.Time `json:"timestamp"`

	// Расширенные данные
	Volume24h float64 `json:"volume_24h,omitempty"`
	Change24h float64 `json:"change_24h,omitempty"`
	High24h   float64 `json:"high_24h,omitempty"`
	Low24h    float64 `json:"low_24h,omitempty"`

	// Trade данные
	LastTradePrice float64 `json:"last_trade_price,omitempty"`
	LastTradeSize  float64 `json:"last_trade_size,omitempty"`
	LastTradeSide  string  `json:"last_trade_side,omitempty"`
	TradeCount     int64   `json:"trade_count,omitempty"`

	// Orderbook данные
	BestBid       float64 `json:"best_bid,omitempty"`
	BestAsk       float64 `json:"best_ask,omitempty"`
	Spread        float64 `json:"spread,omitempty"`
	SpreadPercent float64 `json:"spread_percent,omitempty"`
	BidPressure   float64 `json:"bid_pressure,omitempty"`
	AskPressure   float64 `json:"ask_pressure,omitempty"`

	// Kline данные
	KlineOpen   float64 `json:"kline_open,omitempty"`
	KlineHigh   float64 `json:"kline_high,omitempty"`
	KlineLow    float64 `json:"kline_low,omitempty"`
	KlineClose  float64 `json:"kline_close,omitempty"`
	KlineVolume float64 `json:"kline_volume,omitempty"`

	// Аналитика
	Analytics *AnalyticsMetrics `json:"analytics,omitempty"`
}

func NewEnhancedNATSPublisher(natsURL string) (*EnhancedNATSPublisher, error) {
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

	return &EnhancedNATSPublisher{
		nc:           nc,
		messagesSent: make(map[string]int64),
	}, nil
}

// Реализация EnhancedMarketDataHandler
func (enp *EnhancedNATSPublisher) Start() {
	enp.mu.Lock()
	defer enp.mu.Unlock()
	enp.isRunning = true
	log.Println("🚀 Enhanced NATS Publisher запущен")
}

func (enp *EnhancedNATSPublisher) Stop() {
	enp.mu.Lock()
	defer enp.mu.Unlock()
	enp.isRunning = false
	if enp.nc != nil {
		enp.nc.Close()
	}
	log.Println("🛑 Enhanced NATS Publisher остановлен")
}

func (enp *EnhancedNATSPublisher) OnMarketDataUpdate(symbol string, data *EnhancedMarketData) {
	enp.mu.Lock()
	if !enp.isRunning {
		enp.mu.Unlock()
		return
	}
	enp.mu.Unlock()

	// Конвертируем в NATS формат
	quoteData := enp.convertToQuoteData(data)

	// Публикуем разные типы данных в разные топики
	enp.publishBasicData(quoteData)
	enp.publishTradeData(quoteData)
	enp.publishOrderbookData(quoteData)
	enp.publishKlineData(quoteData)
	enp.publishAnalyticsData(quoteData)
}

func (enp *EnhancedNATSPublisher) convertToQuoteData(data *EnhancedMarketData) *EnhancedQuoteData {
	quote := &EnhancedQuoteData{
		Symbol:    data.Symbol,
		Price:     data.Price,
		Market:    data.Market,
		Timestamp: data.Timestamp,
		Volume24h: data.Volume24h,
		Change24h: data.Change24h,
		High24h:   data.High24h,
		Low24h:    data.Low24h,
		// Analytics: data.Analytics, // <--- ЭТО НЕПРАВИЛЬНО. Нужно копировать данные, а не ссылку
	}

	// Копируем Analytics, если он есть
	if data.Analytics != nil {
		quote.Analytics = &AnalyticsMetrics{ // Создаем новый объект AnalyticsMetrics
			BuyVolume1m:         data.Analytics.BuyVolume1m,
			SellVolume1m:        data.Analytics.SellVolume1m,
			BuyVolume5m:         data.Analytics.BuyVolume5m,
			SellVolume5m:        data.Analytics.SellVolume5m,
			BuySellRatio:        data.Analytics.BuySellRatio,
			BidPressure:         data.Analytics.BidPressure,
			AskPressure:         data.Analytics.AskPressure,
			SpreadPercent:       data.Analytics.SpreadPercent,
			LongLiquidations1m:  data.Analytics.LongLiquidations1m,
			ShortLiquidations1m: data.Analytics.ShortLiquidations1m,
			LiquidationRatio:    data.Analytics.LiquidationRatio,
			RSI14:               data.Analytics.RSI14,
			EMA12:               data.Analytics.EMA12,
			EMA26:               data.Analytics.EMA26,
			MACD:                data.Analytics.MACD,
			MACDSignal:          data.Analytics.MACDSignal,
			BollingerUpper:      data.Analytics.BollingerUpper,
			BollingerMiddle:     data.Analytics.BollingerMiddle,
			BollingerLower:      data.Analytics.BollingerLower,
			PriceVelocity:       data.Analytics.PriceVelocity,
			VolumeVelocity:      data.Analytics.VolumeVelocity,
			Momentum5m:          data.Analytics.Momentum5m,
			MarketSentiment:     data.Analytics.MarketSentiment,
			SentimentScore:      data.Analytics.SentimentScore,
			Signals:             data.Analytics.Signals, // Слайсы копируются по ссылке, но для сигналов это нормально
		}
	}

	// Trade данные
	if data.LastTrade != nil {
		quote.LastTradePrice = data.LastTrade.Price
		quote.LastTradeSize = data.LastTrade.Size
		quote.LastTradeSide = data.LastTrade.Side
	}
	quote.TradeCount = data.TradeCount

	// Orderbook данные
	if data.Orderbook != nil {
		if len(data.Orderbook.Bids) > 0 {
			quote.BestBid = data.Orderbook.Bids[0].Price
		}
		if len(data.Orderbook.Asks) > 0 {
			quote.BestAsk = data.Orderbook.Asks[0].Price
		}
		quote.Spread = data.Spread
		// Давление в стакане уже есть в Analytics, поэтому здесь не нужно дублировать
		// quote.BidPressure = data.Analytics.BidPressure (если data.Analytics == nil будет паника)
		// quote.AskPressure = data.Analytics.AskPressure
		// quote.SpreadPercent = data.Analytics.SpreadPercent
	}

	// Kline данные
	if data.Kline != nil {
		quote.KlineOpen = data.Kline.Open
		quote.KlineHigh = data.Kline.High
		quote.KlineLow = data.Kline.Low
		quote.KlineClose = data.Kline.Close
		quote.KlineVolume = data.Kline.Volume
	}

	return quote
}

func (enp *EnhancedNATSPublisher) publishBasicData(quote *EnhancedQuoteData) {
	// Базовые данные о цене - для основных потребителей
	basicData := map[string]interface{}{
		"symbol":     quote.Symbol,
		"price":      quote.Price,
		"market":     quote.Market,
		"timestamp":  quote.Timestamp,
		"volume_24h": quote.Volume24h,
		"change_24h": quote.Change24h,
		"high_24h":   quote.High24h,
		"low_24h":    quote.Low24h,
	}

	subject := fmt.Sprintf("quotes.%s.%s",
		strings.ToLower(quote.Market),
		strings.ToLower(quote.Symbol))

	enp.publishToNATS(subject, basicData)
}

func (enp *EnhancedNATSPublisher) publishTradeData(quote *EnhancedQuoteData) {
	if quote.LastTradePrice == 0 {
		return
	}

	tradeData := map[string]interface{}{
		"symbol":           quote.Symbol,
		"market":           quote.Market,
		"timestamp":        quote.Timestamp,
		"last_trade_price": quote.LastTradePrice,
		"last_trade_size":  quote.LastTradeSize,
		"last_trade_side":  quote.LastTradeSide,
		"trade_count":      quote.TradeCount,
	}

	subject := fmt.Sprintf("trades.%s.%s",
		strings.ToLower(quote.Market),
		strings.ToLower(quote.Symbol))

	enp.publishToNATS(subject, tradeData)
}

func (enp *EnhancedNATSPublisher) publishOrderbookData(quote *EnhancedQuoteData) {
	if quote.BestBid == 0 || quote.BestAsk == 0 {
		return
	}

	orderbookData := map[string]interface{}{
		"symbol":         quote.Symbol,
		"market":         quote.Market,
		"timestamp":      quote.Timestamp,
		"best_bid":       quote.BestBid,
		"best_ask":       quote.BestAsk,
		"spread":         quote.Spread,
		"spread_percent": quote.SpreadPercent,
		"bid_pressure":   quote.BidPressure,
		"ask_pressure":   quote.AskPressure,
	}

	subject := fmt.Sprintf("orderbook.%s.%s",
		strings.ToLower(quote.Market),
		strings.ToLower(quote.Symbol))

	enp.publishToNATS(subject, orderbookData)
}

func (enp *EnhancedNATSPublisher) publishKlineData(quote *EnhancedQuoteData) {
	if quote.KlineOpen == 0 {
		return
	}

	klineData := map[string]interface{}{
		"symbol":    quote.Symbol,
		"market":    quote.Market,
		"timestamp": quote.Timestamp,
		"open":      quote.KlineOpen,
		"high":      quote.KlineHigh,
		"low":       quote.KlineLow,
		"close":     quote.KlineClose,
		"volume":    quote.KlineVolume,
	}

	subject := fmt.Sprintf("klines.1m.%s.%s",
		strings.ToLower(quote.Market),
		strings.ToLower(quote.Symbol))

	enp.publishToNATS(subject, klineData)
}

func (enp *EnhancedNATSPublisher) publishAnalyticsData(quote *EnhancedQuoteData) {
	if quote.Analytics == nil {
		return // Нечего публиковать, если аналитика пуста
	}

	analyticsData := map[string]interface{}{
		"symbol":                quote.Symbol,
		"market":                quote.Market,
		"timestamp":             quote.Timestamp,
		"buy_volume_1m":         quote.Analytics.BuyVolume1m,
		"sell_volume_1m":        quote.Analytics.SellVolume1m,
		"buy_volume_5m":         quote.Analytics.BuyVolume5m,  // <--- ДОБАВЛЕНО
		"sell_volume_5m":        quote.Analytics.SellVolume5m, // <--- ДОБАВЛЕНО
		"buy_sell_ratio":        quote.Analytics.BuySellRatio,
		"rsi_14":                quote.Analytics.RSI14,
		"ema_12":                quote.Analytics.EMA12,
		"ema_26":                quote.Analytics.EMA26,
		"macd":                  quote.Analytics.MACD,
		"macd_signal":           quote.Analytics.MACDSignal,
		"bollinger_upper":       quote.Analytics.BollingerUpper,
		"bollinger_middle":      quote.Analytics.BollingerMiddle,
		"bollinger_lower":       quote.Analytics.BollingerLower,
		"market_sentiment":      quote.Analytics.MarketSentiment,
		"sentiment_score":       quote.Analytics.SentimentScore,
		"signals":               quote.Analytics.Signals,
		"bid_pressure":          quote.Analytics.BidPressure,         // <--- ДОБАВЛЕНО
		"ask_pressure":          quote.Analytics.AskPressure,         // <--- ДОБАВЛЕНО
		"spread_percent":        quote.Analytics.SpreadPercent,       // <--- ДОБАВЛЕНО
		"long_liquidations_1m":  quote.Analytics.LongLiquidations1m,  // <--- ДОБАВЛЕНО
		"short_liquidations_1m": quote.Analytics.ShortLiquidations1m, // <--- ДОБАВЛЕНО
		"liquidation_ratio":     quote.Analytics.LiquidationRatio,    // <--- ДОБАВЛЕНО
		"price_velocity":        quote.Analytics.PriceVelocity,       // <--- ДОБАВЛЕНО
		"volume_velocity":       quote.Analytics.VolumeVelocity,      // <--- ДОБАВЛЕНО
		"momentum_5m":           quote.Analytics.Momentum5m,          // <--- ДОБАВЛЕНО
	}

	subject := fmt.Sprintf("analytics.%s.%s",
		strings.ToLower(quote.Market),
		strings.ToLower(quote.Symbol))

	enp.publishToNATS(subject, analyticsData)
}

func (enp *EnhancedNATSPublisher) publishToNATS(subject string, data interface{}) {
	jsonData, err := json.Marshal(data)
	if err != nil {
		log.Printf("❌ Ошибка сериализации данных для %s: %v", subject, err)
		return
	}

	err = enp.nc.Publish(subject, jsonData)
	if err != nil {
		log.Printf("❌ Ошибка публикации в %s: %v", subject, err)
		return
	}

	// Обновляем статистику
	enp.statsMux.Lock()
	enp.messagesSent[subject]++
	enp.statsMux.Unlock()
}

func (enp *EnhancedNATSPublisher) GetStats() map[string]int64 {
	enp.statsMux.RLock()
	defer enp.statsMux.RUnlock()

	stats := make(map[string]int64)
	for subject, count := range enp.messagesSent {
		stats[subject] = count
	}
	return stats
}
