package market

import "time"

// QuoteData представляет структуру котировки
type QuoteData struct {
	Symbol    string    `json:"symbol"`
	Price     float64   `json:"price"`
	Market    string    `json:"market"` // "SPOT" или "FUTURES"
	Timestamp time.Time `json:"timestamp"`
}

// EnhancedMarketData - полная структура рыночных данных
type EnhancedMarketData struct {
	Symbol    string    `json:"symbol"`
	Market    string    `json:"market"`
	Timestamp time.Time `json:"timestamp"`

	// Ticker данные
	Price     float64 `json:"price"`
	Volume24h float64 `json:"volume24h"`
	Change24h float64 `json:"change24h"`
	High24h   float64 `json:"high24h"`
	Low24h    float64 `json:"low24h"`

	// Trade данные
	LastTrade    *TradeData  `json:"last_trade,omitempty"`
	TradeCount   int64       `json:"trade_count"`
	TradeHistory []TradeData `json:"trade_history,omitempty"`

	// Orderbook данные
	Orderbook *OrderbookData `json:"orderbook,omitempty"`
	Spread    float64        `json:"spread"`

	// Kline данные
	Kline        *KlineData  `json:"kline,omitempty"`
	KlineHistory []KlineData `json:"kline_history,omitempty"`

	// Liquidation данные
	Liquidations []LiquidationData `json:"liquidations,omitempty"`

	// Аналитика
	Analytics *AnalyticsMetrics `json:"analytics,omitempty"`
}

type TradeData struct {
	ID        string    `json:"id"`
	Price     float64   `json:"price"`
	Size      float64   `json:"size"`
	Side      string    `json:"side"` // Buy/Sell
	Timestamp time.Time `json:"timestamp"`
}

type OrderbookData struct {
	Bids      []PriceLevel `json:"bids"`
	Asks      []PriceLevel `json:"asks"`
	Timestamp time.Time    `json:"timestamp"`
}

type PriceLevel struct {
	Price float64 `json:"price"`
	Size  float64 `json:"size"`
}

type KlineData struct {
	Open      float64   `json:"open"`
	High      float64   `json:"high"`
	Low       float64   `json:"low"`
	Close     float64   `json:"close"`
	Volume    float64   `json:"volume"`
	Timestamp time.Time `json:"timestamp"`
	Interval  string    `json:"interval"`
}

type LiquidationData struct {
	Price     float64   `json:"price"`
	Size      float64   `json:"size"`
	Side      string    `json:"side"` // Buy/Sell
	Timestamp time.Time `json:"timestamp"`
}

// AnalyticsMetrics - метрики для анализа
type AnalyticsMetrics struct {
	// Объемы торгов
	BuyVolume1m  float64 `json:"buy_volume_1m"`
	SellVolume1m float64 `json:"sell_volume_1m"`
	BuyVolume5m  float64 `json:"buy_volume_5m"`
	SellVolume5m float64 `json:"sell_volume_5m"`

	// Активность стакана
	BidPressure   float64 `json:"bid_pressure"`
	AskPressure   float64 `json:"ask_pressure"`
	SpreadPercent float64 `json:"spread_percent"`

	// Ликвидации
	LongLiquidations1m  float64 `json:"long_liquidations_1m"`
	ShortLiquidations1m float64 `json:"short_liquidations_1m"`
	LiquidationRatio    float64 `json:"liquidation_ratio"`

	// Технические индикаторы
	RSI14           float64 `json:"rsi_14"`
	EMA12           float64 `json:"ema_12"`
	EMA26           float64 `json:"ema_26"`
	MACD            float64 `json:"macd"`
	MACDSignal      float64 `json:"macd_signal"`
	BollingerUpper  float64 `json:"bollinger_upper"`
	BollingerMiddle float64 `json:"bollinger_middle"`
	BollingerLower  float64 `json:"bollinger_lower"`

	// Momentum индикаторы
	PriceVelocity  float64 `json:"price_velocity"`
	VolumeVelocity float64 `json:"volume_velocity"`
	Momentum5m     float64 `json:"momentum_5m"`

	// Sentiment
	BuySellRatio    float64 `json:"buy_sell_ratio"`
	MarketSentiment string  `json:"market_sentiment"` // bullish/bearish/neutral
	SentimentScore  float64 `json:"sentiment_score"`

	// Торговые сигналы
	Signals []TradingSignal `json:"signals,omitempty"`
}

type TradingSignal struct {
	Type      string    `json:"type"`     // buy/sell/hold
	Strength  float64   `json:"strength"` // 0-100
	Reason    string    `json:"reason"`
	Timestamp time.Time `json:"timestamp"`
}

// MarketDataPoint представляет точку данных для хранения
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
