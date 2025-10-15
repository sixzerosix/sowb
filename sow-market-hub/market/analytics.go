package market

import (
	"math"
	"sync"
	"time"
)

// RealTimeAnalytics - движок для real-time аналитики
type RealTimeAnalytics struct {
	data          map[string]*EnhancedMarketData
	priceHistory  map[string][]float64
	volumeHistory map[string][]float64
	mu            sync.RWMutex
	isRunning     bool

	// Технические индикаторы
	indicators map[string]*TechnicalIndicators
}

type TechnicalIndicators struct {
	RSI            *RSIIndicator
	EMA12          *EMAIndicator
	EMA26          *EMAIndicator
	MACD           *MACDIndicator
	BollingerBands *BollingerBandsIndicator

	// Кастомные индикаторы
	VolumeProfile  *VolumeProfileIndicator
	OrderFlowDelta *OrderFlowDeltaIndicator
}

// RSI Indicator
type RSIIndicator struct {
	period    int
	gains     []float64
	losses    []float64
	avgGain   float64
	avgLoss   float64
	lastPrice float64
}

// EMA Indicator
type EMAIndicator struct {
	period      int
	multiplier  float64
	ema         float64
	initialized bool
}

// MACD Indicator
type MACDIndicator struct {
	ema12     *EMAIndicator
	ema26     *EMAIndicator
	signal    *EMAIndicator
	histogram float64
}

// Bollinger Bands
type BollingerBandsIndicator struct {
	period    int
	deviation float64
	prices    []float64
	sma       float64
	upper     float64
	middle    float64
	lower     float64
}

// Volume Profile
type VolumeProfileIndicator struct {
	priceRanges   map[float64]float64 // price -> volume
	pocPrice      float64             // Point of Control
	valueAreaHigh float64
	valueAreaLow  float64
}

// Order Flow Delta
type OrderFlowDeltaIndicator struct {
	buyVolume  float64
	sellVolume float64
	delta      float64
	cumDelta   float64
}

func NewRealTimeAnalytics() *RealTimeAnalytics {
	return &RealTimeAnalytics{
		data:          make(map[string]*EnhancedMarketData),
		priceHistory:  make(map[string][]float64),
		volumeHistory: make(map[string][]float64),
		indicators:    make(map[string]*TechnicalIndicators),
	}
}

func (ra *RealTimeAnalytics) Start() {
	ra.isRunning = true
}

func (ra *RealTimeAnalytics) Stop() {
	ra.isRunning = false
}

func (ra *RealTimeAnalytics) UpdateData(key string, data *EnhancedMarketData) {
	ra.mu.Lock()
	defer ra.mu.Unlock()

	ra.data[key] = data

	// Обновляем историю цен
	if ra.priceHistory[key] == nil {
		ra.priceHistory[key] = make([]float64, 0, 1000)
	}
	ra.priceHistory[key] = append(ra.priceHistory[key], data.Price)
	if len(ra.priceHistory[key]) > 1000 {
		ra.priceHistory[key] = ra.priceHistory[key][1:]
	}

	// Обновляем историю объемов
	if ra.volumeHistory[key] == nil {
		ra.volumeHistory[key] = make([]float64, 0, 1000)
	}
	ra.volumeHistory[key] = append(ra.volumeHistory[key], data.Volume24h)
	if len(ra.volumeHistory[key]) > 1000 {
		ra.volumeHistory[key] = ra.volumeHistory[key][1:]
	}

	// Инициализируем индикаторы если нужно
	if ra.indicators[key] == nil {
		ra.indicators[key] = &TechnicalIndicators{
			RSI:            NewRSIIndicator(14),
			EMA12:          NewEMAIndicator(12),
			EMA26:          NewEMAIndicator(26),
			MACD:           NewMACDIndicator(),
			BollingerBands: NewBollingerBandsIndicator(20, 2.0),
			VolumeProfile:  NewVolumeProfileIndicator(),
			OrderFlowDelta: NewOrderFlowDeltaIndicator(),
		}
	}
}

func (ra *RealTimeAnalytics) CalculateMetrics(key string) *AnalyticsMetrics {
	ra.mu.RLock()
	defer ra.mu.RUnlock()

	data := ra.data[key]
	if data == nil {
		return nil
	}

	indicators := ra.indicators[key]
	if indicators == nil {
		return nil
	}

	priceHistory := ra.priceHistory[key]
	if len(priceHistory) < 14 {
		return nil // Недостаточно данных
	}

	metrics := &AnalyticsMetrics{}

	// 1. Обновляем технические индикаторы
	ra.updateTechnicalIndicators(key, data.Price)

	// 2. Анализ объемов торгов
	metrics.BuyVolume1m, metrics.SellVolume1m = ra.calculateVolumeByDirection(data, 1*time.Minute)
	metrics.BuyVolume5m, metrics.SellVolume5m = ra.calculateVolumeByDirection(data, 5*time.Minute)

	if metrics.SellVolume1m > 0 {
		metrics.BuySellRatio = metrics.BuyVolume1m / metrics.SellVolume1m
	}

	// 3. Анализ стакана заявок
	metrics.BidPressure, metrics.AskPressure, metrics.SpreadPercent = ra.calculateOrderbookMetrics(data)

	// 4. Анализ ликвидаций
	metrics.LongLiquidations1m, metrics.ShortLiquidations1m = ra.calculateLiquidations(data, 1*time.Minute)
	if metrics.ShortLiquidations1m > 0 {
		metrics.LiquidationRatio = metrics.LongLiquidations1m / metrics.ShortLiquidations1m
	}

	// 5. Технические индикаторы
	metrics.RSI14 = indicators.RSI.GetValue()
	metrics.EMA12 = indicators.EMA12.GetValue()
	metrics.EMA26 = indicators.EMA26.GetValue()
	metrics.MACD = indicators.MACD.GetMACD()
	metrics.MACDSignal = indicators.MACD.GetSignal()
	metrics.BollingerUpper = indicators.BollingerBands.upper
	metrics.BollingerMiddle = indicators.BollingerBands.middle
	metrics.BollingerLower = indicators.BollingerBands.lower

	// 6. Momentum индикаторы
	metrics.PriceVelocity = ra.calculatePriceVelocity(priceHistory)
	metrics.VolumeVelocity = ra.calculateVolumeVelocity(key)
	metrics.Momentum5m = ra.calculateMomentum(priceHistory, 5)

	// 7. Sentiment анализ
	metrics.SentimentScore = ra.calculateSentimentScore(metrics)
	metrics.MarketSentiment = ra.determineSentiment(metrics)

	// 8. Торговые сигналы
	metrics.Signals = ra.generateTradingSignals(key, metrics)

	return metrics
}

func (ra *RealTimeAnalytics) updateTechnicalIndicators(key string, price float64) {
	indicators := ra.indicators[key]

	// Обновляем RSI
	indicators.RSI.Update(price)

	// Обновляем EMA
	indicators.EMA12.Update(price)
	indicators.EMA26.Update(price)

	// Обновляем MACD
	indicators.MACD.Update(indicators.EMA12.GetValue(), indicators.EMA26.GetValue())

	// Обновляем Bollinger Bands
	indicators.BollingerBands.Update(price)
}

func (ra *RealTimeAnalytics) calculateVolumeByDirection(data *EnhancedMarketData, duration time.Duration) (buyVol, sellVol float64) {
	if data.TradeHistory == nil {
		return 0, 0
	}

	cutoff := time.Now().Add(-duration)

	for _, trade := range data.TradeHistory {
		if trade.Timestamp.After(cutoff) {
			if trade.Side == "Buy" {
				buyVol += trade.Size
			} else {
				sellVol += trade.Size
			}
		}
	}

	return buyVol, sellVol
}

func (ra *RealTimeAnalytics) calculateOrderbookMetrics(data *EnhancedMarketData) (bidPressure, askPressure, spreadPercent float64) {
	if data.Orderbook == nil || len(data.Orderbook.Bids) == 0 || len(data.Orderbook.Asks) == 0 {
		return 0, 0, 0
	}

	// Суммируем объемы в стакане (топ 10 уровней)
	for i := 0; i < min(10, len(data.Orderbook.Bids)); i++ {
		bidPressure += data.Orderbook.Bids[i].Size
	}

	for i := 0; i < min(10, len(data.Orderbook.Asks)); i++ {
		askPressure += data.Orderbook.Asks[i].Size
	}

	// Вычисляем спред в процентах
	bestBid := data.Orderbook.Bids[0].Price
	bestAsk := data.Orderbook.Asks[0].Price
	spread := bestAsk - bestBid
	midPrice := (bestBid + bestAsk) / 2

	if midPrice > 0 {
		spreadPercent = (spread / midPrice) * 100
	}

	return bidPressure, askPressure, spreadPercent
}

func (ra *RealTimeAnalytics) calculateLiquidations(data *EnhancedMarketData, duration time.Duration) (longLiq, shortLiq float64) {
	if data.Liquidations == nil {
		return 0, 0
	}

	cutoff := time.Now().Add(-duration)

	for _, liq := range data.Liquidations {
		if liq.Timestamp.After(cutoff) {
			if liq.Side == "Buy" { // Long liquidation
				longLiq += liq.Size
			} else { // Short liquidation
				shortLiq += liq.Size
			}
		}
	}

	return longLiq, shortLiq
}

func (ra *RealTimeAnalytics) calculatePriceVelocity(prices []float64) float64 {
	if len(prices) < 2 {
		return 0
	}

	// Простая скорость изменения цены
	recent := prices[len(prices)-1]
	previous := prices[len(prices)-2]

	return ((recent - previous) / previous) * 100
}

func (ra *RealTimeAnalytics) calculateVolumeVelocity(key string) float64 {
	volumes := ra.volumeHistory[key]
	if len(volumes) < 2 {
		return 0
	}

	recent := volumes[len(volumes)-1]
	previous := volumes[len(volumes)-2]

	if previous > 0 {
		return ((recent - previous) / previous) * 100
	}
	return 0
}

func (ra *RealTimeAnalytics) calculateMomentum(prices []float64, periods int) float64 {
	if len(prices) < periods+1 {
		return 0
	}

	current := prices[len(prices)-1]
	past := prices[len(prices)-1-periods]

	return ((current - past) / past) * 100
}

func (ra *RealTimeAnalytics) calculateSentimentScore(metrics *AnalyticsMetrics) float64 {
	score := 0.0

	// RSI анализ
	if metrics.RSI14 > 70 {
		score -= 1 // Перекупленность
	} else if metrics.RSI14 < 30 {
		score += 1 // Перепроданность
	}

	// MACD анализ
	if metrics.MACD > metrics.MACDSignal {
		score += 0.5 // Бычий сигнал
	} else {
		score -= 0.5 // Медвежий сигнал
	}

	// Bollinger Bands анализ
	currentPrice := metrics.BollingerMiddle // Используем как текущую цену
	if currentPrice > metrics.BollingerUpper {
		score -= 0.5 // Выше верхней полосы
	} else if currentPrice < metrics.BollingerLower {
		score += 0.5 // Ниже нижней полосы
	}

	// Volume анализ
	if metrics.BuySellRatio > 1.5 {
		score += 1 // Сильное преобладание покупок
	} else if metrics.BuySellRatio < 0.5 {
		score -= 1 // Сильное преобладание продаж
	}

	// Orderbook анализ
	if metrics.BidPressure > metrics.AskPressure*1.5 {
		score += 0.5 // Сильное давление покупателей
	} else if metrics.AskPressure > metrics.BidPressure*1.5 {
		score -= 0.5 // Сильное давление продавцов
	}

	// Liquidation анализ
	if metrics.ShortLiquidations1m > metrics.LongLiquidations1m*2 {
		score += 0.5 // Много коротких ликвидаций = рост
	} else if metrics.LongLiquidations1m > metrics.ShortLiquidations1m*2 {
		score -= 0.5 // Много длинных ликвидаций = падение
	}

	// Нормализуем score от -5 до +5 в диапазон -100 до +100
	return math.Max(-100, math.Min(100, score*20))
}

func (ra *RealTimeAnalytics) determineSentiment(metrics *AnalyticsMetrics) string {
	score := metrics.SentimentScore

	if score >= 60 {
		return "strongly_bullish"
	} else if score >= 20 {
		return "bullish"
	} else if score <= -60 {
		return "strongly_bearish"
	} else if score <= -20 {
		return "bearish"
	}
	return "neutral"
}

func (ra *RealTimeAnalytics) generateTradingSignals(key string, metrics *AnalyticsMetrics) []TradingSignal {
	signals := make([]TradingSignal, 0)
	now := time.Now()

	// RSI Divergence Signal
	if metrics.RSI14 < 30 && metrics.PriceVelocity > 0 {
		signals = append(signals, TradingSignal{
			Type:      "buy",
			Strength:  75,
			Reason:    "RSI oversold with positive price momentum",
			Timestamp: now,
		})
	} else if metrics.RSI14 > 70 && metrics.PriceVelocity < 0 {
		signals = append(signals, TradingSignal{
			Type:      "sell",
			Strength:  75,
			Reason:    "RSI overbought with negative price momentum",
			Timestamp: now,
		})
	}

	// MACD Crossover Signal
	if metrics.MACD > metrics.MACDSignal && metrics.MACD > 0 {
		signals = append(signals, TradingSignal{
			Type:      "buy",
			Strength:  60,
			Reason:    "MACD bullish crossover above zero line",
			Timestamp: now,
		})
	} else if metrics.MACD < metrics.MACDSignal && metrics.MACD < 0 {
		signals = append(signals, TradingSignal{
			Type:      "sell",
			Strength:  60,
			Reason:    "MACD bearish crossover below zero line",
			Timestamp: now,
		})
	}

	// Volume Surge Signal
	if metrics.VolumeVelocity > 50 && metrics.BuySellRatio > 2 {
		signals = append(signals, TradingSignal{
			Type:      "buy",
			Strength:  80,
			Reason:    "Volume surge with strong buy pressure",
			Timestamp: now,
		})
	} else if metrics.VolumeVelocity > 50 && metrics.BuySellRatio < 0.5 {
		signals = append(signals, TradingSignal{
			Type:      "sell",
			Strength:  80,
			Reason:    "Volume surge with strong sell pressure",
			Timestamp: now,
		})
	}

	// Liquidation Cascade Signal
	if metrics.ShortLiquidations1m > 1000000 { // Большие ликвидации коротких позиций
		signals = append(signals, TradingSignal{
			Type:      "buy",
			Strength:  90,
			Reason:    "Large short liquidation cascade",
			Timestamp: now,
		})
	} else if metrics.LongLiquidations1m > 1000000 { // Большие ликвидации длинных позиций
		signals = append(signals, TradingSignal{
			Type:      "sell",
			Strength:  90,
			Reason:    "Large long liquidation cascade",
			Timestamp: now,
		})
	}

	// Orderbook Imbalance Signal
	imbalanceRatio := metrics.BidPressure / (metrics.AskPressure + 0.0001)
	if imbalanceRatio > 3 && metrics.SpreadPercent < 0.1 {
		signals = append(signals, TradingSignal{
			Type:      "buy",
			Strength:  70,
			Reason:    "Strong bid pressure with tight spread",
			Timestamp: now,
		})
	} else if imbalanceRatio < 0.33 && metrics.SpreadPercent < 0.1 {
		signals = append(signals, TradingSignal{
			Type:      "sell",
			Strength:  70,
			Reason:    "Strong ask pressure with tight spread",
			Timestamp: now,
		})
	}

	// Bollinger Bands Squeeze Signal
	bandWidth := (metrics.BollingerUpper - metrics.BollingerLower) / metrics.BollingerMiddle * 100
	if bandWidth < 2 && metrics.VolumeVelocity > 20 { // Сжатие полос + рост объема
		signals = append(signals, TradingSignal{
			Type:      "breakout",
			Strength:  85,
			Reason:    "Bollinger Bands squeeze with volume expansion",
			Timestamp: now,
		})
	}

	return signals
}

// Вспомогательная функция
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
