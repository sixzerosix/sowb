package market

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"strconv"
	"sync"
	"syscall"
	"time"

	"github.com/hirokisan/bybit/v2"
	"github.com/sirupsen/logrus"
)

// EnhancedMarketListener - расширенный слушатель рынка
type EnhancedMarketListener struct {
	config    *Config
	logger    *logrus.Logger
	wsClient  *bybit.WebSocketClient
	ctx       context.Context
	cancel    context.CancelFunc
	wg        sync.WaitGroup
	isRunning bool
	mu        sync.RWMutex

	// Хранилище данных
	marketData map[string]*EnhancedMarketData
	dataMux    sync.RWMutex

	// Аналитика
	analytics *RealTimeAnalytics

	// Обработчики данных
	dataHandlers []EnhancedMarketDataHandler
}

// EnhancedMarketDataHandler - интерфейс для обработки расширенных данных
type EnhancedMarketDataHandler interface {
	OnMarketDataUpdate(symbol string, data *EnhancedMarketData)
	Start()
	Stop()
}

func NewEnhancedMarketListener(config *Config) *EnhancedMarketListener {
	logger := logrus.New()
	logger.SetFormatter(&logrus.JSONFormatter{})

	level, err := logrus.ParseLevel(config.LogLevel)
	if err != nil {
		level = logrus.InfoLevel
	}
	logger.SetLevel(level)

	ctx, cancel := context.WithCancel(context.Background())

	return &EnhancedMarketListener{
		config:     config,
		logger:     logger,
		wsClient:   bybit.NewWebsocketClient(),
		ctx:        ctx,
		cancel:     cancel,
		marketData: make(map[string]*EnhancedMarketData),
		analytics:  NewRealTimeAnalytics(),
	}
}

func (eml *EnhancedMarketListener) AddDataHandler(handler EnhancedMarketDataHandler) {
	eml.dataHandlers = append(eml.dataHandlers, handler)
}

func (eml *EnhancedMarketListener) Start() error {
	eml.mu.Lock()
	if eml.isRunning {
		eml.mu.Unlock()
		return fmt.Errorf("listener already running")
	}
	eml.isRunning = true
	eml.mu.Unlock()

	eml.logger.Info("🚀 Запуск Enhanced Market Listener...")

	// Запускаем аналитику
	eml.analytics.Start()

	// Запускаем обработчики данных
	for _, handler := range eml.dataHandlers {
		handler.Start()
	}

	// Graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-sigChan
		eml.logger.Info("📡 Получен сигнал завершения...")
		eml.Stop()
	}()

	return eml.connectAndListen()
}

func (eml *EnhancedMarketListener) connectAndListen() error {
	var executors []bybit.WebsocketExecutor

	// Подключаемся к SPOT рынку
	if err := eml.setupSpotSubscriptions(&executors); err != nil {
		return fmt.Errorf("ошибка настройки SPOT: %w", err)
	}

	// Подключаемся к FUTURES рынку
	if err := eml.setupFuturesSubscriptions(&executors); err != nil {
		return fmt.Errorf("ошибка настройки FUTURES: %w", err)
	}

	eml.logger.Info("📊 Enhanced мониторинг активирован")

	// Запуск WebSocket клиента
	eml.wg.Add(1)
	go func() {
		defer eml.wg.Done()
		eml.logger.Debug("Starting Bybit WebSocket client...") // <-- ДОБАВЛЕНО
		eml.wsClient.Start(eml.ctx, executors)
		eml.logger.Debug("Bybit WebSocket client stopped.") // <-- ДОБАВЛЕНО
	}()

	// Запуск периодической аналитики
	eml.wg.Add(1)
	go func() {
		defer eml.wg.Done()
		eml.runPeriodicAnalytics()
	}()

	eml.wg.Wait()
	return nil
}

func (eml *EnhancedMarketListener) setupSpotSubscriptions(executors *[]bybit.WebsocketExecutor) error {
	eml.logger.Info("📍 Подключение к V5 API SPOT (Enhanced)...")

	v5SpotPublic, err := eml.wsClient.V5().Public(bybit.CategoryV5Spot)
	if err != nil {
		return err
	}

	for _, symbol := range eml.config.Symbols {
		sym := symbol

		// 1. TICKER подписка
		_, err = v5SpotPublic.SubscribeTicker(
			bybit.V5WebsocketPublicTickerParamKey{Symbol: bybit.SymbolV5(sym)},
			func(response bybit.V5WebsocketPublicTickerResponse) error {
				eml.handleTickerData(sym, "SPOT", response)
				return nil
			},
		)
		if err != nil {
			eml.logger.WithError(err).WithField("symbol", sym).Error("❌ Ошибка подписки на SPOT Ticker")
		}

		// 2. TRADE подписка
		_, err = v5SpotPublic.SubscribeTrade(
			bybit.V5WebsocketPublicTradeParamKey{Symbol: bybit.SymbolV5(sym)},
			func(response bybit.V5WebsocketPublicTradeResponse) error {
				eml.handleTradeData(sym, "SPOT", response)
				return nil
			},
		)
		if err != nil {
			eml.logger.WithError(err).WithField("symbol", sym).Error("❌ Ошибка подписки на SPOT Trade")
		}

		// 3. ORDERBOOK подписка (ИСПРАВЛЕНО ИМЯ ТИПА)
		_, err = v5SpotPublic.SubscribeOrderBook(
			bybit.V5WebsocketPublicOrderBookParamKey{ // <--- ИСПРАВЛЕНО
				Symbol: bybit.SymbolV5(sym),
				Depth:  50,
			},
			func(response bybit.V5WebsocketPublicOrderBookResponse) error { // <--- ИСПРАВЛЕНО
				eml.handleOrderbookData(sym, "SPOT", response)
				return nil
			},
		)
		if err != nil {
			eml.logger.WithError(err).WithField("symbol", sym).Error("❌ Ошибка подписки на SPOT Orderbook")
		}

		// 4. KLINE подписка (1 минута) (ИСПРАВЛЕНО ИМЯ ИНТЕРВАЛА)
		_, err = v5SpotPublic.SubscribeKline(
			bybit.V5WebsocketPublicKlineParamKey{
				Symbol:   bybit.SymbolV5(sym),
				Interval: bybit.SpotInterval1M, // <--- ИСПРАВЛЕНО
			},
			func(response bybit.V5WebsocketPublicKlineResponse) error {
				eml.handleKlineData(sym, "SPOT", response)
				return nil
			},
		)
		if err != nil {
			eml.logger.WithError(err).WithField("symbol", sym).Error("❌ Ошибка подписки на SPOT Kline")
		}

		eml.logger.WithField("symbol", sym).Info("✓ Enhanced SPOT подписки активированы")
	}

	*executors = append(*executors, v5SpotPublic)
	return nil
}

func (eml *EnhancedMarketListener) setupFuturesSubscriptions(executors *[]bybit.WebsocketExecutor) error {
	eml.logger.Info("📍 Подключение к V5 API FUTURES (Enhanced)...")

	v5FuturesPublic, err := eml.wsClient.V5().Public(bybit.CategoryV5Linear)
	if err != nil {
		return err
	}

	for _, symbol := range eml.config.Symbols {
		sym := symbol

		_, err = v5FuturesPublic.SubscribeTicker(
			bybit.V5WebsocketPublicTickerParamKey{Symbol: bybit.SymbolV5(sym)},
			func(response bybit.V5WebsocketPublicTickerResponse) error {
				eml.handleTickerData(sym, "FUTURES", response)
				return nil
			},
		)
		if err != nil {
			eml.logger.WithError(err).WithField("symbol", sym).Error("❌ Ошибка подписки на FUTURES Ticker")
		}

		_, err = v5FuturesPublic.SubscribeTrade(
			bybit.V5WebsocketPublicTradeParamKey{Symbol: bybit.SymbolV5(sym)},
			func(response bybit.V5WebsocketPublicTradeResponse) error {
				eml.handleTradeData(sym, "FUTURES", response)
				return nil
			},
		)
		if err != nil {
			eml.logger.WithError(err).WithField("symbol", sym).Error("❌ Ошибка подписки на FUTURES Trade")
		}

		_, err = v5FuturesPublic.SubscribeOrderBook(
			bybit.V5WebsocketPublicOrderBookParamKey{ // <--- ИСПРАВЛЕНО
				Symbol: bybit.SymbolV5(sym),
				Depth:  50,
			},
			func(response bybit.V5WebsocketPublicOrderBookResponse) error { // <--- ИСПРАВЛЕНО
				eml.handleOrderbookData(sym, "FUTURES", response)
				return nil
			},
		)
		if err != nil {
			eml.logger.WithError(err).WithField("symbol", sym).Error("❌ Ошибка подписки на FUTURES Orderbook")
		}

		_, err = v5FuturesPublic.SubscribeKline(
			bybit.V5WebsocketPublicKlineParamKey{
				Symbol:   bybit.SymbolV5(sym),
				Interval: bybit.Interval1, // <--- ИСПРАВЛЕНО
			},
			func(response bybit.V5WebsocketPublicKlineResponse) error {
				eml.handleKlineData(sym, "FUTURES", response)
				return nil
			},
		)
		if err != nil {
			eml.logger.WithError(err).WithField("symbol", sym).Error("❌ Ошибка подписки на FUTURES Kline")
		}

		// 5. LIQUIDATION подписка (только для FUTURES)
		_, err = v5FuturesPublic.SubscribeLiquidation(
			bybit.V5WebsocketPublicLiquidationParamKey{Symbol: bybit.SymbolV5(sym)},
			func(response bybit.V5WebsocketPublicLiquidationResponse) error {
				eml.handleLiquidationData(sym, response)
				return nil
			},
		)
		if err != nil {
			eml.logger.WithError(err).WithField("symbol", sym).Error("❌ Ошибка подписки на FUTURES Liquidation")
		}

		eml.logger.WithField("symbol", sym).Info("✓ Enhanced FUTURES подписки активированы")
	}

	*executors = append(*executors, v5FuturesPublic)
	return nil
}

// ... (handleTickerData)
func (eml *EnhancedMarketListener) handleTickerData(symbol, market string, response bybit.V5WebsocketPublicTickerResponse) {
	eml.dataMux.Lock()
	defer eml.dataMux.Unlock()

	key := fmt.Sprintf("%s_%s", symbol, market)
	if eml.marketData[key] == nil {
		eml.marketData[key] = &EnhancedMarketData{
			Symbol:       symbol,
			Market:       market,
			TradeHistory: make([]TradeData, 0, 1000),
			KlineHistory: make([]KlineData, 0, 100),
			Liquidations: make([]LiquidationData, 0, 100),
		}
	}

	data := eml.marketData[key]

	// Обновляем ticker данные
	if market == "SPOT" && response.Data.Spot.LastPrice != "" {
		if price, err := parseFloat(response.Data.Spot.LastPrice); err == nil {
			data.Price = price
		}
		if vol, err := parseFloat(response.Data.Spot.Volume24H); err == nil {
			data.Volume24h = vol
		}
		if change, err := parseFloat(response.Data.Spot.Price24HPcnt); err == nil {
			data.Change24h = change // <--- ИСПРАВЛЕНО (убрано * 100)
		}
		if high, err := parseFloat(response.Data.Spot.HighPrice24H); err == nil { // <--- ИСПРАВЛЕНО (HighPrice24h)
			data.High24h = high
		}
		if low, err := parseFloat(response.Data.Spot.LowPrice24H); err == nil { // <--- ИСПРАВЛЕНО (LowPrice24h)
			data.Low24h = low
		}
	} else if market == "FUTURES" && response.Data.LinearInverse.LastPrice != "" {
		if price, err := parseFloat(response.Data.LinearInverse.LastPrice); err == nil {
			data.Price = price
		}
		if vol, err := parseFloat(response.Data.LinearInverse.Volume24h); err == nil {
			data.Volume24h = vol
		}
		if change, err := parseFloat(response.Data.LinearInverse.Price24hPercent); err == nil { // <--- ИСПРАВЛЕНО (Price24hPcnt)
			data.Change24h = change // <--- ИСПРАВЛЕНО (убрано * 100)
		}
		if high, err := parseFloat(response.Data.LinearInverse.HighPrice24h); err == nil {
			data.High24h = high
		}
		if low, err := parseFloat(response.Data.LinearInverse.LowPrice24h); err == nil {
			data.Low24h = low
		}
	}

	data.Timestamp = time.Now()
	eml.logger.Debugf("Processing ticker data for %s_%s: price %.2f, vol %.2f, change %.2f", symbol, market, data.Price, data.Volume24h, data.Change24h) // <-- ДОБАВЛЕНО

	// Обновляем аналитику
	eml.analytics.UpdateData(key, data)

	// Уведомляем обработчики
	eml.notifyHandlers(symbol, data)
}

// ... (handleTradeData)
func (eml *EnhancedMarketListener) handleTradeData(symbol, market string, response bybit.V5WebsocketPublicTradeResponse) {
	eml.dataMux.Lock()
	defer eml.dataMux.Unlock()

	key := fmt.Sprintf("%s_%s", symbol, market)
	if eml.marketData[key] == nil {
		eml.marketData[key] = &EnhancedMarketData{
			Symbol:       symbol,
			Market:       market,
			TradeHistory: make([]TradeData, 0, 1000),
			KlineHistory: make([]KlineData, 0, 100),
			Liquidations: make([]LiquidationData, 0, 100),
		}
	}

	data := eml.marketData[key]

	// Обрабатываем каждую сделку
	for _, trade := range response.Data {
		if price, err := parseFloat(trade.Trade); err == nil { // <--- ИСПРАВЛЕНО (trade.Price)
			if size, err := parseFloat(trade.Value); err == nil { // <--- ИСПРАВЛЕНО (trade.Size)
				tradeData := TradeData{
					ID:        trade.ID, // <--- ИСПРАВЛЕНО (trade.ExecId)
					Price:     price,
					Size:      size,
					Side:      string(trade.Side), // <--- trade.Side уже string
					Timestamp: time.Unix(int64(trade.Timestamp)/1000, 0),
				}

				// Добавляем в историю сделок
				data.TradeHistory = append(data.TradeHistory, tradeData)
				if len(data.TradeHistory) > 1000 {
					data.TradeHistory = data.TradeHistory[1:]
				}

				// Обновляем последнюю сделку
				data.LastTrade = &tradeData
				data.TradeCount++

				// Обновляем OrderFlowDelta
				if indicators := eml.analytics.indicators[key]; indicators != nil && indicators.OrderFlowDelta != nil {
					indicators.OrderFlowDelta.UpdateTrade(tradeData.Side, tradeData.Size)
				}
				eml.logger.Debugf("Processing trade data for %s_%s: price %.2f, size %.2f, side %s", symbol, market, tradeData.Price, tradeData.Size, tradeData.Side) // <-- ДОБАВЛЕНО
			}
		}
	}

	data.Timestamp = time.Now()
	eml.analytics.UpdateData(key, data)
	eml.notifyHandlers(symbol, data)
}

// ... (handleOrderbookData)
func (eml *EnhancedMarketListener) handleOrderbookData(symbol, market string, response bybit.V5WebsocketPublicOrderBookResponse) { // <--- ИСПРАВЛЕНО ИМЯ ТИПА ОТВЕТА
	eml.dataMux.Lock()
	defer eml.dataMux.Unlock()

	key := fmt.Sprintf("%s_%s", symbol, market)
	if eml.marketData[key] == nil {
		eml.marketData[key] = &EnhancedMarketData{
			Symbol:       symbol,
			Market:       market,
			TradeHistory: make([]TradeData, 0, 1000),
			KlineHistory: make([]KlineData, 0, 100),
			Liquidations: make([]LiquidationData, 0, 100),
		}
	}

	data := eml.marketData[key]

	// Обновляем orderbook
	orderbook := &OrderbookData{
		Bids:      make([]PriceLevel, 0, len(response.Data.Bids)),
		Asks:      make([]PriceLevel, 0, len(response.Data.Asks)),
		Timestamp: time.Now(),
	}

	// Парсим bids (ИСПРАВЛЕНО: bid.Price и bid.Size - это поля структуры PriceLevelV5, а не массив)
	for _, bid := range response.Data.Bids {
		if price, err := parseFloat(bid.Price); err == nil {
			if size, err := parseFloat(bid.Size); err == nil {
				orderbook.Bids = append(orderbook.Bids, PriceLevel{
					Price: price,
					Size:  size,
				})
			}
		}
	}

	// Парсим asks (ИСПРАВЛЕНО)
	for _, ask := range response.Data.Asks {
		if price, err := parseFloat(ask.Price); err == nil {
			if size, err := parseFloat(ask.Size); err == nil {
				orderbook.Asks = append(orderbook.Asks, PriceLevel{
					Price: price,
					Size:  size,
				})
			}
		}
	}

	// Вычисляем спред (безопасно — проверяем длины слайсов)
	if len(orderbook.Bids) > 0 && len(orderbook.Asks) > 0 {
		spread := orderbook.Asks[0].Price - orderbook.Bids[0].Price
		data.Spread = spread
	} else {
		data.Spread = 0
	}

	data.Orderbook = orderbook
	data.Timestamp = time.Now()
	// безопасный лог — проверяем длину слайсов перед доступом по [0]
	bestBid, bestAsk := 0.0, 0.0
	if len(orderbook.Bids) > 0 {
		bestBid = orderbook.Bids[0].Price
	}
	if len(orderbook.Asks) > 0 {
		bestAsk = orderbook.Asks[0].Price
	}
	eml.logger.Debugf("Processing orderbook data for %s_%s: best bid %.2f, best ask %.2f, spread %.4f", symbol, market, bestBid, bestAsk, data.Spread)

	eml.analytics.UpdateData(key, data)
	eml.notifyHandlers(symbol, data)
}

// ... (handleKlineData)
func (eml *EnhancedMarketListener) handleKlineData(symbol, market string, response bybit.V5WebsocketPublicKlineResponse) {
	eml.dataMux.Lock()
	defer eml.dataMux.Unlock()

	key := fmt.Sprintf("%s_%s", symbol, market)
	if eml.marketData[key] == nil {
		eml.marketData[key] = &EnhancedMarketData{
			Symbol:       symbol,
			Market:       market,
			TradeHistory: make([]TradeData, 0, 1000),
			KlineHistory: make([]KlineData, 0, 100),
			Liquidations: make([]LiquidationData, 0, 100),
		}
	}

	data := eml.marketData[key]

	// Обрабатываем kline данные
	for _, kline := range response.Data {
		if open, err := parseFloat(kline.Open); err == nil {
			if high, err := parseFloat(kline.High); err == nil {
				if low, err := parseFloat(kline.Low); err == nil {
					if close, err := parseFloat(kline.Close); err == nil {
						if volume, err := parseFloat(kline.Volume); err == nil {
							klineData := KlineData{
								Open:      open,
								High:      high,
								Low:       low,
								Close:     close,
								Volume:    volume,
								Timestamp: time.Unix(kline.Start/1000, 0),
								Interval:  string(kline.Interval), // <--- ИСПРАВЛЕНО (kline.Interval уже string)
							}

							// Добавляем в историю
							data.KlineHistory = append(data.KlineHistory, klineData)
							if len(data.KlineHistory) > 100 {
								data.KlineHistory = data.KlineHistory[1:]
							}

							// Обновляем текущую свечу
							data.Kline = &klineData
						}
					}
				}
			}
		}
	}

	data.Timestamp = time.Now()
	// безопасный лог по Kline (может быть nil)
	kclose, kvolume := 0.0, 0.0
	if data.Kline != nil {
		kclose = data.Kline.Close
		kvolume = data.Kline.Volume
	}
	eml.logger.Debugf("Processing kline data for %s_%s: close %.2f, volume %.2f", symbol, market, kclose, kvolume)
	eml.analytics.UpdateData(key, data)
	eml.notifyHandlers(symbol, data)
}

// ... (handleLiquidationData)
func (eml *EnhancedMarketListener) handleLiquidationData(symbol string, response bybit.V5WebsocketPublicLiquidationResponse) {
	eml.dataMux.Lock()
	defer eml.dataMux.Unlock()

	key := fmt.Sprintf("%s_FUTURES", symbol)
	if eml.marketData[key] == nil {
		eml.marketData[key] = &EnhancedMarketData{
			Symbol:       symbol,
			Market:       "FUTURES",
			TradeHistory: make([]TradeData, 0, 1000),
			KlineHistory: make([]KlineData, 0, 100),
			Liquidations: make([]LiquidationData, 0, 100),
		}
	}

	data := eml.marketData[key]

	// Обрабатываем ликвидации (ИСПРАВЛЕНО: response.Data - это один объект)
	liq := response.Data
	if price, err := parseFloat(liq.Price); err == nil {
		if size, err := parseFloat(liq.Size); err == nil {
			liquidation := LiquidationData{
				Price:     price,
				Size:      size,
				Side:      string(liq.Side), // <--- liq.Side уже string
				Timestamp: time.Unix(int64(liq.UpdatedTime)/1000, 0),
			}

			// Добавляем в историю ликвидаций
			data.Liquidations = append(data.Liquidations, liquidation)
			if len(data.Liquidations) > 100 {
				data.Liquidations = data.Liquidations[1:]
			}
		}
	}
	// Убрана лишняя закрывающая скобка, которая была в предыдущей версии
	lastPrice := 0.0
	lastSize := 0.0
	lastSide := ""
	if len(data.Liquidations) > 0 {
		last := data.Liquidations[len(data.Liquidations)-1]
		lastPrice = last.Price
		lastSize = last.Size
		lastSide = last.Side
	}
	eml.logger.Debugf("Processing liquidation data for %s: price %.2f, size %.2f, side %s", symbol, lastPrice, lastSize, lastSide)

	data.Timestamp = time.Now()
	eml.analytics.UpdateData(key, data)
	eml.notifyHandlers(symbol, data)
}

func (eml *EnhancedMarketListener) runPeriodicAnalytics() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-eml.ctx.Done():
			return
		case <-ticker.C:
			eml.logger.Debug("📊 Triggering analytics calculation.") // <-- ДОБАВЛЕНО
			eml.calculateAnalytics()
		}
	}
}

func (eml *EnhancedMarketListener) calculateAnalytics() {
	eml.dataMux.RLock()
	defer eml.dataMux.RUnlock()

	for key, data := range eml.marketData {
		eml.logger.Debugf("Attempting to calculate analytics for %s. Price history length: %d, Trade history length: %d, Kline history length: %d",
			key, len(eml.analytics.priceHistory[key]), len(data.TradeHistory), len(data.KlineHistory)) // <-- ДОБАВЛЕНО

		if analytics := eml.analytics.CalculateMetrics(key); analytics != nil {
			data.Analytics = analytics
			eml.logger.Debugf("✅ Calculated analytics for %s: RSI=%.2f, Sentiment=%s, Signals count: %d", key, analytics.RSI14, analytics.MarketSentiment, len(analytics.Signals)) // <-- ДОБАВЛЕНО
			eml.notifyHandlers(data.Symbol, data)
		} else {
			eml.logger.Debugf("❌ Analytics not ready or failed for %s (not enough data or nil result).", key) // <-- ДОБАВЛЕНО
		}
	}
}

func (eml *EnhancedMarketListener) notifyHandlers(symbol string, data *EnhancedMarketData) {
	for _, handler := range eml.dataHandlers {
		go handler.OnMarketDataUpdate(symbol, data)
	}
}

func (eml *EnhancedMarketListener) Stop() {
	eml.mu.Lock()
	defer eml.mu.Unlock()

	if !eml.isRunning {
		return
	}

	eml.logger.Info("🛑 Завершение Enhanced Market Listener...")

	for _, handler := range eml.dataHandlers {
		handler.Stop()
	}

	eml.analytics.Stop()

	eml.cancel()
	eml.isRunning = false

	done := make(chan struct{})
	go func() {
		eml.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		eml.logger.Info("✅ Enhanced Market Listener корректно завершен")
	case <-time.After(10 * time.Second):
		eml.logger.Warn("⚠️ Таймаут при завершении Enhanced Market Listener")
	}
}

func parseFloat(s string) (float64, error) {
	return strconv.ParseFloat(s, 64)
}
