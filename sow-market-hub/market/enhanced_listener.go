package market

import (
	"context"
	"fmt"
	"strconv"
	"sync"
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
		eml.wsClient.Start(eml.ctx, executors)
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
		v5SpotPublic.SubscribeTicker(
			bybit.V5WebsocketPublicTickerParamKey{Symbol: bybit.SymbolV5(sym)},
			func(response bybit.V5WebsocketPublicTickerResponse) error {
				eml.handleTickerData(sym, "SPOT", response)
				return nil
			},
		)

		// 2. TRADE подписка
		v5SpotPublic.SubscribeTrade(
			bybit.V5WebsocketPublicTradeParamKey{Symbol: bybit.SymbolV5(sym)},
			func(response bybit.V5WebsocketPublicTradeResponse) error {
				eml.handleTradeData(sym, "SPOT", response)
				return nil
			},
		)

		// 3. ORDERBOOK подписка
		v5SpotPublic.SubscribeOrderBook(
			// bybit.V5WebsocketPublicOrderbookParamKey{
			bybit.V5WebsocketPublicOrderBookParamKey{
				Symbol: bybit.SymbolV5(sym),
				Depth:  50,
			},
			func(response bybit.V5WebsocketPublicOrderBookResponse) error {
				eml.handleOrderbookData(sym, "SPOT", response)
				return nil
			},
		)

		// 4. KLINE подписка (1 минута)
		v5SpotPublic.SubscribeKline(
			bybit.V5WebsocketPublicKlineParamKey{
				Symbol:   bybit.SymbolV5(sym),
				Interval: bybit.Interval1,
			},
			func(response bybit.V5WebsocketPublicKlineResponse) error {
				eml.handleKlineData(sym, "SPOT", response)
				return nil
			},
		)

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

		// Аналогичные подписки для FUTURES
		v5FuturesPublic.SubscribeTicker(
			bybit.V5WebsocketPublicTickerParamKey{Symbol: bybit.SymbolV5(sym)},
			func(response bybit.V5WebsocketPublicTickerResponse) error {
				eml.handleTickerData(sym, "FUTURES", response)
				return nil
			},
		)

		v5FuturesPublic.SubscribeTrade(
			bybit.V5WebsocketPublicTradeParamKey{Symbol: bybit.SymbolV5(sym)},
			func(response bybit.V5WebsocketPublicTradeResponse) error {
				eml.handleTradeData(sym, "FUTURES", response)
				return nil
			},
		)

		v5FuturesPublic.SubscribeOrderBook(
			bybit.V5WebsocketPublicOrderBookParamKey{
				Symbol: bybit.SymbolV5(sym),
				Depth:  50,
			},
			func(response bybit.V5WebsocketPublicOrderBookResponse) error {
				eml.handleOrderbookData(sym, "FUTURES", response)
				return nil
			},
		)

		v5FuturesPublic.SubscribeKline(
			bybit.V5WebsocketPublicKlineParamKey{
				Symbol:   bybit.SymbolV5(sym),
				Interval: bybit.Interval1,
			},
			func(response bybit.V5WebsocketPublicKlineResponse) error {
				eml.handleKlineData(sym, "FUTURES", response)
				return nil
			},
		)

		// 5. LIQUIDATION подписка (только для FUTURES)
		v5FuturesPublic.SubscribeLiquidation(
			bybit.V5WebsocketPublicLiquidationParamKey{Symbol: bybit.SymbolV5(sym)},
			func(response bybit.V5WebsocketPublicLiquidationResponse) error {
				eml.handleLiquidationData(sym, response)
				return nil
			},
		)

		eml.logger.WithField("symbol", sym).Info("✓ Enhanced FUTURES подписки активированы")
	}

	*executors = append(*executors, v5FuturesPublic)
	return nil
}

// Обработчики данных
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
			data.Change24h = change * 100
		}
	} else if market == "FUTURES" && response.Data.LinearInverse.LastPrice != "" {
		if price, err := parseFloat(response.Data.LinearInverse.LastPrice); err == nil {
			data.Price = price
		}
		if vol, err := parseFloat(response.Data.LinearInverse.Volume24h); err == nil {
			data.Volume24h = vol
		}
		if change, err := parseFloat(response.Data.LinearInverse.Price24hPercent); err == nil {
			data.Change24h = change * 100
		}
	}

	data.Timestamp = time.Now()

	// Запускаем аналитику
	eml.analytics.UpdateData(key, data)

	// Уведомляем обработчики
	eml.notifyHandlers(symbol, data)
}

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
		if price, err := parseFloat(trade.Trade); err == nil {
			if size, err := parseFloat(trade.Value); err == nil {
				tradeData := TradeData{
					ID:        trade.ID,
					Price:     price,
					Size:      size,
					Side:      string(trade.Side),
					Timestamp: time.Unix(int64(trade.Timestamp/1000), 0), // Check
				}

				// Добавляем в историю сделок
				data.TradeHistory = append(data.TradeHistory, tradeData)
				if len(data.TradeHistory) > 1000 {
					data.TradeHistory = data.TradeHistory[1:]
				}

				// Обновляем последнюю сделку
				data.LastTrade = &tradeData
				data.TradeCount++
			}
		}
	}

	data.Timestamp = time.Now()
	eml.analytics.UpdateData(key, data)
	eml.notifyHandlers(symbol, data)
}

func (eml *EnhancedMarketListener) handleOrderbookData(symbol, market string, response bybit.V5WebsocketPublicOrderBookResponse) {
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

	// Парсим bids
	for _, bid := range response.Data.Bids {
		if len(bid.Size) >= 2 {
			if price, err := parseFloat(bid.Price); err == nil {
				if size, err := parseFloat(bid.Size); err == nil {
					orderbook.Bids = append(orderbook.Bids, PriceLevel{
						Price: price,
						Size:  size,
					})
				}
			}
		}
	}

	// Парсим asks
	for _, ask := range response.Data.Asks {
		if len(ask.Size) >= 2 {
			if price, err := parseFloat(ask.Price); err == nil {
				if size, err := parseFloat(ask.Size); err == nil {
					orderbook.Asks = append(orderbook.Asks, PriceLevel{
						Price: price,
						Size:  size,
					})
				}
			}
		}
	}

	// Вычисляем спред
	if len(orderbook.Bids) > 0 && len(orderbook.Asks) > 0 {
		spread := orderbook.Asks[0].Price - orderbook.Bids[0].Price
		data.Spread = spread
	}

	data.Orderbook = orderbook
	data.Timestamp = time.Now()

	eml.analytics.UpdateData(key, data)
	eml.notifyHandlers(symbol, data)
}

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
								Interval:  string(kline.Interval),
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
	eml.analytics.UpdateData(key, data)
	eml.notifyHandlers(symbol, data)
}

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

	// Обрабатываем ликвидации
	liq := response.Data
	if price, err := parseFloat(liq.Price); err == nil {
		if size, err := parseFloat(liq.Size); err == nil {
			liquidation := LiquidationData{
				Price:     price,
				Size:      size,
				Side:      string(liq.Side),
				Timestamp: time.Unix(int64(liq.UpdatedTime/1000), 0),
			}

			// Добавляем в историю ликвидаций
			data.Liquidations = append(data.Liquidations, liquidation)
			if len(data.Liquidations) > 100 {
				data.Liquidations = data.Liquidations[1:]
			}
		}
	}
	// Removed the extra closing brace

	data.Timestamp = time.Now()
	eml.analytics.UpdateData(key, data)
	eml.notifyHandlers(symbol, data)
}

func (eml *EnhancedMarketListener) runPeriodicAnalytics() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-eml.ctx.Done():
			return
		case <-ticker.C:
			eml.calculateAnalytics()
		}
	}
}

func (eml *EnhancedMarketListener) calculateAnalytics() {
	eml.dataMux.RLock()
	defer eml.dataMux.RUnlock()

	for key, data := range eml.marketData {
		if analytics := eml.analytics.CalculateMetrics(key); analytics != nil {
			data.Analytics = analytics
			eml.notifyHandlers(data.Symbol, data)
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

	// Останавливаем обработчики
	for _, handler := range eml.dataHandlers {
		handler.Stop()
	}

	// Останавливаем аналитику
	eml.analytics.Stop()

	eml.cancel()
	eml.isRunning = false

	// Ждем завершения
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

// Вспомогательные функции
func parseFloat(s string) (float64, error) {
	// Реализация парсинга строки в float64
	// Можно использовать strconv.ParseFloat(s, 64)
	return strconv.ParseFloat(s, 64)
}
