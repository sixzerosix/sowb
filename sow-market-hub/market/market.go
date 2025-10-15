package market

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/hirokisan/bybit/v2"
	"github.com/sirupsen/logrus"
)

// Config структура для конфигурации
type Config struct {
	Symbols          []string `json:"symbols"`
	ReconnectTimeout int      `json:"reconnect_timeout_seconds"`
	MaxRetries       int      `json:"max_retries"`
	LogLevel         string   `json:"log_level"`
}

// MarketDataHandler интерфейс для обработки рыночных данных
type MarketDataHandler interface {
	OnSpotPriceUpdate(symbol, price string)
	OnFuturesPriceUpdate(symbol, price string)
	Start()
	Stop()
}

// MarketListener основная структура
type MarketListener struct {
	config      *Config
	logger      *logrus.Logger
	wsClient    *bybit.WebSocketClient
	ctx         context.Context
	cancel      context.CancelFunc
	wg          sync.WaitGroup
	isRunning   bool
	mu          sync.RWMutex
	dataHandler MarketDataHandler // Добавляем handler для данных
}

// NewMarketListener создает новый экземпляр
func NewMarketListener() *MarketListener {
	logger := logrus.New()
	logger.SetFormatter(&logrus.JSONFormatter{})

	// Загружаем конфиг или используем дефолтный
	config := LoadConfig()

	// Устанавливаем уровень логирования
	level, err := logrus.ParseLevel(config.LogLevel)
	if err != nil {
		level = logrus.InfoLevel
	}

	// ОТКЛЮЧАЕМ ВЫВОД ЛОГОВ В КОНСОЛЬ ДЛЯ ДАШБОРДА
	logger.SetOutput(os.NewFile(0, os.DevNull)) // Отправляем логи в никуда
	logger.SetLevel(level)

	ctx, cancel := context.WithCancel(context.Background())

	return &MarketListener{
		config:   config,
		logger:   logger,
		wsClient: bybit.NewWebsocketClient(),
		ctx:      ctx,
		cancel:   cancel,
	}
}

// SetDataHandler устанавливает обработчик данных
func (ml *MarketListener) SetDataHandler(handler MarketDataHandler) {
	ml.dataHandler = handler
}

// LoadConfig загружает конфигурацию
func LoadConfig() *Config {
	// Пытаемся загрузить из файла
	if data, err := os.ReadFile("config.json"); err == nil {
		var config Config
		if json.Unmarshal(data, &config) == nil {
			return &config
		}
	}

	// Дефолтная конфигурация
	return &Config{
		Symbols: []string{
			"BTCUSDT",
			"ETHUSDT",
			"BNBUSDT",
			"SOLUSDT",
		},
		ReconnectTimeout: 5,
		MaxRetries:       10,
		LogLevel:         "info",
	}
}

// Start запускает мониторинг
func (ml *MarketListener) Start() error {
	ml.mu.Lock()
	if ml.isRunning {
		ml.mu.Unlock()
		return fmt.Errorf("listener already running")
	}
	ml.isRunning = true
	ml.mu.Unlock()

	ml.logger.Info("🚀 Запуск Market Listener...")

	// Запускаем data handler если есть
	if ml.dataHandler != nil {
		ml.dataHandler.Start()
	}

	// Graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-sigChan
		ml.logger.Info("📡 Получен сигнал завершения...")
		ml.Stop()
	}()

	// Запускаем с retry логикой
	return ml.startWithRetry()
}

// startWithRetry запуск с повторными попытками
func (ml *MarketListener) startWithRetry() error {
	retries := 0

	for retries < ml.config.MaxRetries {
		select {
		case <-ml.ctx.Done():
			return nil
		default:
		}

		if retries > 0 {
			ml.logger.WithField("retry", retries).Warn("🔄 Повторная попытка подключения...")
			time.Sleep(time.Duration(ml.config.ReconnectTimeout) * time.Second)
		}

		if err := ml.connectAndListen(); err != nil {
			retries++
			ml.logger.WithError(err).WithField("retry", retries).Error("❌ Ошибка подключения")
			continue
		}

		// Успешное подключение - сбрасываем счетчик
		retries = 0
	}

	return fmt.Errorf("превышено максимальное количество попыток подключения")
}

// connectAndListen основная логика подключения
func (ml *MarketListener) connectAndListen() error {
	var executors []bybit.WebsocketExecutor

	// Подключение к СПОТ
	if err := ml.setupSpotConnection(&executors); err != nil {
		return fmt.Errorf("ошибка настройки СПОТ: %w", err)
	}

	// Подключение к ФЬЮЧЕРСАМ
	if err := ml.setupFuturesConnection(&executors); err != nil {
		return fmt.Errorf("ошибка настройки ФЬЮЧЕРСОВ: %w", err)
	}

	ml.logger.Info("📊 Мониторинг активирован (СПОТ + ФЬЮЧЕРСЫ)")

	// Запуск WebSocket клиента
	ml.wg.Add(1)
	go func() {
		defer ml.wg.Done()
		ml.wsClient.Start(ml.ctx, executors)
	}()

	// Ожидание завершения
	ml.wg.Wait()
	return nil
}

// setupSpotConnection настройка подключения к споту
func (ml *MarketListener) setupSpotConnection(executors *[]bybit.WebsocketExecutor) error {
	ml.logger.Info("📍 Подключение к V5 API СПОТ...")

	v5SpotPublic, err := ml.wsClient.V5().Public(bybit.CategoryV5Spot)
	if err != nil {
		return err
	}

	successCount := 0
	for _, symbol := range ml.config.Symbols {
		sym := symbol

		_, err := v5SpotPublic.SubscribeTicker(
			bybit.V5WebsocketPublicTickerParamKey{
				Symbol: bybit.SymbolV5(sym),
			},
			func(response bybit.V5WebsocketPublicTickerResponse) error {
				ml.handleSpotTicker(sym, response)
				return nil
			},
		)

		if err != nil {
			ml.logger.WithError(err).WithField("symbol", sym).Error("❌ Ошибка подписки на СПОТ тикер")
			continue
		}

		ml.logger.WithField("symbol", sym).Info("✓ Подписаны на СПОТ тикер")
		successCount++
	}

	if successCount == 0 {
		return fmt.Errorf("не удалось подписаться ни на один СПОТ символ")
	}

	*executors = append(*executors, v5SpotPublic)
	return nil
}

// setupFuturesConnection настройка подключения к фьючерсам
func (ml *MarketListener) setupFuturesConnection(executors *[]bybit.WebsocketExecutor) error {
	ml.logger.Info("📍 Подключение к V5 API ДЕРИВАТИВЫ...")

	v5LinearPublic, err := ml.wsClient.V5().Public(bybit.CategoryV5Linear)
	if err != nil {
		return err
	}

	successCount := 0
	for _, symbol := range ml.config.Symbols {
		sym := symbol

		_, err := v5LinearPublic.SubscribeTicker(
			bybit.V5WebsocketPublicTickerParamKey{
				Symbol: bybit.SymbolV5(sym),
			},
			func(response bybit.V5WebsocketPublicTickerResponse) error {
				ml.handleFuturesTicker(sym, response)
				return nil
			},
		)

		if err != nil {
			ml.logger.WithError(err).WithField("symbol", sym).Error("❌ Ошибка подписки на FUTURES тикер")
			continue
		}

		ml.logger.WithField("symbol", sym).Info("✓ Подписаны на FUTURES тикер")
		successCount++
	}

	if successCount == 0 {
		return fmt.Errorf("не удалось подписаться ни на один FUTURES символ")
	}

	*executors = append(*executors, v5LinearPublic)
	return nil
}

// handleSpotTicker обработка данных спота
func (ml *MarketListener) handleSpotTicker(symbol string, response bybit.V5WebsocketPublicTickerResponse) {
	price := response.Data.Spot.LastPrice
	if price == "" {
		ml.logger.WithField("symbol", symbol).Debug("Пустая цена в СПОТ тикере")
		return
	}

	ml.logger.WithFields(logrus.Fields{
		"market": "SPOT",
		"symbol": symbol,
		"price":  price,
	}).Info("💰 Обновление цены")

	// Отправляем данные в handler если есть, иначе выводим в консоль
	if ml.dataHandler != nil {
		ml.dataHandler.OnSpotPriceUpdate(symbol, price)
	} else {
		fmt.Printf("SPOT:%s - Цена: %s\n", symbol, price)
	}
}

// handleFuturesTicker обработка данных фьючерсов
func (ml *MarketListener) handleFuturesTicker(symbol string, response bybit.V5WebsocketPublicTickerResponse) {
	price := response.Data.LinearInverse.LastPrice
	if price == "" {
		ml.logger.WithField("symbol", symbol).Debug("Пустая цена в FUTURES тикере")
		return
	}

	ml.logger.WithFields(logrus.Fields{
		"market": "FUTURES",
		"symbol": symbol,
		"price":  price,
	}).Info("💰 Обновление цены")

	// Отправляем данные в handler если есть, иначе выводим в консоль
	if ml.dataHandler != nil {
		ml.dataHandler.OnFuturesPriceUpdate(symbol, price)
	} else {
		fmt.Printf("FUTURES:%s - Цена: %s\n", symbol, price)
	}
}

// Stop останавливает мониторинг
func (ml *MarketListener) Stop() {
	ml.mu.Lock()
	defer ml.mu.Unlock()

	if !ml.isRunning {
		return
	}

	ml.logger.Info("🛑 Завершение работы...")

	// Останавливаем data handler если есть
	if ml.dataHandler != nil {
		ml.dataHandler.Stop()
	}

	ml.cancel()
	ml.isRunning = false

	// Ждем завершения горутин с таймаутом
	done := make(chan struct{})
	go func() {
		ml.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		ml.logger.Info("✅ Корректное завершение работы")
	case <-time.After(10 * time.Second):
		ml.logger.Warn("⚠️ Таймаут при завершении работы")
	}
}

// StartListener публичная функция для обратной совместимости
func StartListener() {
	listener := NewMarketListener()
	if err := listener.Start(); err != nil {
		log.Fatalf("Критическая ошибка: %v", err)
	}
}

// StartListenerWithDashboard запускает с консольным дашбордом
// StartListenerWithDashboard запускает с консольным дашбордом БЕЗ ЛОГОВ
func StartListenerWithDashboard() {
	listener := NewMarketListener()

	// Отключаем логи полностью для дашборда
	listener.logger.SetOutput(io.Discard)

	dashboard := NewConsoleDashboard()
	listener.SetDataHandler(dashboard)

	if err := listener.Start(); err != nil {
		log.Fatalf("Критическая ошибка: %v", err)
	}
}

// StartListenerWithNATS запускает с публикацией в NATS
func StartListenerWithNATS(natsURL string) {
	listener := NewMarketListener()

	natsPublisher, err := NewNATSPublisher(natsURL)
	if err != nil {
		log.Fatalf("Ошибка создания NATS publisher: %v", err)
	}

	// Подключаем NATS publisher как обработчик данных
	listener.SetDataHandler(natsPublisher)

	if err := listener.Start(); err != nil {
		log.Fatalf("Критическая ошибка: %v", err)
	}
}

// CombinedHandler комбинирует несколько обработчиков
type CombinedHandler struct {
	handlers []MarketDataHandler
}

// NewCombinedHandler создает комбинированный обработчик
func NewCombinedHandler(handlers ...MarketDataHandler) *CombinedHandler {
	return &CombinedHandler{
		handlers: handlers,
	}
}

// Start запускает все обработчики
func (ch *CombinedHandler) Start() {
	for _, handler := range ch.handlers {
		handler.Start()
	}
}

// Stop останавливает все обработчики
func (ch *CombinedHandler) Stop() {
	for _, handler := range ch.handlers {
		handler.Stop()
	}
}

// OnSpotPriceUpdate передает данные всем обработчикам
func (ch *CombinedHandler) OnSpotPriceUpdate(symbol, price string) {
	for _, handler := range ch.handlers {
		handler.OnSpotPriceUpdate(symbol, price)
	}
}

// OnFuturesPriceUpdate передает данные всем обработчикам
func (ch *CombinedHandler) OnFuturesPriceUpdate(symbol, price string) {
	for _, handler := range ch.handlers {
		handler.OnFuturesPriceUpdate(symbol, price)
	}
}

// StartListenerWithBoth запускает и с дашбордом, и с NATS одновременно
func StartListenerWithBoth(natsURL string) {
	listener := NewMarketListener()

	// Создаем оба обработчика
	dashboard := NewConsoleDashboard()
	natsPublisher, err := NewNATSPublisher(natsURL)
	if err != nil {
		log.Fatalf("Ошибка создания NATS publisher: %v", err)
	}

	// Создаем комбинированный обработчик
	combinedHandler := NewCombinedHandler(dashboard, natsPublisher)
	listener.SetDataHandler(combinedHandler)

	if err := listener.Start(); err != nil {
		log.Fatalf("Критическая ошибка: %v", err)
	}
}
