package market

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/fatih/color"
)

// PriceData структура для хранения данных о цене
type PriceData struct {
	Symbol       string
	SpotPrice    string
	FuturePrice  string
	LastUpdate   time.Time
	SpotChange   string
	FutureChange string
}

// ConsoleDashboard структура для управления консольным дашбордом
type ConsoleDashboard struct {
	data      map[string]*PriceData
	mu        sync.RWMutex
	isRunning bool
	stopChan  chan struct{}
}

// NewConsoleDashboard создает новый дашборд
func NewConsoleDashboard() *ConsoleDashboard {
	return &ConsoleDashboard{
		data:     make(map[string]*PriceData),
		stopChan: make(chan struct{}),
	}
}

// clearScreen очищает экран
func clearScreen() {
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.Command("cmd", "/c", "cls")
	} else {
		cmd = exec.Command("clear")
	}
	cmd.Stdout = os.Stdout
	cmd.Run()
}

// moveCursor перемещает курсор в начало
func moveCursor() {
	fmt.Print("\033[H\033[2J") // ANSI escape codes для очистки и перемещения курсора
}

// Start запускает дашборд (реализация интерфейса MarketDataHandler)
func (cd *ConsoleDashboard) Start() {
	cd.mu.Lock()
	if cd.isRunning {
		cd.mu.Unlock()
		return
	}
	cd.isRunning = true
	cd.mu.Unlock()

	// Очищаем экран в начале
	clearScreen()

	// Запускаем обновление каждые 500ms для стабильности
	go func() {
		ticker := time.NewTicker(500 * time.Millisecond)
		defer ticker.Stop()

		for {
			select {
			case <-cd.stopChan:
				return
			case <-ticker.C:
				cd.render()
			}
		}
	}()
}

// Stop останавливает дашборд (реализация интерфейса MarketDataHandler)
func (cd *ConsoleDashboard) Stop() {
	cd.mu.Lock()
	defer cd.mu.Unlock()

	if !cd.isRunning {
		return
	}

	cd.isRunning = false
	close(cd.stopChan)
}

// OnSpotPriceUpdate обновляет спот цену (реализация интерфейса MarketDataHandler)
func (cd *ConsoleDashboard) OnSpotPriceUpdate(symbol, price string) {
	cd.mu.Lock()
	defer cd.mu.Unlock()

	if cd.data[symbol] == nil {
		cd.data[symbol] = &PriceData{Symbol: symbol}
	}

	// Сравниваем с предыдущей ценой для определения тренда
	oldPrice := cd.data[symbol].SpotPrice
	cd.data[symbol].SpotPrice = price
	cd.data[symbol].LastUpdate = time.Now()

	if oldPrice != "" && oldPrice != price {
		if price > oldPrice {
			cd.data[symbol].SpotChange = "↗"
		} else {
			cd.data[symbol].SpotChange = "↘"
		}
	} else {
		cd.data[symbol].SpotChange = "→"
	}
}

// OnFuturesPriceUpdate обновляет фьючерсную цену (реализация интерфейса MarketDataHandler)
func (cd *ConsoleDashboard) OnFuturesPriceUpdate(symbol, price string) {
	cd.mu.Lock()
	defer cd.mu.Unlock()

	if cd.data[symbol] == nil {
		cd.data[symbol] = &PriceData{Symbol: symbol}
	}

	// Сравниваем с предыдущей ценой для определения тренда
	oldPrice := cd.data[symbol].FuturePrice
	cd.data[symbol].FuturePrice = price
	cd.data[symbol].LastUpdate = time.Now()

	if oldPrice != "" && oldPrice != price {
		if price > oldPrice {
			cd.data[symbol].FutureChange = "↗"
		} else {
			cd.data[symbol].FutureChange = "↘"
		}
	} else {
		cd.data[symbol].FutureChange = "→"
	}
}

// render отрисовывает дашборд
func (cd *ConsoleDashboard) render() {
	cd.mu.RLock()
	defer cd.mu.RUnlock()

	// Перемещаем курсор в начало вместо очистки экрана
	moveCursor()

	var output strings.Builder

	// Заголовок
	title := color.New(color.FgCyan, color.Bold).Sprint("🚀 BYBIT MARKET DASHBOARD")
	timestamp := color.New(color.FgWhite).Sprintf("Last Update: %s", time.Now().Format("15:04:05"))

	output.WriteString(fmt.Sprintf("%s\n", title))
	output.WriteString(fmt.Sprintf("%s\n", timestamp))
	output.WriteString(strings.Repeat("═", 80) + "\n")

	// Заголовки таблицы
	header := fmt.Sprintf("%-12s │ %-25s │ %-25s │ %-10s",
		color.New(color.FgYellow, color.Bold).Sprint("SYMBOL"),
		color.New(color.FgGreen, color.Bold).Sprint("SPOT PRICE"),
		color.New(color.FgBlue, color.Bold).Sprint("FUTURES PRICE"),
		color.New(color.FgMagenta, color.Bold).Sprint("STATUS"))

	output.WriteString(header + "\n")
	output.WriteString(strings.Repeat("─", 80) + "\n")

	// Сортируем символы для стабильного отображения
	symbols := make([]string, 0, len(cd.data))
	for symbol := range cd.data {
		symbols = append(symbols, symbol)
	}
	sort.Strings(symbols)

	// Отображаем данные
	for _, symbol := range symbols {
		data := cd.data[symbol]
		if data == nil {
			continue
		}

		// Форматируем цены с цветами и стрелками
		spotPrice := cd.formatPrice(data.SpotPrice, data.SpotChange, color.FgGreen)
		futurePrice := cd.formatPrice(data.FuturePrice, data.FutureChange, color.FgBlue)

		// Статус подключения
		status := cd.getConnectionStatus(data)

		row := fmt.Sprintf("%-12s │ %-35s │ %-35s │ %-10s",
			color.New(color.FgWhite, color.Bold).Sprint(symbol),
			spotPrice,
			futurePrice,
			status)

		output.WriteString(row + "\n")
	}

	output.WriteString(strings.Repeat("═", 80) + "\n")

	// Статистика
	stats := cd.getStats()
	output.WriteString(stats + "\n")

	// Добавляем пустые строки для стабильности отображения
	for i := 0; i < 10; i++ {
		output.WriteString(strings.Repeat(" ", 80) + "\n")
	}

	// Выводим весь блок сразу
	fmt.Print(output.String())
}

// formatPrice форматирует цену с цветом и стрелкой
func (cd *ConsoleDashboard) formatPrice(price, change string, priceColor color.Attribute) string {
	if price == "" {
		return color.New(color.FgRed).Sprint("N/A")
	}

	// Определяем цвет стрелки
	var arrowColor color.Attribute
	switch change {
	case "↗":
		arrowColor = color.FgGreen
	case "↘":
		arrowColor = color.FgRed
	default:
		arrowColor = color.FgYellow
	}

	priceStr := color.New(priceColor, color.Bold).Sprint(price)
	arrow := color.New(arrowColor).Sprint(change)

	return fmt.Sprintf("%s %s", priceStr, arrow)
}

// getConnectionStatus возвращает статус подключения
func (cd *ConsoleDashboard) getConnectionStatus(data *PriceData) string {
	now := time.Now()
	timeSinceUpdate := now.Sub(data.LastUpdate)

	if timeSinceUpdate < 5*time.Second {
		return color.New(color.FgGreen).Sprint("🟢 LIVE")
	} else if timeSinceUpdate < 30*time.Second {
		return color.New(color.FgYellow).Sprint("🟡 SLOW")
	} else {
		return color.New(color.FgRed).Sprint("🔴 DEAD")
	}
}

// getStats возвращает общую статистику
func (cd *ConsoleDashboard) getStats() string {
	totalSymbols := len(cd.data)
	activeConnections := 0

	for _, data := range cd.data {
		if time.Since(data.LastUpdate) < 30*time.Second {
			activeConnections++
		}
	}

	stats := fmt.Sprintf("📊 Total Symbols: %s | Active Connections: %s | Press Ctrl+C to exit",
		color.New(color.FgCyan, color.Bold).Sprint(totalSymbols),
		color.New(color.FgGreen, color.Bold).Sprint(activeConnections))

	return stats
}
