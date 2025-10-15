package main

import (
	"log"
	"sow-market-hub/market" // Убедись, что путь правильный
)

func main() {
	// --- ЗАГРУЖАЕМ КОНФИГУРАЦИЮ ИЗ ФАЙЛА config.json ---
	// Вместо хардкода, используем функцию loadConfig из пакета market
	// config := market.LoadConfig() // Предполагается, что market.LoadConfig() публична и загружает config.json

	// Если нужно настроить логирование для этого конкретного main.go
	// log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)
	// log.Printf("Loaded config: %+v", config) // Для отладки

	// --- ЗАКОММЕНТИРОВАННЫЙ КОД ДЛЯ БАЗ ДАННЫХ ---
	// Этот код вызывал панику, так как Dragonfly/TimescaleDB не были запущены.
	// listener := market.NewEnhancedMarketListener(config) // Это больше не нужно здесь напрямую

	// writer, err := market.NewTimescaleWriter("localhost:6379", "postgres://postgres:postgres@localhost:5432/timescaledb?sslmode=disable")
	// if err != nil {
	// 	panic(fmt.Sprintf("Failed to create TimescaleDB writer: %v", err))
	// }
	// writer.Start(time.Second * 5)

	// if err := listener.Start(); err != nil {
	// 	panic(fmt.Sprintf("Failed to start enhanced market listener: %v", err))
	// }
	// --- КОНЕЦ ЗАКОММЕНТИРОВАННОГО КОДА ---

	// --- ЗАПУСКАЕМ PUBLISHER, КОТОРЫЙ БУДЕТ ПИСАТЬ ТОЛЬКО В NATS ---
	// Эта функция уже использует NewMarketListener() и NewNATSPublisher()
	// и загружает конфиг внутри себя (если LoadConfig() вызывается в NewMarketListener).
	// Если ты хочешь передать конкретный конфиг, то нужно будет изменить StartListenerWithNATS
	// или создать новый wrapper. Пока оставим как есть, он возьмет дефолтный/из файла.
	// log.Println("🚀 Запуск Publisher в NATS...")
	// market.StartListenerWithNATS("nats://31.207.77.179:4222") // Используем твой NATS URL

	// // Блокируем main горутину, чтобы программа не завершилась сразу
	// select {}
	config := market.LoadConfig() // Загрузка конфига
	listener := market.NewEnhancedMarketListener(config)

	// --- ОБЯЗАТЕЛЬНО ДОБАВЬ EnhancedNATSPublisher как обработчик ---
	natsPublisher, err := market.NewEnhancedNATSPublisher("nats://31.207.77.179:4222")
	if err != nil {
		log.Fatalf("Ошибка создания Enhanced NATS publisher: %v", err)
	}
	listener.AddDataHandler(natsPublisher)
	// -------------------------------------------------------------

	// --- И ЗАПУСТИ ЕГО ---
	if err := listener.Start(); err != nil {
		log.Fatalf("Критическая ошибка: %v", err)
	}
	select {}

}
