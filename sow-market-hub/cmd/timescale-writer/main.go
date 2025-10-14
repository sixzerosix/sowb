package main

import (
	"os"
	"os/signal"
	"sow-market-hub/market"
	"syscall"
	"time"
)

func main() {
	// Создаем TimescaleDB writer
	writer, err := market.NewTimescaleWriter(
		"localhost:6379", // Dragonfly
		"postgres://user:pass@localhost:5432/timescaledb", // TimescaleDB
	)
	if err != nil {
		panic(err)
	}

	// Запускаем с интервалом 30 секунд
	writer.Start(30 * time.Second)

	// Graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	<-sigChan

	writer.Stop()
}
