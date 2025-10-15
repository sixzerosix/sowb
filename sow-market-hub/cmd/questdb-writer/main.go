package main

import (
	"os"
	"os/signal"
	"sow-market-hub/market"
	"syscall"
	"time"
)

func main() {
	// Создаем QuestDB writer
	writer, err := market.NewQuestDBWriter(
		"localhost:6379", // Dragonfly
		"postgres://admin:quest@localhost:8812/qdb?sslmode=disable", // QuestDB
	)
	if err != nil {
		panic(err)
	}

	// Запускаем с интервалом 10 секунд (QuestDB очень быстрый)
	writer.Start(10 * time.Second)

	// Graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	<-sigChan

	writer.Stop()
}
