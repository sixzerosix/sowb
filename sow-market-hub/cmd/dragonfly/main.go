package main

import (
	"os"
	"os/signal"
	"sow-market-hub/market"
	"syscall"
)

func main() {
	// Запускаем Dragonfly consumer
	consumer, err := market.StartDragonflyConsumer(
		"nats://31.207.77.179:4222", // NATS URL
		"31.207.77.179:6379",        // Dragonfly URL
	)
	if err != nil {
		panic(err)
	}

	// Graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	<-sigChan

	consumer.Stop()
}
