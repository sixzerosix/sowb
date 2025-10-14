package main

import (
	"os"
	"os/signal"
	"sow-market-hub/market"
	"syscall"
)

func main() {
	// Создаем subscriber
	subscriber, err := market.StartSubscriber("nats://31.207.77.179:4222")
	if err != nil {
		panic(err)
	}

	// Запускаем
	subscriber.Start()

	// Graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	<-sigChan

	// Останавливаем
	subscriber.Stop()
}
