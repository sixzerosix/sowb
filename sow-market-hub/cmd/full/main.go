package main

import "sow-market-hub/market"

func main() {
	// Дашборд + NATS publisher
	market.StartListenerWithBoth("nats://31.207.77.179:4222")
}
