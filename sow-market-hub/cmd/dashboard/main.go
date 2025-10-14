package main

import "sow-market-hub/market"

func main() {
	// Только дашборд (без NATS)
	market.StartListenerWithDashboard()
}
