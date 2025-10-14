package main

import "sow-market-hub/market"

func main() {
	market.StartListenerWithNATS("nats://31.207.77.179:4222")
}
