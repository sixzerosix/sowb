package main

import (
	"fmt"
	"sow-wallet/pkj/currency"
)

func main() {
	amount, err := currency.NormalizeFromCurrency("USD", 22244)
	if err != nil {
		panic(err)
	}
	fmt.Printf("USD %d.%d", amount[0], amount[1])
}
