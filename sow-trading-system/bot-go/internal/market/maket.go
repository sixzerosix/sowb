package market

// Main configs
const (
	PRICEURL = "https://fapi.binance.com/fapi/v2/ticker/price"
)

type Coin struct {
	symbol string
	price  float64
}

func fetchPriceData() ([]Coin, error) {

}

func Flow() {

}
