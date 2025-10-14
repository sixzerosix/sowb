package market

import "time"

// QuoteData представляет структуру котировки
type QuoteData struct {
	Symbol    string    `json:"symbol"`
	Price     float64   `json:"price"`
	Market    string    `json:"market"` // "SPOT" или "FUTURES"
	Timestamp time.Time `json:"timestamp"`
}
