package currency

import (
	"fmt"
	"strings"
)

const (
	USD int64 = 100
	EUR int64 = 100
	JPY int64 = 1
	GBP int64 = 100
	CHF int64 = 100
	CNY int64 = 100
	AUD int64 = 100
	CAD int64 = 100
	SEK int64 = 100
	RUB int64 = 100
)

type CurrencyOperations interface {
	// Convert(symbol string, amount int64, convertToCurrencySymbol string)
	SymbolDivider() (int64, error)
	NormalizeToCurrency(amountIntegerPart int64, amountFractionalPart int64) (int64, error)
	NormalizeFromCurrency(ammount int64) ([]int64, error)
}

type Currency struct {
	Country     string
	FullName    string
	Symbol      string
	Description string
}

func SymbolDivider(symbol string) (int64, error) {

	switch strings.ToUpper(symbol) {
	case "USD":
		return USD, nil
	case "EUR":
		return EUR, nil
	case "JPY":
		return JPY, nil
	case "GBP":
		return GBP, nil
	case "CHF":
		return CHF, nil
	case "CNY":
		return CNY, nil
	case "AUD":
		return AUD, nil
	case "CAD":
		return CAD, nil
	case "SEK":
		return SEK, nil
	case "RUB":
		return RUB, nil
	default:
		return 0, fmt.Errorf("symbol %s is not found", strings.ToUpper(symbol))
	}
}

func NormalizeFromCurrency(symbol string, amount int64) ([]int64, error) {
	divider, err := SymbolDivider(symbol)
	if err != nil {
		return nil, err
	}
	integerPart := amount / int64(divider)
	fractionalPart := amount % int64(divider)

	normalized := []int64{
		integerPart,
		fractionalPart,
	}

	return normalized, nil
}

func NormalizeToCurrency(symbol string, amountIntegerPart int64, amountFractionalPart int64) (int64, error) {
	divider, err := SymbolDivider(symbol)
	if err != nil {
		return 0, err
	}

	normalized := amountIntegerPart*int64(divider) + int64(amountFractionalPart)

	return normalized, nil
}

func (c *Currency) SymbolDivider() (int64, error) {
	return SymbolDivider(c.Symbol)
}

func (c *Currency) NormalizeToCurrency(amountIntegerPart int64, amountFractionalPart int64) (int64, error) {
	return NormalizeToCurrency(c.Symbol, amountIntegerPart, amountFractionalPart)
}

func (c *Currency) NormalizeFromCurrency(amount int64) ([]int64, error) {
	return NormalizeFromCurrency(c.Symbol, amount)
}
