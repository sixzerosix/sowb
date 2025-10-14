package services

import (
	"fmt"
	"log"
	"strings"
)

const (
	RUB CURR = 100
)

// Cyrrencies type
type CURR int64

type Operation interface {
	Transfer(walletID int64, symbol string, amount CURR) (bool, error)
	MyBalance(symbol string) (CURR, error)
}

// Test wallets
var wallets = []Wallet{
	Wallet{
		ID: 1,
		Balance: Balance{
			RUB: 0,
		},
	},
	Wallet{
		ID: 2,
		Balance: Balance{
			RUB: 1000,
		},
	},
}

type Wallet struct {
	ID int64
	Balance
}

type Balance struct {
	RUB CURR
}

func symbolToBalance(symbol string, ammount CURR) CURR {
	switch strings.ToUpper(symbol) {
	case "RUB":
		normalizedAmount := ammount * RUB
		return normalizedAmount
	default:
		return 0
	}
}

func symbolFromBalance(symbol string, ammount CURR) CURR {
	switch strings.ToUpper(symbol) {
	case "RUB":
		normalizedAmount := ammount / RUB
		return normalizedAmount
	default:
		return 0
	}
}

func NewWallet() (*Wallet, error) {
	id := len(wallets) + 1
	wallet := &Wallet{
		ID:      int64(id),
		Balance: Balance{},
	}
	return wallet, nil
}

func getWallet(walletID int64) (*Wallet, error) {
	for _, wallet := range wallets {
		if walletID == wallet.ID {
			return &wallet, nil
		}
	}
	return nil, fmt.Errorf("Wallet width ID is not define")
}

func (w *Wallet) Transfer(recipientWalletID int64, symbol string, amount CURR) (bool, error) {
	balance, err := w.MyBalance(symbol)
	if err != nil {
		log.Fatal(err)
	}
	if balance < amount {
		return false, fmt.Errorf("Insufficient funds on the balance sheet")
	}

	recipientWallet, err := getWallet(recipientWalletID)
	if err != nil {
		log.Fatal(err)
	}
	w.Balance.RUB -= amount
	recipientWallet.Balance.RUB += amount

	fmt.Printf("Transfer %s %d from wallet %d to wallet %d is successful", symbol, amount, w.ID, recipientWalletID)
	return true, nil
}

func ExecuteTransfer(op Operation, recipientWalletID int64, symbol string, amount CURR) error {
	success, err := op.Transfer(recipientWalletID, symbol, amount)
	if err != nil {
		return err
	}
	if !success {
		return fmt.Errorf("transfer unsuccessful")
	}
	return nil
}

func (w *Wallet) MyBalance(symbol string) (CURR, error) {
	balance := symbolFromBalance(symbol, w.Balance.RUB)
	return balance, nil
}

// Test
func Run() {
	ExecuteTransfer(&wallets[0], 2, "RUB", 1)
}
