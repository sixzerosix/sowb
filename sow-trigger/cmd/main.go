package main

import (
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
)

const (
	WorkerBuferSize = 10000
)

type Trigger struct {
	ID        int64     `json:"id"`
	UUID      uuid.UUID `json:"uuid"`
	Condition string    `json:"condition"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

type Worker struct {
	ID       int64     `json:"id"`
	UUID     uuid.UUID `json:"uuid"`
	Triggers []Trigger
}

type WorkerPool struct {
	ID   int64     `json:"id"`
	UUID uuid.UUID `json:"uuid"`
}

func CreateTrigger() {}

func CreateWorker() {}

func (w *Worker) WorkerInit() {

}

func evaluateCondition(condition string, a int64, b int64) bool {
	switch condition {
	case ">":
		return a > b
	case "<":
		return a < b
	default:
		return false
	}
}

func main() {
	start := time.Now()
	numsChan := make(chan int64, WorkerBuferSize)
	wg := sync.WaitGroup{}
	mu := sync.Mutex{}

	var acc int64 = 0

	for i := 0; i < cap(numsChan); i++ {
		wg.Add(1)
		go func(num int64) {
			defer wg.Done()
			mu.Lock()
			acc += num
			numsChan <- acc
			mu.Unlock()
		}(int64(i))
	}

	go func() {
		wg.Wait()
		close(numsChan)
	}()

	for n := range numsChan {
		cinditionResult := evaluateCondition(">", n, 1000)
		fmt.Printf("[%d > 1000][%t]\n", n, cinditionResult)
	}

	fmt.Printf("%d\n", acc)
	fmt.Println(time.Since(start).Seconds())
}
