package service

import (
	"fmt"
	"math/rand"
	"sync"
	"time"

	"github.com/google/uuid"
)

/*
Бизнес логика
*/

type DataSet map[string]int64

type Trigger struct {
	UUID uuid.UUID
	Data DataSet
}

func generateUUID() uuid.UUID {
	return uuid.New()
}

var rnd = rand.New(rand.NewSource(time.Now().UnixNano()))

func rndInt(max int64) int64 {
	return rnd.Int63n(max)
}

func generateRndData() DataSet {
	return DataSet{
		"key": rndInt(100),
	}
}

func NewTrigger(data DataSet) (*Trigger, error) {
	if len(data) <= 0 {
		return nil, fmt.Errorf("data is empty")
	}

	trigger := &Trigger{
		UUID: generateUUID(),
		Data: data,
	}

	return trigger, nil
}

func GeneratorTriggers(count int64) (*[]Trigger, error) {
	if count < 0 {
		return nil, fmt.Errorf("the minimum number to generate must be 1 or more")
	}

	var triggers []Trigger

	for i := 0; i < int(count); i++ {
		newTrigger, err := NewTrigger(generateRndData())
		if err != nil {
			panic(err)
		}
		triggers = append(triggers, *newTrigger)
	}

	return &triggers, nil
}

func AddPercent(num int64, percent int64, cb func(num int64) int64) int64 {
	return cb(num) + (num / 100 * percent)
}

func Worker() {
	var countTrg int64 = 100_000

	var counter int
	strart := time.Now()
	trgCh := make(chan Trigger, countTrg)
	trgs, err := GeneratorTriggers(countTrg)
	if err != nil {
		panic(err)
	}
	lt := len(*trgs)
	var wg sync.WaitGroup
	var mu sync.Mutex

	for _, t := range *trgs {
		wg.Add(1)
		go func(trigger Trigger) {
			defer wg.Done()

			mu.Lock()
			trgCh <- trigger
			mu.Unlock()
		}(t)

	}
	wg.Wait()

	// go func() {
	for {
		var rndTo = rand.New(rand.NewSource(time.Now().UnixNano()))
		r := rndTo.Int63n(100)

		for i := 0; i < lt; i++ {
			t := <-trgCh
			counter++

			// fmt.Printf("[LOG:%s]: %d >= %d \n", t.UUID, t.Data["key"], r)

			if t.Data["key"] <= r {
				// fmt.Printf("[%d][DONE:%s]: %d >= %d \n", lt, t.UUID, t.Data["key"], r)

				lt--
				continue
			}

			trgCh <- t
		}

		if lt == 0 {
			fmt.Printf("%d\n", counter)
			fmt.Println(time.Now().Sub(strart).Seconds())
			return
		}

		// fmt.Printf("[%d]\n", lt)
		// time.Sleep(time.Nanosecond)
	}

	// }()
	close(trgCh)
}
