package model

import "github.com/google/uuid"

type DataSet map[string]int64

type Trigger struct {
	UUID uuid.UUID
	Data DataSet
}
