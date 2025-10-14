package trigger

import (
	"fmt"
	"time"

	"github.com/google/uuid"
)

type Trigger struct {
	UUID        uuid.UUID
	Name        string      // Название
	Description string      // Описание
	Conditions  func() bool // Услоаие при котором сработате триггер
	Action      func()      // Функция которая сработатет после выполнения условия
	Status      int         // Статус выполнения: выполнен, отменён, активный
	IsActive    bool        // Состояние
	CreatedAt   int64       // Время создания
	UpdatedAt   int64       // Время когда было последнее обновление
	ActivatedAt int64       // Время когда триггер сработал
}

func Run() {
	for {
		fmt.Print("Trigger is running")
		time.Sleep(1 * time.Second)
	}
}
