package flow

import (
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/gorilla/websocket"
)

var wss = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		origin := r.Header.Get("Origin")
		// Разрешаем только определённые домены
		allowedOrigins := map[string]bool{
			"http://127.0.0.1:5500": true,
			"http://localhost:3006": true,
		}
		return allowedOrigins[origin]
	},
}

func Connection(w http.ResponseWriter, r *http.Request) {
	connection, err := wss.Upgrade(w, r, nil)
	if err != nil {
		log.Fatal(err)
	}
	// Закрываем соеденение
	defer connection.Close()

	fmt.Println("New websocket connection")

	for {
		msgType, msg, err := connection.ReadMessage()
		if err != nil {
			log.Fatal(err)
			break
		}

		fmt.Println("Get message: ", string(msg))

		for i := 0; ; i++ {
			err = connection.WriteMessage(msgType, []byte(fmt.Sprintf("Hello: %d", i)))
			if err != nil {
				log.Fatal(err)
				break
			}

			time.Sleep(10 * time.Millisecond)
		}

	}

}

func Run() {
	http.HandleFunc("/ws", Connection)

	fmt.Println("Websocket server is running")
	http.ListenAndServe(":3006", nil)
}
