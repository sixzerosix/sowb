package main

import (
	"fmt"

	"github.com/nats-io/nats.go"
)

func main() {

	nc, _ := nats.Connect("nats://31.207.77.179:4222")
	nc.Subscribe("analytics.>", func(m *nats.Msg) {
		fmt.Printf("subj=%s data=%s\n", m.Subject, string(m.Data))
	})
	// select {}
}

// type AgentSettings struct {
// 	Temperature float64
// 	MaxTokens   int
// }
// type Agent struct {
// 	Name         string
// 	Role         string
// 	SystemPrompt string

// 	Settings AgentSettings
// }

// func (a *Agent) GetInfo(format *string) string {
// 	if format == nil {
// 		*format = "text"
// 	}
// 	switch *format {
// 	case "json":
// 		return fmt.Sprintf(`{"Name": "%s", "Role": "%s", "SystemPrompt": "%s", "Settings": {"Temperature": %f, "MaxTokens": %d}}`,
// 			a.Name, a.Role, a.SystemPrompt, a.Settings.Temperature, a.Settings.MaxTokens)
// 	default:
// 		return fmt.Sprintf("Name: %s, Role: %s, SystemPrompt: %s, Settings: {Temperature: %f, MaxTokens: %d}",
// 			a.Name, a.Role, a.SystemPrompt, a.Settings.Temperature, a.Settings.MaxTokens)
// 	}
// }
// a := &Agent{
// 	Name:         "Ghost",
// 	Role:         "Assassin",
// 	SystemPrompt: "You are a stealthy assassin.",
// 	Settings: AgentSettings{
// 		Temperature: 0.7,
// 		MaxTokens:   150,
// 	},
// }
// s := ""

// a.GetInfo(&s)
