package main

import (
  "fmt"
  corelang "github.com/kinga-ai/kinga-core/lang/go"
)

func main() {
  fmt.Printf(\"Inzwa sees %d supported languages: %v\\n\", len(corelang.Codes()), corelang.Codes())
}
