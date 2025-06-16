#!/bin/bash

curl -fsSL https://ollama.com/install.sh | sh

ollama serve

ollama pull gemma3:27b

