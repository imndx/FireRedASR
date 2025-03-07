#!/bin/bash

# 等待Ollama服务启动
echo "Waiting for Ollama service to start..."
until $(curl --output /dev/null --silent --fail http://ollama:11434/api/health); do
  printf '.'
  sleep 5
done
echo "Ollama service is up!"

# 拉取qwen2.5:3b模型
echo "Pulling qwen2.5:3b model..."
curl -X POST http://ollama:11434/api/pull -d '{"name": "qwen2.5:3b"}'
echo "Model pulled successfully"

echo "Ollama initialization completed!"
