@echo off
cd /d e:\ollm-project\rust-llm
echo Running rust-llm...
target\debug\rust-llm.exe --model e:\models\qwen\Qwen3.5-9B-Q4_K_M.gguf --use-gpu --prompt "hello" --max-tokens 10 --temperature 0.7 --top-p 0.9 --top-k 40 --repeat-penalty 1.1 --context-size 2048 --batch-size 1
echo Done.
pause
