# SkyRescue human-instruction LLM benchmark

Fixed parameters: temperature=0, top_p=1, max_tokens=512; one response per model and instruction.
The same raw response is reused for direct JSON parsing, schema validation, and full SkyRescue compilation.

| Model | API success | Direct JSON | Schema pass | Slot accuracy | Exact record | Compiler outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deepseek-v4-flash | 1.0000 | 1.0000 | 0.9400 | 0.6586 | 0.0200 | 0.6400 |
| qwen3-30b-a3b-instruct-2507 | 1.0000 | 1.0000 | 1.0000 | 0.6214 | 0.0000 | 0.7600 |
