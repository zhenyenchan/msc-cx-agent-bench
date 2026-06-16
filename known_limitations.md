# Known Limitations

This document tracks observed failure modes of the agent during development.

## 1. Scope refusal
- e.g. what is the capital of France
- The agent sometimes calls a tool before refusing an out-of-scope question
- Final answer is correct but the path is inefficient
- Path efficiency metric needed

## 2. Unsupported query granularity
- e.g. what is the top trending model of brand X's product Y with the best reviews?
- The agent does not detect when a question asks for granularity below what the data supports e.g. specific products and companies. 
- Currently maps the question to the nearest available filter and returns a misleading answer
- Metric needed to measure whether the answer is grounded in data

## 3. Invalid tool name hallucination
- The agent sometimes invents a fake function name e.g. "function_name"
- Metric needed to measure tool validity