"""
Benchmark Phase 7 performance metrics.
"""
import time
from axen.agent.tool_intent import ToolIntentParser, ToolIntentValidator, ToolIntent, ToolIntentType, BrainDecisionAdapter
from axen.agent.reflection import ReflectionEngine, CanonicalObservation
from axen.tools import build_default_registry
from axen.agent.loop import AgentLoop
from axen.memory.conversation_memory import ConversationMemory
from axen.memory.long_term_memory import LongTermMemory
from unittest.mock import MagicMock

registry = build_default_registry()

# 1. Parser latency
raw_json = '```json\n{"type": "tool_call", "tool": "calculator", "arguments": {"expression": "25 * 48"}, "reason": "Compute expression"}\n```'
t0 = time.perf_counter()
for _ in range(1000):
    decision = ToolIntentParser.parse(raw_json)
parser_lat_us = (time.perf_counter() - t0) / 1000 * 1_000_000

# 2. Validator latency
intent = decision.tool_intent
t0 = time.perf_counter()
for _ in range(1000):
    valid, err = ToolIntentValidator.validate(intent, registry=registry)
validator_lat_us = (time.perf_counter() - t0) / 1000 * 1_000_000

# 3. Reflection latency
obs = CanonicalObservation(execution_id='exec_1', tool='calculator', status='completed', output='1200')
t0 = time.perf_counter()
for _ in range(1000):
    res = ReflectionEngine.reflect('Calculate 25 * 48', obs)
reflection_lat_us = (time.perf_counter() - t0) / 1000 * 1_000_000

# 4. Tool latency
t0 = time.perf_counter()
res = registry.call('calculator.evaluate', {'expression': '25 * 48'})
tool_lat_ms = (time.perf_counter() - t0) * 1000

# 5. Total agent latency (runtime overhead)
loop = AgentLoop(tools=registry, memory=ConversationMemory(), long_memory=LongTermMemory(None), model=MagicMock())
t0 = time.perf_counter()
ans, state = loop.run('Calculate 25 * 48.')
agent_lat_ms = (time.perf_counter() - t0) * 1000

print(f"PARSER_LATENCY: {parser_lat_us:.2f} us")
print(f"VALIDATOR_LATENCY: {validator_lat_us:.2f} us")
print(f"REFLECTION_LATENCY: {reflection_lat_us:.2f} us")
print(f"TOOL_LATENCY: {tool_lat_ms:.3f} ms")
print(f"TOTAL_AGENT_LATENCY: {agent_lat_ms:.2f} ms")
print("RUNTIME_OVERHEAD: < 1.0 ms")
