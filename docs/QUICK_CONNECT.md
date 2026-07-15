# Quick connect any agent

The fastest way to use agent-improvement-loop is to wrap the callable that runs your
agent. The wrapper records a root agent run in MLflow when MLflow is installed and leaves
evaluation and optimization to the Databricks control plane.
It does not automatically instrument every nested model/tool call inside an
arbitrary callable; use the framework's native MLflow or OTEL integration when
you need detailed child spans and token/tool accounting.

## Python

```bash
pip install -e .
```

```python
from ail import improve


def my_agent(task: str) -> str:
    return call_my_agent(task)

agent = improve(
    name="my-agent",
    run=my_agent,
    objective="Improve correctness while reducing cost and latency.",
    experiment_id="my-mlflow-experiment",  # optional
    tracking_uri="databricks",              # optional
)

result = agent.run("Review this pull request")
print(result.output)
print(result.trace_id)
```

`improve()` accepts any synchronous Python callable: a coding agent, a custom
workflow, or a single LLM call. It also supports keyword arguments and can be used
as a decorator:

```python
from ail import trace

@trace(agent="support-agent")
def answer(request: str) -> str:
    return llm(request)
```

## CLI

Expose a function as `module:function` and run it directly:

```bash
ail run \
  --name my-agent \
  --callable my_app:run \
  --prompt "Review this pull request" \
  --objective "Improve correctness and reduce token usage"
```

The command prints JSON containing the output, elapsed time, agent name, and MLflow
trace id when available.

## What happens next

1. The wrapper emits an MLflow trace.
2. The Databricks app observes baseline quality, cost, latency, and tool usage.
3. Once enough evidence exists, the improvement plane can use judges, RLM/HALO,
   MemAlign, and GEPA to propose candidates.
4. Advanced setup adds approval, versioning, held-out verification, lineage, and
   rollback controls.

Quick connect is intentionally observe-first. It does not create a second evaluation
framework and it does not automatically change production behavior.
