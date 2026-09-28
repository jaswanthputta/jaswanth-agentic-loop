# Agentic patterns

## ReAct

ReAct interleaves a model's reasoning or decision summary with a concrete action, then places the action's observation back into context for the next decision. This lets a system adapt its next tool call to evidence rather than execute a fixed script. This agent uses a compact ReAct cycle: `reason` selects one registered tool, `act` returns its observation, and that observation is carried into the next `perceive`/`reason` pass. The plan includes only a brief decision summary; it does not request hidden chain-of-thought.

## Reflexion

Reflexion adds a verbal self-evaluation after an action and stores that feedback so later attempts can improve. It is useful where a first answer may be incomplete and a system needs explicit progress criteria rather than treating any output as success. This agent uses `reflect` to score progress, state what is still missing, and persist the result in session memory. The next planning step receives both the latest reflection and semantically recalled prior steps.

## Chain-of-Thought (CoT)

Chain-of-Thought prompting asks a model to produce intermediate reasoning steps before an answer. It can help decompose difficult tasks, but unrestricted reasoning traces are not necessary for tool control and can be verbose or expose internal reasoning. This loop does not request or store private step-by-step reasoning; it uses concise decision summaries, explicit tool inputs, and inspectable results instead.

## Tree of Thoughts (ToT)

Tree of Thoughts explores multiple candidate reasoning branches, evaluates intermediate states, and can backtrack or continue promising branches. It can help with search problems that have several plausible solution paths, but requires additional evaluation and search-budget logic. This use case has a simpler linear workflow with review-and-revise, so the loop does not implement ToT branching.

## LATS

Language Agent Tree Search (LATS) combines LLM-generated candidate actions with tree search, often using value estimates and Monte Carlo Tree Search-style exploration to select promising trajectories. It is appropriate when action sequences have meaningful alternatives and outcomes can be evaluated. The question-generation workflow is sufficiently served by sequential generation and coverage review, so this implementation does not add LATS search overhead.

## Choice for this use case

The loop combines **ReAct and Reflexion**. ReAct makes analysis, question generation, and coverage review distinct tool actions; Reflexion keeps the work iterative and records what the next pass should improve. This pairing directly supports the user goal—questions grounded in a job description and checked for coverage—without the unnecessary branching cost of ToT/LATS. The demo deliberately demonstrates three passes: extract requirements, draft questions, and review coverage.
