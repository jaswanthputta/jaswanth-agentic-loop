# Memory design

The agent uses **ChromaDB** as a persistent vector store. Each reflection is saved as a document with session and iteration metadata. Recall is filtered by session ID and uses a stable, lightweight feature-hashing embedding (tokens and character trigrams), so the vector store works without downloading a separate embedding model. For production use with multilingual or semantically distant descriptions, replace the local embedder with a dedicated embedding model.

At the start of every `reason` call, the loop recalls up to three prior reflection/action records and passes them into the planner prompt. After every `reflect` call, it stores the current action, result, and reflection. The included demo shows this across iterations: after iteration 1 stores extracted requirements, iteration 2 recalls that record and uses those requirements as the input to question generation; iteration 3 recalls the stored work before choosing coverage review. Memory read/write errors are logged as warnings, and the loop continues without persistence when the backend is unavailable.

Use `--session-id` to resume a memory session, or omit it to get a fresh session. Chroma data is stored under `.agent_memory` by default.
