# State of the Art Outline for Review

This outline is intended as review material before editing `Thesis.tex`.
It focuses on the research areas that frame this thesis: LLM-based data agents, text-to-SQL, agent orchestration, inference-time optimization, evaluation of generated analytical artifacts, and efficiency-aware benchmarking.

The article version is limited to approximately 15 pages, so the State of the Art should occupy at most four pages.
The section should not be a tool catalogue.
Its goal is to explain the scientific and technical context behind the thesis:
modern LLM agents can perform multi-step data analysis, but their quality, reliability, latency, and energy cost depend on model choice, prompting strategy, workflow design, and evaluation methodology.

## Suggested Structure

1. LLM agents for structured data analysis
2. Text-to-SQL and database-grounded generation
3. Model families for local agent inference
4. Agent orchestration and inference-time reasoning strategies
5. Evaluation of data-agent outputs
6. Efficiency-aware benchmarking of local LLM agents
7. Short synthesis and research gap

## 1. LLM Agents for Structured Data Analysis

Target length: about half a page.

Purpose:
Introduce the broader move from single-turn chatbots to tool-using systems that can interact with external computational resources.
In this thesis, the relevant class of systems is data-analysis agents: agents that transform natural language business questions into executable database queries, interpret the results, and optionally produce visualization code.

Points to cover:

- LLMs are increasingly used as interfaces for data analysis because they can map natural language requests to formal operations.
- A data-analysis agent differs from a simple question-answering system because it must execute tools and maintain intermediate state.
- The output is not only text: it includes SQL, retrieved tables, analytical summaries, and visualizations.
- This makes the task closer to an analytical workflow than to a single LLM response.
- The central challenge is reliability: the agent must correctly identify the required data, generate executable SQL, avoid hallucinated conclusions, and produce valid chart specifications or code.

Connection to this work:
Use this subsection to introduce the Sales Agent as an example of an LLM-powered analytical workflow over a relational retail dataset stored in Parquet files.
The agent's tools correspond to the main phases of a business analyst's workflow: data retrieval, interpretation, and visualization.

## 2. Text-to-SQL and Database-Grounded Generation

Target length: about three quarters of a page.

Purpose:
Position the lookup step within the text-to-SQL literature.
The generated SQL query is the foundation of the whole agent: if the lookup is wrong, later text and visualization steps may become fluent but incorrect.

Points to cover:

- Text-to-SQL maps natural language questions to executable database queries.
- Accuracy depends on schema understanding, column grounding, join selection, aggregation logic, filtering, grouping, and date handling.
- Multi-table schemas make the problem harder because the model must identify which tables are relevant and how they should be joined.
- SQL equivalence is difficult to evaluate through string matching because different queries can return the same correct result.
- In data-agent settings, the more meaningful correctness target is usually the returned table, not the literal SQL string.
- Common failure modes include wrong joins, wrong aggregation level, missing filters, confusing average-of-rows with average-of-aggregates, incorrect date extraction, and non-pivoted results when the requested output needs a wide table.

Connection to this work:
This thesis treats SQL generation as a dedicated agent step.
The lookup step receives schema descriptions, generates DuckDB SQL, executes the query over Parquet tables, and stores both the SQL and the resulting dataframe for later evaluation and downstream generation.
The benchmark difficulty rubric should be introduced here as a natural consequence of text-to-SQL complexity: flat aggregations, time bucketing, joins, nested aggregation, conditional metrics, ratios, and year-over-year comparisons.

## 3. Model Families for Local Agent Inference

Target length: about half a page.

Purpose:
Introduce the model-side variables that matter for energy-performance analysis.
The thesis experiments compare not only hyperparameters, but also different model families: smaller versus larger models, thinking/reasoning versus instruction-oriented models, and dense versus mixture-of-experts architectures.

Points to cover:

- Model size is a first-order factor in latency and energy consumption, but larger models do not always provide proportional quality gains for every agent step.
- Thinking or reasoning-oriented models are optimized to spend more inference compute on intermediate reasoning. They may improve difficult SQL generation or multi-step analytical interpretation, but can also increase latency and energy.
- Non-thinking or instruction-oriented models often produce shorter, more direct outputs. They may be more efficient for simple routing, formatting, or chart-generation tasks, but may fail more often on complex reasoning cases.
- Dense models activate most parameters during inference. Their compute cost tends to scale directly with total parameter count.
- Mixture-of-Experts (MoE) models contain many parameters but activate only a subset of experts per token. This can produce a different quality/energy profile: large representational capacity with lower active compute than an equally large dense model.
- Local inference makes these differences especially important because GPU energy, memory pressure, throughput, and latency are observable in the experimental setup.
- The same model may not be optimal for every agent step. SQL generation, textual analysis, and visualization code generation may prefer different trade-offs between reasoning depth, output determinism, and cost.

Connection to this work:
The experiments compare baseline behavior across models and then study how each model responds to per-step parameter changes.
The model set should be framed by role: small thinking models as low-cost reasoning candidates, larger thinking or MoE models as capacity/efficiency candidates, and larger non-thinking dense models as instruction-following controls.
This allows the thesis to discuss not only absolute quality, but quality per kWh and quality per second.

## 4. Agent Orchestration and Inference-Time Reasoning Strategies

Target length: about one page.

Purpose:
Explain why workflow structure and inference-time controls matter, not only model choice.
This prepares the reader for the thesis experiments on per-step hyperparameters, token budgets, best-of-N, and Chain-of-Thought refinement.

Points to cover:

- Agent frameworks organize LLM calls, tools, and state transitions into explicit workflows.
- Graph-based orchestration is useful when an agent must decide which step to execute next and when to stop.
- LangGraph is relevant because it represents an LLM application as a stateful graph with nodes, conditional edges, loops, and shared state.
- Chain-of-Thought prompting encourages the model to make intermediate reasoning explicit before producing a final answer.
- Self-consistency and best-of-N strategies generate multiple candidate outputs and select one through consensus or an evaluator.
- Inference-time compute can be allocated unevenly across steps. SQL generation, textual analysis, and visualization generation have different failure modes and may benefit from different sampling or refinement strategies.
- Token budget is an important control. Too few tokens can truncate reasoning or code; too many tokens may increase latency and energy, especially for reasoning models.
- Thinking models make token-budget and refinement choices especially important because additional output budget may be used for reasoning rather than only for the final answer.
- Best-of-N and CoT should be interpreted as deliberate compute expansion: they may increase quality, but they also multiply or extend LLM calls, so their benefit must be measured against energy and latency.
- The agent should therefore expose per-step parameters such as temperature, top-p, top-k, maximum tokens, best-of-N candidate count, and CoT refinement iterations.

Connection to this work:
The thesis uses a LangGraph workflow with distinct nodes for tool selection, lookup, analysis, and visualization.
Each step has its own configurable `StepConfig`, allowing experiments that vary one part of the agent while keeping the others fixed.
This supports step-isolated analysis rather than treating the entire agent as an opaque black box.

## 5. Evaluation of Data-Agent Outputs

Target length: about one page.

Purpose:
Show that evaluating a data-analysis agent is harder than evaluating a single chat response.
The output is multimodal and each component needs a different quality signal: table correctness for lookup, factual coverage for analysis, and chart/code correctness for visualization.

Points to cover:

- SQL/data retrieval quality should be evaluated on the returned table, because many SQL strings can be semantically equivalent.
- Exact table comparison is useful but brittle. It can fail when the model returns correct values with different aliases, column order, boolean labels, or long-vs-wide layouts.
- Graded table similarity is more appropriate for benchmark analysis when outputs are semantically close but not syntactically identical.
- Textual analysis should be judged for factual accuracy, numerical correctness, and coverage of the relevant conclusions, not by surface-level n-gram overlap alone.
- LLM-as-judge evaluation is useful for semantic scoring, but it introduces judge bias, parse failures, and model-dependent variability that should be reported.
- Visualization quality requires checking chart type, axes, grouping, title, data mapping, functional equivalence, and whether generated plotting code executes.
- Evaluation should include failure diagnostics, not only aggregate scores, because agent failures are often step-specific.

Connection to this work:
The benchmark contains ground-truth SQL, expected CSV data, reference analysis text, chart configuration, and chart code.
The lookup step is evaluated with a deterministic table similarity score, while analysis and visualization use judge-based scores.
The ground truth is used only for tracking benchmark quality and is not exposed to the agent during generation or best-of-N selection.

## 6. Efficiency-Aware Benchmarking of Local LLM Agents

Target length: about three quarters of a page.

Purpose:
Place the thesis in the literature on efficient and sustainable AI, but keep the focus on inference-time agent execution rather than model training.

Points to cover:

- LLM quality should be interpreted together with computational cost, latency, and energy consumption.
- For local models, inference energy is especially relevant because the GPU is directly under experimental control.
- Energy-performance comparisons should account for model family. A small thinking model, a large dense non-thinking model, and a large MoE thinking model may occupy very different points on the quality/energy frontier.
- CodeCarbon is a practical software tool for estimating CPU, GPU, RAM energy, and CO2 emissions during code execution.
- Run-level energy is useful, but per-step energy is more informative in modular agents because different workflow nodes may dominate cost.
- Energy-aware evaluation should report not only the best-quality configuration, but also Pareto-efficient configurations that preserve most of the quality at lower time or energy cost.
- Useful derived metrics include accuracy per kWh, accuracy gain per second, delta quality versus baseline, and delta energy versus baseline.
- Sequential single-GPU execution is a reasonable experimental choice when the goal is controlled comparison between configurations rather than load testing.

Connection to this work:
The experiments measure elapsed time, total energy, GPU energy, per-step LLM energy, and quality metrics.
The design compares baseline models, step-specific parameter sensitivity, maximum-token ladders, compute expansion through best-of-N and CoT, and final Pareto configurations.
The final discussion should distinguish highest-quality configurations from efficient configurations that achieve similar quality with lower energy.

## 7. Synthesis and Research Gap

Target length: one short final paragraph.

The State of the Art should end by synthesizing the gap, not by listing tools.
A possible direction:

Existing research provides the building blocks for LLM-based data agents: text-to-SQL methods for database grounding, graph-based orchestration for multi-step tool use, model families with different inference-time behavior, prompting and self-consistency methods for improving reasoning, semantic evaluation strategies for generated text and code, and software-based tools for estimating energy consumption.
However, these aspects are often studied separately.
This thesis combines them in a single reproducible setting: a multi-step data-analysis agent evaluated across data correctness, textual analysis quality, visualization quality, latency, and energy consumption, with deterministic step-isolated experiments that identify which model families and which forms of additional inference compute improve the energy-performance trade-off.

## Notes for Later Implementation

- Keep direct comparison to specific prior theses out of the main structure of the State of the Art.
- Mention project lineage only briefly in the introduction or methodology if needed.
- Do not spend much space describing Phoenix or JMeter, since they are not used in this thesis.
- Use citations to support the general research areas: LangGraph/agent orchestration, Chain-of-Thought, self-consistency, LLM-as-judge evaluation, text-to-SQL, dense and mixture-of-experts LLMs, reasoning/thinking model behavior, visualization/code generation evaluation, and CodeCarbon or energy-aware AI.
