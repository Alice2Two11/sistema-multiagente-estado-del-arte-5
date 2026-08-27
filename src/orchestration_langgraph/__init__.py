"""MAIN 5: variante experimental de orquestación con LangGraph.

Consume exclusivamente ``src/orchestration/stage_execution.py``,
``src/orchestration/stage_constructors.py`` y
``src/orchestration/pending_reconciliation.py`` -- nunca
``src/orchestration/pipeline_orchestrator.py``. Ver
``pipeline_graph.py::run_pipeline_via_langgraph`` como punto de entrada
equivalente a ``pipeline_orchestrator.run_pipeline``.
"""
