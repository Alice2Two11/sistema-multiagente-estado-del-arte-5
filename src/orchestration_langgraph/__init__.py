"""MAIN 5: LangGraph, único motor de orquestación del sistema (desde el
Bloque 6 -- la máquina de estados propia, ``pipeline_orchestrator.py``, se
retiró definitivamente una vez validada la equivalencia estructural).

Consume exclusivamente ``src/orchestration/stage_execution.py``,
``src/orchestration/stage_constructors.py`` y
``src/orchestration/pending_reconciliation.py``. Ver
``pipeline_graph.py::run_pipeline_via_langgraph`` como el único punto de
entrada de orquestación global del sistema.
"""
