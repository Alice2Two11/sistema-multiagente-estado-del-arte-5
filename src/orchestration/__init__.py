"""Orquestador de pipeline construido sobre los componentes migrados en ``src/``.

Este paquete no reemplaza ``StateStore`` ni el contrato PREPARE/EXECUTE/COMMIT/
RESUME ya implementado por cada etapa; únicamente los invoca en el orden
correcto, etapa por etapa, reutilizando exactamente las mismas funciones que
ya usan los notebooks operativos.
"""
