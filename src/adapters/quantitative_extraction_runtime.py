from __future__ import annotations
import importlib.util, json, os
from pathlib import Path
from typing import Any
from src.contracts.agent_input import AgentInput, AgentContext, ArtifactReference, ExecutionMode
from src.io.credentials import load_runtime_credential
from src.state.fingerprints import sha256_file
from src.config.quantitative_extraction_policy_config import STAGE_NAME, validate_quantitative_policy
from src.capabilities.quantitative_extraction import QuantitativeExtractionCapability, QuantitativeExtractionDependencies

def _load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def load_quantitative_configuration(project_dir:str|Path):
    root=Path(project_dir).resolve(); active=json.loads((root/'active_experiment.json').read_text(encoding='utf-8'))
    exp=active['active_experiment_id']; exp_dir=root/exp; outputs=exp_dir/'05_outputs'; kb=outputs/'02_scientific_knowledge_base'; chunks=exp_dir/'03_chunks'/'chunks_clean_for_rag.csv'
    policy=validate_quantitative_policy(active.get('quantitative_extraction_policy',{}))
    return {'project_dir':root,'experiment_id':exp,'run_id':active.get('run_id',exp),'experiment_dir':exp_dir,'output_dir':kb,'model':active['openai_model'],'policy':policy,'paths':{'scientific_knowledge_base_csv':kb/'scientific_knowledge_base.csv','scientific_knowledge_base_jsonl':kb/'scientific_knowledge_base.jsonl','scientific_extraction_manifest':outputs/'01_scientific_extraction'/'scientific_extraction_manifest.json','chunks_clean_for_rag_csv':chunks}}

def build_quantitative_agent_input(configuration):
    deps={name:ArtifactReference(path=str(path),hash=sha256_file(path)) for name,path in configuration['paths'].items() if path.exists()}
    return AgentInput(experiment_id=configuration['experiment_id'],run_id=configuration['run_id'],stage_name=STAGE_NAME,attempt_number=1,mode=ExecutionMode.FULL_RUN,agent_context=AgentContext(allowed_tools=('llm','source_chunks','atomic_write'),output_directory=str(configuration['output_dir']),runtime_resources={'model':configuration['model']}),dependencies=deps,policy=configuration['policy'])

def build_quantitative_capability(configuration, *, llm_factory:Any=None, human_message_factory:Any=None, json_parser:Any=None):
    repair_mode = bool(configuration['policy'].get('deterministic_flattening_repair', False))
    if repair_mode:
        return QuantitativeExtractionCapability(
            QuantitativeExtractionDependencies(
                llm=None,
                human_message_factory=lambda **kwargs: kwargs,
                json_parser=lambda text: json.loads(text),
            )
        )
    load_runtime_credential('OPENAI_API_KEY',project_dir=configuration['project_dir'])
    if llm_factory is None:
        from langchain_openai import ChatOpenAI
        llm_factory=ChatOpenAI
    if human_message_factory is None:
        from langchain_core.messages import HumanMessage
        human_message_factory=HumanMessage
    if json_parser is None:
        module=_load_module('project_llm_utils_03b',configuration['project_dir']/'src'/'llm_utils.py'); json_parser=module.parse_json_safely
    llm=llm_factory(model=configuration['model'],temperature=configuration['policy']['temperature'])
    return QuantitativeExtractionCapability(QuantitativeExtractionDependencies(llm=llm,human_message_factory=human_message_factory,json_parser=json_parser))
