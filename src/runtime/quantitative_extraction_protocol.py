from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping
from src.contracts.agent_input import AgentInput
from src.contracts.agent_result import AgentResult
from src.state.fingerprints import build_stage_fingerprints
from src.state.pipeline_state import PipelineState
from src.state.state_store import StateStore, PrepareResult, ResumeResolution

@dataclass(frozen=True)
class QuantitativeTransactionResult:
    prepare: PrepareResult
    agent_result: AgentResult
    persisted_result_path: str
    committed_state: PipelineState

def build_quantitative_fingerprints(agent_input):
    return build_stage_fingerprints(input_data={'experiment_id':agent_input.experiment_id,'run_id':agent_input.run_id,'stage_name':agent_input.stage_name,'attempt_number':agent_input.attempt_number},config_data=dict(agent_input.policy),dependencies_data={k:v.to_dict() for k,v in agent_input.dependencies.items()})

def execute_quantitative_transaction(*, store:StateStore, capability:Any, agent_input:AgentInput, observations:Mapping[str,Any]|None=None):
    prepared=store.prepare_execution(target_stage=agent_input.stage_name,intended_action='EXECUTE_QUANTITATIVE_EXTRACTION',attempt_number=agent_input.attempt_number)
    result=capability.execute(agent_input)
    persisted=store.persist_agent_result(prepared.decision_id,result)
    state=store.commit_execution(decision_id=prepared.decision_id,result=result,stage_name=agent_input.stage_name,fingerprints=build_quantitative_fingerprints(agent_input),observations=dict(observations or {}))
    return QuantitativeTransactionResult(prepared,result,str(persisted),state)

def execute_quantitative_runtime_transaction(*, store:StateStore, build_execution:Any, observations=None):
    prepared=store.prepare_execution(target_stage='03B_extraccion_cuantitativa_kb',intended_action='EXECUTE_QUANTITATIVE_EXTRACTION',attempt_number=1)
    try:
        capability,agent_input=build_execution(); result=capability.execute(agent_input); fingerprints=build_quantitative_fingerprints(agent_input)
    except Exception as exc:
        from src.capabilities.quantitative_extraction import QuantitativeExtractionCapability, QuantitativeExtractionDependencies
        class Failing:
            def invoke(self,*a,**k): raise exc
        dummy=AgentInput
        # use shared capability failure conversion through a minimal invalid input is not safe; construct directly
        from src.contracts.agent_result import AgentWarning,DecisionInfo,ExecutionStatus,QualityStatus,RequestedTransition,ToolUsage,TransitionAction,WarningSeverity
        safe=str(exc); code=('DEPENDENCY_NOT_FOUND' if isinstance(exc,FileNotFoundError) else ('CREDENTIAL_NOT_FOUND' if 'credencial' in safe.casefold() or 'OPENAI_API_KEY' in safe else 'RUNTIME_DEPENDENCY_FAILED'))
        result=AgentResult(execution_status=ExecutionStatus.FAILED,quality_status=QualityStatus.REJECTED,decision=DecisionInfo(code='QUANTITATIVE_RUNTIME_FAILED',rationale='Falló la preparación de 03B.'),quality_metrics={'technical':{},'scientific':{}},warnings=(AgentWarning(code=code,severity=WarningSeverity.ERROR,blocking=True,message=safe),),failure_reason_codes=(code,),requested_transition=RequestedTransition(action=TransitionAction.HALT_STAGE,target_stage=None,reason_code=code,requires_human_confirmation=False),output_artifacts={},tool_usage=ToolUsage(),attempt_number=1,started_at=datetime.now(timezone.utc).isoformat(),completed_at=datetime.now(timezone.utc).isoformat(),error={'type':type(exc).__name__,'message':safe,'stage':'03B_extraccion_cuantitativa_kb'})
        fingerprints=build_stage_fingerprints(input_data={'stage_name':'03B_extraccion_cuantitativa_kb','attempt_number':1},config_data={'runtime_resolution':'FAILED'},dependencies_data={})
    persisted=store.persist_agent_result(prepared.decision_id,result); state=store.commit_execution(decision_id=prepared.decision_id,result=result,stage_name='03B_extraccion_cuantitativa_kb',fingerprints=fingerprints,observations=dict(observations or {}))
    return QuantitativeTransactionResult(prepared,result,str(persisted),state)

def resolve_quantitative_resume(*,store:StateStore,agent_input:AgentInput,observations=None)->ResumeResolution:
    return store.resolve_resume(stage_name=agent_input.stage_name,fingerprints=build_quantitative_fingerprints(agent_input),observations=dict(observations or {}))
