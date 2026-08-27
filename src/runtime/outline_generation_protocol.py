from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime,timezone
from src.state.fingerprints import build_stage_fingerprints
from src.contracts.agent_result import *
@dataclass(frozen=True)
class OutlineTransactionResult: prepare:object;agent_result:object;persisted_result_path:str;committed_state:object
def build_outline_fingerprints(agent_input):return build_stage_fingerprints(input_data={'experiment_id':agent_input.experiment_id,'run_id':agent_input.run_id,'stage_name':agent_input.stage_name,'attempt_number':agent_input.attempt_number},config_data=dict(agent_input.policy),dependencies_data={k:v.to_dict() for k,v in agent_input.dependencies.items()})
def execute_outline_transaction(*,store,agent,agent_input,observations=None):
 prep=store.prepare_execution(target_stage=agent_input.stage_name,intended_action='EXECUTE_OUTLINE_GENERATION',attempt_number=agent_input.attempt_number);result=agent.execute(agent_input);p=store.persist_agent_result(prep.decision_id,result);s=store.commit_execution(decision_id=prep.decision_id,result=result,stage_name=agent_input.stage_name,fingerprints=build_outline_fingerprints(agent_input),observations=dict(observations or {}));return OutlineTransactionResult(prep,result,str(p),s)
def execute_outline_runtime_transaction(*,store,build_execution,attempt_number=1,observations=None):
 prep=store.prepare_execution(target_stage='05_generador_esquema',intended_action='EXECUTE_OUTLINE_GENERATION',attempt_number=attempt_number)
 try:agent,ai=build_execution();result=agent.execute(ai);fp=build_outline_fingerprints(ai)
 except Exception as exc:
  now=datetime.now(timezone.utc).isoformat();code='CREDENTIAL_NOT_FOUND' if 'OPENAI_API_KEY' in str(exc) else ('OUTLINE_INPUT_NOT_FOUND' if isinstance(exc,FileNotFoundError) else 'RUNTIME_DEPENDENCY_FAILED');result=AgentResult(execution_status=ExecutionStatus.FAILED,quality_status=QualityStatus.REJECTED,decision=DecisionInfo(code='OUTLINE_RUNTIME_FAILED',rationale='Falló preparación 05.'),quality_metrics={'scientific':{},'technical':{}},warnings=(AgentWarning(code=code,severity=WarningSeverity.ERROR,blocking=True,message=str(exc)),),failure_reason_codes=(code,),requested_transition=RequestedTransition(action=TransitionAction.HALT_STAGE,target_stage=None,reason_code=code,requires_human_confirmation=False),output_artifacts={},tool_usage=ToolUsage(),attempt_number=attempt_number,started_at=now,completed_at=now,error={'type':type(exc).__name__,'message':str(exc),'stage':'05_generador_esquema'});fp=build_stage_fingerprints(input_data={'stage_name':'05_generador_esquema','attempt_number':attempt_number},config_data={'runtime_resolution':'FAILED'},dependencies_data={})
 p=store.persist_agent_result(prep.decision_id,result);s=store.commit_execution(decision_id=prep.decision_id,result=result,stage_name='05_generador_esquema',fingerprints=fp,observations=dict(observations or {}));return OutlineTransactionResult(prep,result,str(p),s)
def resolve_outline_resume(*,store,agent_input,observations=None):return store.resolve_resume(stage_name=agent_input.stage_name,fingerprints=build_outline_fingerprints(agent_input),observations=dict(observations or {}))
