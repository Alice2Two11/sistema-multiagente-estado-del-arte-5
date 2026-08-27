from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime,timezone
from src.state.fingerprints import build_stage_fingerprints
from src.contracts.agent_result import *
@dataclass(frozen=True)
class ThematicTransactionResult:
    prepare:object; agent_result:object; persisted_result_path:str; committed_state:object

def build_thematic_fingerprints(agent_input):
    return build_stage_fingerprints(input_data={'experiment_id':agent_input.experiment_id,'run_id':agent_input.run_id,'stage_name':agent_input.stage_name,'attempt_number':agent_input.attempt_number},config_data=dict(agent_input.policy),dependencies_data={k:v.to_dict() for k,v in agent_input.dependencies.items()})
def execute_thematic_transaction(*,store,agent,agent_input,observations=None):
    prep=store.prepare_execution(target_stage=agent_input.stage_name,intended_action='EXECUTE_THEMATIC_ANALYSIS',attempt_number=agent_input.attempt_number)
    result=agent.execute(agent_input); path=store.persist_agent_result(prep.decision_id,result); state=store.commit_execution(decision_id=prep.decision_id,result=result,stage_name=agent_input.stage_name,fingerprints=build_thematic_fingerprints(agent_input),observations=dict(observations or {})); return ThematicTransactionResult(prep,result,str(path),state)
def execute_thematic_runtime_transaction(*,store,build_execution,attempt_number=1,observations=None):
    prep=store.prepare_execution(target_stage='04_agente_analisis_tematico',intended_action='EXECUTE_THEMATIC_ANALYSIS',attempt_number=attempt_number)
    try:
        agent,agent_input=build_execution(); result=agent.execute(agent_input); fp=build_thematic_fingerprints(agent_input)
    except Exception as exc:
        code='DEPENDENCY_NOT_FOUND' if isinstance(exc,FileNotFoundError) else ('CREDENTIAL_NOT_FOUND' if 'OPENAI_API_KEY' in str(exc) else 'RUNTIME_DEPENDENCY_FAILED'); now=datetime.now(timezone.utc).isoformat()
        result=AgentResult(execution_status=ExecutionStatus.FAILED,quality_status=QualityStatus.REJECTED,decision=DecisionInfo(code='THEMATIC_RUNTIME_FAILED',rationale='Falló la preparación de 04.'),quality_metrics={'technical':{},'scientific':{}},warnings=(AgentWarning(code=code,severity=WarningSeverity.ERROR,blocking=True,message=str(exc)),),failure_reason_codes=(code,),requested_transition=RequestedTransition(action=TransitionAction.HALT_STAGE,target_stage=None,reason_code=code,requires_human_confirmation=False),output_artifacts={},tool_usage=ToolUsage(),attempt_number=attempt_number,started_at=now,completed_at=now,error={'type':type(exc).__name__,'message':str(exc),'stage':'04_agente_analisis_tematico'}); fp=build_stage_fingerprints(input_data={'stage_name':'04_agente_analisis_tematico','attempt_number':attempt_number},config_data={'runtime_resolution':'FAILED'},dependencies_data={})
    p=store.persist_agent_result(prep.decision_id,result); s=store.commit_execution(decision_id=prep.decision_id,result=result,stage_name='04_agente_analisis_tematico',fingerprints=fp,observations=dict(observations or {})); return ThematicTransactionResult(prep,result,str(p),s)
def resolve_thematic_resume(*,store,agent_input,observations=None): return store.resolve_resume(stage_name=agent_input.stage_name,fingerprints=build_thematic_fingerprints(agent_input),observations=dict(observations or {}))


def execute_deterministic_thematic_repair_transaction(*, store, output_dir, attempt_number=2, observations=None):
    """Persist a deterministic repair as its own transaction without invoking an LLM."""
    from src.tools.thematic_analysis.deterministic_repair import execute_deterministic_thematic_repair
    from src.state.fingerprints import build_stage_fingerprints
    prep=store.prepare_execution(target_stage='04_agente_analisis_tematico',intended_action='DETERMINISTIC_THEMATIC_FLATTENING_REPAIR',attempt_number=attempt_number)
    result=execute_deterministic_thematic_repair(output_dir=output_dir,attempt_number=attempt_number)
    path=store.persist_agent_result(prep.decision_id,result)
    fingerprints=build_stage_fingerprints(
        input_data={'stage_name':'04_agente_analisis_tematico','attempt_number':attempt_number,'repair_mode':'deterministic_thematic_flattening'},
        config_data={'openai_called':False,'alias_version':'v16_deterministic_alias_mapping_1'},
        dependencies_data={k:v.to_dict() for k,v in result.output_artifacts.items() if k in {'analysis','raw','kb_final','kb_excluded'}},
    )
    state=store.commit_execution(decision_id=prep.decision_id,result=result,stage_name='04_agente_analisis_tematico',fingerprints=fingerprints,observations=dict(observations or {}))
    return ThematicTransactionResult(prep,result,str(path),state)
