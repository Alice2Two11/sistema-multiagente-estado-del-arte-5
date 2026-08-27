from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime,timezone
from src.state.fingerprints import build_stage_fingerprints
from src.contracts.agent_result import *
@dataclass(frozen=True)
class DraftWritingTransactionResult:prepare:object;agent_result:object;persisted_result_path:str;committed_state:object
def build_draft_fingerprints(agent_input):return build_stage_fingerprints(input_data={'experiment_id':agent_input.experiment_id,'run_id':agent_input.run_id,'stage_name':agent_input.stage_name,'attempt_number':agent_input.attempt_number},config_data=dict(agent_input.policy),dependencies_data={k:v.to_dict() for k,v in agent_input.dependencies.items()})
def execute_draft_transaction(*,store,agent,agent_input,observations=None):
    prep=store.prepare_execution(target_stage=agent_input.stage_name,intended_action='EXECUTE_DRAFT_WRITING',attempt_number=agent_input.attempt_number);result=agent.execute(agent_input);p=store.persist_agent_result(prep.decision_id,result);s=store.commit_execution(decision_id=prep.decision_id,result=result,stage_name=agent_input.stage_name,fingerprints=build_draft_fingerprints(agent_input),observations=dict(observations or {}));return DraftWritingTransactionResult(prep,result,str(p),s)
def resolve_draft_resume(*,store,agent_input,observations=None):return store.resolve_resume(stage_name=agent_input.stage_name,fingerprints=build_draft_fingerprints(agent_input),observations=dict(observations or {}))
