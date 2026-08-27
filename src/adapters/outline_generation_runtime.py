from __future__ import annotations
from dataclasses import dataclass
@dataclass
class OutlineGenerationRuntime:
 invoke_fn: object
 parse_fallback: object|None=None
 def invoke(self,prompt): return self.invoke_fn(prompt)
 def parse(self,raw):
  from src.tools.outline_generation.response_parsing import extract_first_valid_json
  return extract_first_valid_json(raw,self.parse_fallback)
def build_openai_outline_runtime(model, *, temperature, project_dir=None, llm_factory=None, human_message_factory=None):
 from src.io.credentials import load_runtime_credential
 load_runtime_credential('OPENAI_API_KEY', project_dir=project_dir)
 if llm_factory is None:
  from langchain_openai import ChatOpenAI
  llm_factory=ChatOpenAI
 if human_message_factory is None:
  from langchain_core.messages import HumanMessage
  human_message_factory=HumanMessage
 llm=llm_factory(model=model,temperature=temperature)
 return OutlineGenerationRuntime(lambda prompt: llm.invoke([human_message_factory(content=prompt)]).content)

from pathlib import Path
import json
from src.contracts.agent_input import AgentInput,AgentContext,ArtifactReference,ExecutionMode,PreviousAttemptSummary
from src.state.fingerprints import sha256_file,fingerprint_mapping
from src.config.outline_generation_policy_config import get_outline_generation_policy
from src.agents.outline_generation_agent import OutlineGenerationAgent

def resolve_pipeline_state_path(project_dir,experiment_id):
 root=Path(project_dir).resolve();exp=root/experiment_id;canonical=exp/'05_outputs'/'00_orchestrator_planner'/'pipeline_state.json'
 if canonical.is_file():return canonical
 raise FileNotFoundError(f'pipeline_state.json no encontrado en {canonical}')
def load_outline_configuration(project_dir,attempt_number=1):
 root=Path(project_dir).resolve();active=json.loads((root/'active_experiment.json').read_text(encoding='utf-8'));eid=active['active_experiment_id'];exp=root/eid;outputs=exp/'05_outputs';thematic=outputs/'03_thematic_analysis';out=outputs/'04_outline';gp=active.get('generation_profile',{});policy=get_outline_generation_policy(active.get('outline_generation_policy',{}));policy.update({'experiment_profile':active.get('experiment_profile',{}),'topic_profile':active.get('topic_profile',{}),'generation_profile':gp,'rag_policy':active.get('rag_policy',{}),'min_sections':gp['min_sections'],'max_sections':gp['max_sections'],'length_profile':gp.get('length_profile',gp.get('profile','')),'output_language':gp.get('output_language',active.get('experiment_profile',{}).get('output_language','español académico')),'writing_mode':gp.get('writing_mode',gp.get('mode','critical')),'focus_mode':gp.get('focus_mode','balanced'),'citation_style':gp.get('citation_style','IEEE')})
 paths={'thematic_analysis_json':thematic/'thematic_analysis.json','thematic_analysis_manifest':thematic/'thematic_analysis_manifest.json','thematic_validation_report':thematic/'thematic_validation_report.json','themes_summary_csv':thematic/'themes_summary.csv','research_gaps_csv':thematic/'research_gaps.csv','comparative_table_papers_csv':thematic/'comparative_table_papers.csv','kb_final_for_thematic_analysis_csv':thematic/'kb_final_for_thematic_analysis.csv','suggested_structure_json':thematic/'suggested_state_of_art_structure.json','suggested_structure_csv':thematic/'suggested_state_of_art_structure.csv','kb_excluded':thematic/'kb_excluded_from_thematic_analysis.csv'}
 # CONFIG-D (Stage 05): openai_model es responsabilidad de
 # 00_setup_config.ipynb -- 04/07 ya lo exigen fail-closed; 05 usaba
 # active.get("openai_model", "gpt-4o-mini"), un fallback hardcodeado
 # silencioso que aquí queda eliminado.
 openai_model=active.get('openai_model')
 if not isinstance(openai_model,str) or not openai_model.strip():
  raise ValueError("active_experiment.json['openai_model'] debe ser un string no vacío (00_setup_config.ipynb es su autoridad).")
 return {'project_dir':root,'experiment_id':eid,'run_id':active.get('run_id',eid),'attempt_number':int(attempt_number),'model':openai_model,'policy':policy,'output_dir':out,'state_path':resolve_pipeline_state_path(root,eid),'paths':paths,'experiment_dir':exp}
def _previous_outline_attempt(cfg):
 if cfg['attempt_number']!=2:return None
 p=Path(cfg['state_path']);payload=json.loads(p.read_text(encoding='utf-8'));stage=payload.get('stages',{}).get('05_generador_esquema',{})
 if stage.get('requested_transition',{}).get('action')!='RETRY':raise RuntimeError('El intento 2 requiere una transición RETRY persistida.')
 return PreviousAttemptSummary(quality_status=stage.get('quality_status','NEEDS_REVISION'),failure_reason_codes=tuple(stage.get('failure_reason_codes',[])),blocking_warnings=tuple(str(x.get('code','')) for x in stage.get('warnings',[]) if x.get('blocking') and x.get('code')),previous_artifacts={})
def build_outline_agent_input(cfg):
 paths=cfg['paths'];required=('thematic_analysis_json','thematic_analysis_manifest','thematic_validation_report','themes_summary_csv','research_gaps_csv','comparative_table_papers_csv','kb_final_for_thematic_analysis_csv');deps={}
 for n in required:
  p=Path(paths[n]);
  if not p.is_file():raise FileNotFoundError(f'{n} no existe: {p}')
  deps[n]=ArtifactReference(str(p),sha256_file(p))
 if Path(paths['suggested_structure_json']).is_file():deps['suggested_structure_json']=ArtifactReference(str(paths['suggested_structure_json']),sha256_file(paths['suggested_structure_json']))
 elif Path(paths['suggested_structure_csv']).is_file():deps['suggested_structure_csv']=ArtifactReference(str(paths['suggested_structure_csv']),sha256_file(paths['suggested_structure_csv']))
 else:raise FileNotFoundError('No existe estructura sugerida JSON ni CSV.')
 for n in ('kb_excluded',):
  if Path(paths[n]).is_file():deps[n]=ArtifactReference(str(paths[n]),sha256_file(paths[n]))
 signature={'stage':'05_generador_esquema','stage_version':cfg['policy']['stage_version'],'experiment_id':cfg['experiment_id'],'experiment_dir':str(cfg['experiment_dir']),'openai_model':cfg['model'],'topic_profile':cfg['policy'].get('topic_profile',{}),'experiment_profile':cfg['policy'].get('experiment_profile',{}),'generation_profile':cfg['policy'].get('generation_profile',{}),'rag_policy':cfg['policy'].get('rag_policy',{}),'min_sections':cfg['policy'].get('min_sections'),'max_sections':cfg['policy'].get('max_sections'),'output_language':cfg['policy'].get('output_language'),'writing_mode':cfg['policy'].get('writing_mode'),'focus_mode':cfg['policy'].get('focus_mode'),'citation_style':cfg['policy'].get('citation_style'),'paths':{k:v.path for k,v in deps.items()},'hashes':{k:v.hash for k,v in deps.items()},'outline_prompt_version':cfg['policy']['prompt_version'],'outline_schema_version':cfg['policy']['schema_version'],'outline_validation_version':cfg['policy']['validation_version']}
 cfg['policy']['current_fingerprint']=fingerprint_mapping(signature)
 return AgentInput(experiment_id=cfg['experiment_id'],run_id=cfg['run_id'],stage_name='05_generador_esquema',attempt_number=cfg['attempt_number'],mode=ExecutionMode.FULL_RUN,agent_context=AgentContext(allowed_tools=('llm','atomic_write','outline_validation'),output_directory=str(cfg['output_dir']),runtime_resources={'model':cfg['model']}),dependencies=deps,policy=cfg['policy'],previous_attempt=_previous_outline_attempt(cfg))
def build_real_outline_execution(project_dir,attempt_number=1):
 cfg=load_outline_configuration(project_dir,attempt_number);return OutlineGenerationAgent(build_openai_outline_runtime(cfg['model'],temperature=float(cfg['policy']['temperature']),project_dir=cfg['project_dir'])),build_outline_agent_input(cfg),cfg
