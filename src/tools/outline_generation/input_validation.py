from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from src.state.fingerprints import sha256_file
REQUIRED=('thematic_analysis_json','thematic_analysis_manifest','thematic_validation_report','themes_summary_csv','research_gaps_csv','comparative_table_papers_csv','kb_final_for_thematic_analysis_csv')
def _json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def validate_outline_dependencies(agent_input):
 if agent_input.stage_name!='05_generador_esquema': raise ValueError('INVALID_CONFIGURATION')
 if agent_input.attempt_number not in (1,2): raise ValueError('INVALID_CONFIGURATION')
 deps=agent_input.dependencies
 for name in REQUIRED:
  if name not in deps: raise FileNotFoundError(f'OUTLINE_INPUT_NOT_FOUND:{name}')
  p=Path(deps[name].path)
  if not p.is_file(): raise FileNotFoundError(f'OUTLINE_INPUT_NOT_FOUND:{name}:{p}')
  if sha256_file(p)!=deps[name].hash: raise ValueError(f'DEPENDENCY_HASH_MISMATCH:{name}')
 thematic=_json(deps['thematic_analysis_json'].path)
 if not isinstance(thematic,dict): raise ValueError('INVALID_THEMATIC_ANALYSIS_INPUT')
 manifest=_json(deps['thematic_analysis_manifest'].path)
 if manifest.get('experiment_id') not in (None,agent_input.experiment_id): raise ValueError('THEMATIC_MANIFEST_MISMATCH')
 if manifest.get('run_id') not in (None,agent_input.run_id): raise ValueError('THEMATIC_MANIFEST_MISMATCH')
 validation=_json(deps['thematic_validation_report'].path)
 kb=pd.read_csv(deps['kb_final_for_thematic_analysis_csv'].path)
 if kb.empty: raise ValueError('EMPTY_OUTLINE_KB')
 for c in ('source_filename','title'):
  if c not in kb.columns: raise ValueError('INVALID_OUTLINE_KB_SCHEMA')
 bad=kb['title'].fillna('').astype(str).str.strip().str.lower().isin(['','error','no especificado','nan'])
 if bad.any(): raise ValueError('INVALID_SOURCE_TITLE')
 if kb['source_filename'].astype(str).str.contains('ground.?truth',case=False,regex=True).any(): raise ValueError('GROUND_TRUTH_POLICY_VIOLATION')
 structure=None
 if 'suggested_structure_json' in deps and Path(deps['suggested_structure_json'].path).is_file():
  structure=_json(deps['suggested_structure_json'].path)
 elif 'suggested_structure_csv' in deps and Path(deps['suggested_structure_csv'].path).is_file():
  structure=pd.read_csv(deps['suggested_structure_csv'].path).to_dict(orient='records')
 else: raise FileNotFoundError('OUTLINE_INPUT_NOT_FOUND:suggested_structure')
 return {'thematic':thematic,'manifest':manifest,'validation':validation,'themes':pd.read_csv(deps['themes_summary_csv'].path),'gaps':pd.read_csv(deps['research_gaps_csv'].path),'comparative':pd.read_csv(deps['comparative_table_papers_csv'].path),'kb':kb,'structure':structure}
