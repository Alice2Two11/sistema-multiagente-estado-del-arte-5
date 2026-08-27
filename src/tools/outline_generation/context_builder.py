from __future__ import annotations
import json,re
import pandas as pd
def clean_value(value,max_chars=1800):
 if value is None or (isinstance(value,float) and pd.isna(value)): return ''
 if isinstance(value,(list,dict)): text=json.dumps(value,ensure_ascii=False)
 else: text=str(value)
 return re.sub(r'\s+',' ',text).strip()[:max_chars]
def df_to_records_clean(df,max_chars=1800,max_rows=None):
 if max_rows is not None: df=df.head(max_rows)
 return [{c:clean_value(r.get(c),max_chars) for c in df.columns} for _,r in df.iterrows()]
KB_FIELDS=['source_filename','title','paper_type','research_problem','objective','task_type','target_domain','target_variable_or_object','temporal_horizon_or_scope','methods_or_models','method_families','datasets_or_case_study','input_variables_or_data_sources','evaluation_metrics','reported_best_method_or_model','main_results','limitations_or_gaps','contributions','contribution','relevance_level','relevance_reason','quant_methods_or_models_03B','quant_model_families_03B','quant_dataset_names_03B','quant_case_studies_03B','quant_data_types_03B','quant_temporal_resolution_03B','quant_spatial_resolution_03B','quant_forecast_horizon_03B','quant_metrics_03B','quant_values_03B','quant_result_count_03B','quant_literal_values_confirmed_03B']
def build_outline_context(bundle,agent_input):
 p=agent_input.policy; kb=bundle['kb']; fields=[c for c in KB_FIELDS if c in kb.columns]; m=int(p['max_field_chars'])
 return {'experiment_id':agent_input.experiment_id,'experiment_profile':p.get('experiment_profile',{}),'topic_profile':p.get('topic_profile',{}),'generation_profile':p.get('generation_profile',{}),'rag_policy':p.get('rag_policy',{}),'section_constraints':{'min_sections':p.get('min_sections',4),'max_sections':p.get('max_sections',5),'length_profile':p.get('length_profile',''),'output_language':p.get('output_language','español académico'),'writing_mode':p.get('writing_mode','critical'),'focus_mode':p.get('focus_mode','balanced'),'citation_style':p.get('citation_style','IEEE')},'thematic_analysis':bundle['thematic'],'thematic_validation_report':bundle['validation'],'themes_summary':df_to_records_clean(bundle['themes'],m),'research_gaps':df_to_records_clean(bundle['gaps'],m),'suggested_structure_from_thematic_analysis':bundle['structure'] if isinstance(bundle['structure'],list) else bundle['structure'].get('suggested_state_of_art_structure',bundle['structure']),'comparative_table':df_to_records_clean(bundle['comparative'],m),'scientific_kb_final_compact':df_to_records_clean(kb[fields],m),'valid_source_filenames':sorted(kb.source_filename.astype(str).tolist())}
