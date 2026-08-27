from __future__ import annotations
import json

def clean_value(v,max_chars=3500):
    if v is None:return ''
    s=str(v).strip()
    return s[:max_chars]
def compact_for_thematic_analysis(df,max_field_chars=3500):
    fields=['source_filename','title','research_problem','objective','target_domain','task_type','method_families','methods_or_models','evaluation_metrics','main_results','limitations_or_gaps','contribution','relevance_reason']
    extra=[c for c in df.columns if c.endswith('_03B') or c.startswith('quant_')]
    rows=[]
    for _,r in df.iterrows(): rows.append({c:clean_value(r.get(c,''),max_field_chars) for c in fields+extra if c in df.columns})
    return rows
