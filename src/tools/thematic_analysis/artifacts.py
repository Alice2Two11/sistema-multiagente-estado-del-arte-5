from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from src.io.atomic_write import atomic_write_json, atomic_write_text, atomic_write_csv
from src.contracts.agent_input import ArtifactReference

ARTIFACT_FILENAMES={
'kb_final':'kb_final_for_thematic_analysis.csv','kb_excluded':'kb_excluded_from_thematic_analysis.csv','raw':'thematic_analysis_raw.txt','analysis':'thematic_analysis.json','themes':'themes_summary.csv','gaps':'research_gaps.csv','structure_csv':'suggested_state_of_art_structure.csv','structure_json':'suggested_state_of_art_structure.json','dimensions':'comparative_dimensions.csv','comparative':'comparative_table_papers.csv','validation':'thematic_validation_report.json','manifest':'thematic_analysis_manifest.json'}

def _csv(path,df):
    return atomic_write_csv(path,df.to_dict(orient='records'),fieldnames=list(df.columns))

def build_thematic_tables(data):
    themes=pd.DataFrame(data['themes'],columns=['theme_id','theme_name','description','representative_papers'])
    gaps=pd.DataFrame(data['research_gaps'],columns=['gap_id','description','basis','supporting_sources'])
    structure=pd.DataFrame(data['suggested_state_of_art_structure'],columns=['section_id','section_title','description','recommended_sources'])
    dims=pd.DataFrame(data['comparative_dimensions'],columns=['dimension','description','relevant_sources'])
    return themes,gaps,structure,dims

def thematic_table_counts(data):
    themes,gaps,structure,dims=build_thematic_tables(data)
    return {
        'flattened_theme_rows':len(themes),
        'flattened_gap_rows':len(gaps),
        'flattened_structure_rows':len(structure),
        'flattened_comparative_dimension_rows':len(dims),
        'flattened_theme_semantic_rows':sum(bool(str(row.get('theme_name') or '').strip()) and bool(row.get('representative_papers')) for row in data['themes']),
        'flattened_gap_semantic_rows':sum(bool(str(row.get('description') or '').strip()) and bool(row.get('supporting_sources')) for row in data['research_gaps']),
        'flattened_structure_semantic_rows':sum(bool(str(row.get('section_title') or '').strip()) for row in data['suggested_state_of_art_structure']),
        'flattened_comparative_dimension_semantic_rows':sum(bool(str(row.get('dimension') or '').strip()) and bool(row.get('relevant_sources')) for row in data['comparative_dimensions']),
    }

def write_thematic_artifacts(output_dir,df_final,df_excluded,raw,data,validation,manifest):
    d=Path(output_dir); d.mkdir(parents=True,exist_ok=True); r={}
    themes,gaps,structure,dims=build_thematic_tables(data)
    writers=[('kb_final',lambda p:_csv(p,df_final)),('kb_excluded',lambda p:_csv(p,df_excluded)),('raw',lambda p:atomic_write_text(p,raw)),('analysis',lambda p:atomic_write_json(p,data)),('themes',lambda p:_csv(p,themes)),('gaps',lambda p:_csv(p,gaps)),('structure_csv',lambda p:_csv(p,structure)),('structure_json',lambda p:atomic_write_json(p,data['suggested_state_of_art_structure'])),('dimensions',lambda p:_csv(p,dims)),('comparative',lambda p:_csv(p,df_final)),('validation',lambda p:atomic_write_json(p,validation))]
    for name,fn in writers:
        res=fn(d/ARTIFACT_FILENAMES[name]);r[name]=ArtifactReference(res.path,res.hash)
    manifest=dict(manifest);manifest['outputs']={k:v.to_dict() for k,v in r.items()}
    res=atomic_write_json(d/ARTIFACT_FILENAMES['manifest'],manifest);r['manifest']=ArtifactReference(res.path,res.hash)
    return r

def write_deterministic_thematic_repair_artifacts(output_dir,data,validation,manifest):
    """Regenerate only derived thematic artifacts; preserve the original thematic_analysis.json."""
    d=Path(output_dir); d.mkdir(parents=True,exist_ok=True); r={}
    themes,gaps,structure,dims=build_thematic_tables(data)
    writers=[
        ('themes',lambda p:_csv(p,themes)),
        ('gaps',lambda p:_csv(p,gaps)),
        ('structure_csv',lambda p:_csv(p,structure)),
        ('structure_json',lambda p:atomic_write_json(p,data['suggested_state_of_art_structure'])),
        ('dimensions',lambda p:_csv(p,dims)),
        ('validation',lambda p:atomic_write_json(p,validation)),
    ]
    for name,fn in writers:
        res=fn(d/ARTIFACT_FILENAMES[name]);r[name]=ArtifactReference(res.path,res.hash)
    manifest=dict(manifest)
    existing_outputs={}
    existing_manifest_path=d/ARTIFACT_FILENAMES['manifest']
    if existing_manifest_path.is_file():
        try:
            existing_outputs=json.loads(existing_manifest_path.read_text(encoding='utf-8')).get('outputs',{})
        except Exception:
            existing_outputs={}
    manifest['outputs']={**existing_outputs,**{k:v.to_dict() for k,v in r.items()}}
    res=atomic_write_json(existing_manifest_path,manifest);r['manifest']=ArtifactReference(res.path,res.hash)
    return r
