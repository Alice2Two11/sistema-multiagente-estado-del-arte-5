from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from src.io.atomic_write import atomic_write_json,atomic_write_text,atomic_write_csv
from src.contracts.agent_result import ArtifactReference
from .source_repair import as_list
NAMES=('state_of_art_outline.json','state_of_art_outline_raw.txt','state_of_art_outline.md','outline_sections.csv','outline_paper_mapping.csv','outline_validation_report.json','outline_generation_manifest.json')
def _md(o):
 l=[f"# {o.get('title','Esquema del estado del arte')}",'',f"**Tema:** {o.get('topic','')}",'',f"**Objetivo:** {o.get('objective','')}",'','## Estrategia narrativa','',str(o.get('narrative_strategy','')),'','## Secciones propuestas','']
 for s in as_list(o.get('sections')):
  if not isinstance(s,dict):continue
  l += [f"### {s.get('section_id','')}. {s.get('section_title','')}",'',f"**Tipo:** {s.get('section_type','')}",'',f"**Propósito:** {s.get('purpose','')}",'','**Argumentos clave:**']+[f'- {x}' for x in as_list(s.get('key_arguments'))]+['','**Papers sugeridos:**']
  for p in as_list(s.get('papers_to_use')):
   l.append(f"- `{p.get('source_filename','')}` — {p.get('title','')}. Motivo: {p.get('reason_to_use','')}" if isinstance(p,dict) else f'- {p}')
  l += ['','**Necesidades de evidencia:**']+[f'- {x}' for x in as_list(s.get('evidence_needs'))]+['']
  if s.get('transition_to_next'):l += [f"**Transición:** {s.get('transition_to_next')}",'']
 l += ['## Cobertura de papers','']
 for i in as_list(o.get('paper_coverage_summary')):
  if isinstance(i,dict):l.append(f"- `{i.get('source_filename','')}` — {i.get('title','')}. Secciones: {', '.join(map(str,as_list(i.get('used_in_sections'))))}. Rol: {i.get('role','')}")
 l += ['','## Guías globales de redacción','']+[f'- {x}' for x in as_list(o.get('global_writing_guidelines'))]+['','## Riesgos o advertencias','']+[f'- {x}' for x in as_list(o.get('risks_or_warnings'))]
 return '\n'.join(l)
def write_outline_artifacts(output_dir,outline,raw,validation,manifest):
 d=Path(output_dir);d.mkdir(parents=True,exist_ok=True); rows=[];mapping=[]
 for s in as_list(outline.get('sections')):
  if not isinstance(s,dict):continue
  papers=as_list(s.get('papers_to_use'));rows.append({'section_id':s.get('section_id'),'section_title':s.get('section_title'),'section_type':s.get('section_type'),'purpose':s.get('purpose'),'themes_used':'; '.join(map(str,as_list(s.get('themes_used')))),'key_arguments':'; '.join(map(str,as_list(s.get('key_arguments')))),'num_papers_to_use':len(papers),'evidence_needs':'; '.join(map(str,as_list(s.get('evidence_needs')))),'expected_output':s.get('expected_output'),'transition_to_next':s.get('transition_to_next')})
  for p in papers:
   mapping.append({'section_id':s.get('section_id'),'section_title':s.get('section_title'),'source_filename':p.get('source_filename'),'title':p.get('title'),'reason_to_use':p.get('reason_to_use')} if isinstance(p,dict) else {'section_id':s.get('section_id'),'section_title':s.get('section_title'),'source_filename':'','title':str(p),'reason_to_use':''})
 results={}
 for name,res in [('outline',atomic_write_json(d/NAMES[0],outline)),('raw',atomic_write_text(d/NAMES[1],raw)),('markdown',atomic_write_text(d/NAMES[2],_md(outline))),('sections',atomic_write_csv(d/NAMES[3],rows,fieldnames=list(rows[0]) if rows else ['section_id','section_title','section_type','purpose','themes_used','key_arguments','num_papers_to_use','evidence_needs','expected_output','transition_to_next'])),('mapping',atomic_write_csv(d/NAMES[4],mapping,fieldnames=list(mapping[0]) if mapping else ['section_id','section_title','source_filename','title','reason_to_use'])),('validation',atomic_write_json(d/NAMES[5],validation))]:results[name]=ArtifactReference(res.path,res.hash)
 manifest['outputs']={k:v.to_dict() for k,v in results.items()}; mr=atomic_write_json(d/NAMES[6],manifest);results['manifest']=ArtifactReference(mr.path,mr.hash);return results
