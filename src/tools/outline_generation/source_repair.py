from __future__ import annotations
from difflib import get_close_matches
import re
from src.tools.shared.section_source_requirement import section_is_source_free_organizational
def as_list(v): return [] if v is None else (v if isinstance(v,list) else [v])
def norm_text(v): return re.sub(r'\s+',' ',str(v or '').strip().lower())
def section_allows_empty_papers(sec):
 """Única definición de qué secciones pueden legítimamente no citar
 papers -- reutiliza classify_section_source_requirement (``src/
 tools/shared/section_source_requirement.py``), la MISMA fuente que
 consume Stage 06 para su gate de evidencia, para que ambas etapas
 nunca puedan divergir."""
 return section_is_source_free_organizational(sec)
def repair_outline_sources(outline,valid_sources,source_to_title,title_to_source,cutoff=0.55):
 repairs=[]; unresolved=[]; titles=list(title_to_source)
 for sec in as_list(outline.get('sections',[])):
  if not isinstance(sec,dict): continue
  clean=[]
  for p in as_list(sec.get('papers_to_use',[])):
   if not isinstance(p,dict): continue
   source=str(p.get('source_filename','')).strip(); title=str(p.get('title','')).strip()
   if source in valid_sources:
    if not title:p['title']=source_to_title.get(source,'')
    clean.append(p); continue
   matches=get_close_matches(title,titles,n=1,cutoff=cutoff)
   if matches:
    mt=matches[0]; ns=title_to_source[mt]; repairs.append({'section_id':sec.get('section_id'),'old_source_filename':source,'generated_title':title,'matched_title':mt,'new_source_filename':ns}); p['source_filename']=ns;p['title']=mt;clean.append(p)
   else: unresolved.append({'section_id':sec.get('section_id'),'source_filename':source,'title':title})
  sec['papers_to_use']=clean
 return repairs,unresolved
def repair_coverage_summary(outline,valid_sources,source_to_title,title_to_source,cutoff=0.55):
 repairs=[];unresolved=[];clean=[];titles=list(title_to_source)
 for item in as_list(outline.get('paper_coverage_summary',[])):
  if not isinstance(item,dict):continue
  source=str(item.get('source_filename','')).strip();title=str(item.get('title','')).strip()
  if source in valid_sources:
   if not title:item['title']=source_to_title.get(source,'')
   clean.append(item);continue
  matches=get_close_matches(title,titles,n=1,cutoff=cutoff)
  if matches:
   mt=matches[0];ns=title_to_source[mt];repairs.append({'old_source_filename':source,'generated_title':title,'matched_title':mt,'new_source_filename':ns});item['source_filename']=ns;item['title']=mt;clean.append(item)
  else:unresolved.append({'source_filename':source,'title':title})
 outline['paper_coverage_summary']=clean;return repairs,unresolved


def _theme_papers(theme,valid_sources):
 out=[]
 for p in as_list(theme.get('representative_papers',[]) if isinstance(theme,dict) else []):
  if not isinstance(p,dict):continue
  source=str(p.get('source_filename','')).strip()
  # Fail-closed: un paper referenciado por 04 que ya no existe en el
  # conjunto validado de fuentes (valid_sources, el mismo que usa el
  # resto de la validación de 05) NUNCA se propaga -- nunca se asume
  # que sigue siendo válido.
  if source and source in valid_sources:
   out.append({'source_filename':source,'title':p.get('title') or ''})
 return out


def repair_empty_section_papers(outline,themes,valid_sources):
 """Reparación determinista y segura para secciones NO exentas
 (``section_allows_empty_papers`` es False -- introducción/discusión/
 conclusiones/gaps sí pueden legítimamente no citar papers) que el LLM
 dejó con ``papers_to_use=[]``.

 Usa EXCLUSIVAMENTE ``themes[].representative_papers`` -- el mapping
 real y ya auditado que produjo ``04_agente_analisis_tematico``
 (``thematic_analysis.json``) -- nunca texto libre, nunca similitud
 semántica de contenido, nunca el KB completo sin pasar por ese
 mapping. Regla, en dos niveles, ambos deterministas:

 1. Si el título/tipo de la sección coincide TEXTUALMENTE (contención
    de subcadena normalizada, en cualquier dirección -- nunca fuzzy)
    con el nombre/descripción de un tema, se asignan los papers
    representativos de ESE tema únicamente.
 2. Si ninguna sección coincide con ningún tema (típico de una sección
    transversal/fundamentos que no corresponde a un único tema, ej.
    "datasets y métricas" -- se aplica a TODOS los temas por igual), se
    asigna la UNIÓN de los papers representativos de TODOS los temas
    -- el conjunto completo que 04 ya estableció como relevante para
    el corpus, nunca un subconjunto adivinado.

 Nunca inventa un ``source_filename`` que no esté en ``valid_sources``.
 Si no hay NINGÚN tema con papers representativos disponibles, la
 sección queda tal cual (vacía) -- fail-closed, la validación posterior
 la seguirá marcando como problemática, correctamente.

 Devuelve ``(repairs, sections_without_evidence)`` -- ``repairs`` es la
 traza auditable (qué sección recibió qué papers y de qué tema/unión),
 ``sections_without_evidence`` son las que no pudieron repararse por
 falta real de evidencia upstream."""

 themes=as_list(themes) if isinstance(themes,list) else []
 valid_themes=[t for t in themes if isinstance(t,dict)]
 union_papers={}
 for t in valid_themes:
  for p in _theme_papers(t,valid_sources):
   union_papers.setdefault(p['source_filename'],p)
 repairs=[];no_evidence=[]
 for sec in as_list(outline.get('sections',[])):
  if not isinstance(sec,dict):continue
  if section_allows_empty_papers(sec):continue
  if as_list(sec.get('papers_to_use',[])):continue
  title_norm=norm_text(sec.get('section_title'))
  matched_theme=None
  for t in valid_themes:
   theme_norm=norm_text(t.get('theme_name') or t.get('theme') or t.get('description'))
   if not theme_norm or not title_norm:continue
   if theme_norm in title_norm or title_norm in theme_norm:
    matched_theme=t;break
  if matched_theme is not None:
   papers=_theme_papers(matched_theme,valid_sources)
   source_desc={'mode':'THEME_TITLE_MATCH','theme_id':matched_theme.get('theme_id') or matched_theme.get('theme_name') or matched_theme.get('theme')}
  else:
   papers=list(union_papers.values())
   source_desc={'mode':'ALL_THEMES_UNION','theme_count':len(valid_themes)}
  if not papers:
   no_evidence.append({'section_id':sec.get('section_id'),'section_title':sec.get('section_title')})
   continue
  sec['papers_to_use']=[dict(p) for p in papers]
  repairs.append({'section_id':sec.get('section_id'),'section_title':sec.get('section_title'),
                   'assigned_paper_count':len(papers),'assigned_source_filenames':sorted(p['source_filename'] for p in papers),
                   **source_desc})
 return repairs,no_evidence
