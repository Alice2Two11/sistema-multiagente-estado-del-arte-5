from __future__ import annotations
from .source_repair import as_list, norm_text, section_allows_empty_papers
def validate_outline(outline,valid_sources,min_sections,max_sections,source_repairs,unresolved_sources,coverage_repairs,unresolved_coverage):
 required_top=['title','objective','narrative_strategy','sections','paper_coverage_summary']; missing_top=[k for k in required_top if k not in outline]
 sections=outline.get('sections',[]); sections=sections if isinstance(sections,list) else []; before=len(sections);trim=False
 if len(sections)>max_sections:
  outline['sections']=sections[:max_sections];sections=outline['sections'];trim=True
  # El trim debe ocurrir ANTES de leer paper_coverage_summary/section
  # references -- de lo contrario used_in_sections queda con IDs de
  # secciones ya eliminadas (S6/S7...), referencias obsoletas que
  # ningún consumidor posterior podría resolver. Se reconstruye aquí
  # mismo, inmediatamente después del trim, sobre el MISMO outline que
  # se persiste -- nunca se deja para un paso separado que podría
  # olvidarse.
  kept_ids={str(s.get('section_id','')).strip() for s in sections if isinstance(s,dict)}
  for item in as_list(outline.get('paper_coverage_summary',[])):
   if not isinstance(item,dict):continue
   used_in=item.get('used_in_sections')
   if isinstance(used_in,list):
    item['used_in_sections']=[sid for sid in used_in if str(sid).strip() in kept_ids]
 n=len(sections); count_valid=min_sections<=n<=max_sections; required=['section_id','section_title','section_type','purpose','key_arguments','evidence_needs']
 missing_rows=[];invalid=[];problematic=[];allowed=[];used=set()
 for sec in sections:
  if not isinstance(sec,dict):continue
  miss=[k for k in required if sec.get(k) in [None,'',[],{}]]; papers=as_list(sec.get('papers_to_use',[])); allow=section_allows_empty_papers(sec)
  if not allow and not papers:miss.append('papers_to_use');problematic.append({'section_id':sec.get('section_id'),'section_title':sec.get('section_title'),'section_type':sec.get('section_type')})
  if allow and not papers:allowed.append({'section_id':sec.get('section_id'),'section_title':sec.get('section_title'),'section_type':sec.get('section_type')})
  for p in papers:
   if isinstance(p,dict):
    src=str(p.get('source_filename','')).strip()
    if src:used.add(src)
    if src and src not in valid_sources:invalid.append({'section_id':sec.get('section_id'),'section_title':sec.get('section_title'),'source_filename':src,'title':p.get('title','')})
  if miss:missing_rows.append({'section_id':sec.get('section_id'),'section_title':sec.get('section_title'),'missing_fields':miss})
 coverage=as_list(outline.get('paper_coverage_summary',[])); cs={str(i.get('source_filename','')).strip() for i in coverage if isinstance(i,dict) and str(i.get('source_filename','')).strip()}; invalid_cov=sorted(x for x in cs if x not in valid_sources)
 ok=not missing_top and count_valid and not missing_rows and not invalid and not invalid_cov and not unresolved_sources and not unresolved_coverage
 return {'stage':'05_generador_esquema','n_sections':n,'section_count_before_trim':before,'sections_trimmed_to_max':trim,'min_sections':min_sections,'max_sections':max_sections,'section_count_valid':count_valid,'missing_top_keys':missing_top,'sections_missing_required_fields':missing_rows,'empty_papers_to_use_allowed':allowed,'empty_papers_to_use_problematic':problematic,'invalid_section_sources':invalid,'invalid_coverage_sources':invalid_cov,'source_repairs':source_repairs,'coverage_repairs':coverage_repairs,'unresolved_sources':unresolved_sources,'unresolved_coverage':unresolved_coverage,'papers_available_count':len(valid_sources),'papers_used_count':len(used),'papers_used':sorted(used),'coverage_summary_count':len(coverage),'coverage_summary_sources_count':len(cs),'validation_ok':ok}
def reason_codes(report):
 codes=[]
 if report['missing_top_keys']:codes.append('INVALID_OUTLINE_SCHEMA')
 if not report['section_count_valid']:codes.append('SECTION_COUNT_OUT_OF_RANGE')
 if report['sections_missing_required_fields']:codes.append('MISSING_SECTION_FIELDS')
 if report['empty_papers_to_use_problematic']:codes.append('MISSING_REQUIRED_SECTION_PAPERS')
 if report['invalid_section_sources']:codes.append('INVALID_SECTION_SOURCE')
 if report['invalid_coverage_sources']:codes.append('INVALID_COVERAGE_SOURCE')
 if report['unresolved_sources']:codes.append('UNRESOLVED_SECTION_SOURCE')
 if report['unresolved_coverage']:codes.append('UNRESOLVED_COVERAGE_SOURCE')
 return tuple(codes)
