from __future__ import annotations
import re

def norm(s): return re.sub(r'\s+',' ',str(s or '').strip()).casefold()
def _paper(p):
    if isinstance(p,str): return p,''
    if isinstance(p,dict): return str(p.get('source_filename','')),str(p.get('title',''))
    return '',''
def validate_references(data,df):
    valid=set(df.source_filename.astype(str)); titles=dict(zip(df.source_filename.astype(str),df.title.astype(str)))
    issues=[]; total=valid_refs=title_total=title_ok=0
    def check(source,title,invalid_code,empty_code=None):
        nonlocal total,valid_refs,title_total,title_ok
        total+=1
        if not source:
            issues.append(empty_code or invalid_code); return False
        if source not in valid: issues.append(invalid_code); return False
        valid_refs+=1
        if title:
            title_total+=1
            if norm(title)==norm(titles.get(source,'')): title_ok+=1
            else: issues.append('TITLE_MISMATCH')
        return True
    for t in data['themes']:
        reps=t.get('representative_papers',[])
        if not reps: issues.append('EMPTY_REPRESENTATIVE_SOURCE')
        for p in reps:
            s,tt=_paper(p); check(s,tt,'INVALID_REPRESENTATIVE_SOURCE','EMPTY_REPRESENTATIVE_SOURCE')
    for s in data['suggested_state_of_art_structure']:
        for src in s.get('recommended_sources',[]) or []: check(str(src),'','INVALID_STRUCTURE_SOURCE')
    for g in data['research_gaps']:
        srcs=g.get('supporting_sources',[]) or []
        if not srcs: issues.append('MISSING_GAP_EVIDENCE')
        for src in srcs: check(str(src),'','INVALID_GAP_SOURCE')
    for d in data['comparative_dimensions']:
        srcs=d.get('relevant_sources',[]) or []
        if not srcs: issues.append('MISSING_COMPARATIVE_EVIDENCE')
        for src in srcs: check(str(src),'','INVALID_COMPARATIVE_SOURCE')
    return issues,{'reference_total':total,'valid_references':valid_refs,'title_total':title_total,'title_matches':title_ok},titles
