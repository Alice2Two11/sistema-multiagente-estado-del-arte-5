from __future__ import annotations

def build_repair_plan(codes,data=None):
    plan=[]
    mapping={
      'TITLE_MISMATCH':'CORRECT_TITLE_FROM_CLOSED_MAP','INVALID_REPRESENTATIVE_SOURCE':'REMOVE_INVALID_REFERENCE',
      'MISSING_THEME_EVIDENCE':'REGENERATE_THEME_ONLY','UNSUPPORTED_THEME':'REGENERATE_THEME_ONLY',
      'MISSING_GAP_EVIDENCE':'REGENERATE_OR_DROP_GAP_ONLY','UNSUPPORTED_RESEARCH_GAP':'REGENERATE_OR_DROP_GAP_ONLY',
      'MISSING_COMPARATIVE_EVIDENCE':'REGENERATE_DIMENSION_ONLY','INVALID_COMPARATIVE_DIMENSION':'REGENERATE_DIMENSION_ONLY',
'STRUCTURE_TOO_SHORT':'REGENERATE_STRUCTURE_ONLY','STRUCTURE_TOO_LONG':'REGENERATE_STRUCTURE_ONLY'}
    for c in dict.fromkeys(codes): plan.append({'reason_code':c,'strategy':mapping.get(c,'REPAIR_AFFECTED_BLOCK_ONLY')})
    return plan

def apply_deterministic_repairs(data,title_map,valid_sources):
    repairs=[]; out=data
    for t in out.get('themes',[]):
        clean=[]
        for p in t.get('representative_papers',[]) or []:
            if isinstance(p,dict) and p.get('source_filename') in valid_sources:
                src=p['source_filename']; expected=title_map.get(src,'')
                if expected and p.get('title')!=expected: p=dict(p);p['title']=expected;repairs.append({'type':'TITLE_REPAIRED','source_filename':src})
                clean.append(p)
            elif isinstance(p,str) and p in valid_sources: clean.append({'source_filename':p,'title':title_map.get(p,'')})
            else: repairs.append({'type':'INVALID_REFERENCE_REMOVED','value':repr(p)})
        t['representative_papers']=clean
    return out,repairs
