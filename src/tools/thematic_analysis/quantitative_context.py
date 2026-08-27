from __future__ import annotations
import pandas as pd

def integrate_quantitative_context(df, bundle):
    result=df.copy(); meta={'quantitative_context_available':False,'quantitative_context_used':False,'quantitative_context_quality_status':None}
    if not bundle.quantitative_manifest: return result,meta,[]
    meta['quantitative_context_available']=True; meta['quantitative_context_quality_status']=bundle.quantitative_manifest.get('quality_status') or bundle.quantitative_manifest.get('result',{}).get('quality_status')
    warnings=[]
    if meta['quantitative_context_quality_status']=='APPROVED_WITH_WARNINGS': warnings.append('QUANTITATIVE_CONTEXT_USED_WITH_WARNINGS')
    comp=pd.read_csv(bundle.quantitative_files['quantitative_comparative_table'])
    summary=pd.read_csv(bundle.quantitative_files['dataset_technique_summary'])
    if 'source_filename' in comp.columns and not comp.empty:
        agg=comp.groupby('source_filename').agg(quant_result_count_03B=('source_filename','size'))
        result=result.merge(agg,left_on='source_filename',right_index=True,how='left')
    if 'source_filename' in summary.columns and not summary.empty:
        cols=[c for c in summary.columns if c!='source_filename']
        result=result.merge(summary[['source_filename']+cols],on='source_filename',how='left',suffixes=('','_03B'))
    meta['quantitative_context_used']=True
    return result,meta,warnings
