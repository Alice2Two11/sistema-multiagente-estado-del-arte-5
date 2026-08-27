from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Callable
from .input_validation import safe_str
from .prompting import build_quant_prompt

def ensure_quantitative_object(value):
    if isinstance(value,dict): return value
    if isinstance(value,list) and len(value)==1 and isinstance(value[0],dict): return value[0]
    raise ValueError("INVALID_LLM_OUTPUT: la respuesta debe ser un objeto JSON único.")

def extract_quantitative_records(df_kb, *, llm:Any, human_message_factory:Callable[...,Any], json_parser:Callable[[str],Any]):
    results=[]; raw=[]; errors=[]; calls=0
    for _,row in df_kb.iterrows():
        source=safe_str(row.get("source_filename","")); response_text=""
        try:
            response=llm.invoke([human_message_factory(content=build_quant_prompt(row))]); calls+=1
            response_text=str(response.content).strip(); parsed=ensure_quantitative_object(json_parser(response_text))
            parsed.setdefault("source_filename",source); parsed.setdefault("paper_title",safe_str(row.get("title","")))
            for k in ("techniques","datasets","quantitative_results"): parsed.setdefault(k,[])
            parsed.setdefault("notes",""); status="ok"; error=""
        except Exception as exc:
            parsed={"source_filename":source,"paper_title":safe_str(row.get("title","")),"techniques":[],"datasets":[],"quantitative_results":[],"notes":"extraction_error"}
            status="error"; error=str(exc); errors.append({"source_filename":source,"error_type":"LLM_EXTRACTION_ERROR","error_code":"INVALID_LLM_OUTPUT","error_message":str(exc),"raw_path":"","raw_value":response_text,"discarded":True,"created_at":datetime.now(timezone.utc).isoformat()})
        results.append(parsed); raw.append({"status":status,"source_filename":source,"raw_response":response_text,"parsed":parsed,"error_message":error})
    return results, raw, errors, calls
