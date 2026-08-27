from __future__ import annotations
import pandas as pd

def _b(v): return str(v).strip().casefold() in {'true','1','yes','si','sí'}
def filter_corpus(df):
    include=df['include_in_state_of_art'].map(_b)
    if 'relevance_level' in df.columns:
        include &= df['relevance_level'].fillna('').astype(str).str.casefold().isin({'alta','media','high','medium','relevante','relevant'})
    final=df[include].copy(); excluded=df[~include].copy()
    if final.empty: raise ValueError('EMPTY_THEMATIC_CORPUS')
    return final,excluded
