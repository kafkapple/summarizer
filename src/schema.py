from typing import List, Dict, Any, Optional

def get_default_summary_schema() -> Dict[str, Any]:
    """Returns the default dictionary structure for a summary result."""
    return {
        'one_sentence_summary': "",
        'full_summary': "",
        'core_summary': [],       # List of dicts, e.g., [{'title': 't1', 'points': ['p1', 'p2']}]
        'sections_summary': "",
        'keywords': [],           # List of dicts, e.g., [{'term': 'kw1'}, ...]
        'sections': [],           # List of dicts, e.g., [{'title': 't1', 'summary': 's1'}, ...]
        'chapters': [],           # List of dicts (for very long texts)
        'metadata': {},           # Dict for metadata like title, url, etc.
        'model': {                # Dict for LLM info
            'provider': None,
            'model': None,
            'output_language': None
        },
        'summary_strategy_used': None, # Name of the strategy actually used
        'full_text': None,        # Optional: Full original text (if included)
        'all_bullet_points': ""   # String containing all bullet points from all sections
    }

def get_minimal_summary_schema() -> Dict[str, Any]:
    """Returns a minimal structure, typically for chain outputs or simple summaries."""
    # Start with the default schema and remove/modify fields
    schema = get_default_summary_schema().copy() # Use copy to avoid modifying original
    
    # Fields often not present in minimal summaries
    schema.pop('core_summary', None) # Keep structure consistent, but expect empty list
    schema.pop('all_bullet_points', None)
    schema.pop('sections_summary', None)
    schema.pop('keywords', None)
    schema.pop('sections', None)
    schema.pop('chapters', None)
    schema.pop('full_text', None) # Usually not included here
    
    # Set expected default types/values for minimal fields
    schema['one_sentence_summary'] = ""
    schema['full_summary'] = "" # LangChain chains often return string here
    
    return schema

def get_refine_summary_schema() -> Dict[str, Any]:
    """Returns the structure expected specifically from the refine LLM call."""
    return {
        'one_sentence_summary': "",
        'full_summary': "",
        # Other fields like core_summary are not expected from this specific call
        # The 'refine' LLM task only returns these two fields.
    }
