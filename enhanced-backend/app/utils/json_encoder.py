"""
Custom JSON encoder for handling special values like NaN, Infinity
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, date

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that properly handles NaN, Infinity, and other special values."""
    
    def default(self, obj):
        # Handle numpy and pandas special values
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            if np.isnan(obj) or np.isinf(obj):
                return None  # Convert NaN/Infinity to None (null in JSON)
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        elif isinstance(obj, pd.Series):
            return obj.to_list()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        
        # Let the base class default method raise the TypeError
        return super().default(obj)

def dumps(obj, **kwargs):
    """Wrapper around json.dumps that uses the CustomJSONEncoder."""
    return json.dumps(obj, cls=CustomJSONEncoder, **kwargs)
