import io
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
from collections import OrderedDict

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://excel-axiom-react.vercel.app","http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage with LRU cache (limit to 10 datasets max)
class LRUCache:
    def __init__(self, max_size=10):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def set(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def delete(self, key):
        if key in self.cache:
            del self.cache[key]
    
    def size(self):
        return len(self.cache)

datasets_cache = LRUCache(max_size=10)

# --- Data Models ---
class ProcessRequest(BaseModel):
    session_id: str
    filters: Dict[str, List[Any]] = {}
    search_term: str = ""
    case_sensitive_search: bool = False
    new_column_config: Optional[Dict[str, Any]] = None
    page: int = 1
    page_size: int = 500  # Optimized for 75k rows
    sort_column: Optional[str] = None
    sort_direction: str = "asc"

# --- Helper Functions ---

def make_unique_column_names(columns: List[Any]) -> List[str]:
    unique_names: List[str] = []
    seen: Dict[str, int] = {}
    for raw_name in columns:
        base_name = str(raw_name).strip() if raw_name is not None else ""
        if base_name == "nan":
            base_name = ""
        candidate = base_name or "column"
        if candidate not in seen:
            seen[candidate] = 0
            unique_names.append(candidate)
        else:
            seen[candidate] += 1
            unique_names.append(f"{candidate}_{seen[candidate]}")
    return unique_names

def sanitize_dataframe_for_json(df: pd.DataFrame) -> pd.DataFrame:
    sanitized = df.copy()

    if not sanitized.empty:
        sanitized = sanitized.dropna(how="all")
        sanitized = sanitized.dropna(how="all", axis=1)

    sanitized.columns = make_unique_column_names(list(sanitized.columns))

    for column_name in sanitized.columns:
        column = sanitized[column_name]
        if pd.api.types.is_datetime64_any_dtype(column):
            try:
                sanitized[column_name] = (
                    column.dt.tz_localize(None)
                    .dt.strftime("%Y-%m-%dT%H:%M:%S")
                )
            except Exception:
                sanitized[column_name] = column.astype(str)
        elif column.dtype == object:
            sanitized[column_name] = column.map(
                lambda v: (
                    v.isoformat() if isinstance(v, (pd.Timestamp, datetime)) else v
                )
            )

    for column_name in sanitized.select_dtypes(include=[object]).columns:
        sanitized[column_name] = sanitized[column_name].map(
            lambda v: v.strip() if isinstance(v, str) else v
        )

    sanitized = sanitized.replace([np.inf, -np.inf], None)
    sanitized = sanitized.where(pd.notna(sanitized), None)

    return sanitized

def clean_nan_values(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj

def read_table_from_bytes(data: bytes, filename: str) -> Optional[pd.DataFrame]:
    name = filename.lower()
    buffer = io.BytesIO(data)
    try:
        if name.endswith(".csv"):
            try:
                # Read CSV in chunks for large files
                return pd.read_csv(buffer, encoding="utf-8-sig", low_memory=False)
            except Exception:
                buffer.seek(0)
                return pd.read_csv(buffer, encoding="latin-1", low_memory=False)
        elif name.endswith((".xlsx", ".xls")):
            # Use openpyxl engine with read_only mode for large files
            return pd.read_excel(buffer, engine='openpyxl')
        else:
            return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def apply_search_filter(df: pd.DataFrame, search_term: str, case_sensitive: bool = False) -> pd.DataFrame:
    if not search_term or not search_term.strip():
        return df
    
    search_term_processed = search_term if case_sensitive else search_term.lower()
    
    # Optimize search by only checking string columns
    string_cols = df.select_dtypes(include=[object]).columns
    if len(string_cols) == 0:
        string_cols = df.columns
    
    mask = df[string_cols].astype(str).apply(
        lambda row: row.str.contains(search_term_processed, case=case_sensitive, na=False).any(),
        axis=1
    )
    return df[mask]

def get_sample_for_filters(df: pd.DataFrame, column: str, max_unique: int = 1000) -> List[Any]:
    """For columns with too many unique values, return a sample"""
    unique_values = df[column].dropna().unique()
    if len(unique_values) > max_unique:
        # For large datasets, sample and sort
        sample = pd.Series(unique_values).sample(n=max_unique, random_state=42).tolist()
        try:
            return sorted(sample)
        except TypeError:
            return sample
    else:
        try:
            return sorted(unique_values.tolist())
        except TypeError:
            return unique_values.tolist()

# --- API Endpoints ---

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    # Check file size (limit to 50MB)
    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)
    
    if file_size_mb > 50:
        raise HTTPException(status_code=400, detail=f"File too large: {file_size_mb:.1f}MB. Maximum is 50MB.")
    
    df = read_table_from_bytes(contents, file.filename)

    if df is None:
        raise HTTPException(status_code=400, detail="Unsupported or invalid file type.")

    # Normalize DataFrame
    df = sanitize_dataframe_for_json(df)

    # Generate session ID
    session_id = str(uuid.uuid4())
    
    # Store dataset
    datasets_cache.set(session_id, df)

    # Build filter options (optimized for large datasets)
    column_filters = {}
    for column in df.columns:
        column_filters[column] = get_sample_for_filters(df, column)
    
    # Calculate basic stats
    total_rows = len(df)
    total_columns = len(df.columns)
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    return {
        "session_id": session_id,
        "columns": list(df.columns),
        "column_filters": column_filters,
        "stats": {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "memory_usage_mb": round(memory_usage_mb, 2),
            "file_size_mb": round(file_size_mb, 2)
        }
    }

@app.post("/api/process")
async def process_data(request: ProcessRequest):
    # Retrieve dataset
    df = datasets_cache.get(request.session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session expired. Please upload the file again.")
    
    # Make a working copy
    df = df.copy()
    
    # Apply new column creation
    if request.new_column_config and request.new_column_config.get("name"):
        config = request.new_column_config
        name = config["name"]
        mode = config["mode"]
        
        try:
            if mode == "Constant value":
                df[name] = config["value"]
            elif mode == "Expression":
                df[name] = df.eval(config["expression"])
            
            # Update cached dataset
            datasets_cache.set(request.session_id, df.copy())
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to add column: {str(e)}")

    # Apply search filter
    filtered_df = apply_search_filter(df, request.search_term, request.case_sensitive_search)

    # Apply column filters
    if request.filters:
        for column_name, selected_values in request.filters.items():
            if selected_values and column_name in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[column_name].isin(selected_values)]

    # Apply sorting
    if request.sort_column and request.sort_column in filtered_df.columns:
        ascending = request.sort_direction.lower() == "asc"
        filtered_df = filtered_df.sort_values(by=request.sort_column, ascending=ascending)

    # Calculate pagination
    total_rows = len(filtered_df)
    total_pages = max(1, (total_rows + request.page_size - 1) // request.page_size)
    
    # Clamp page number
    current_page = max(1, min(request.page, total_pages))
    
    # Extract page
    start_idx = (current_page - 1) * request.page_size
    end_idx = min(start_idx + request.page_size, total_rows)
    paginated_df = filtered_df.iloc[start_idx:end_idx]

    # Sanitize for JSON
    paginated_df = sanitize_dataframe_for_json(paginated_df)
    
    # Convert to dict
    data_dict = paginated_df.to_dict(orient='records')
    cleaned_data = clean_nan_values(data_dict)

    return {
        "data": cleaned_data,
        "pagination": {
            "current_page": current_page,
            "page_size": request.page_size,
            "total_rows": total_rows,
            "total_pages": total_pages,
            "has_next": current_page < total_pages,
            "has_previous": current_page > 1,
            "showing_from": start_idx + 1 if total_rows > 0 else 0,
            "showing_to": end_idx
        },
        "columns": list(filtered_df.columns)
    }

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    df = datasets_cache.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    datasets_cache.delete(session_id)
    return {"message": "Session deleted successfully"}

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "active_sessions": datasets_cache.size()
    }

@app.get("/api/export/{session_id}")
async def export_data(
    session_id: str,
    format: str = "csv",
    filters: Optional[str] = None,
    search_term: Optional[str] = None
):
    """Export filtered data (useful for downloading results)"""
    df = datasets_cache.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Apply filters if provided
    # (In production, parse filters from query string)
    
    if format == "csv":
        output = io.StringIO()
        df.to_csv(output, index=False)
        return {"data": output.getvalue(), "filename": f"export_{session_id}.csv"}
    else:
        raise HTTPException(status_code=400, detail="Unsupported export format")
