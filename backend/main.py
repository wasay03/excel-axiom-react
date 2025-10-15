import io
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# IMPORTANT: Allow your frontend (running on localhost:3000) to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://excel-axiom-react.vercel.app","http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models for API requests ---
class ProcessRequest(BaseModel):
    data: List[Dict[str, Any]]
    filters: Dict[str, List[Any]]
    search_term: str = ""
    case_sensitive_search: bool = False
    new_column_config: Optional[Dict[str, Any]] = None

# --- Replicated Functions ---

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
    # Create a copy to avoid mutating the original DataFrame upstream
    sanitized = df.copy()

    # Drop completely empty rows and columns
    if not sanitized.empty:
        sanitized = sanitized.dropna(how="all")
        sanitized = sanitized.dropna(how="all", axis=1)

    # Ensure column names are strings, trimmed, and unique
    sanitized.columns = make_unique_column_names(list(sanitized.columns))

    # Convert timezone-aware datetimes to naive and then to ISO strings
    for column_name in sanitized.columns:
        column = sanitized[column_name]
        if pd.api.types.is_datetime64_any_dtype(column):
            # Remove timezone if present then format to ISO 8601
            try:
                sanitized[column_name] = (
                    column.dt.tz_localize(None)  # type: ignore[attr-defined]
                    .dt.strftime("%Y-%m-%dT%H:%M:%S")
                )
            except Exception:
                sanitized[column_name] = column.astype(str)
        elif column.dtype == object:
            # Handle mixed objects containing Timestamp or datetime
            sanitized[column_name] = column.map(
                lambda v: (
                    v.isoformat() if isinstance(v, (pd.Timestamp, datetime)) else v
                )
            )

    # Trim whitespace from string cells
    for column_name in sanitized.select_dtypes(include=[object]).columns:
        sanitized[column_name] = sanitized[column_name].map(
            lambda v: v.strip() if isinstance(v, str) else v
        )

    # Replace +/- inf with None so it can be JSON serialized
    sanitized = sanitized.replace([np.inf, -np.inf], None)

    # Replace NaN/NaT with None
    sanitized = sanitized.where(pd.notna(sanitized), None)

    return sanitized

def clean_nan_values(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    # Use pandas isna to catch NaN/NaT
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    # Convert lingering pandas Timestamps/datetime to ISO strings
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj

def read_table_from_bytes(data: bytes, filename: str) -> Optional[pd.DataFrame]:
    name = filename.lower()
    buffer = io.BytesIO(data)
    try:
        if name.endswith(".csv"):
            # Try UTF-8 with BOM first, then fallback to latin-1 to tolerate odd encodings
            try:
                return pd.read_csv(buffer, encoding="utf-8-sig")
            except Exception:
                buffer.seek(0)
                return pd.read_csv(buffer, encoding="latin-1")
        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(buffer)
        else:
            return None
    except Exception as e:
        return None

def apply_search_filter(df: pd.DataFrame, search_term: str, case_sensitive: bool = False) -> pd.DataFrame:
    if not search_term or not search_term.strip():
        return df
    
    search_term_processed = search_term if case_sensitive else search_term.lower()
    
    mask = df.apply(
        lambda row: row.astype(str).str.contains(search_term_processed, case=case_sensitive, na=False).any(),
        axis=1
    )
    return df[mask]


# --- API Endpoints ---

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    df = read_table_from_bytes(contents, file.filename)

    if df is None:
        raise HTTPException(status_code=400, detail="Unsupported or invalid file type.")

    # Normalize DataFrame to make it safe for JSON serialization and downstream logic
    df = sanitize_dataframe_for_json(df)

    column_filters = {}
    for column in df.columns:
        unique_values = df[column].dropna().unique().tolist()
        try:
            column_filters[column] = sorted(unique_values)
        except TypeError:
            column_filters[column] = unique_values
    
    # Convert DataFrame to dict and clean any remaining NaN values
    data_dict = df.to_dict(orient='records')
    cleaned_data = clean_nan_values(data_dict)
    
    return {
        "data": cleaned_data,
        "columns": list(df.columns),
        "column_filters": column_filters
    }

@app.post("/api/process")
async def process_data(request: ProcessRequest):
    if not request.data:
        return {"data": [], "rows": 0, "columns": 0}

    df = pd.DataFrame(request.data)
    
    if request.new_column_config and request.new_column_config.get("name"):
        config = request.new_column_config
        name = config["name"]
        mode = config["mode"]
        
        try:
            if mode == "Constant value":
                df[name] = config["value"]
            elif mode == "Expression":
                df[name] = df.eval(config["expression"])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to add column: {str(e)}")

    filtered_df = apply_search_filter(df, request.search_term, request.case_sensitive_search)

    if request.filters:
        for column_name, selected_values in request.filters.items():
            if selected_values and column_name in filtered_df.columns:
                all_possible_values = df[column_name].dropna().unique()
                if len(selected_values) < len(all_possible_values):
                     filtered_df = filtered_df[filtered_df[column_name].isin(selected_values)]

    # Handle NaN values before JSON serialization
    # Normalize the filtered frame to ensure JSON-safe content
    filtered_df = sanitize_dataframe_for_json(filtered_df)
    
    # Convert DataFrame to dict and ensure no NaN values remain
    data_dict = filtered_df.to_dict(orient='records')
    cleaned_data = clean_nan_values(data_dict)

    return {
        "data": cleaned_data,
        "rows": len(filtered_df),
        "columns": len(filtered_df.columns),
    }