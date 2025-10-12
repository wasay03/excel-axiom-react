import io
from typing import Optional, List, Dict, Any

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# IMPORTANT: Allow your frontend (running on localhost:3000) to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
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

def read_table_from_bytes(data: bytes, filename: str) -> Optional[pd.DataFrame]:
    name = filename.lower()
    buffer = io.BytesIO(data)
    try:
        if name.endswith(".csv"):
            return pd.read_csv(buffer)
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

    df = df.where(pd.notna(df), None)

    column_filters = {}
    for column in df.columns:
        unique_values = df[column].dropna().unique().tolist()
        try:
            column_filters[column] = sorted(unique_values)
        except TypeError:
            column_filters[column] = unique_values
            
    data = df.to_dict(orient='records')
    
    return {
        "data": data,
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

    filtered_df = filtered_df.where(pd.notna(filtered_df), None)

    return {
        "data": filtered_df.to_dict(orient='records'),
        "rows": len(filtered_df),
        "columns": len(filtered_df.columns),
    }