'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { Box, Typography, Button, CircularProgress, Alert, TextField, Checkbox, FormControlLabel, RadioGroup, Radio, Select, MenuItem, InputLabel, FormControl, Collapse, SelectChangeEvent } from '@mui/material';
import { AgGridReact } from 'ag-grid-react';
import { ColDef, GridApi, GridReadyEvent, RowSelectedEvent, GetRowIdParams } from 'ag-grid-community';
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css';

// --- Type Definitions ---
interface RowData {
  [key: string]: any;
  // Add a unique ID for robust row handling
  internal_id: number;
}

interface ColumnFilterMap {
  [key: string]: (string | number | boolean)[];
}

interface ActiveFilterMap {
  [key: string]: (string | number | boolean)[];
}

// --- API Service Functions ---
const API_URL = 'https://excel-axiom-react-hro4-5a72zyzkw-wasay03s-projects.vercel.app/api';

async function uploadFile(file: File): Promise<any> {
  const formData = new FormData();
  formData.append('file', file);
  const response = await fetch(`${API_URL}/upload`, { method: 'POST', body: formData });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'File upload failed. Unsupported or invalid file type.');
  }
  return response.json();
}

async function processDataAPI(payload: object): Promise<any> {
  const response = await fetch(`${API_URL}/process`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Processing failed');
  }
  return response.json();
}

// --- Main Page Component ---
export default function Home() {
  // --- STATE MANAGEMENT ---
  const [baseData, setBaseData] = useState<RowData[]>([]);
  const [processedData, setProcessedData] = useState<RowData[]>([]);
  const [columns, setColumns] = useState<ColDef[]>([]);
  const [columnFilters, setColumnFilters] = useState<ColumnFilterMap>({});
  const [activeFilters, setActiveFilters] = useState<ActiveFilterMap>({});
  const [searchTerm, setSearchTerm] = useState('');
  const [caseSensitive, setCaseSensitive] = useState(false);
  const [hiddenColumns, setHiddenColumns] = useState<string[]>([]);
  const [hiddenRows, setHiddenRows] = useState<Set<number>>(new Set());
  
  // UI & Loading State
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [gridApi, setGridApi] = useState<GridApi | null>(null);
  const debounceTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Add Column State
  const [addColumnExpanded, setAddColumnExpanded] = useState(false);
  const [newColName, setNewColName] = useState('');
  const [newColMode, setNewColMode] = useState('Constant value');
  const [newColConstValue, setNewColConstValue] = useState('');
  const [newColExpression, setNewColExpression] = useState('');

  // --- CORE DATA HANDLING ---
  const processData = useCallback(async (newColumnConfig: object | null = null) => {
    if (baseData.length === 0) return;

    setLoading(true);
    setError('');

    const dataToProcess = newColumnConfig ? baseData : baseData;

    try {
      const payload = {
        data: dataToProcess.map(({ internal_id, ...rest }) => rest), // Remove internal_id before sending to backend
        filters: activeFilters,
        search_term: searchTerm,
        case_sensitive_search: caseSensitive,
        new_column_config: newColumnConfig,
      };

      const result = await processDataAPI(payload);
      
      const dataWithIds = result.data.map((row: object, index: number) => ({ ...row, internal_id: index }));

      if (newColumnConfig) {
        setBaseData(dataWithIds);
        const newColumns = Object.keys(result.data[0] || {}).map(c => ({ headerName: c, field: c, filter: true }));
        const currentHideCol = columns.find(c => c.field === 'hide');
        setColumns(currentHideCol ? [currentHideCol, ...newColumns] : newColumns);
      }
      
      setProcessedData(dataWithIds);

    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [baseData, activeFilters, searchTerm, caseSensitive, columns]);

  useEffect(() => {
    if (debounceTimeoutRef.current) {
      clearTimeout(debounceTimeoutRef.current);
    }
    debounceTimeoutRef.current = setTimeout(() => {
      processData();
    }, 500);
  }, [searchTerm, caseSensitive, activeFilters, processData]);


  // --- EVENT HANDLERS ---
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setError('');
    try {
      const result = await uploadFile(file);
      const dataWithIds = result.data.map((row: object, index: number) => ({ ...row, internal_id: index }));

      setBaseData(dataWithIds);
      setProcessedData(dataWithIds);
      
      const hideColumn: ColDef = {
          headerName: 'Hide', field: 'hide', checkboxSelection: true, headerCheckboxSelection: true, width: 80
      };
      const dataColumns: ColDef[] = result.columns.map((c: string) => ({ headerName: c, field: c, filter: true }));

      setColumns([hideColumn, ...dataColumns]);
      setColumnFilters(result.column_filters);
      setActiveFilters({});
      setSearchTerm('');
      setHiddenColumns([]);
      setHiddenRows(new Set());
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
      event.target.value = ''; // Allows re-uploading the same file
    }
  };

  const handleFilterChange = (column: string, selectedValues: string[]) => {
      const newActiveFilters = { ...activeFilters };
      if (selectedValues.length === 0 || selectedValues.length === (columnFilters[column]?.length || 0)) {
        delete newActiveFilters[column];
      } else {
        newActiveFilters[column] = selectedValues;
      }
      setActiveFilters(newActiveFilters);
  };
  
  const handleAddColumn = () => {
    if (!newColName) {
        setError("New column name cannot be empty.");
        return;
    }
    const config = {
        name: newColName,
        mode: newColMode,
        value: newColConstValue,
        expression: newColExpression,
    };
    processData(config);
    setNewColName('');
    setNewColConstValue('');
    setNewColExpression('');
  };

  const onGridReady = (params: GridReadyEvent) => setGridApi(params.api);

  const exportToCsv = () => {
    if (!gridApi) return;
    gridApi.exportDataAsCsv({ fileName: 'filtered_data.csv' });
  };
  
  useEffect(() => {
    if (!gridApi) return;
    const allColIds = columns.map(c => c.field as string);
    gridApi.setColumnsVisible(allColIds, true); // Show all
    gridApi.setColumnsVisible(hiddenColumns, false); // Then hide selected
  }, [hiddenColumns, gridApi, columns]);

  const onRowSelected = (event: RowSelectedEvent) => {
    const rowId = event.node.data.internal_id;
    const newHiddenRows = new Set(hiddenRows);
    if (event.node.isSelected()) {
        newHiddenRows.add(rowId);
    } else {
        newHiddenRows.delete(rowId);
    }
    setHiddenRows(newHiddenRows);
  };
  
  const getRowId = useCallback((params: GetRowIdParams<RowData>) => String(params.data.internal_id), []);


  // --- RENDER ---
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh', bgcolor: '#f5f5f5' }}>
      <Box sx={{ p: 2, bgcolor: 'white', borderBottom: '1px solid #ddd', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0 }}>
        <Typography variant="h5" fontWeight="bold">Excel Axiom</Typography>
        <Button variant="contained" component="label" disabled={loading}>
          {loading && !baseData.length ? <CircularProgress size={24} color="inherit" /> : 'Upload File'}
          <input type="file" hidden onChange={handleFileUpload} accept=".csv, .xlsx, .xls" />
        </Button>
      </Box>

      <Box sx={{ display: 'flex', flexGrow: 1, overflow: 'hidden' }}>
        <Box sx={{ width: 280, p: 2, bgcolor: 'white', borderRight: '1px solid #ddd', overflowY: 'auto', flexShrink: 0 }}>
          <Typography variant="h6">Filters</Typography>
          {Object.keys(columnFilters).length > 0 ? (
            Object.entries(columnFilters).map(([col, values]) => (
              <FormControl key={col} fullWidth margin="normal" size="small">
                <InputLabel>{col}</InputLabel>
                <Select
                  multiple
                  value={(activeFilters[col] || []).map(String)}
                  onChange={(e: SelectChangeEvent<string[]>) => handleFilterChange(col, e.target.value as string[])}
                  renderValue={(selected) => (selected as string[]).join(', ')}
                >
                  {values.map(val => <MenuItem key={String(val)} value={String(val)}><Checkbox checked={(activeFilters[col] || []).includes(String(val))} />{String(val)}</MenuItem>)}
                </Select>
              </FormControl>
            ))
          ) : (
            <Typography variant="caption">Upload a file to see filters.</Typography>
          )}

          <Typography variant="h6" sx={{ mt: 2 }}>Hide Columns</Typography>
          <FormControl fullWidth margin="normal" size="small" disabled={columns.length <= 1}>
            <InputLabel>Select Columns</InputLabel>
            <Select
              multiple
              value={hiddenColumns}
              onChange={(e: SelectChangeEvent<string[]>) => setHiddenColumns(e.target.value as string[])}
              renderValue={(selected) => (selected as string[]).join(', ')}
            >
              {columns.filter(c => c.field !== 'hide').map(col => <MenuItem key={col.field} value={col.field}><Checkbox checked={hiddenColumns.includes(col.field!)} />{col.headerName}</MenuItem>)}
            </Select>
          </FormControl>

        </Box>

        <Box component="main" sx={{ flexGrow: 1, p: 2, display: 'flex', flexDirection: 'column', overflow: 'auto' }}>
          {error && <Alert severity="error" onClose={() => setError('')} sx={{ mb: 2 }}>{error}</Alert>}
          
          {baseData.length === 0 ? (
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%'}}>
                <Typography>Upload an Excel/CSV file from the top right to begin.</Typography>
            </Box>
          ) : (
            <>
              <Box sx={{ bgcolor: 'white', p: 2, borderRadius: 1, border: '1px solid #ddd', mb: 2 }}>
                  <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                    <TextField
                      fullWidth
                      label="Search Across All Columns"
                      variant="outlined"
                      size="small"
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                    />
                    <FormControlLabel control={<Checkbox checked={caseSensitive} onChange={(e) => setCaseSensitive(e.target.checked)} />} label="Case Sensitive" />
                  </Box>
                  <Button sx={{mt: 2}} onClick={() => setAddColumnExpanded(!addColumnExpanded)}>
                    {addColumnExpanded ? 'Collapse' : 'Add New Column'}
                  </Button>
                  <Collapse in={addColumnExpanded}>
                    <Box sx={{ borderTop: '1px solid #eee', mt: 2, pt: 2}}>
                        <RadioGroup row value={newColMode} onChange={(e) => setNewColMode(e.target.value)}>
                            <FormControlLabel value="Constant value" control={<Radio />} label="Constant Value" />
                            <FormControlLabel value="Expression" control={<Radio />} label="Expression" />
                        </RadioGroup>
                        <TextField fullWidth size="small" label="New Column Name" value={newColName} onChange={(e) => setNewColName(e.target.value)} sx={{ my: 1 }} />
                        {newColMode === 'Constant value' ? (
                            <TextField fullWidth size="small" label="Value" value={newColConstValue} onChange={(e) => setNewColConstValue(e.target.value)} />
                        ) : (
                            <TextField fullWidth size="small" label="Expression (e.g. Price * Quantity)" value={newColExpression} onChange={(e) => setNewColExpression(e.target.value)} />
                        )}
                        <Button variant="contained" onClick={handleAddColumn} sx={{ mt: 2 }} disabled={!newColName}>Add Column</Button>
                    </Box>
                  </Collapse>
              </Box>

              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="h6">Filtered Data {loading && <CircularProgress size={20} sx={{ml:1}}/>}</Typography>
                  <div>
                    <Button variant="outlined" onClick={() => setHiddenRows(new Set())} sx={{ mr: 1 }} disabled={hiddenRows.size === 0}>Restore Hidden Rows</Button>
                    <Button variant="outlined" onClick={exportToCsv} disabled={processedData.length === 0}>Download CSV</Button>
                  </div>
              </Box>
              <Box className="ag-theme-alpine" sx={{ flexGrow: 1, width: '100%', mt: 1 }}>
                  <AgGridReact
                      onGridReady={onGridReady}
                      rowData={processedData.filter(row => !hiddenRows.has(row.internal_id))}
                      columnDefs={columns}
                      defaultColDef={{ sortable: true, resizable: true, filter: true }}
                      rowSelection="multiple"
                      suppressRowClickSelection={true}
                      onRowSelected={onRowSelected}
                      getRowId={getRowId}
                  />
              </Box>
              <Typography variant="caption" sx={{ mt: 1, flexShrink: 0 }}>
                Showing {(processedData.length - hiddenRows.size).toLocaleString()} of {baseData.length.toLocaleString()} rows
              </Typography>
            </>
          )}
        </Box>
      </Box>
    </Box>
  );
}