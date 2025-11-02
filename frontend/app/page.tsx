'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  CircularProgress,
  Alert,
  TextField,
  Checkbox,
  FormControlLabel,
  RadioGroup,
  Radio,
  Select,
  MenuItem,
  InputLabel,
  FormControl,
  Collapse,
  SelectChangeEvent,
  Divider,
  Paper,
  Pagination
} from '@mui/material';
import { AgGridReact } from 'ag-grid-react';
import { ColDef, GridApi, GridReadyEvent, RowSelectedEvent, GetRowIdParams } from 'ag-grid-community';
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css';

// --- Type Definitions ---
interface RowData {
  [key: string]: any;
  internal_id: number;
}

interface ColumnFilterMap {
  [key: string]: (string | number | boolean)[];
}

interface ActiveFilterMap {
  [key: string]: (string | number | boolean)[];
}

interface PaginationInfo {
  current_page: number;
  page_size: number;
  total_rows: number;
  total_pages: number;
  has_next: boolean;
  has_previous: boolean;
  showing_from: number;
  showing_to: number;
}

// --- API Service Functions ---
const API_URL = 'https://excel-axiom-react-hro4.vercel.app/api';
// const API_URL = 'http://localhost:8000/api';
async function uploadFile(file: File, onProgress?: (progress: number) => void): Promise<any> {
  const CHUNK_SIZE = 3 * 1024 * 1024; // 3MB chunks (under Vercel's 4.5MB limit)
  const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
  
  // For small files, use direct upload
  if (file.size < 4 * 1024 * 1024) {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${API_URL}/upload`, { method: 'POST', body: formData });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'File upload failed.');
    }
    return response.json();
  }
  
  // For large files, use chunked upload
  const uploadId = `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  
  for (let i = 0; i < totalChunks; i++) {
    const start = i * CHUNK_SIZE;
    const end = Math.min(start + CHUNK_SIZE, file.size);
    const chunk = file.slice(start, end);
    
    const formData = new FormData();
    formData.append('chunk', chunk);
    formData.append('chunk_index', i.toString());
    formData.append('total_chunks', totalChunks.toString());
    formData.append('filename', file.name);
    formData.append('upload_id', uploadId);
    
    const response = await fetch(`${API_URL}/upload-chunk`, { 
      method: 'POST', 
      body: formData 
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Chunk upload failed.');
    }
    
    // Update progress
    if (onProgress) {
      onProgress(Math.round(((i + 1) / totalChunks) * 100));
    }
    
    // If this was the last chunk, return the result
    if (i === totalChunks - 1) {
      return response.json();
    }
  }
  
  throw new Error('Upload failed unexpectedly');
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

async function deleteSession(sessionId: string): Promise<void> {
  await fetch(`${API_URL}/session/${sessionId}`, { method: 'DELETE' });
}

// --- Main Page Component ---
export default function Home() {
  // --- STATE MANAGEMENT ---
  const [sessionId, setSessionId] = useState<string>('');
  const [processedData, setProcessedData] = useState<RowData[]>([]);
  const [columns, setColumns] = useState<ColDef[]>([]);
  const [columnFilters, setColumnFilters] = useState<ColumnFilterMap>({});
  const [activeFilters, setActiveFilters] = useState<ActiveFilterMap>({});
  const [searchTerm, setSearchTerm] = useState('');
  const [caseSensitive, setCaseSensitive] = useState(false);
  const [hiddenColumns, setHiddenColumns] = useState<string[]>([]);
  const [hiddenRows, setHiddenRows] = useState<Set<number>>(new Set());

  const [pagination, setPagination] = useState<PaginationInfo | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(500);

  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState('');
  const [gridApi, setGridApi] = useState<GridApi | null>(null);
  const debounceTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const [addColumnExpanded, setAddColumnExpanded] = useState(false);
  const [newColName, setNewColName] = useState('');
  const [newColMode, setNewColMode] = useState('Constant value');
  const [newColConstValue, setNewColConstValue] = useState('');
  const [newColExpression, setNewColExpression] = useState('');

  // --- CORE DATA HANDLING ---
  const processData = useCallback(async (
    newColumnConfig: object | null = null,
    page: number = currentPage,
    resetPage: boolean = false
  ) => {
    if (!sessionId) return;

    setLoading(true);
    setError('');

    try {
      const payload = {
        session_id: sessionId,
        filters: activeFilters,
        search_term: searchTerm,
        case_sensitive_search: caseSensitive,
        new_column_config: newColumnConfig,
        page: resetPage ? 1 : page,
        page_size: pageSize,
      };

      const result = await processDataAPI(payload);

      const dataWithIds = result.data.map((row: object, index: number) => ({
        ...row,
        internal_id: (result.pagination.showing_from - 1) + index,
      }));

      if (newColumnConfig) {
        // Update columns when new column is added
        const newColumns = result.columns.map((c: string) => ({
          headerName: c,
          field: c,
          filter: true,
        }));
        const hideColumn: ColDef = {
          headerName: 'Hide',
          field: 'hide',
          checkboxSelection: true,
          headerCheckboxSelection: true,
          width: 80,
        };
        setColumns([hideColumn, ...newColumns]);
      }

      setProcessedData(dataWithIds);
      setPagination(result.pagination);
      
      if (resetPage) {
        setCurrentPage(1);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [sessionId, activeFilters, searchTerm, caseSensitive, currentPage, pageSize]);

  // Debounced search and filter
  useEffect(() => {
    if (!sessionId) return;
    
    if (debounceTimeoutRef.current) clearTimeout(debounceTimeoutRef.current);
    debounceTimeoutRef.current = setTimeout(() => {
      processData(null, 1, true); // Reset to page 1 when filters change
    }, 500);
  }, [searchTerm, caseSensitive, activeFilters, sessionId]);

  // --- EVENT HANDLERS ---
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setError('');
    setUploadProgress(0);
    
    // Cleanup old session
    if (sessionId) {
      try {
        await deleteSession(sessionId);
      } catch (e) {
        console.warn('Failed to delete old session:', e);
      }
    }

    try {
      const result = await uploadFile(file, (progress) => {
        setUploadProgress(progress);
      });
      
      setSessionId(result.session_id);
      setColumnFilters(result.column_filters);
      setActiveFilters({});
      setSearchTerm('');
      setHiddenColumns([]);
      setHiddenRows(new Set());
      setCurrentPage(1);

      // Set up columns
      const hideColumn: ColDef = {
        headerName: 'Hide',
        field: 'hide',
        checkboxSelection: true,
        headerCheckboxSelection: true,
        width: 80,
      };
      const dataColumns: ColDef[] = result.columns.map((c: string) => ({
        headerName: c,
        field: c,
        filter: true,
      }));
      setColumns([hideColumn, ...dataColumns]);

      // Fetch first page of data
      const processPayload = {
        session_id: result.session_id,
        filters: {},
        search_term: '',
        case_sensitive_search: false,
        page: 1,
        page_size: pageSize,
      };
      
      const processResult = await processDataAPI(processPayload);
      const dataWithIds = processResult.data.map((row: object, index: number) => ({
        ...row,
        internal_id: index,
      }));
      
      setProcessedData(dataWithIds);
      setPagination(processResult.pagination);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
      setUploadProgress(0);
      event.target.value = '';
    }
  };

  const handleFilterChange = (column: string, selectedValues: string[]) => {
    const newActiveFilters = { ...activeFilters };
    if (
      selectedValues.length === 0 ||
      selectedValues.length === (columnFilters[column]?.length || 0)
    ) {
      delete newActiveFilters[column];
    } else {
      newActiveFilters[column] = selectedValues;
    }
    setActiveFilters(newActiveFilters);
  };

  const handleAddColumn = () => {
    if (!newColName) {
      setError('New column name cannot be empty.');
      return;
    }
    const config = {
      name: newColName,
      mode: newColMode,
      value: newColConstValue,
      expression: newColExpression,
    };
    processData(config, currentPage);
    setNewColName('');
    setNewColConstValue('');
    setNewColExpression('');
  };

  const handlePageChange = (_event: React.ChangeEvent<unknown>, page: number) => {
    setCurrentPage(page);
    processData(null, page);
    // Scroll to top of grid
    if (gridApi) {
      gridApi.ensureIndexVisible(0);
    }
  };

  const onGridReady = (params: GridReadyEvent) => setGridApi(params.api);

  const exportToCsv = () => {
    if (!gridApi) return;
    gridApi.exportDataAsCsv({ fileName: 'filtered_data.csv' });
  };

  useEffect(() => {
    if (!gridApi) return;
    const allColIds = columns.map((c) => c.field as string);
    gridApi.setColumnsVisible(allColIds, true);
    gridApi.setColumnsVisible(hiddenColumns, false);
  }, [hiddenColumns, gridApi, columns]);

  const onRowSelected = (event: RowSelectedEvent) => {
    const rowId = event.node.data.internal_id;
    const newHiddenRows = new Set(hiddenRows);
    if (event.node.isSelected()) newHiddenRows.add(rowId);
    else newHiddenRows.delete(rowId);
    setHiddenRows(newHiddenRows);
  };

  const getRowId = useCallback((params: GetRowIdParams<RowData>) => String(params.data.internal_id), []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (sessionId) {
        deleteSession(sessionId).catch(console.warn);
      }
    };
  }, [sessionId]);

  // --- RENDER ---
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh', bgcolor: '#fafafa' }}>
      {/* Header */}
      <Box
        sx={{
          p: 2,
          bgcolor: '#1a73e8',
          color: 'white',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexShrink: 0,
          boxShadow: 2,
        }}
      >
        <Typography variant="h5" fontWeight="bold">
          GEDAT : Gomalian Exome Data Analysis Tool
        </Typography>
        <Button
          variant="contained"
          component="label"
          disabled={loading}
          sx={{
            bgcolor: 'white',
            color: '#1a73e8',
            fontWeight: 600,
            '&:hover': { bgcolor: '#e8f0fe' },
          }}
        >
          {loading && !sessionId ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <CircularProgress size={24} color="inherit" />
              {uploadProgress > 0 && <Typography variant="caption">{uploadProgress}%</Typography>}
            </Box>
          ) : (
            'Upload File'
          )}
          <input type="file" hidden onChange={handleFileUpload} accept=".csv, .xlsx, .xls" />
        </Button>
      </Box>

      <Box sx={{ display: 'flex', flexGrow: 1, overflow: 'hidden' }}>
        {/* Sidebar Filters */}
        <Paper
          elevation={3}
          sx={{
            width: 300,
            p: 2,
            bgcolor: '#fdfdfd',
            borderRight: '1px solid #ddd',
            overflowY: 'auto',
            flexShrink: 0,
          }}
        >
          <Typography variant="h6" sx={{ fontWeight: 600, color: '#333', mb: 1 }}>
            Filters
          </Typography>
          {Object.keys(columnFilters).length > 0 ? (
            Object.entries(columnFilters).map(([col, values]) => (
              <FormControl key={col} fullWidth margin="normal" size="small">
                <InputLabel>{col}</InputLabel>
                <Select
                  multiple
                  value={(activeFilters[col] || []).map(String)}
                  onChange={(e: SelectChangeEvent<string[]>) =>
                    handleFilterChange(col, e.target.value as string[])
                  }
                  renderValue={(selected) => (selected as string[]).join(', ')}
                >
                  {values.map((val) => (
                    <MenuItem key={String(val)} value={String(val)}>
                      <Checkbox
                        checked={(activeFilters[col] || []).includes(String(val))}
                      />
                      {String(val)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            ))
          ) : (
            <Typography variant="body2" color="text.secondary">
              Upload a file to see filters.
            </Typography>
          )}

          <Divider sx={{ my: 2 }} />

          <Typography variant="h6" sx={{ fontWeight: 600, color: '#333' }}>
            Hide Columns
          </Typography>
          <FormControl fullWidth margin="normal" size="small" disabled={columns.length <= 1}>
            <InputLabel>Select Columns</InputLabel>
            <Select
              multiple
              value={hiddenColumns}
              onChange={(e: SelectChangeEvent<string[]>) =>
                setHiddenColumns(e.target.value as string[])
              }
              renderValue={(selected) => (selected as string[]).join(', ')}
            >
              {columns
                .filter((c) => c.field !== 'hide')
                .map((col) => (
                  <MenuItem key={col.field} value={col.field}>
                    <Checkbox checked={hiddenColumns.includes(col.field!)} />
                    {col.headerName}
                  </MenuItem>
                ))}
            </Select>
          </FormControl>
        </Paper>

        {/* Main Content */}
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            p: 3,
            display: 'flex',
            flexDirection: 'column',
            overflow: 'auto',
          }}
        >
          {error && (
            <Alert severity="error" onClose={() => setError('')} sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {!sessionId ? (
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                height: '100%',
                color: '#555',
              }}
            >
              <Typography variant="h6">
                Upload an Excel/CSV file from the top right to begin.
              </Typography>
            </Box>
          ) : (
            <>
              <Paper elevation={2} sx={{ p: 2, mb: 3, borderRadius: 2 }}>
                <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                  <TextField
                    fullWidth
                    label="Search Across All Columns"
                    variant="outlined"
                    size="small"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                  />
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={caseSensitive}
                        onChange={(e) => setCaseSensitive(e.target.checked)}
                      />
                    }
                    label="Case Sensitive"
                  />
                </Box>

                <Button
                  sx={{ mt: 2, fontWeight: 600 }}
                  onClick={() => setAddColumnExpanded(!addColumnExpanded)}
                >
                  {addColumnExpanded ? 'Collapse' : 'Add New Column'}
                </Button>
                <Collapse in={addColumnExpanded}>
                  <Box sx={{ borderTop: '1px solid #eee', mt: 2, pt: 2 }}>
                    <RadioGroup
                      row
                      value={newColMode}
                      onChange={(e) => setNewColMode(e.target.value)}
                    >
                      <FormControlLabel
                        value="Constant value"
                        control={<Radio />}
                        label="Constant Value"
                      />
                      <FormControlLabel
                        value="Expression"
                        control={<Radio />}
                        label="Expression"
                      />
                    </RadioGroup>
                    <TextField
                      fullWidth
                      size="small"
                      label="New Column Name"
                      value={newColName}
                      onChange={(e) => setNewColName(e.target.value)}
                      sx={{ my: 1 }}
                    />
                    {newColMode === 'Constant value' ? (
                      <TextField
                        fullWidth
                        size="small"
                        label="Value"
                        value={newColConstValue}
                        onChange={(e) => setNewColConstValue(e.target.value)}
                      />
                    ) : (
                      <TextField
                        fullWidth
                        size="small"
                        label="Expression (e.g. Price * Quantity)"
                        value={newColExpression}
                        onChange={(e) => setNewColExpression(e.target.value)}
                      />
                    )}
                    <Button
                      variant="contained"
                      onClick={handleAddColumn}
                      sx={{ mt: 2 }}
                      disabled={!newColName}
                    >
                      Add Column
                    </Button>
                  </Box>
                </Collapse>
              </Paper>

              <Box
                sx={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  mb: 1,
                }}
              >
                <Typography variant="h6" color="#333" fontWeight={600}>
                  Filtered Data {loading && <CircularProgress size={18} sx={{ ml: 1 }} />}
                </Typography>
                <div>
                  <Button
                    variant="outlined"
                    onClick={() => setHiddenRows(new Set())}
                    sx={{ mr: 1 }}
                    disabled={hiddenRows.size === 0}
                  >
                    Restore Hidden Rows
                  </Button>
                  <Button
                    variant="contained"
                    onClick={exportToCsv}
                    disabled={processedData.length === 0}
                    sx={{ bgcolor: '#1a73e8' }}
                  >
                    Download CSV
                  </Button>
                </div>
              </Box>

              <Paper elevation={1} className="ag-theme-alpine" sx={{ flexGrow: 1, width: '100%' }}>
                <AgGridReact
                  onGridReady={onGridReady}
                  rowData={processedData.filter((row) => !hiddenRows.has(row.internal_id))}
                  columnDefs={columns}
                  defaultColDef={{ sortable: true, resizable: true, filter: true }}
                  rowSelection="multiple"
                  suppressRowClickSelection={true}
                  onRowSelected={onRowSelected}
                  getRowId={getRowId}
                />
              </Paper>

              {pagination && (
                <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="caption" sx={{ color: '#444' }}>
                    Showing {pagination.showing_from.toLocaleString()} - {pagination.showing_to.toLocaleString()} of{' '}
                    {pagination.total_rows.toLocaleString()} rows
                    {hiddenRows.size > 0 && ` (${hiddenRows.size} hidden)`}
                  </Typography>
                  
                  <Pagination
                    count={pagination.total_pages}
                    page={currentPage}
                    onChange={handlePageChange}
                    color="primary"
                    showFirstButton
                    showLastButton
                    disabled={loading}
                  />
                </Box>
              )}
            </>
          )}
        </Box>
      </Box>
    </Box>
  );
}
