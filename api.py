"""
FastAPI REST API for Synthetic Data Generator

Provides endpoints for:
- Data upload and analysis
- Configuration management
- Data generation
- Quality and privacy validation
- Result download
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import io
import json
import uuid
from datetime import datetime
from pathlib import Path
import logging
import os

# Import our modules
from src.config import Config, ConfigLoader, get_default_config
from src.orchestrator import DataOrchestrator, PipelineType
from src.schema import SchemaAnalyzer
from src.validation import QualityValidator, PrivacyValidator
from src.generators import NumericGenerator, TextGenerator, PIIGenerator, TemporalGenerator, CategoricalGenerator
from src.utils import FileHandler, setup_logging

# Configure logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Synthetic Data Generator API",
    description="Generate high-quality synthetic data with privacy guarantees",
    version="1.0.0"
)

# CORS middleware
allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8501").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lightweight persistence (filesystem-backed)
RUNTIME_DIR = Path(os.getenv("SDG_RUNTIME_DIR", ".runtime"))
DATA_DIR = RUNTIME_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

jobs_store: Dict[str, Dict[str, Any]] = {}
data_store: Dict[str, pd.DataFrame] = {}

def _data_path(data_id: str) -> Path:
    return DATA_DIR / f"{data_id}.parquet"

def save_data(data_id: str, df: pd.DataFrame):
    data_store[data_id] = df
    df.to_parquet(_data_path(data_id), index=False)

def load_data(data_id: str) -> Optional[pd.DataFrame]:
    if data_id in data_store:
        return data_store[data_id]
    path = _data_path(data_id)
    if path.exists():
        df = pd.read_parquet(path)
        data_store[data_id] = df
        return df
    return None

def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    expected = os.getenv("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# Pydantic models
class GenerationConfig(BaseModel):
    """Generation configuration"""
    num_rows: int = Field(1000, ge=1, le=1000000,
                          description="Number of rows to generate")
    seed: Optional[int] = Field(
        None, description="Random seed for reproducibility")
    batch_size: int = Field(100, ge=10, le=10000,
                            description="Batch size for generation")
    enable_parallel: bool = Field(
        True, description="Enable parallel processing")
    preserve_correlations: bool = Field(
        True, description="Preserve numeric correlations")
    distribution_fitting: str = Field(
        "auto", description="Distribution fitting method")


class ValidationConfig(BaseModel):
    """Validation configuration"""
    enable_quality_checks: bool = Field(
        True, description="Enable quality validation")
    enable_privacy_checks: bool = Field(
        True, description="Enable privacy validation")
    quality_threshold: float = Field(
        0.8, ge=0.0, le=1.0, description="Quality threshold")
    k_anonymity: int = Field(5, ge=2, description="K-anonymity requirement")


class JobStatus(BaseModel):
    """Job status response"""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: float = Field(0.0, ge=0.0, le=100.0)
    message: str = ""
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class AnalysisResult(BaseModel):
    """Schema analysis result"""
    num_rows: int
    num_columns: int
    columns: List[Dict[str, Any]]
    memory_mb: float
    missing_values: int


class GenerationResult(BaseModel):
    """Generation result"""
    job_id: str
    num_rows: int
    num_columns: int
    generation_time: float
    data_id: str


class ValidationResult(BaseModel):
    """Validation result"""
    quality_score: Optional[float] = None
    quality_passed: Optional[bool] = None
    privacy_risk: Optional[str] = None
    privacy_passed: Optional[bool] = None
    details: Dict[str, Any] = Field(default_factory=dict)


# Helper functions
def create_job(job_type: str) -> str:
    """Create a new job"""
    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        'job_id': job_id,
        'type': job_type,
        'status': 'pending',
        'progress': 0.0,
        'message': 'Job created',
        'created_at': datetime.now().isoformat(),
        'completed_at': None,
        'result': None
    }
    return job_id


def update_job(job_id: str, status: str, progress: float = None, message: str = None, result: Any = None):
    """Update job status"""
    if job_id not in jobs_store:
        return

    jobs_store[job_id]['status'] = status
    if progress is not None:
        jobs_store[job_id]['progress'] = progress
    if message is not None:
        jobs_store[job_id]['message'] = message
    if result is not None:
        jobs_store[job_id]['result'] = result
    if status in ['completed', 'failed']:
        jobs_store[job_id]['completed_at'] = datetime.now().isoformat()


def read_uploaded_file(file: UploadFile) -> pd.DataFrame:
    """Read uploaded file into DataFrame"""
    try:
        file_extension = file.filename.split('.')[-1].lower()

        if file_extension == 'csv':
            return pd.read_csv(io.BytesIO(file.file.read()))
        elif file_extension in ['xlsx', 'xls']:
            return pd.read_excel(io.BytesIO(file.file.read()))
        elif file_extension == 'json':
            return pd.read_json(io.BytesIO(file.file.read()))
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise HTTPException(
            status_code=400, detail=f"Error reading file: {str(e)}")


# API Endpoints

@app.get("/", tags=["General"])
async def root():
    """API root endpoint"""
    return {
        "name": "Synthetic Data Generator API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "upload": "/upload",
            "analyze": "/analyze/{data_id}",
            "generate": "/generate",
            "validate": "/validate/{data_id}",
            "download": "/download/{data_id}",
            "jobs": "/jobs/{job_id}"
        }
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len([j for j in jobs_store.values() if j['status'] == 'running']),
        "total_jobs": len(jobs_store),
        "data_store_size": len(data_store)
    }


@app.post("/upload", tags=["Data"])
async def upload_data(file: UploadFile = File(...), x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)
    """
    Upload reference data

    Accepts CSV, Excel, or JSON files
    Returns a data_id for subsequent operations
    """
    try:
        # Read file
        data = read_uploaded_file(file)

        # Generate data ID
        data_id = str(uuid.uuid4())
        save_data(data_id, data)

        logger.info(f"Uploaded data: {data_id}, shape: {data.shape}")

        return {
            "data_id": data_id,
            "filename": file.filename,
            "num_rows": len(data),
            "num_columns": len(data.columns),
            "columns": data.columns.tolist(),
            "memory_mb": data.memory_usage(deep=True).sum() / 1024**2
        }

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/analyze/{data_id}", response_model=AnalysisResult, tags=["Data"])
async def analyze_data(data_id: str):
    """
    Analyze uploaded data schema

    Returns detailed column information and statistics
    """
    data = load_data(data_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Data not found")

    try:

        # Analyze schema
        analyzer = SchemaAnalyzer()
        profiles = analyzer.analyze_dataframe(data)

        # Build response
        columns_info = []
        for col_name, profile in profiles.items():
            col_info = {
                'name': col_name,
                'type': profile.inferred_type,
                'confidence': profile.type_confidence.confidence if profile.type_confidence else 0.0,
                'completeness': profile.completeness,
                'uniqueness': profile.uniqueness,
                'contains_pii': profile.contains_pii,
            }

            if profile.inferred_type == 'numeric':
                col_info.update({
                    'min': profile.min_value,
                    'max': profile.max_value,
                    'mean': profile.mean,
                    'std': profile.std
                })

            columns_info.append(col_info)

        return AnalysisResult(
            num_rows=len(data),
            num_columns=len(data.columns),
            columns=columns_info,
            memory_mb=data.memory_usage(deep=True).sum() / 1024**2,
            missing_values=data.isna().sum().sum()
        )

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerationResult, tags=["Generation"])
async def generate_synthetic_data(
    data_id: str = Query(..., description="Reference data ID"),
    config: GenerationConfig = None,
    background_tasks: BackgroundTasks = None,
    x_api_key: Optional[str] = Header(default=None)
):
    require_api_key(x_api_key)
    """
    Generate synthetic data

    Uses reference data to generate synthetic dataset
    Returns job_id for tracking progress
    """
    reference_data = load_data(data_id)
    if reference_data is None:
        raise HTTPException(status_code=404, detail="Reference data not found")

    try:

        # Use default config if not provided
        if config is None:
            config = GenerationConfig()

        # Create configuration object
        app_config = get_default_config()
        app_config.generation.num_rows = config.num_rows
        app_config.generation.seed = config.seed
        app_config.generation.batch_size = config.batch_size
        app_config.generation.enable_parallel = config.enable_parallel
        app_config.numeric.preserve_correlations = config.preserve_correlations
        app_config.numeric.distribution_fitting = config.distribution_fitting

        import time
        start_time = time.time()

        orchestrator = DataOrchestrator(app_config)
        orchestrator.register_pipeline(PipelineType.NUMERIC, NumericGenerator(app_config))
        orchestrator.register_pipeline(PipelineType.TEXT, TextGenerator(app_config))
        orchestrator.register_pipeline(PipelineType.PII, PIIGenerator(app_config))
        orchestrator.register_pipeline(PipelineType.TEMPORAL, TemporalGenerator(app_config))
        orchestrator.register_pipeline(PipelineType.HYBRID, CategoricalGenerator(app_config))

        schema = orchestrator.analyze_schema(reference_data)
        result = orchestrator.generate(
            num_rows=config.num_rows,
            reference_data=reference_data,
            schema=schema,
        )
        synthetic_data = result.data
        generation_time = result.generation_time if result else (time.time() - start_time)

        # Store synthetic data
        synthetic_id = str(uuid.uuid4())
        save_data(synthetic_id, synthetic_data)

        logger.info(
            f"Generated synthetic data: {synthetic_id}, shape: {synthetic_data.shape}")

        return GenerationResult(
            job_id=synthetic_id,
            num_rows=len(synthetic_data),
            num_columns=len(synthetic_data.columns),
            generation_time=generation_time,
            data_id=synthetic_id
        )

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))




def _run_generation_job(job_id: str, data_id: str, config: GenerationConfig):
    """Background generation worker."""
    try:
        update_job(job_id, status='running', progress=5.0, message='Preparing generation')
        reference_data = load_data(data_id)
        if reference_data is None:
            raise ValueError('Reference data not found')

        app_config = get_default_config()
        app_config.generation.num_rows = config.num_rows
        app_config.generation.seed = config.seed
        app_config.generation.batch_size = config.batch_size
        app_config.generation.enable_parallel = config.enable_parallel
        app_config.numeric.preserve_correlations = config.preserve_correlations
        app_config.numeric.distribution_fitting = config.distribution_fitting

        orchestrator = DataOrchestrator(app_config)
        orchestrator.register_pipeline(PipelineType.NUMERIC, NumericGenerator(app_config))
        orchestrator.register_pipeline(PipelineType.TEXT, TextGenerator(app_config))
        orchestrator.register_pipeline(PipelineType.PII, PIIGenerator(app_config))
        orchestrator.register_pipeline(PipelineType.TEMPORAL, TemporalGenerator(app_config))
        orchestrator.register_pipeline(PipelineType.HYBRID, CategoricalGenerator(app_config))

        update_job(job_id, status='running', progress=30.0, message='Analyzing schema')
        schema = orchestrator.analyze_schema(reference_data)
        update_job(job_id, status='running', progress=60.0, message='Generating data')
        result = orchestrator.generate(num_rows=config.num_rows, reference_data=reference_data, schema=schema)
        synthetic_data = result.data

        synthetic_id = str(uuid.uuid4())
        save_data(synthetic_id, synthetic_data)

        update_job(job_id, status='completed', progress=100.0, message='Generation completed', result={
            'data_id': synthetic_id,
            'num_rows': len(synthetic_data),
            'num_columns': len(synthetic_data.columns),
            'generation_time': result.generation_time,
        })
    except Exception as e:
        logger.error(f'Async generation error: {e}')
        update_job(job_id, status='failed', progress=100.0, message=str(e))


@app.post("/generate/async", response_model=JobStatus, tags=["Generation"])
async def generate_synthetic_data_async(
    data_id: str = Query(..., description="Reference data ID"),
    config: GenerationConfig = None,
    background_tasks: BackgroundTasks = None,
    x_api_key: Optional[str] = Header(default=None),
):
    require_api_key(x_api_key)
    if load_data(data_id) is None:
        raise HTTPException(status_code=404, detail="Reference data not found")

    if config is None:
        config = GenerationConfig()

    job_id = create_job('generation_async')
    update_job(job_id, status='pending', progress=0.0, message='Job queued')
    background_tasks.add_task(_run_generation_job, job_id, data_id, config)
    return JobStatus(**jobs_store[job_id])

@app.post("/validate/{data_id}", response_model=ValidationResult, tags=["Validation"])
async def validate_synthetic_data(
    data_id: str,
    reference_id: str = Query(..., description="Reference data ID"),
    config: ValidationConfig = None
):
    """
    Validate synthetic data

    Runs quality and privacy validation checks
    """
    synthetic = load_data(data_id)
    if synthetic is None:
        raise HTTPException(status_code=404, detail="Synthetic data not found")
    reference = load_data(reference_id)
    if reference is None:
        raise HTTPException(status_code=404, detail="Reference data not found")

    try:

        # Use default config if not provided
        if config is None:
            config = ValidationConfig()

        # Create configuration object
        app_config = get_default_config()
        app_config.validation.enable_quality_checks = config.enable_quality_checks
        app_config.validation.enable_privacy_checks = config.enable_privacy_checks
        app_config.validation.quality_threshold = config.quality_threshold
        app_config.validation.k_anonymity = config.k_anonymity

        result = ValidationResult()

        # Quality validation
        if config.enable_quality_checks:
            validator = QualityValidator(app_config)
            quality_report = validator.validate(reference, synthetic)

            result.quality_score = quality_report.overall_score
            result.quality_passed = quality_report.passed
            result.details['quality'] = quality_report.to_dict()

        # Privacy validation
        if config.enable_privacy_checks:
            validator = PrivacyValidator(app_config)
            privacy_report = validator.validate(synthetic, reference)

            result.privacy_risk = privacy_report.overall_risk
            result.privacy_passed = privacy_report.passed
            result.details['privacy'] = privacy_report.to_dict()

        return result

    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{data_id}", tags=["Data"])
async def download_data(
    data_id: str,
    format: str = Query("csv", pattern="^(csv|json|excel|parquet)$")
):
    """
    Download synthetic data

    Supports CSV, JSON, Excel, and Parquet formats
    """
    data = load_data(data_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Data not found")

    try:

        if format == "csv":
            buffer = io.StringIO()
            data.to_csv(buffer, index=False)
            content = buffer.getvalue()
            media_type = "text/csv"
            filename = f"synthetic_data_{data_id[:8]}.csv"

        elif format == "json":
            content = data.to_json(orient='records', indent=2)
            media_type = "application/json"
            filename = f"synthetic_data_{data_id[:8]}.json"

        elif format == "excel":
            buffer = io.BytesIO()
            data.to_excel(buffer, index=False)
            content = buffer.getvalue()
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename = f"synthetic_data_{data_id[:8]}.xlsx"

        elif format == "parquet":
            buffer = io.BytesIO()
            data.to_parquet(buffer)
            content = buffer.getvalue()
            media_type = "application/octet-stream"
            filename = f"synthetic_data_{data_id[:8]}.parquet"

        return StreamingResponse(
            io.BytesIO(content.encode() if isinstance(
                content, str) else content),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}", response_model=JobStatus, tags=["Jobs"])
async def get_job_status(job_id: str):
    """
    Get job status

    Returns current status and progress of a generation job
    """
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatus(**jobs_store[job_id])


@app.get("/jobs", tags=["Jobs"])
async def list_jobs(
    status: Optional[str] = Query(
        None, pattern="^(pending|running|completed|failed)$"),
    limit: int = Query(10, ge=1, le=100)
):
    """
    List all jobs

    Optionally filter by status
    """
    jobs = list(jobs_store.values())

    if status:
        jobs = [j for j in jobs if j['status'] == status]

    # Sort by created_at descending
    jobs.sort(key=lambda x: x['created_at'], reverse=True)

    return {
        "total": len(jobs),
        "jobs": jobs[:limit]
    }


@app.delete("/data/{data_id}", tags=["Data"])
async def delete_data(data_id: str, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)
    """
    Delete stored data

    Removes data from memory
    """
    existing = load_data(data_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Data not found")

    data_store.pop(data_id, None)
    path = _data_path(data_id)
    if path.exists():
        path.unlink()

    return {
        "message": "Data deleted successfully",
        "data_id": data_id
    }


@app.get("/presets", tags=["Configuration"])
async def list_presets():
    """
    List available configuration presets
    """
    loader = ConfigLoader()
    presets = loader.list_presets()

    return {
        "presets": presets,
        "count": len(presets)
    }


@app.get("/presets/{preset_name}", tags=["Configuration"])
async def get_preset(preset_name: str):
    """
    Get a configuration preset
    """
    try:
        loader = ConfigLoader()
        config = loader.load_preset(preset_name)

        return {
            "preset_name": preset_name,
            "config": config.to_dict()
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# Run with: uvicorn api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
