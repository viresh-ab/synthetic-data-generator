from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import io, json

from src.config import get_default_config
from src.orchestrator import DataOrchestrator, PipelineType
from src.schema import SchemaAnalyzer
from src.validation import QualityValidator, PrivacyValidator
from src.generators import (
    NumericGenerator, TextGenerator,
    PIIGenerator, TemporalGenerator, CategoricalGenerator
)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

STATE = {
    "reference": None,
    "synthetic": None,
    "schema": None,
    "config": get_default_config(),
    "quality": None,
    "privacy": None,
}

# ---------------- HOME ----------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# ---------------- UPLOAD ----------------
@app.get("/upload", response_class=HTMLResponse)
def upload_ui(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile):
    df = pd.read_csv(file.file)
    STATE["reference"] = df
    preview = df.head(10).to_html(classes="table table-sm")
    return templates.TemplateResponse(
        "upload.html",
        {"request": request, "preview": preview, "rows": len(df), "cols": len(df.columns)}
    )

# ---------------- CONFIGURE ----------------
@app.get("/configure", response_class=HTMLResponse)
def configure_ui(request: Request):
    return templates.TemplateResponse(
        "configure.html",
        {"request": request, "config": STATE["config"]}
    )

@app.post("/configure", response_class=HTMLResponse)
def save_config(
    request: Request,
    num_rows: int = Form(...),
    batch_size: int = Form(...),
    seed: int = Form(...),
    preserve_correlations: bool = Form(False),
):
    cfg = STATE["config"]
    cfg.generation.num_rows = num_rows
    cfg.generation.batch_size = batch_size
    cfg.generation.seed = seed
    cfg.numeric.preserve_correlations = preserve_correlations
    return templates.TemplateResponse(
        "configure.html",
        {"request": request, "config": cfg, "saved": True}
    )

# ---------------- GENERATE ----------------
@app.get("/generate", response_class=HTMLResponse)
def generate_ui(request: Request):
    return templates.TemplateResponse("generate.html", {"request": request})

@app.post("/generate", response_class=HTMLResponse)
def generate_data(request: Request):
    ref = STATE["reference"]
    cfg = STATE["config"]

    orch = DataOrchestrator(cfg)
    schema = orch.analyze_schema(ref)

    orch.register_pipeline(PipelineType.NUMERIC, NumericGenerator(cfg))
    orch.register_pipeline(PipelineType.TEXT, TextGenerator(cfg))
    orch.register_pipeline(PipelineType.PII, PIIGenerator(cfg))
    orch.register_pipeline(PipelineType.TEMPORAL, TemporalGenerator(cfg))
    orch.register_pipeline(PipelineType.HYBRID, CategoricalGenerator(cfg))

    result = orch.generate(cfg.generation.num_rows, ref, schema)

    STATE["synthetic"] = result.data
    preview = result.data.head(10).to_html(classes="table table-sm")

    return templates.TemplateResponse(
        "generate.html",
        {"request": request, "preview": preview}
    )

# ---------------- VALIDATE ----------------
@app.get("/validate", response_class=HTMLResponse)
def validate_ui(request: Request):
    return templates.TemplateResponse("validate.html", {"request": request})

@app.post("/validate", response_class=HTMLResponse)
def run_validation(request: Request):
    ref, syn, cfg = STATE["reference"], STATE["synthetic"], STATE["config"]

    q = QualityValidator(cfg).validate(ref, syn)
    p = PrivacyValidator(cfg).validate(syn, ref)

    STATE["quality"], STATE["privacy"] = q, p

    return templates.TemplateResponse(
        "validate.html",
        {"request": request, "quality": q, "privacy": p}
    )

# ---------------- EXPORT ----------------
@app.get("/export", response_class=HTMLResponse)
def export_ui(request: Request):
    return templates.TemplateResponse("export.html", {"request": request})

@app.get("/download")
def download(format: str):
    df = STATE["synthetic"]

    if format == "csv":
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return StreamingResponse(
            io.BytesIO(buf.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=synthetic.csv"}
        )