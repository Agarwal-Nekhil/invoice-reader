import json
import logging
import os
import secrets
import uuid
from enum import Enum

from dotenv import load_dotenv
from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.routing import APIRoute
from fastapi.security import APIKeyHeader
from mistralai import Mistral
from pdf2image import convert_from_path
from pydantic import BaseModel
from pypdf import PdfReader
import pytesseract
from starlette.requests import Request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("invoice_reader")

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_KEY")
APP_API_KEY = os.getenv("API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_KEY environment variable not set. Please check your .env file.")
if not APP_API_KEY:
    raise ValueError("API_KEY environment variable not set. Please check your .env file.")

# --- Settings (constants, not environment variables) ---
MODEL_NAME = "mistral-small-latest"
MAX_PAGES = 2                       # pages read from each PDF
MIN_TEXT_CHARS = 100                # fewer characters than this in the text layer means a scan
MISSING_THRESHOLD = 0.3             # retry with OCR when more than this share of the fields is empty
MAX_PROMPT_CHARS = 16000            # the document text is cut here before the LLM call
LLM_TIMEOUT_MS = 60_000
OCR_TIMEOUT_S = 60                  # for Poppler and for Tesseract, each
MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_TEMPLATE_BYTES = 64 * 1024
MAX_REQUEST_BYTES = MAX_PDF_BYTES + MAX_TEMPLATE_BYTES + 64 * 1024   # the two files plus multipart overhead
MAX_STORED_RESULTS = 1000           # oldest results are dropped beyond this
EMPTY_VALUES = {"", "n/a", "null", "none", "not found"}   # what a model writes when it found nothing
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_invoices")

client = Mistral(api_key=MISTRAL_API_KEY)

app = FastAPI(
    title="Invoice Reader API",
    description="Fills a JSON template from a PDF invoice: text layer first, OCR when needed, one LLM call."
)

# --- API key ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def check_api_key(api_key: str | None) -> None:
    """Raises 401 unless the X-API-Key value matches (compared in constant time)."""
    if api_key is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    if not secrets.compare_digest(api_key.encode(), APP_API_KEY.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "APIKey"},
        )

async def get_api_key(api_key: str = Depends(api_key_header)) -> str:
    """Puts the key field in /docs and checks it again. KeyFirstRoute does the real gatekeeping."""
    check_api_key(api_key)
    return api_key

class KeyFirstRoute(APIRoute):
    """FastAPI reads a multipart body before it runs dependencies, so check the key and the size first."""

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def route_handler(request: Request):
            check_api_key(request.headers.get("X-API-Key"))
            length = request.headers.get("content-length", "")
            if length.isdigit() and int(length) > MAX_REQUEST_BYTES:
                raise HTTPException(status_code=413, detail="The request is too large.")
            return await handler(request)

        return route_handler

router = APIRouter(route_class=KeyFirstRoute, dependencies=[Depends(get_api_key)])

# --- Job state, kept in memory ---
class InvoiceStatus(str, Enum):
    RECEIVED = "received"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class JobError(Exception):
    """A failure whose message is safe to show the caller."""

invoice_processing_tasks: dict[str, dict] = {}

def register_task(invoice_id: str) -> None:
    invoice_processing_tasks[invoice_id] = {
        "status": InvoiceStatus.RECEIVED,
        "filled_json": None,
        "flag_array": None,
        "error_detail": None,
        "method_used": None,
    }
    while len(invoice_processing_tasks) > MAX_STORED_RESULTS:
        invoice_processing_tasks.pop(next(iter(invoice_processing_tasks)))

def update_task(invoice_id: str, **fields) -> None:
    task = invoice_processing_tasks.get(invoice_id)  # None if it was dropped to stay under the cap
    if task is not None:
        task.update(fields)

# --- API models ---
class UploadInvoiceResponse(BaseModel):
    invoice_id: str
    status: InvoiceStatus
    message: str

class InvoiceProcessingStatusResponse(BaseModel):
    invoice_id: str
    status: InvoiceStatus
    message: str
    filled_json: dict[str, str] | None = None
    flag_array: list[int] | None = None
    error_detail: str | None = None

# --- Reading the PDF ---
def extract_text_from_pdf(pdf_path: str) -> str:
    """Text layer of the first pages ("" if the file cannot be read)."""
    try:
        with open(pdf_path, "rb") as file:
            pages = PdfReader(file).pages[:MAX_PAGES]
            return "\n\n".join(page.extract_text() or "" for page in pages)
    except Exception as e:
        log.warning("Could not read the text layer (%s).", e)
        return ""

def is_text_enough(text: str, min_length: int = MIN_TEXT_CHARS) -> bool:
    return len(text.strip()) >= min_length

def ocr_pdf_to_text(pdf_path: str) -> str:
    """OCR of the first pages with Tesseract (OEM 3, PSM 6: one uniform block of text per page)."""
    try:
        # On Windows, you might need to point pytesseract at tesseract.exe:
        # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        images = convert_from_path(pdf_path, last_page=MAX_PAGES, timeout=OCR_TIMEOUT_S)
        return "\n\n".join(
            pytesseract.image_to_string(img, config="--oem 3 --psm 6", timeout=OCR_TIMEOUT_S) for img in images
        )
    except Exception as e:
        log.warning("OCR failed: %s", e)
        return ""

# --- Filling the template ---
def extract_all_fields_at_once(document_text: str, json_template: dict) -> str | None:
    """One LLM call for the whole template; the text is cut at MAX_PROMPT_CHARS. Returns the raw reply."""
    if len(document_text) > MAX_PROMPT_CHARS:
        document_text = document_text[:MAX_PROMPT_CHARS] + "\n[TRUNCATED - Invoice content continues...]"

    prompt = f"""
You are a highly accurate structured data extraction assistant. Your job is to extract structured fields from invoices.

Your task is to read the DOCUMENT provided below and fill in the JSON template with the extracted values. Return ONLY valid JSON matching the template.

Strict rules:
- If a field is not found in the document, leave its value as an empty string.
- Do NOT include any extra text or commentary.
- Do NOT change the field names or JSON structure.
- Return ONLY the JSON object as the answer, with correct JSON syntax.

DOCUMENT:
{document_text}

JSON TEMPLATE:
{json.dumps(json_template, indent=2)}

YOUR RESPONSE:
"""
    response = client.chat.complete(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        timeout_ms=LLM_TIMEOUT_MS,
    )
    return response.choices[0].message.content

def parse_reply(content) -> dict:
    """The JSON object in a model reply, or {} if there is none. Tolerates code fences and words around it."""
    if not isinstance(content, str):
        return {}
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end < start:
        return {}
    try:
        parsed = json.loads(content[start:end + 1])
    except (json.JSONDecodeError, RecursionError):
        return {}
    return parsed if isinstance(parsed, dict) else {}

def as_text(value) -> str:
    """The status response types every value as a string: lists and numbers become text, "n/a" and nulls become ""."""
    if value is None or value in ([], {}):
        return ""
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value).strip()
    return "" if text.lower() in EMPTY_VALUES else text

def run_extraction_pipeline(raw_text: str, json_template: dict) -> tuple[dict[str, str], list[int]]:
    """Returns the filled template and one flag per field, in template order (1 found, 0 missing)."""
    parsed = parse_reply(extract_all_fields_at_once(raw_text, json_template))
    if not parsed:
        log.warning("The model reply had no usable JSON object.")

    filled_json = {k: as_text(parsed.get(k, "")) for k in json_template}
    flag_array = [1 if filled_json[k] else 0 for k in json_template]
    return filled_json, flag_array

def process_invoice_document_background(
    invoice_id: str,
    pdf_path: str,
    json_template: dict,
    missing_threshold: float = MISSING_THRESHOLD,
):
    """Runs as a background task: text layer, OCR if needed, one LLM call, one OCR retry if too much is empty."""
    try:
        update_task(invoice_id, status=InvoiceStatus.PROCESSING)

        method_used = "text"
        text = extract_text_from_pdf(pdf_path)
        if not is_text_enough(text):
            ocr_text = ocr_pdf_to_text(pdf_path)
            if len(ocr_text.strip()) > len(text.strip()):  # a short text layer beats OCR that found nothing
                text, method_used = ocr_text, "ocr"
        if not text.strip():
            raise JobError("No text could be read from the PDF. If it is a scan, check that Tesseract and Poppler are installed.")

        filled_json, flag_array = run_extraction_pipeline(text, json_template)
        missing_rate = flag_array.count(0) / len(flag_array) if flag_array else 1.0

        if method_used == "text" and missing_rate > missing_threshold:
            log.info("Invoice %s: %.0f%% of the fields are empty, reading it again with OCR.", invoice_id, missing_rate * 100)
            try:
                ocr_text = ocr_pdf_to_text(pdf_path)
                if ocr_text.strip():
                    ocr_json, ocr_flags = run_extraction_pipeline(ocr_text, json_template)
                    if sum(ocr_flags) > sum(flag_array):  # keep the text-layer result unless OCR found more
                        filled_json, flag_array, method_used = ocr_json, ocr_flags, "ocr"
            except Exception:
                log.exception("Invoice %s: the OCR retry failed, keeping the first result.", invoice_id)

        update_task(
            invoice_id,
            status=InvoiceStatus.COMPLETED,
            filled_json=filled_json,
            flag_array=flag_array,
            method_used=method_used,
        )
    except JobError as e:
        log.warning("Invoice %s failed: %s", invoice_id, e)
        update_task(invoice_id, status=InvoiceStatus.FAILED, error_detail=str(e))
    except Exception as e:
        log.exception("Invoice %s failed.", invoice_id)
        update_task(
            invoice_id,
            status=InvoiceStatus.FAILED,
            error_detail=f"Processing failed ({type(e).__name__}). See the service log.",
        )
    finally:
        try:
            os.remove(pdf_path)
        except FileNotFoundError:
            pass
        except OSError as e:
            log.warning("Could not remove %s: %s", pdf_path, e)

# --- Endpoints ---
async def read_limited(upload: UploadFile, limit: int, too_big: str) -> bytes:
    data = await upload.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=too_big)
    return data

def parse_template(raw: bytes) -> dict:
    try:
        template = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON template provided. Please ensure it's valid JSON.",
        )
    if not isinstance(template, dict) or not template:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The JSON template must be a JSON object with at least one field.",
        )
    return template

@router.post("/upload-invoice", response_model=UploadInvoiceResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_invoice(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(..., description="PDF invoice file"),
    json_template_file: UploadFile = File(..., description="JSON template file"),
):
    """Takes a PDF and a JSON template and starts the job. Returns an invoice ID to poll."""
    pdf_bytes = await read_limited(pdf_file, MAX_PDF_BYTES, "The PDF is larger than 10 MB.")
    template_bytes = await read_limited(json_template_file, MAX_TEMPLATE_BYTES, "The template is larger than 64 KB.")
    if b"%PDF-" not in pdf_bytes[:1024]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The uploaded file is not a PDF.")
    json_template = parse_template(template_bytes)

    invoice_id = str(uuid.uuid4())
    pdf_path = os.path.join(TEMP_DIR, f"{invoice_id}.pdf")
    try:
        os.makedirs(TEMP_DIR, exist_ok=True)
        with open(pdf_path, "wb") as buffer:
            buffer.write(pdf_bytes)
    except OSError:
        log.exception("Could not save the upload for invoice %s.", invoice_id)
        try:
            os.remove(pdf_path)
        except OSError:
            pass
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not save the uploaded file.")

    register_task(invoice_id)
    background_tasks.add_task(process_invoice_document_background, invoice_id, pdf_path, json_template)
    return UploadInvoiceResponse(
        invoice_id=invoice_id,
        status=InvoiceStatus.RECEIVED,
        message="Invoice received and processing initiated.",
    )

@router.get("/invoice-status/{invoice_id}", response_model=InvoiceProcessingStatusResponse)
async def get_invoice_status(invoice_id: str):
    """Status of a job. Once completed it carries the filled template and the flags; if it failed, error_detail says why."""
    task_info = invoice_processing_tasks.get(invoice_id)
    if not task_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Invoice with ID '{invoice_id}' is unknown, or its result is no longer kept.",
        )

    current_status = task_info["status"]
    if current_status == InvoiceStatus.COMPLETED:
        return InvoiceProcessingStatusResponse(
            invoice_id=invoice_id,
            status=current_status,
            message="Processing complete.",
            filled_json=task_info["filled_json"],
            flag_array=task_info["flag_array"],
        )
    if current_status == InvoiceStatus.FAILED:
        return InvoiceProcessingStatusResponse(
            invoice_id=invoice_id,
            status=current_status,
            message="Processing failed.",
            error_detail=task_info["error_detail"],
        )
    return InvoiceProcessingStatusResponse(
        invoice_id=invoice_id,
        status=current_status,
        message=f"Invoice processing is {current_status.value}.",
    )

app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Invoice Reader API is running. Check /docs for API documentation."}
