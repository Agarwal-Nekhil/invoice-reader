# invoice-reader

Send a PDF invoice and a JSON template, get the template back filled in, with a found or missing flag for every
field. A small FastAPI service reads the PDF's text layer, falls back to OCR for scans, and has an LLM (Mistral
Small) fill the template. The template can have any field names.

I built it in 2025 during my AI/ML internship, starting from a basic RAG prototype I was handed. This repo is the
July 2025 FastAPI and Docker version. The version that went live later ran on Google Cloud Functions and is not in
this repo (see [What came next](#what-came-next)).

## How it reads an invoice

```
POST /upload-invoice   (PDF + JSON template, X-API-Key header)
      |
      v
FastAPI saves the PDF and replies at once: 202 + invoice_id ----> the client polls
      |  (the rest runs as a background task)                     GET /invoice-status/{invoice_id}
      v                                                           received, processing, completed, failed
text layer of the first two pages (PyPDF2)
      |
      |  under 100 characters of text? ---- yes ----> OCR the pages instead (Tesseract)
      v
one Mistral call fills the template; anything not in the document stays ""
      |
      v
flag per field: 1 found, 0 missing
      |
      |  more than 30% missing and the text layer was used? ---- yes ----> OCR the pages, ask once more
      v
result kept in memory, temporary PDF deleted
```

Why it is built this way:

- **Text layer first.** Reading the text a PDF already carries is quick and does not misread characters the way OCR
  can. OCR runs only when the pages have almost no text, which is what a scan looks like.
- **Retry with OCR.** If more than 30% of the fields come back empty from the text layer, the OCR text may do better,
  so it reads the pages again with OCR and asks once more. This happens at most once per invoice.
- **One LLM call, no retrieval.** An invoice of one or two pages fits in the model's context window, so the whole
  text goes in one call (cut at 16,000 characters as a guard). The earlier version chunked the text, embedded it and
  sent the top five chunks to the model; that step added a vector store and an embedding model for no gain on
  documents this short. It is kept in `with_retrieval/` for comparison.
- **A flag per field.** The caller can tell which values were found and which to check by hand.
- **202 and polling.** The upload returns straight away and the work runs in the background, so a slow OCR pass or
  model call does not hold the request open.

## Run it

You need Docker and a Mistral API key.

```sh
cp .env.example .env    # then set MISTRAL_KEY and API_KEY in .env (see Configuration)
docker build -t invoice-reader service
docker run --rm --env-file .env -p 8000:8000 invoice-reader
```

From a second terminal, in the repo root. Use the same value you put in `API_KEY`:

```sh
curl -H "X-API-Key: your-api-key" \
  -F "pdf_file=@samples/invoice_text.pdf" \
  -F "json_template_file=@samples/template.json" \
  http://localhost:8000/upload-invoice
```

On Windows PowerShell use `curl.exe` and put the command on one line.

The reply is HTTP 202 with an ID:

```json
{
  "invoice_id": "3f6d2a52-8c1b-4e1f-9a57-0c6f7b8e2d41",
  "status": "received",
  "message": "Invoice received and processing initiated."
}
```

Poll with that ID until `status` is `completed`:

```sh
curl -H "X-API-Key: your-api-key" http://localhost:8000/invoice-status/3f6d2a52-8c1b-4e1f-9a57-0c6f7b8e2d41
```

Example of a completed response for `samples/invoice_text.pdf`. I wrote it by hand from the sample invoice to show
the shape; the exact wording of each value comes from the model on your run. `flag_array` has one entry per template
field, in template order.

```json
{
  "invoice_id": "3f6d2a52-8c1b-4e1f-9a57-0c6f7b8e2d41",
  "status": "completed",
  "message": "Processing complete.",
  "filled_json": {
    "invoice_number": "INV-2023-0456",
    "date": "2023-11-15",
    "vendor_name": "Acme Supplies",
    "vendor_address": "123 Business Rd, New York, NY",
    "customer_name": "John Doe",
    "customer_address": "456 Customer Ave, Boston, MA",
    "total_amount": "1,299.99",
    "due_date": "2023-12-15",
    "line_items": "Widget A, 2 units at 500 each; Widget B, 1 unit at 299.99"
  },
  "flag_array": [1, 1, 1, 1, 1, 1, 1, 1, 1],
  "error_detail": null
}
```

`samples/invoice_image.pdf` has no text layer, so it takes the OCR path. Interactive API docs are at
http://localhost:8000/docs.

### Without Docker

You need Python 3.11, and Tesseract and Poppler on your PATH (Debian or Ubuntu: `apt install tesseract-ocr
poppler-utils`; macOS: `brew install tesseract poppler`; Windows: install both and add them to PATH).

```sh
cp .env.example .env    # then set MISTRAL_KEY and API_KEY in .env
cd service
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --port 8000
```

## The API

| Call | Send | Reply |
|---|---|---|
| `POST /upload-invoice` | multipart form: `pdf_file` and `json_template_file` | 202 with `invoice_id`. 400 if the template is not a JSON object. 401 for a wrong key. |
| `GET /invoice-status/{invoice_id}` | nothing | 200 with `status`; once `completed` it also has `filled_json` and `flag_array`. 404 for an unknown ID. 500 with the reason if the job failed. |

Both calls need the `X-API-Key` header.

## Configuration

| Variable | Needed | What it is |
|---|---|---|
| `MISTRAL_KEY` | yes | Your Mistral API key. |
| `API_KEY` | yes | The key callers send in `X-API-Key`. Any long random string. |

Copy `.env.example` to `.env` and fill both in. `.env` is ignored by git, and the service stops at startup if
either value is empty. The invoice text is sent to the Mistral API, so do not run invoices through it unless that is
allowed.

These are constants in `service/main.py`, not environment variables: the model (`mistral-small-latest`), the
100-character OCR threshold, the 30% retry threshold, the first two pages, and the 16,000-character cap.

## Tests

The tests replace the LLM call with a canned reply, so they need no key and no network. They cover the routing (text
layer, OCR, the 30% retry rule), the per-field flags and the API (key check, 202, polling). The OCR test runs only
where Tesseract and Poppler are installed, as in the Docker image.

```sh
cd service
pip install -r requirements.txt httpx
python -m unittest -v
```

## Limits

- Tested on about 50 to 100 real invoices. They are not in this repo; the samples here are made up. Handwritten
  Indian invoices were the hard case: when entries ignore the printed rows and columns, turning the page into plain
  text loses which value belongs to which field.
- I also tried a vision model (Gemini) that reads the page image straight into the JSON. It was tested locally only
  and weighed against its cost.
- Results live in memory, so a restart loses them, and several worker processes would not share them.
- Only the first two pages are read, and OCR is set up for English only.
- Templates are flat: field names with empty values. Every value comes back as a string, line items included.
- No CI.

## What came next

This repo is the July 2025 version. The code that followed is not in it.

- **AWS Lambda.** The service was split into router, OCR and text functions. Tesseract and Poppler as Lambda layers
  ran into Lambda's 250 MB unzipped size limit, so it never went live there.
- **Google Cloud Functions, the version that went live.** Four functions linked by Pub/Sub (upload, routing, OCR,
  status), with Firestore, Secret Manager and Terraform. The routing function read the text layer and filled the
  template, and stopped there when few fields were missing; the OCR function ran EasyOCR on the rest.
- **Reading scans better.** Layout-aware EasyOCR (text with its position, rebuilt into lines) read 19% more text on
  a test invoice. It also ran the OCR function out of memory; the function logs showed it, and I raised the memory
  limit. I added a rule for tables to the prompt.

## What is here

- `service/`: the service to run (`main.py`), its Dockerfile, requirements and tests.
- `with_retrieval/`: the same service with the retrieval step I was handed (chunks, e5-large-v2 embeddings in Chroma,
  top five to the LLM), with my text-first, OCR and retry steps around it.
- `samples/`: a made-up invoice with a text layer, a made-up invoice as an image, and an example template.
- `.env.example`: the two variables the service needs.

## Stack

Python 3.11, FastAPI, PyPDF2, Tesseract (pytesseract, pdf2image), the Mistral API, Docker. `with_retrieval/` also
uses Chroma and Hugging Face embeddings.
