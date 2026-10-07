# invoice-reader

Turns an invoice PDF into the JSON fields you ask for. You send the PDF and a JSON template with any field names; a
small FastAPI service reads the text, falls back to OCR when the PDF is a scan, and has an LLM (Mistral Small) fill
the template. Built in 2025 during my AI/ML internship, starting from a basic RAG prototype I was handed.

## How it works

`dev_test_check/` is the version to run:

1. Read the text layer of the first two pages (PyPDF2, with LangChain's PyPDFLoader as a fallback).
2. Under 100 characters of text means a scan: OCR the pages instead (Tesseract, via pdf2image).
3. One LLM call fills the template. Anything not in the document stays an empty string.
4. If more than 30% of the fields come back empty from the text layer, retry once with OCR.
5. The result carries a flag per field (1 found, 0 missing), so the caller knows what to check by hand.

The upload returns an ID straight away (HTTP 202) and the work runs in the background; you poll for the result.
Both endpoints need an `X-API-Key` header.

`dev_test_integrated/` keeps the retrieval approach I was handed: split the document into chunks, embed them
(e5-large-v2 in Chroma) and send the top five chunks to the LLM, with my text-first, OCR and retry steps around it.
`dev_test_check/` drops retrieval and sends the text in one call, since an invoice of one or two pages fits in the
model's context.

## Run it

```bash
cd dev_test_check
printf "MISTRAL_KEY=your-mistral-key\nAPI_KEY=any-key-you-choose\n" > .env
docker build -t invoice-reader .
docker run --env-file .env -p 8000:8000 invoice-reader
```

Then, from the repo root:

```bash
curl -H "X-API-Key: any-key-you-choose" \
  -F "pdf_file=@sample_invoice_text.pdf" \
  -F "json_template_file=@template.json" \
  http://localhost:8000/upload-invoice

curl -H "X-API-Key: any-key-you-choose" http://localhost:8000/invoice-status/<invoice_id>
```

Interactive API docs are at http://localhost:8000/docs.

## What is here

- `dev_test_check/`: the service (text first, OCR fallback, one LLM call, OCR retry) and its Dockerfile.
- `dev_test_integrated/`: the same service with the retrieval step I was handed.
- `template.json`: an example template (invoice number, dates, vendor, customer, total, line items).
- `sample_invoice_text.pdf` and `sample_invoice_image.pdf`: made-up invoices, one with a text layer and one as an
  image, to try both paths.

## Limits

- Tested on a handful of real invoices. Handwritten ones were the hard case: when entries ignore the printed rows
  and columns, turning the page into plain text loses which value belongs to which field.
- Results live in memory, so they are lost when the service restarts (the later version below keeps them in a
  database).
- Only the first two pages are read.
- Every value comes back as a string, line items included.
- No tests or CI yet.

## What came next

This repo is the July 2025 version. The code that followed is not in it.

- **AWS Lambda.** The service moved to Lambda with an upload function and a background worker, but the OCR tools
  kept running into Lambda's package size limit (250 MB unzipped), and the container route never got Tesseract
  working.
- **Google Cloud Functions, the version that went live.** Three functions linked by Pub/Sub: one stores the PDF and
  template and returns an ID; a router reads the text layer and has the LLM fill the template, stopping there when
  few fields are missing; an OCR function runs EasyOCR on scans only. Results in Firestore, keys in Secret Manager,
  infrastructure in Terraform.
- **Reading scans better.** Layout-aware EasyOCR (text with its position, rebuilt into lines) read 19% more text on a
  test invoice. It also pushed the OCR function past its 4 GB memory limit; the logs showed it, and it moved to 8 GB.
  A rule for tables in the prompt found 2 more of 45 fields on another test invoice.
- **A one-step trial.** Gemini reading the page images straight to JSON, tested locally and weighed on cost.

## Stack

Python 3.11, FastAPI, PyPDF2, Tesseract (pytesseract, pdf2image), Mistral API, Docker. The first version also uses
LangChain, Chroma and Hugging Face embeddings.
