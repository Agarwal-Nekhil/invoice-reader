# invoice-reader

Turns an invoice PDF into the JSON fields you ask for. You send the PDF and a JSON template with any field names; a
small FastAPI service reads the text, falls back to OCR when the PDF is a scan, and has an LLM (Mistral Small) fill
the template. Built in 2025, during my AI/ML internship.

## How it works

`dev_test_check/` is the version to run:

1. Read the text layer of the first two pages (PyPDF2, with LangChain's PyPDFLoader as a fallback).
2. Under 100 characters of text means a scan: OCR the pages instead (Tesseract, via pdf2image).
3. One LLM call fills the template. Anything not in the document stays an empty string.
4. If more than 30% of the fields come back empty from the text layer, retry once with OCR.
5. The result carries a flag per field (1 found, 0 missing), so the caller knows what to check by hand.

The upload returns an ID straight away (HTTP 202) and the work runs in the background; you poll for the result.
Both endpoints need an `X-API-Key` header.

`dev_test_integrated/` is the first version. It split the document into chunks, embedded them (e5-large-v2 in
Chroma) and retrieved the top five chunks before the LLM call. The second version drops retrieval and sends the text
in one call, since an invoice of one or two pages fits in the model's context.

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
- `dev_test_integrated/`: the first version, with retrieval.
- `template.json`: an example template (invoice number, dates, vendor, customer, total, line items).
- `sample_invoice_text.pdf` and `sample_invoice_image.pdf`: made-up invoices, one with a text layer and one as an
  image, to try both paths.

## Limits

- Results live in memory, so they are lost when the service restarts; a real deployment would keep them in a
  database.
- Only the first two pages are read.
- Every value comes back as a string, line items included.
- No tests or CI yet.

## Stack

Python 3.11, FastAPI, PyPDF2, Tesseract (pytesseract, pdf2image), Mistral API, Docker. The first version also uses
LangChain, Chroma and Hugging Face embeddings.
