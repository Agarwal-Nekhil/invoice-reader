"""Offline tests for the invoice service: no Mistral key, no network.

Run from this folder (needs the packages in requirements.txt plus httpx):

    python -m unittest -v

The LLM call is replaced with a canned reply. The OCR test runs only where
Tesseract and Poppler are installed (they are in the Docker image).
"""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# main.py refuses to start without these two, so set them before importing it
os.environ["MISTRAL_KEY"] = "test-mistral-key"
os.environ["API_KEY"] = "test-api-key"

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
TEMPLATE = {"invoice_number": "", "date": "", "vendor_name": "", "total_amount": ""}
HEADERS = {"X-API-Key": "test-api-key"}


class HelperTests(unittest.TestCase):
    def test_code_fences_are_stripped(self):
        reply = '```json\n{"a": "1"}\n```'
        self.assertEqual(main.strip_markdown_code_block(reply), '{"a": "1"}')

    def test_text_layer_threshold_is_100_characters(self):
        self.assertFalse(main.is_text_enough("x" * 99))
        self.assertTrue(main.is_text_enough("x" * 100))
        self.assertFalse(main.is_text_enough(" " * 200))


class ExtractionTests(unittest.TestCase):
    def run_pipeline(self, llm_reply):
        with mock.patch.object(main, "extract_all_fields_at_once", return_value=llm_reply):
            return main.run_extraction_pipeline("some invoice text", TEMPLATE)

    def test_flags_follow_template_order(self):
        reply = json.dumps({"invoice_number": "INV-1", "date": "", "vendor_name": "Acme", "total_amount": ""})
        filled, flags = self.run_pipeline(reply)
        self.assertEqual(flags, [1, 0, 1, 0])
        self.assertEqual(filled["vendor_name"], "Acme")

    def test_fields_the_model_left_out_come_back_empty(self):
        filled, flags = self.run_pipeline(json.dumps({"invoice_number": "INV-1"}))
        self.assertEqual(list(filled), list(TEMPLATE))
        self.assertEqual(flags, [1, 0, 0, 0])

    def test_fenced_reply_is_parsed(self):
        filled, flags = self.run_pipeline('```json\n{"invoice_number": "INV-1"}\n```')
        self.assertEqual(filled["invoice_number"], "INV-1")

    def test_unparseable_reply_gives_an_empty_template(self):
        filled, flags = self.run_pipeline("Sorry, I cannot read this.")
        self.assertEqual(filled, {k: "" for k in TEMPLATE})
        self.assertEqual(flags, [0, 0, 0, 0])


class RoutingTests(unittest.TestCase):
    """Which text source is used, and when OCR is tried again."""

    LONG_TEXT = "invoice text " * 20
    ALL_FOUND = ({}, [1] * 10)

    def setUp(self):
        pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        pdf.close()
        self.pdf_path = pdf.name
        main.invoice_processing_tasks["job"] = {
            "status": main.InvoiceStatus.RECEIVED,
            "filled_json": None,
            "flag_array": None,
            "error_detail": None,
            "method_used": None,
        }

    def tearDown(self):
        main.invoice_processing_tasks.clear()
        if os.path.exists(self.pdf_path):
            os.remove(self.pdf_path)

    def run_job(self, text, replies, ocr_text="text from OCR " * 20):
        with mock.patch.object(main, "extract_text_from_pdf", return_value=text), \
                mock.patch.object(main, "ocr_pdf_to_text", return_value=ocr_text) as ocr, \
                mock.patch.object(main, "run_extraction_pipeline", side_effect=replies) as pipeline:
            main.process_invoice_document_background("job", self.pdf_path, TEMPLATE)
        return ocr, pipeline, main.invoice_processing_tasks["job"]

    def test_good_text_layer_skips_ocr(self):
        ocr, pipeline, task = self.run_job(self.LONG_TEXT, [self.ALL_FOUND])
        ocr.assert_not_called()
        self.assertEqual(pipeline.call_count, 1)
        self.assertEqual(task["status"], main.InvoiceStatus.COMPLETED)
        self.assertEqual(task["method_used"], "text")

    def test_page_with_almost_no_text_goes_to_ocr(self):
        ocr, pipeline, task = self.run_job("x" * 99, [self.ALL_FOUND])
        ocr.assert_called_once()
        self.assertEqual(pipeline.call_count, 1)
        self.assertEqual(task["method_used"], "ocr")

    def test_retries_with_ocr_when_more_than_30_percent_is_empty(self):
        four_of_ten_empty = ({}, [1] * 6 + [0] * 4)
        ocr, pipeline, task = self.run_job(self.LONG_TEXT, [four_of_ten_empty, self.ALL_FOUND])
        ocr.assert_called_once()
        self.assertEqual(pipeline.call_count, 2)
        self.assertEqual(task["flag_array"], [1] * 10)
        self.assertEqual(task["method_used"], "ocr")

    def test_exactly_30_percent_empty_is_not_retried(self):
        three_of_ten_empty = ({}, [1] * 7 + [0] * 3)
        ocr, pipeline, task = self.run_job(self.LONG_TEXT, [three_of_ten_empty])
        ocr.assert_not_called()
        self.assertEqual(task["method_used"], "text")

    def test_no_second_retry_after_ocr(self):
        mostly_empty = ({}, [0] * 10)
        ocr, pipeline, task = self.run_job("x" * 10, [mostly_empty])
        self.assertEqual(pipeline.call_count, 1)
        self.assertEqual(task["status"], main.InvoiceStatus.COMPLETED)

    def test_failure_is_recorded_and_the_temp_pdf_is_removed(self):
        ocr, pipeline, task = self.run_job(self.LONG_TEXT, [RuntimeError("model unavailable")])
        self.assertEqual(task["status"], main.InvoiceStatus.FAILED)
        self.assertIn("model unavailable", task["error_detail"])
        self.assertFalse(os.path.exists(self.pdf_path))


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.template = json.loads((SAMPLES / "template.json").read_text())

    def upload(self, headers=HEADERS, template_bytes=None, pdf_name="invoice_text.pdf"):
        if template_bytes is None:
            template_bytes = json.dumps(self.template).encode()
        with open(SAMPLES / pdf_name, "rb") as pdf:
            return self.client.post(
                "/upload-invoice",
                headers=headers,
                files={
                    "pdf_file": ("invoice.pdf", pdf, "application/pdf"),
                    "json_template_file": ("template.json", template_bytes, "application/json"),
                },
            )

    def test_wrong_key_is_rejected(self):
        self.assertEqual(self.upload(headers={"X-API-Key": "nope"}).status_code, 401)

    def test_missing_key_is_rejected(self):
        self.assertIn(self.upload(headers={}).status_code, (401, 403))

    def test_template_must_be_a_json_object(self):
        self.assertEqual(self.upload(template_bytes=b"not json").status_code, 400)
        self.assertEqual(self.upload(template_bytes=b'["a", "b"]').status_code, 400)

    def test_unknown_id_is_404(self):
        self.assertEqual(self.client.get("/invoice-status/nope", headers=HEADERS).status_code, 404)

    def test_text_sample_end_to_end(self):
        reply = json.dumps({
            "invoice_number": "INV-2023-0456",
            "date": "2023-11-15",
            "vendor_name": "Acme Supplies",
            "total_amount": "1299.99",
        })
        with mock.patch.object(main, "extract_all_fields_at_once", return_value=reply):
            response = self.upload()  # the background job runs before the test client returns
        self.assertEqual(response.status_code, 202)
        invoice_id = response.json()["invoice_id"]

        status = self.client.get(f"/invoice-status/{invoice_id}", headers=HEADERS)
        self.assertEqual(status.status_code, 200)
        body = status.json()
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["filled_json"]["invoice_number"], "INV-2023-0456")
        # template order: number, date, vendor name, vendor address, customer name,
        # customer address, total, due date, line items
        self.assertEqual(body["flag_array"], [1, 1, 1, 0, 0, 0, 1, 0, 0])
        self.assertEqual(main.invoice_processing_tasks[invoice_id]["method_used"], "text")

    def test_image_sample_has_no_text_layer(self):
        text = main.extract_text_from_pdf(str(SAMPLES / "invoice_image.pdf"))
        self.assertFalse(main.is_text_enough(text))

    @unittest.skipUnless(shutil.which("tesseract") and shutil.which("pdftoppm"),
                         "Tesseract and Poppler are not installed")
    def test_ocr_reads_the_image_sample(self):
        text = main.ocr_pdf_to_text(str(SAMPLES / "invoice_image.pdf"))
        self.assertIn("INV-2023-0456", text)


if __name__ == "__main__":
    unittest.main()
