"""Offline tests for the invoice service: no Mistral key, no network.

Run from this folder (needs the packages in requirements.txt plus httpx):

    python -m unittest -v

The LLM call is replaced with a canned reply. The OCR test runs only where
Tesseract and Poppler are installed.
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


class ReplyParsingTests(unittest.TestCase):
    def test_plain_and_fenced_replies(self):
        self.assertEqual(main.parse_reply('{"a": "1"}'), {"a": "1"})
        self.assertEqual(main.parse_reply('```json\n{"a": "1"}\n```'), {"a": "1"})

    def test_words_around_the_object_are_ignored(self):
        self.assertEqual(main.parse_reply('Here is the JSON:\n```json\n{"a": "1"}\n```\nNote: done.'), {"a": "1"})

    def test_one_line_fence_and_nested_braces(self):
        self.assertEqual(main.parse_reply('```json {"a": "1"} ```'), {"a": "1"})
        self.assertEqual(main.parse_reply('{"a": {"b": "x}"}}'), {"a": {"b": "x}"}})

    def test_anything_else_gives_an_empty_dict(self):
        for reply in ('["a"]', "no json here", '{"a": ', "", None, [{"type": "text", "text": "{}"}]):
            self.assertEqual(main.parse_reply(reply), {}, reply)


class ValueTextTests(unittest.TestCase):
    def test_lists_numbers_and_nulls_become_text(self):
        self.assertEqual(main.as_text(456), "456")
        self.assertEqual(main.as_text(None), "")
        self.assertEqual(main.as_text([]), "")
        self.assertEqual(main.as_text([{"item": "Widget A"}]), '[{"item": "Widget A"}]')

    def test_blank_looking_values_count_as_missing(self):
        for value in ("", "  ", "\n", "N/A", "null", "None", "Not found"):
            self.assertEqual(main.as_text(value), "", repr(value))
        self.assertEqual(main.as_text("  Acme Supplies "), "Acme Supplies")

    def test_text_layer_threshold_is_100_characters(self):
        self.assertFalse(main.is_text_enough("x" * 99))
        self.assertTrue(main.is_text_enough("x" * 100))
        self.assertFalse(main.is_text_enough(" " * 200))


class LimitsTests(unittest.TestCase):
    def call_model(self, text):
        fake_client = mock.MagicMock()
        fake_client.chat.complete.return_value.choices[0].message.content = "{}"
        with mock.patch.object(main, "client", fake_client):
            main.extract_all_fields_at_once(text, TEMPLATE)
        return fake_client.chat.complete.call_args.kwargs

    def test_the_model_call_has_a_timeout(self):
        self.assertEqual(self.call_model("text")["timeout_ms"], main.LLM_TIMEOUT_MS)

    def test_long_documents_are_cut_before_the_model_call(self):
        prompt = self.call_model("a" * (main.MAX_PROMPT_CHARS + 500))["messages"][0]["content"]
        self.assertIn("[TRUNCATED", prompt)
        self.assertNotIn("a" * (main.MAX_PROMPT_CHARS + 1), prompt)

    def test_only_the_newest_results_are_kept(self):
        with mock.patch.object(main, "MAX_STORED_RESULTS", 2):
            for name in ("first", "second", "third"):
                main.register_task(name)
        try:
            self.assertEqual(list(main.invoice_processing_tasks), ["second", "third"])
        finally:
            main.invoice_processing_tasks.clear()

    def test_updating_a_dropped_job_does_nothing(self):
        main.update_task("never-registered", status=main.InvoiceStatus.FAILED)
        self.assertNotIn("never-registered", main.invoice_processing_tasks)


class TextLayerTests(unittest.TestCase):
    def test_text_sample_has_a_text_layer(self):
        text = main.extract_text_from_pdf(str(SAMPLES / "invoice_text.pdf"))
        self.assertIn("INV-2023-0456", text)
        self.assertTrue(main.is_text_enough(text))

    def test_a_file_that_cannot_be_read_gives_no_text(self):
        with tempfile.TemporaryDirectory() as folder:
            broken = Path(folder) / "broken.pdf"
            broken.write_bytes(b"%PDF-1.4 this is not a real pdf")
            self.assertEqual(main.extract_text_from_pdf(str(broken)), "")

    def test_image_sample_has_no_text_layer(self):
        text = main.extract_text_from_pdf(str(SAMPLES / "invoice_image.pdf"))
        self.assertFalse(main.is_text_enough(text))

    @unittest.skipUnless(shutil.which("tesseract") and shutil.which("pdftoppm"),
                         "Tesseract and Poppler are not installed")
    def test_ocr_reads_the_image_sample(self):
        text = main.ocr_pdf_to_text(str(SAMPLES / "invoice_image.pdf"))
        self.assertIn("INV-2023-0456", text)


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

    def test_a_fenced_reply_with_a_preface_is_parsed(self):
        filled, flags = self.run_pipeline('Sure!\n```json\n{"invoice_number": "INV-1"}\n```')
        self.assertEqual(filled["invoice_number"], "INV-1")

    def test_lists_numbers_and_nulls_in_the_reply_become_text(self):
        reply = json.dumps({"invoice_number": 456, "date": None, "vendor_name": "N/A",
                            "total_amount": [{"item": "Widget A", "qty": 2}]})
        filled, flags = self.run_pipeline(reply)
        self.assertEqual(filled["invoice_number"], "456")
        self.assertEqual(filled["total_amount"], '[{"item": "Widget A", "qty": 2}]')
        self.assertEqual(flags, [1, 0, 0, 1])
        self.assertTrue(all(isinstance(v, str) for v in filled.values()))

    def test_a_reply_without_a_json_object_gives_an_empty_template(self):
        for reply in ("Sorry, I cannot read this.", '["INV-1", "Acme"]', None):
            filled, flags = self.run_pipeline(reply)
            self.assertEqual(filled, {k: "" for k in TEMPLATE})
            self.assertEqual(flags, [0, 0, 0, 0])


class RoutingTests(unittest.TestCase):
    """Which text source is used, and when OCR is tried again."""

    LONG_TEXT = "invoice text " * 20
    ALL_FOUND = ({}, [1] * 10)
    FOUR_OF_TEN_EMPTY = ({}, [1] * 6 + [0] * 4)

    def setUp(self):
        pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        pdf.close()
        self.pdf_path = pdf.name
        main.register_task("job")

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

    def test_a_short_text_layer_is_kept_when_ocr_reads_nothing(self):
        ocr, pipeline, task = self.run_job("Invoice 42 Acme Total 99", [self.ALL_FOUND], ocr_text="")
        self.assertEqual(task["status"], main.InvoiceStatus.COMPLETED)
        self.assertEqual(task["method_used"], "text")

    def test_retries_with_ocr_when_more_than_30_percent_is_empty(self):
        ocr, pipeline, task = self.run_job(self.LONG_TEXT, [self.FOUR_OF_TEN_EMPTY, self.ALL_FOUND])
        ocr.assert_called_once()
        self.assertEqual(pipeline.call_count, 2)
        self.assertEqual(task["flag_array"], [1] * 10)
        self.assertEqual(task["method_used"], "ocr")

    def test_exactly_30_percent_empty_is_not_retried(self):
        three_of_ten_empty = ({}, [1] * 7 + [0] * 3)
        ocr, pipeline, task = self.run_job(self.LONG_TEXT, [three_of_ten_empty])
        ocr.assert_not_called()
        self.assertEqual(task["method_used"], "text")

    def test_retry_keeps_the_text_result_when_ocr_finds_fewer_fields(self):
        worse_ocr_result = ({"kept": "ocr"}, [1] * 2 + [0] * 8)
        text_result = ({"kept": "text"}, [1] * 6 + [0] * 4)
        ocr, pipeline, task = self.run_job(self.LONG_TEXT, [text_result, worse_ocr_result])
        self.assertEqual(pipeline.call_count, 2)
        self.assertEqual(task["filled_json"], {"kept": "text"})
        self.assertEqual(task["method_used"], "text")

    def test_retry_is_skipped_when_ocr_reads_nothing(self):
        ocr, pipeline, task = self.run_job(self.LONG_TEXT, [self.FOUR_OF_TEN_EMPTY], ocr_text="")
        self.assertEqual(pipeline.call_count, 1)
        self.assertEqual(task["status"], main.InvoiceStatus.COMPLETED)
        self.assertEqual(task["method_used"], "text")

    def test_a_failing_retry_keeps_the_first_result(self):
        ocr, pipeline, task = self.run_job(self.LONG_TEXT, [self.FOUR_OF_TEN_EMPTY, TimeoutError("model timed out")])
        self.assertEqual(task["status"], main.InvoiceStatus.COMPLETED)
        self.assertEqual(task["flag_array"], [1] * 6 + [0] * 4)
        self.assertEqual(task["method_used"], "text")

    def test_no_second_retry_after_ocr(self):
        mostly_empty = ({}, [0] * 10)
        ocr, pipeline, task = self.run_job("x" * 10, [mostly_empty])
        self.assertEqual(pipeline.call_count, 1)
        self.assertEqual(task["status"], main.InvoiceStatus.COMPLETED)

    def test_a_pdf_with_no_readable_text_fails_without_calling_the_model(self):
        ocr, pipeline, task = self.run_job("", [], ocr_text="")
        pipeline.assert_not_called()
        self.assertEqual(task["status"], main.InvoiceStatus.FAILED)
        self.assertIn("No text could be read", task["error_detail"])

    def test_an_unexpected_error_is_recorded_without_its_message(self):
        ocr, pipeline, task = self.run_job(self.LONG_TEXT, [RuntimeError("secret detail at /app/x.pdf")])
        self.assertEqual(task["status"], main.InvoiceStatus.FAILED)
        self.assertIn("RuntimeError", task["error_detail"])
        self.assertNotIn("secret detail", task["error_detail"])
        self.assertFalse(os.path.exists(self.pdf_path), "the temporary PDF should be removed")


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.template = json.loads((SAMPLES / "template.json").read_text())
        self.temp_dir = tempfile.TemporaryDirectory()
        patcher = mock.patch.object(main, "TEMP_DIR", self.temp_dir.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.temp_dir.cleanup)
        self.addCleanup(main.invoice_processing_tasks.clear)

    def upload(self, headers=HEADERS, template_bytes=None, pdf_bytes=None):
        if template_bytes is None:
            template_bytes = json.dumps(self.template).encode()
        if pdf_bytes is None:
            pdf_bytes = (SAMPLES / "invoice_text.pdf").read_bytes()
        return self.client.post(
            "/upload-invoice",
            headers=headers,
            files={
                "pdf_file": ("invoice.pdf", pdf_bytes, "application/pdf"),
                "json_template_file": ("template.json", template_bytes, "application/json"),
            },
        )

    def upload_with_reply(self, reply, ocr_text="text from OCR"):
        """Uploads the text sample; the model returns `reply`. OCR is mocked because a reply with many empty
        fields triggers the OCR retry, and the machine running the tests may not have Tesseract."""
        with mock.patch.object(main, "extract_all_fields_at_once", return_value=reply), \
                mock.patch.object(main, "ocr_pdf_to_text", return_value=ocr_text):
            return self.upload()  # the background job runs before the test client returns

    def status_of(self, invoice_id):
        return self.client.get(f"/invoice-status/{invoice_id}", headers=HEADERS)

    # --- who may call, and what they may send ---
    def test_wrong_or_missing_key_is_rejected(self):
        self.assertEqual(self.upload(headers={"X-API-Key": "nope"}).status_code, 401)
        self.assertEqual(self.upload(headers={}).status_code, 401)

    def test_status_needs_the_key_too(self):
        self.assertEqual(self.client.get("/invoice-status/anything").status_code, 401)
        self.assertEqual(self.client.get("/invoice-status/anything", headers={"X-API-Key": "nope"}).status_code, 401)

    def test_the_key_is_checked_before_the_body_is_read(self):
        headers = {"X-API-Key": "nope", "Content-Type": "multipart/form-data; boundary=zzz"}
        response = self.client.post("/upload-invoice", headers=headers, content=b"this is not multipart")
        self.assertEqual(response.status_code, 401)

    def test_a_request_that_announces_too_much_is_rejected_early(self):
        with mock.patch.object(main, "MAX_REQUEST_BYTES", 100):
            self.assertEqual(self.upload().status_code, 413)

    def test_oversized_files_are_rejected(self):
        with mock.patch.object(main, "MAX_PDF_BYTES", 100):
            self.assertEqual(self.upload().status_code, 413)
        with mock.patch.object(main, "MAX_TEMPLATE_BYTES", 10):
            self.assertEqual(self.upload().status_code, 413)

    def test_a_file_that_is_not_a_pdf_is_rejected(self):
        self.assertEqual(self.upload(pdf_bytes=b"just some text").status_code, 400)

    def test_template_must_be_a_json_object_with_a_field(self):
        for bad in (b"not json", b'["a", "b"]', b"{}", b"[" * 40000, b"\xff\xfe"):
            self.assertEqual(self.upload(template_bytes=bad).status_code, 400, bad[:20])

    def test_unknown_id_is_404(self):
        self.assertEqual(self.status_of("nope").status_code, 404)

    # --- jobs ---
    def test_text_sample_end_to_end(self):
        reply = json.dumps({
            "invoice_number": "INV-2023-0456",
            "date": "2023-11-15",
            "vendor_name": "Acme Supplies",
            "vendor_address": "123 Business Rd, New York, NY",
            "customer_name": "John Doe",
            "customer_address": "456 Customer Ave, Boston, MA",
            "total_amount": "1,299.99",
            "due_date": "2023-12-15",
        })  # line_items left out on purpose: one empty field of nine is under the 30% retry line
        response = self.upload_with_reply(reply)
        self.assertEqual(response.status_code, 202)
        invoice_id = response.json()["invoice_id"]

        status = self.status_of(invoice_id)
        self.assertEqual(status.status_code, 200)
        body = status.json()
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["filled_json"]["invoice_number"], "INV-2023-0456")
        # template order: number, date, vendor name, vendor address, customer name,
        # customer address, total, due date, line items
        self.assertEqual(body["flag_array"], [1, 1, 1, 1, 1, 1, 1, 1, 0])
        self.assertEqual(main.invoice_processing_tasks[invoice_id]["method_used"], "text")
        self.assertEqual(os.listdir(self.temp_dir.name), [], "the temporary PDF should be removed")

    def test_status_call_survives_a_model_that_returns_a_list(self):
        reply = json.dumps({"invoice_number": "INV-1", "line_items": [{"item": "Widget A"}], "total_amount": 1299.99})
        invoice_id = self.upload_with_reply(reply).json()["invoice_id"]
        status = self.status_of(invoice_id)
        self.assertEqual(status.status_code, 200)
        filled = status.json()["filled_json"]
        self.assertEqual(filled["total_amount"], "1299.99")
        self.assertEqual(filled["line_items"], '[{"item": "Widget A"}]')

    def test_a_failed_job_is_reported_with_a_safe_message(self):
        with mock.patch.object(main, "extract_all_fields_at_once", side_effect=RuntimeError("secret detail at /app/x.pdf")):
            invoice_id = self.upload().json()["invoice_id"]
        status = self.status_of(invoice_id)
        self.assertEqual(status.status_code, 200)
        body = status.json()
        self.assertEqual(body["status"], "failed")
        self.assertIn("RuntimeError", body["error_detail"])
        self.assertNotIn("secret detail", status.text)
        self.assertIsNone(body["filled_json"])


if __name__ == "__main__":
    unittest.main()
