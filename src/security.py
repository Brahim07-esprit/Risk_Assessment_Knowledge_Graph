import hashlib
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from functools import wraps
from typing import BinaryIO, Dict, Optional

import magic
import PyPDF2
import streamlit as st

logger = logging.getLogger(__name__)


class SecurityManager:
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.file_validator = FileValidator()

    def validate_file_upload(self, file: BinaryIO,
                             filename: str) -> Dict[str, any]:
        return self.file_validator.validate(file, filename)

    def sanitize_input(self, text: str, max_length: int = 10000) -> str:
        if not text:
            return ""

        text = text.replace("\0", "")

        text = text[:max_length]

        text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]", "", text)

        return text

    def check_rate_limit(self, user_id: str, action: str) -> bool:
        return self.rate_limiter.check_limit(user_id, action)


class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
        self.limits = {
            "upload": {"per_minute": 5, "per_hour": 20},
            "process": {"per_minute": 10, "per_hour": 100},
            "api_call": {"per_minute": 30, "per_hour": 500},
        }

    def check_limit(self, user_id: str, action: str) -> bool:
        if not os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true":
            return True

        now = datetime.now()
        key = f"{user_id}:{action}"

        self.requests[key] = [
            timestamp
            for timestamp in self.requests[key]
            if now - timestamp < timedelta(hours=1)
        ]

        limits = self.limits.get(action, {"per_minute": 60, "per_hour": 1000})

        recent_minute = [
            t for t in self.requests[key] if now - t < timedelta(minutes=1)
        ]
        if len(recent_minute) >= limits["per_minute"]:
            return False

        if len(self.requests[key]) >= limits["per_hour"]:
            return False

        self.requests[key].append(now)
        return True


class FileValidator:
    def __init__(self):
        self.max_size_mb = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
        self.allowed_types = os.getenv("ALLOWED_FILE_TYPES", "pdf").split(",")

    def validate(self, file: BinaryIO, filename: str) -> Dict[str, any]:
        result = {"valid": True, "errors": [], "warnings": [], "file_info": {}}

        try:
            ext = filename.lower().split(".")[-1]
            if ext not in self.allowed_types:
                result["valid"] = False
                result["errors"].append(
                    f"File type '{ext}' not allowed. Allowed: {self.allowed_types}"
                )
                return result

            file_content = file.read()
            file.seek(0)

            file_size_mb = len(file_content) / (1024 * 1024)
            result["file_info"]["size_mb"] = round(file_size_mb, 2)

            if file_size_mb > self.max_size_mb:
                result["valid"] = False
                result["errors"].append(
                    f"File size {file_size_mb:.1f}MB exceeds limit of {self.max_size_mb}MB"
                )
                return result

            try:
                mime = magic.from_buffer(file_content, mime=True)
                result["file_info"]["mime_type"] = mime

                if "pdf" in ext and mime != "application/pdf":
                    result["valid"] = False
                    result["errors"].append(
                        f"File claims to be PDF but has MIME type: {mime}"
                    )
                    return result
            except BaseException:
                result["warnings"].append("Could not verify MIME type")

            if ext == "pdf":
                pdf_result = self._validate_pdf(file_content)
                result["file_info"].update(pdf_result["info"])
                result["errors"].extend(pdf_result["errors"])
                result["warnings"].extend(pdf_result["warnings"])
                if pdf_result["errors"]:
                    result["valid"] = False

            result["file_info"]["hash"] = hashlib.sha256(
                file_content).hexdigest()

        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Validation error: {str(e)}")

        return result

    def _validate_pdf(self, content: bytes) -> Dict:    
        result = {"info": {}, "errors": [], "warnings": []}

        try:
            from io import BytesIO

            pdf_file = BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            if pdf_reader.is_encrypted:
                result["warnings"].append("PDF is encrypted")
                try:
                    pdf_reader.decrypt("")
                except BaseException:
                    result["errors"].append("Cannot read encrypted PDF")
                    return result

            num_pages = len(pdf_reader.pages)
            result["info"]["pages"] = num_pages

            if num_pages == 0:
                result["errors"].append("PDF has no pages")
            elif num_pages > 1000:
                result["warnings"].append(
                    f"Large PDF with {num_pages} pages may take time to process"
                )

            if pdf_reader.metadata and "/JavaScript" in str(
                    pdf_reader.metadata):
                result["warnings"].append("PDF contains JavaScript")

            try:
                first_page_text = pdf_reader.pages[0].extract_text()
                if len(first_page_text.strip()) < 10:
                    result["warnings"].append(
                        "PDF appears to have little or no extractable text"
                    )
            except BaseException:
                result["warnings"].append(
                    "Could not extract text from first page")

        except Exception as e:
            result["errors"].append(f"PDF parsing error: {str(e)}")

        return result


def require_auth(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if os.getenv("ENABLE_AUTH", "false").lower() == "true":
            if (
                "authenticated" not in st.session_state
                or not st.session_state.authenticated
            ):
                st.error("⚠️ Authentication required")
                st.stop()
        return func(*args, **kwargs)

    return wrapper


def sanitize_api_key(api_key: str) -> bool:
    if not api_key:
        return False

    if api_key.startswith("sk-") and len(api_key) > 20:
        return True

    if len(api_key) > 40:
        return True

    return False


def get_user_id() -> str:
    if "user_id" not in st.session_state:
        st.session_state.user_id = hashlib.md5(
            f"{time.time()}{os.urandom(16).hex()}".encode()
        ).hexdigest()
    return st.session_state.user_id
