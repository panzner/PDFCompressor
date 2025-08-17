#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF Compressor – vector-preserving by default, with selective + full raster targets.

This build ensures:
- If you check “Convert to grayscale” OR “Force rasterize if needed”, the app
  won’t exit early after in-place recompress; it WILL rasterize to honor that choice.
- Rasterizer fix: render pages in RGB with alpha=True, then composite to white in PIL
  and only then convert to grayscale — avoids blacked-out logos/transparency artifacts.
- Robustness tweak: in-place image conversions composite alpha to white before JPEG encode.
- Presets & expert floors (Min DPI / Min JPEG Q) remain, and routing logic retains:
  * In-place image recompress (never grows the file; safe skips).
  * Selective rasterization for heavy pages when chasing a target.
  * Full rasterization when requested or still needed for targets.
  * Ghostscript path for real B/W (CCITT/JBIG2) scans.
  * Final never-grow guard and optional qpdf linearization.
"""

import sys
import os
import io
import shutil
import tempfile
import logging
import warnings

import psutil
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QSpinBox, QDoubleSpinBox, QMessageBox, QLineEdit,
    QGroupBox, QRadioButton, QButtonGroup, QCheckBox, QTextEdit, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

# Prefer pypdf; fall back to PyPDF2
try:
    import pypdf as PyPDF2
    HAS_PYPDF2 = True
except Exception:
    try:
        import PyPDF2  # type: ignore
        HAS_PYPDF2 = True
    except Exception:
        HAS_PYPDF2 = False

try:
    import fitz  # PyMuPDF
except Exception as e:
    raise SystemExit("pymupdf is required. Install with: pip install pymupdf") from e

# Reduce noisy library logs/warnings
warnings.filterwarnings("ignore", message=".*wrong pointing object.*")
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("PyPDF2").setLevel(logging.ERROR)


# ------------------------- Small helpers -------------------------

def file_size_bytes(path: str) -> int:
    try:
        return os.path.getsize(path)
    except Exception:
        return 0

def available_gb(path: str) -> float:
    try:
        return shutil.disk_usage(path).free / (1024**3)
    except Exception:
        return 0.0

def _subprocess_rc(cmd, timeout=30):
    import subprocess
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode
    except Exception:
        return 1


# ------------------------- Dependencies -------------------------

class DependencyChecker:
    @staticmethod
    def check_poppler():
        return _subprocess_rc(['pdftoppm', '-h']) == 0

    @staticmethod
    def check_pdfimages():
        rc = _subprocess_rc(['pdfimages', '-v'])
        return rc in (0, 1)

    @staticmethod
    def check_qpdf():
        return _subprocess_rc(['qpdf', '--version']) == 0

    @staticmethod
    def check_ghostscript():
        return _subprocess_rc(['gs', '-v']) == 0

    @staticmethod
    def check_pymupdf():
        try:
            import fitz  # noqa
            return True
        except Exception:
            return False

    @staticmethod
    def check_available_memory():
        try:
            return psutil.virtual_memory().available / (1024**3)
        except Exception:
            return 0

    @staticmethod
    def get_free_disk_space(path):
        try:
            return shutil.disk_usage(path).free / (1024**3)
        except Exception:
            return 0


# ------------------------- Analyzer -------------------------

class PDFAnalyzer:
    @staticmethod
    def _page_has_large_images(reader_page, area_threshold=0.6):
        try:
            w = float(reader_page.mediabox.width)
            h = float(reader_page.mediabox.height)
            page_area = max(1.0, w * h)
            resources = reader_page.get('/Resources', {})
            xobj = resources.get('/XObject', {})
            if not hasattr(xobj, 'items'):
                return False
            for _, ref in xobj.items():
                obj = ref.get_object()
                if obj.get('/Subtype') == '/Image':
                    iw = int(obj.get('/Width', 0))
                    ih = int(obj.get('/Height', 0))
                    if iw * ih >= area_threshold * page_area * 12.25:  # ~3.5 px/pt
                        return True
            return False
        except Exception:
            return False

    @staticmethod
    def is_single_image_scan(reader_page, extracted_text):
        try:
            resources = reader_page.get('/Resources', {})
            xobj = resources.get('/XObject', {})
            if not hasattr(xobj, 'items'):
                return False
            imgs = []
            for _, ref in xobj.items():
                obj = ref.get_object()
                if obj.get('/Subtype') == '/Image':
                    iw = int(obj.get('/Width', 0))
                    ih = int(obj.get('/Height', 0))
                    imgs.append((iw, ih))
            if len(imgs) != 1:
                return False
            has_large = PDFAnalyzer._page_has_large_images(reader_page, area_threshold=0.85)
            low_text = len((extracted_text or "").strip()) < 10
            return has_large and low_text
        except Exception:
            return False

    @staticmethod
    def has_bilevel_images(pdf_path, sample_pages=5):
        if not HAS_PYPDF2:
            return False
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                n = min(sample_pages, len(reader.pages))
                for i in range(n):
                    page = reader.pages[i]
                    resources = page.get('/Resources', {})
                    xobj = resources.get('/XObject', {})
                    if hasattr(xobj, 'items'):
                        for _, ref in xobj.items():
                            img = ref.get_object()
                            if img.get('/Subtype') == '/Image':
                                filt = img.get('/Filter', None)
                                if filt is None:
                                    continue
                                if not isinstance(filt, (list, tuple)):
                                    filt = [filt]
                                filters = {str(x) for x in filt}
                                if '/CCITTFaxDecode' in filters or '/JBIG2Decode' in filters:
                                    return True
            return False
        except Exception:
            return False

    @staticmethod
    def _probe_with_pdfimages(pdf_path, sample_limit=10):
        if not DependencyChecker.check_pdfimages():
            return {'has_bilevel': False, 'image_dominant_pages': 0}
        import subprocess
        try:
            r = subprocess.run(['pdfimages', '-list', pdf_path],
                               capture_output=True, text=True, timeout=20)
            out = r.stdout.splitlines()
            has_bilevel = False
            big_rows = 0
            seen = 0
            for line in out:
                line = line.strip()
                if not line or line.startswith('page'):
                    continue
                parts = line.split()
                if len(parts) < 10:
                    continue
                try:
                    width = int(parts[3]); height = int(parts[4]); enc = parts[9].lower()
                except Exception:
                    continue
                if enc in ('ccitt', 'jbig2'):
                    has_bilevel = True
                if (width * height) >= int(0.8 * 1_000_000):
                    big_rows += 1
                seen += 1
                if seen >= sample_limit:
                    break
            return {'has_bilevel': has_bilevel, 'image_dominant_pages': big_rows}
        except Exception:
            return {'has_bilevel': False, 'image_dominant_pages': 0}

    @staticmethod
    def bilevel_ppi(pdf_path):
        if not DependencyChecker.check_pdfimages():
            return None
        import subprocess, statistics
        try:
            r = subprocess.run(['pdfimages', '-list', pdf_path],
                               capture_output=True, text=True, timeout=20)
            xppis, yppis = [], []
            for line in r.stdout.splitlines():
                line = line.strip()
                if not line or line.startswith('page'):
                    continue
                parts = line.split()
                if len(parts) < 13:
                    continue
                enc = parts[9].lower()
                if enc not in ('ccitt', 'jbig2'):
                    continue
                try:
                    xppi = int(parts[11]); yppi = int(parts[12])
                    if xppi > 0 and yppi > 0:
                        xppis.append(xppi); yppis.append(yppi)
                except Exception:
                    continue
            if xppis and yppis:
                mins = [min(x, y) for x, y in zip(xppis, yppis)]
                return int(statistics.median(mins))
            return None
        except Exception:
            return None

    @staticmethod
    def is_text_based_pdf(pdf_path):
        def _fallback_mupdf(p):
            try:
                with fitz.open(p) as d:
                    n = min(5, d.page_count)
                    total = 0
                    for i in range(n):
                        try:
                            txt = d.load_page(i).get_text("text") or ""
                        except Exception:
                            txt = ""
                        total += len(txt.strip())
                    return total / max(1, n) > 50
            except Exception:
                return False

        if not HAS_PYPDF2:
            return _fallback_mupdf(pdf_path)

        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                n = min(5, total_pages)
                text_chars = 0
                image_dominant_hits = 0
                single_scan_hits = 0
                for i in range(n):
                    page = reader.pages[i]
                    txt = ""
                    try:
                        txt = page.extract_text() or ""
                    except Exception:
                        return _fallback_mupdf(pdf_path)
                    text_chars += len(txt.strip())
                    if PDFAnalyzer._page_has_large_images(page, area_threshold=0.8):
                        image_dominant_hits += 1
                    if PDFAnalyzer.is_single_image_scan(page, txt):
                        single_scan_hits += 1
                avg = text_chars / max(1, n)
                mostly_text = avg > 50
                mostly_images = image_dominant_hits >= max(1, n // 2)
                mostly_single_scans = single_scan_hits >= max(1, n // 2)
                if mostly_single_scans:
                    return False
                if mostly_images and not mostly_text:
                    return False
                return mostly_text
        except Exception:
            return _fallback_mupdf(pdf_path)

    @staticmethod
    def get_pdf_info(pdf_path):
        info = {
            'pages': 0,
            'size_mb': file_size_bytes(pdf_path) / (1024**2),
            'is_encrypted': False,
            'is_text_based': False,
            'has_bilevel': False,
            'image_dominant_pages': 0,
            'bilevel_ppi': None,
        }
        try:
            with fitz.open(pdf_path) as doc:
                info['pages'] = doc.page_count
                info['is_encrypted'] = doc.is_encrypted
        except Exception:
            info['is_encrypted'] = True

        try:
            info['is_text_based'] = PDFAnalyzer.is_text_based_pdf(pdf_path)
        except Exception:
            pass

        try:
            info['has_bilevel'] = PDFAnalyzer.has_bilevel_images(pdf_path)
            probe = PDFAnalyzer._probe_with_pdfimages(pdf_path)
            info['has_bilevel'] = info['has_bilevel'] or probe['has_bilevel']
            info['image_dominant_pages'] = probe['image_dominant_pages']
            info['bilevel_ppi'] = PDFAnalyzer.bilevel_ppi(pdf_path)
        except Exception:
            pass

        return info


# ------------------------- In-place Image Optimizer -------------------------

class InPlaceImageOptimizer:
    """
    Safe in-place image recompression using PyMuPDF.

    - Skips bilevel (bpc==1), CCITT/JBIG2/JPX images, and images with soft masks.
    - Skips tiny drawn images.
    - Caps effective DPI where oversized; no upsampling.
    - Encodes JPEG with progressive on; 4:4:4 if preserving color fidelity, else 4:2:0.
    - Only replaces if new stream is at least 5% smaller (or >1 KB smaller).
    - Global controller binary-searches JPEG quality to meet a target; never returns larger than original.
    - Robustness: composite alpha to white before JPEG encode to avoid dark artifacts.
    """
    def __init__(self, preserve_color_fidelity: bool, grayscale: bool, min_jpeg_q: int,
                 progress_cb=None, status_cb=None):
        self.preserve_color_fidelity = preserve_color_fidelity
        self.grayscale = grayscale
        self.min_jpeg_q = max(10, min(90, int(min_jpeg_q)))
        self.progress_cb = progress_cb or (lambda *_: None)
        self.status_cb = status_cb or (lambda *_: None)

    @staticmethod
    def _img_ext(doc: fitz.Document, xref: int) -> str:
        try:
            meta = doc.extract_image(xref)
            return (meta.get("ext") or "").lower()
        except Exception:
            return ""

    @staticmethod
    def _is_problematic_format(ext: str) -> bool:
        return ext in {"jbig2", "ccitt", "jpx", "jp2"}

    @staticmethod
    def _effective_display_points(page: fitz.Page, xref: int) -> tuple[int, int]:
        rects = page.get_image_rects(xref)
        if not rects:
            return (0, 0)
        r = max(rects, key=lambda rr: rr.get_area())
        return int(r.width), int(r.height)

    @staticmethod
    def _pixmap_to_pil(pm: fitz.Pixmap) -> Image.Image:
        # Composite alpha to white if present
        if pm.alpha:
            img_rgba = Image.frombytes("RGBA", (pm.width, pm.height), pm.samples)
            bg = Image.new("RGB", (pm.width, pm.height), (255, 255, 255))
            bg.paste(img_rgba, mask=img_rgba.split()[-1])
            return bg
        # grayscale or RGB
        if pm.colorspace is None or pm.n <= 1:
            return Image.frombytes("L", (pm.width, pm.height), pm.samples)
        if pm.n == 3:
            return Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
        # other spaces -> RGB
        pm2 = fitz.Pixmap(fitz.csRGB, pm)
        return Image.frombytes("RGB", (pm2.width, pm2.height), pm2.samples)

    def _downscale_for_display(self, pil_img: Image.Image, disp_w_pts: int, disp_h_pts: int, max_dpi: int | None) -> Image.Image:
        if not max_dpi or disp_w_pts <= 0 or disp_h_pts <= 0:
            return pil_img
        disp_w_in = disp_w_pts / 72.0
        disp_h_in = disp_h_pts / 72.0
        if disp_w_in <= 0 or disp_h_in <= 0:
            return pil_img
        eff_dpi_x = pil_img.width / disp_w_in
        eff_dpi_y = pil_img.height / disp_h_in
        eff_dpi = max(eff_dpi_x, eff_dpi_y)
        if eff_dpi <= max_dpi:
            return pil_img
        scale = max_dpi / eff_dpi
        new_w = max(1, int(pil_img.width * scale))
        new_h = max(1, int(pil_img.height * scale))
        if new_w >= pil_img.width or new_h >= pil_img.height:
            return pil_img
        return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def _encode_jpeg(self, pil_img: Image.Image, quality: int) -> bytes:
        params = {
            "format": "JPEG",
            "quality": int(quality),
            "optimize": True,
            "progressive": True,
            "subsampling": 0 if self.preserve_color_fidelity else "4:2:0"
        }
        if self.grayscale:
            pil_img = pil_img.convert("L")
        out = io.BytesIO()
        pil_img.save(out, **params)
        return out.getvalue()

    def _info_list(self, page: fitz.Page):
        try:
            return page.get_image_info(xrefs=True)
        except TypeError:
            return page.get_image_info()

    def _rewrite_once(self, src_pdf: str, quality: int, max_dpi: int | None) -> bytes:
        orig_bytes = file_size_bytes(src_pdf)
        with fitz.open(src_pdf) as doc:
            for pno in range(doc.page_count):
                page = doc.load_page(pno)
                info_list = self._info_list(page)
                if not info_list:
                    continue
                for info in info_list:
                    xref = info.get("xref")
                    if not xref:
                        continue
                    bpc = int(info.get("bpc") or 0)
                    if bpc == 1:
                        continue  # bilevel
                    ext = self._img_ext(doc, xref)
                    if self._is_problematic_format(ext):
                        continue
                    if info.get("smask") not in (0, None):
                        continue  # soft-masked; skip
                    disp_w_pts, disp_h_pts = self._effective_display_points(page, xref)
                    if disp_w_pts * disp_h_pts < 72 * 72:
                        continue

                    try:
                        pm = fitz.Pixmap(doc, xref)
                    except Exception:
                        continue
                    pil = self._pixmap_to_pil(pm)
                    pil = self._downscale_for_display(pil, disp_w_pts, disp_h_pts, max_dpi)
                    new_stream = self._encode_jpeg(pil, quality)

                    try:
                        old_len = len(doc.xref_stream_raw(xref) or b"")
                    except Exception:
                        old_len = 0

                    if old_len == 0 or len(new_stream) < max(int(old_len * 0.95), old_len - 1024):
                        try:
                            page.replace_image(xref, stream=new_stream)
                        except Exception:
                            try:
                                doc.update_stream(xref, new_stream)
                            except Exception:
                                pass

            try:
                fitz.Tools().store_shrink(100)
            except Exception:
                pass

            out = io.BytesIO()
            doc.save(out, garbage=3, deflate=True, ascii=False)
            data = out.getvalue()
        if len(data) >= orig_bytes:
            with open(src_pdf, "rb") as f:
                return f.read()
        return data

    def rewrite_to_target(self, src_pdf: str, *, target_bytes: int | None,
                          pct_of_original: float | None,
                          hard_max_dpi: int | None) -> bytes:
        orig_sz = file_size_bytes(src_pdf)
        if pct_of_original is not None:
            target_bytes = int(max(1, orig_sz * pct_of_original))
        if target_bytes is None:
            q = max(self.min_jpeg_q, 75 if not self.preserve_color_fidelity else 85)
            return self._rewrite_once(src_pdf, q, hard_max_dpi)

        lo = max(self.min_jpeg_q, 35 if not self.preserve_color_fidelity else 50)
        hi = 90 if self.preserve_color_fidelity else 85
        best_bytes, best_q = None, None
        smallest_bytes, smallest_q = None, None

        for _ in range(8):
            q = (lo + hi) // 2
            self.status_cb(f"In-place recompress: trying JPEG quality {q}...")
            pdf_bytes = self._rewrite_once(src_pdf, q, hard_max_dpi)
            sz = len(pdf_bytes)

            if sz < orig_sz and (smallest_bytes is None or sz < len(smallest_bytes)):
                smallest_bytes = pdf_bytes
                smallest_q = q

            if sz <= target_bytes:
                best_bytes = pdf_bytes
                best_q = q
                lo = min(q + 1, hi)
            else:
                hi = max(q - 1, lo)

        if best_bytes is not None:
            self.status_cb(f"In-place recompress met target at quality {best_q}.")
            return best_bytes

        if smallest_bytes is not None:
            self.status_cb(f"Target not fully reachable in-place; returning best smaller result (quality {smallest_q}).")
            return smallest_bytes

        self.status_cb("In-place recompress could not shrink file; returning original.")
        with open(src_pdf, "rb") as f:
            return f.read()


# ------------------------- Worker -------------------------

class PDFCompressorWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)
    status_update = pyqtSignal(str)
    memory_usage = pyqtSignal(float)

    def __init__(self, input_path, output_path, compression_settings):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.s = compression_settings
        self.cancelled = False
        self.pdf_info = None

    def cancel(self):
        self.cancelled = True

    def check_cancellation(self):
        if self.cancelled:
            raise InterruptedError("Operation cancelled by user")

    def _emit_memory(self):
        try:
            memory_mb = psutil.Process().memory_info().rss / (1024**2)
            self.memory_usage.emit(memory_mb)
        except Exception:
            pass

    def _dpi_for_percentage(self, pct: int) -> int:
        pct = max(1, min(100, int(pct)))
        if pct >= 90: return 300
        if pct >= 80: return 250
        if pct >= 70: return 225
        if pct >= 60: return 200
        if pct >= 50: return 175
        if pct >= 40: return 160
        if pct >= 30: return 140
        if pct >= 20: return 120
        return 100

    # Ghostscript path for mono scans
    def compress_with_ghostscript(self):
        import subprocess
        pct = int(self.s['value']) if self.s['mode'] == 'percentage' else 60
        baseline = self._dpi_for_percentage(pct) - 25
        baseline = max(110, min(350, baseline))
        src_ppi = self.pdf_info.get('bilevel_ppi', None)
        if src_ppi:
            factor = 0.65 if pct == 50 else (0.75 if pct >= 60 else 0.6)
            mono_dpi = int(max(110, min(350, min(baseline, src_ppi * factor))))
        else:
            mono_dpi = baseline
        color_dpi = max(120, mono_dpi // 2)

        cmd = [
            'gs', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.6',
            '-dDetectDuplicateImages=true', '-dCompressFonts=true',
            '-dAutoRotatePages=/None',
            '-dDownsampleMonoImages=true', '-dMonoImageDownsampleType=/Average',
            f'-dMonoImageResolution={mono_dpi}', '-dAutoFilterMonoImages=false',
            '-sMonoImageFilter=/CCITTFaxEncode',
            '-dDownsampleColorImages=true', '-dColorImageDownsampleType=/Average',
            f'-dColorImageResolution={color_dpi}', '-dAutoFilterColorImages=false',
            '-sColorImageFilter=/DCTEncode',
            f'-sOutputFile={self.output_path}', self.input_path
        ]
        self.status_update.emit(f"Ghostscript mono DPI={mono_dpi}, color DPI={color_dpi}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise Exception(f"Ghostscript failed: {r.stderr[:400]}")

    # ----- Raster builders (with alpha composite to white before grayscale) -----

    def _render_page_pil(self, page: fitz.Page, dpi: int, grayscale: bool) -> Image.Image:
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        # Render with alpha channel, composite ourselves to white to avoid artifacts
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=True)
        if pix.alpha:
            img_rgba = Image.frombytes("RGBA", (pix.width, pix.height), pix.samples)
            bg = Image.new("RGB", (pix.width, pix.height), (255, 255, 255))
            bg.paste(img_rgba, mask=img_rgba.split()[-1])
            img = bg
        else:
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        if grayscale:
            img = img.convert("L")
        return img

    def _encode_jpeg_bytes(self, pil_img: Image.Image, quality: int, preserve_color: bool) -> bytes:
        params = {
            "format": "JPEG",
            "quality": int(quality),
            "optimize": True,
            "progressive": True,
            "subsampling": 0 if preserve_color else "4:2:0"
        }
        out = io.BytesIO()
        pil_img.save(out, **params)
        return out.getvalue()

    def _build_pdf_from_raster(self, dpi: int, quality: int, preserve_color: bool, grayscale: bool) -> bytes:
        src = fitz.open(self.input_path)
        out = fitz.open()
        pages = src.page_count
        for i in range(pages):
            p = src.load_page(i)
            w, h = p.rect.width, p.rect.height
            pil = self._render_page_pil(p, dpi, grayscale)
            jpg = self._encode_jpeg_bytes(pil, quality, preserve_color)
            newp = out.new_page(width=w, height=h)
            newp.insert_image(p.rect, stream=jpg)
            self.progress.emit(85 + int(10 * (i + 1) / max(1, pages)))
            self._emit_memory()
        buf = io.BytesIO()
        out.save(buf, garbage=3, deflate=True)
        out.close()
        src.close()
        return buf.getvalue()

    def _raster_to_target(self, target_bytes: int | None, preserve_color: bool, grayscale: bool,
                          min_dpi_floor: int, min_q_floor: int) -> bytes:
        mode = self.s['mode']
        if mode == 'percentage':
            dpi0 = self._dpi_for_percentage(int(self.s['value']))
        elif mode == 'resolution':
            dpi0 = int(self.s['value'])
        else:
            dpi0 = 200

        default_floor = 200 if preserve_color else 150
        min_dpi = max(min_dpi_floor, 50)
        dpi0 = max(dpi0, default_floor, min_dpi)

        original = file_size_bytes(self.input_path)

        if target_bytes is None:
            q = max(min_q_floor, 72 if preserve_color else 68)
            out = self._build_pdf_from_raster(dpi0, q, preserve_color, grayscale)
            return out if len(out) < original else open(self.input_path, "rb").read()

        candidates_under = []
        smallest_any = None
        attempts = 6
        for attempt in range(attempts):
            dpi = max(min_dpi, int(dpi0 * (1.0 - 0.15 * attempt)))
            self.status_update.emit(f"Raster target search: DPI {dpi} (attempt {attempt+1}/{attempts})")

            hi = 90 if preserve_color else 80
            lo = max(min_q_floor, 30 if not preserve_color else 40)
            best_under = None
            best_under_size = None

            for _ in range(6):
                q = (lo + hi) // 2
                pdf_bytes = self._build_pdf_from_raster(dpi, q, preserve_color, grayscale)
                sz = len(pdf_bytes)

                if smallest_any is None or sz < len(smallest_any):
                    smallest_any = pdf_bytes

                if sz <= target_bytes:
                    if best_under is None or sz > best_under_size:
                        best_under = pdf_bytes
                        best_under_size = sz
                    lo = min(q + 1, hi)
                else:
                    hi = max(q - 1, lo)

            if best_under is not None:
                delta = target_bytes - best_under_size
                candidates_under.append((delta, best_under_size, best_under))

            if dpi == min_dpi:
                break

        if candidates_under:
            candidates_under.sort(key=lambda t: (t[0], -t[1]))
            return candidates_under[0][2]

        if smallest_any is not None and len(smallest_any) < original:
            self.status_update.emit("Raster couldn't hit target; returning best smaller raster result.")
            return smallest_any

        with open(self.input_path, "rb") as f:
            return f.read()

    # ----- Selective rasterization (keeps vector pages unless needed) -----

    def _render_page_jpeg_bytes(self, doc, page_idx: int, dpi: int, quality: int,
                                preserve_color: bool, grayscale: bool) -> bytes:
        page = doc.load_page(page_idx)
        pil = self._render_page_pil(page, dpi, grayscale)
        return self._encode_jpeg_bytes(pil, quality, preserve_color)

    def _build_mixed_pdf(self, base_pdf_path: str, rasterize_set: set, dpi: int, quality: int,
                         preserve_color: bool, grayscale: bool) -> bytes:
        src = fitz.open(base_pdf_path)
        out = fitz.open()
        for i in range(src.page_count):
            if i in rasterize_set:
                p = src.load_page(i)
                w, h = p.rect.width, p.rect.height
                jpg = self._render_page_jpeg_bytes(src, i, dpi, quality, preserve_color, grayscale)
                newp = out.new_page(width=w, height=h)
                newp.insert_image(p.rect, stream=jpg)
            else:
                out.insert_pdf(src, from_page=i, to_page=i)
        buf = io.BytesIO()
        out.save(buf, garbage=3, deflate=True)
        out.close()
        src.close()
        return buf.getvalue()

    def _selective_rasterize_to_target(self, base_pdf_bytes: bytes, target_bytes: int,
                                       preserve_color: bool, grayscale: bool,
                                       optimizer: InPlaceImageOptimizer, max_eff_dpi: int,
                                       min_dpi_floor: int, min_q_floor: int) -> bytes | None:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(base_pdf_bytes)
            tmp_path = tmp.name

        try:
            doc = fitz.open(tmp_path); pages = doc.page_count; doc.close()
        except Exception:
            try: os.unlink(tmp_path)
            except Exception: pass
            return None

        current_size = len(base_pdf_bytes)
        if current_size <= target_bytes:
            try: os.unlink(tmp_path)
            except Exception: pass
            return base_pdf_bytes

        if self.s["mode"] == "percentage":
            render_dpi = self._dpi_for_percentage(int(self.s["value"]))
        elif self.s["mode"] == "resolution":
            render_dpi = int(self.s["value"])
        else:
            render_dpi = 200
        default_floor = 200 if preserve_color else 150
        render_dpi = max(render_dpi, default_floor, min_dpi_floor)
        base_quality = max(min_q_floor, 72 if preserve_color else 68)

        self.status_update.emit("Selective raster: probing page sizes...")
        raster_sizes = []
        try:
            d = fitz.open(tmp_path)
            for i in range(pages):
                jpg = self._render_page_jpeg_bytes(d, i, render_dpi, base_quality, preserve_color, grayscale)
                raster_sizes.append(len(jpg))
                self.progress.emit(75 + int(7 * (i + 1) / max(1, pages)))
            d.close()
        except Exception:
            try: os.unlink(tmp_path)
            except Exception: pass
            return None

        avg_page_now = current_size / max(1, pages)
        benefits = [(i, (avg_page_now - raster_sizes[i])) for i in range(pages)]
        benefits.sort(key=lambda t: t[1], reverse=True)

        need = current_size - target_bytes
        chosen = set()
        got = 0
        cap_pages = max(1, int(0.5 * pages))
        step = max(1, pages // 10)
        idx = 0
        best_bytes = None

        while got < need and idx < len(benefits) and len(chosen) < cap_pages:
            batch = []
            for _ in range(step):
                if idx >= len(benefits): break
                pno, gain = benefits[idx]; idx += 1
                if gain <= 0: continue
                chosen.add(pno); batch.append(pno); got += max(0, gain)
            if not batch: break

            self.status_update.emit(f"Selective raster: building with {len(chosen)} rasterized pages...")
            mixed = self._build_mixed_pdf(tmp_path, chosen, render_dpi, base_quality, preserve_color, grayscale)
            mixed_size = len(mixed)
            if best_bytes is None or mixed_size < len(best_bytes):
                best_bytes = mixed
            if mixed_size <= target_bytes:
                os.unlink(tmp_path)
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as mid:
                    mid.write(mixed); mid_path = mid.name
                tightened = optimizer.rewrite_to_target(mid_path, target_bytes=target_bytes,
                                                        pct_of_original=None, hard_max_dpi=max_eff_dpi)
                os.unlink(mid_path)
                return tightened if len(tightened) <= len(mixed) else mixed

            if mixed_size < current_size:
                current_size = mixed_size
                need = current_size - target_bytes

        if best_bytes is not None:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as mid:
                mid.write(best_bytes); mid_path = mid.name
            tightened = optimizer.rewrite_to_target(mid_path, target_bytes=target_bytes,
                                                    pct_of_original=None, hard_max_dpi=max_eff_dpi)
            os.unlink(mid_path)
            if len(tightened) < len(best_bytes):
                best_bytes = tightened

        try: os.unlink(tmp_path)
        except Exception: pass
        if best_bytes is None or len(best_bytes) >= len(base_pdf_bytes):
            return None
        return best_bytes

    def _maybe_linearize(self, out_path):
        try:
            if not DependencyChecker.check_qpdf():
                return
            before = file_size_bytes(out_path)
            tmp_lin = out_path + ".lin.pdf"
            import subprocess
            r = subprocess.run(['qpdf', '--linearize', out_path, tmp_lin],
                               capture_output=True, text=True)
            if r.returncode == 0 and os.path.exists(tmp_lin):
                after = file_size_bytes(tmp_lin)
                if after <= before:
                    os.replace(tmp_lin, out_path)
                else:
                    os.remove(tmp_lin)
        except Exception:
            pass

    def run(self):
        try:
            # Analyze
            self.pdf_info = PDFAnalyzer.get_pdf_info(self.input_path)
            self.status_update.emit(
                f"Analyzed: pages={self.pdf_info['pages']}, "
                f"size={self.pdf_info['size_mb']:.2f} MB, "
                f"text_based={self.pdf_info['is_text_based']}, "
                f"image_pages(sample)={self.pdf_info['image_dominant_pages']}, "
                f"has_bilevel={self.pdf_info['has_bilevel']}, "
                f"bilevel_ppi={self.pdf_info.get('bilevel_ppi', None)}"
            )

            if self.pdf_info['is_encrypted']:
                raise Exception("PDF is password protected. Please decrypt it first.")

            out_dir = os.path.dirname(self.output_path) or os.getcwd()
            est_temp = self.pdf_info['size_mb'] * 2 + 64
            if available_gb(out_dir) * 1024 < est_temp:
                raise Exception(f"Insufficient disk space. Need ~{est_temp:.1f} MB free near output directory.")

            mode = self.s['mode']
            val = self.s['value']
            preserve_color = self.s.get('preserve_color_fidelity', False)
            grayscale = self.s.get('grayscale', False)
            force_raster = self.s.get('force_image_compression', False)
            min_dpi_floor = int(self.s.get('min_dpi', 96))
            min_q_floor = int(self.s.get('min_jpeg_q', 30))
            min_dpi_floor = max(50, min(300, min_dpi_floor))
            min_q_floor = max(10, min(90, min_q_floor))

            original_bytes = file_size_bytes(self.input_path)
            if mode == 'percentage':
                target_bytes = int(original_bytes * (float(val) / 100.0))
                target_desc = f"~{val:.0f}% of original"
            elif mode == 'size':
                target_bytes = int(float(val) * 1024 * 1024)
                target_desc = f"{val:.1f} MB"
            elif mode == 'reduction':
                target_bytes = int(original_bytes * (1 - float(val) / 100.0))
                target_desc = f"{val:.0f}% smaller"
            else:
                target_bytes = None
                target_desc = "gentle recompress"

            self.status_update.emit(f"Target: {target_desc}")
            self.progress.emit(5)

            # Route: true B/W scans -> Ghostscript (unless user demanded full raster)
            is_scan = (self.pdf_info.get('has_bilevel', False) and self.pdf_info.get('image_dominant_pages', 0) >= 1)
            if is_scan and DependencyChecker.check_ghostscript() and not force_raster and not grayscale:
                self.status_update.emit("B/W scan detected; using Ghostscript path...")
                self.compress_with_ghostscript()
                if file_size_bytes(self.output_path) > original_bytes:
                    shutil.copy2(self.input_path, self.output_path)
                    self.status_update.emit("Ghostscript increased size; original kept.")
                self.progress.emit(100)
                self.finished.emit(True, "PDF compression completed.")
                return

            # Should we rasterize regardless of in-place success?
            needs_raster = bool(force_raster or grayscale)

            # In-place optimizer (try first only if we are not forced to raster)
            in_place_bytes = None
            in_place_size = None
            if not needs_raster:
                optimizer = InPlaceImageOptimizer(
                    preserve_color_fidelity=preserve_color,
                    grayscale=grayscale,
                    min_jpeg_q=min_q_floor,
                    progress_cb=self.progress.emit,
                    status_cb=self.status_update.emit
                )
                max_eff_dpi = 300 if not preserve_color else 360
                in_place_bytes = optimizer.rewrite_to_target(
                    self.input_path,
                    target_bytes=target_bytes,
                    pct_of_original=(float(val)/100.0 if mode == 'percentage' else None),
                    hard_max_dpi=max_eff_dpi
                )
                in_place_size = len(in_place_bytes)
                met_target = (target_bytes is None) or (in_place_size <= target_bytes)
                grew = in_place_size > original_bytes

                # Only take early exit if not forcing raster and not grayscaling
                if met_target and not grew and not needs_raster:
                    with open(self.output_path, "wb") as f:
                        f.write(in_place_bytes)
                    self._maybe_linearize(self.output_path)
                    self.progress.emit(100)
                    self.finished.emit(True, "PDF compression completed.")
                    return

            # If we still have a target and didn’t meet it (or we must raster), try selective/full raster
            optimizer = InPlaceImageOptimizer(
                preserve_color_fidelity=preserve_color,
                grayscale=grayscale,
                min_jpeg_q=min_q_floor,
                progress_cb=self.progress.emit,
                status_cb=self.status_update.emit
            )
            max_eff_dpi = 300 if not preserve_color else 360

            # If grayscale is requested we guarantee grayscale by full raster pass
            if grayscale or force_raster:
                self.status_update.emit("Rasterizing document to honor settings (grayscale/force raster)...")
                raster_bytes = self._raster_to_target(target_bytes, preserve_color, grayscale,
                                                      min_dpi_floor=min_dpi_floor, min_q_floor=min_q_floor)
                # Choose smallest vs original; if target exists, prefer closest under target
                candidates = [(raster_bytes, len(raster_bytes))]
                if in_place_bytes is not None:
                    candidates.append((in_place_bytes, len(in_place_bytes)))
                if target_bytes is None:
                    chosen = min(candidates, key=lambda t: t[1])[0]
                else:
                    under = [(b, s) for (b, s) in candidates if s <= target_bytes]
                    if under:
                        chosen = max(under, key=lambda t: t[1])[0]
                    else:
                        chosen = min(candidates, key=lambda t: t[1])[0]
                if len(chosen) > original_bytes:
                    # do not grow the file
                    chosen = in_place_bytes if (in_place_bytes is not None and len(in_place_bytes) < original_bytes) else open(self.input_path, "rb").read()

                with open(self.output_path, "wb") as f:
                    f.write(chosen)
                self._maybe_linearize(self.output_path)
                self.progress.emit(100)
                self.finished.emit(True, "Done.")
                return

            # Otherwise (no grayscale/force), if we still have target, attempt selective raster first
            if target_bytes is not None and in_place_bytes is not None:
                self.status_update.emit("Attempting selective rasterization on heavy pages...")
                sel = self._selective_rasterize_to_target(
                    base_pdf_bytes=in_place_bytes,
                    target_bytes=target_bytes,
                    preserve_color=preserve_color,
                    grayscale=grayscale,
                    optimizer=optimizer,
                    max_eff_dpi=max_eff_dpi,
                    min_dpi_floor=min_dpi_floor,
                    min_q_floor=min_q_floor
                )
                if sel is not None:
                    final_bytes = sel
                    if len(final_bytes) > original_bytes:
                        final_bytes = in_place_bytes if len(in_place_bytes) < original_bytes else open(self.input_path, "rb").read()
                    with open(self.output_path, "wb") as f:
                        f.write(final_bytes)
                    self._maybe_linearize(self.output_path)
                    self.progress.emit(100)
                    self.finished.emit(True, "Selective rasterization completed.")
                    return

            # If still over target, do a full raster tuned to hit it
            if target_bytes is not None:
                self.status_update.emit("Full rasterization toward target...")
                raster_bytes = self._raster_to_target(target_bytes, preserve_color, grayscale,
                                                      min_dpi_floor=min_dpi_floor, min_q_floor=min_q_floor)
                candidates = []
                if in_place_bytes is not None:
                    candidates.append((in_place_bytes, len(in_place_bytes)))
                candidates.append((raster_bytes, len(raster_bytes)))
                under = [(b, s) for (b, s) in candidates if s <= target_bytes]
                chosen = (max(under, key=lambda t: t[1])[0] if under else min(candidates, key=lambda t: t[1])[0])
                if len(chosen) > original_bytes:
                    chosen = in_place_bytes if (in_place_bytes is not None and len(in_place_bytes) < original_bytes) else open(self.input_path, "rb").read()
                with open(self.output_path, "wb") as f:
                    f.write(chosen)
                self._maybe_linearize(self.output_path)
                self.progress.emit(100)
                self.finished.emit(True, "Done.")
                return

            # No better option: keep best in-place if it shrank; else original
            final_bytes = in_place_bytes if (in_place_bytes is not None and len(in_place_bytes) < original_bytes) else open(self.input_path, "rb").read()
            with open(self.output_path, "wb") as f:
                f.write(final_bytes)
            self.status_update.emit("Saved best achievable result without rasterizing.")
            self.progress.emit(100)
            self.finished.emit(True, "Done.")

        except InterruptedError:
            self.finished.emit(False, "Operation cancelled by user")
        except Exception as e:
            logging.exception("Compression error")
            self.finished.emit(False, f"Error: {e}")


# ------------------------- UI -------------------------

class DropArea(QWidget):
    file_dropped = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        layout = QVBoxLayout()
        label = QLabel("Drag a PDF here or click to select")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        self.label = label
        self.setLayout(layout)
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls() and event.mimeData().urls()[0].path().lower().endswith(".pdf"):
            event.acceptProposedAction()
    def dropEvent(self, event: QDropEvent):
        self.file_dropped.emit(event.mimeData().urls()[0].toLocalFile())
    def mousePressEvent(self, _):
        path, _ = QFileDialog.getOpenFileName(self, "Select PDF", "", "PDF Files (*.pdf)")
        if path:
            self.file_dropped.emit(path)

class PDFCompressorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.input_file_path = None
        self.worker = None
        self.pdf_info = None

        self.setWindowTitle("PDF Compressor")
        self.setMinimumSize(860, 680)

        self._build_ui()
        self._check_dependencies()

        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self._show_resources)
        self.memory_timer.start(2000)

    # ---- Presets ----
    def _preset_defs(self):
        return {
            "Manual / Custom": {
                "mode": "percentage", "value": 50.0,
                "grayscale": False, "preserve_color": False, "force_raster": False,
                "min_dpi": 96, "min_q": 30
            },
            "Sharable Doc (1 MB)": {
                "mode": "size", "value": 1.0,
                "grayscale": True, "preserve_color": False, "force_raster": False,
                "min_dpi": 96, "min_q": 28
            },
            "Email-safe (0.5 MB)": {
                "mode": "size", "value": 0.5,
                "grayscale": True, "preserve_color": False, "force_raster": True,
                "min_dpi": 72, "min_q": 25
            },
            "Web-view (50%)": {
                "mode": "percentage", "value": 50.0,
                "grayscale": False, "preserve_color": False, "force_raster": False,
                "min_dpi": 96, "min_q": 30
            },
            "High fidelity (75%)": {
                "mode": "percentage", "value": 75.0,
                "grayscale": False, "preserve_color": True, "force_raster": False,
                "min_dpi": 200, "min_q": 50
            },
            "Smallest reasonable (35%)": {
                "mode": "percentage", "value": 35.0,
                "grayscale": True, "preserve_color": False, "force_raster": True,
                "min_dpi": 72, "min_q": 22
            },
            "Presentation images (20 MB)": {
                "mode": "size", "value": 20.0,
                "grayscale": False, "preserve_color": True, "force_raster": False,
                "min_dpi": 200, "min_q": 60
            },
        }

    def _apply_preset_to_ui(self, preset_name: str):
        p = self.presets.get(preset_name)
        if not p:
            return
        mode = p["mode"]
        if mode == "percentage":
            self.percentage_radio.setChecked(True)
        elif mode == "size":
            self.size_radio.setChecked(True)
        elif mode == "resolution":
            self.resolution_radio.setChecked(True)
        else:
            self.reduction_radio.setChecked(True)

        self._update_value_range(mode)
        self.value_input.blockSignals(True)
        self.value_input.setValue(float(p["value"]))
        self.value_input.blockSignals(False)

        self.grayscale_cb.setChecked(bool(p["grayscale"]))
        self.preserve_color_cb.setChecked(bool(p["preserve_color"]))
        self.force_image_cb.setChecked(bool(p["force_raster"]))

        self.min_dpi_spin.setValue(int(p["min_dpi"]))
        self.min_q_spin.setValue(int(p["min_q"]))

        self.status_label.setText(f"Preset applied: {preset_name}")

    def _build_ui(self):
        self.presets = self._preset_defs()

        main = QWidget()
        layout = QVBoxLayout()

        # System status
        sys_group = QGroupBox("System Status")
        sys_h = QHBoxLayout()
        self.dependency_status = QLabel("Checking dependencies...")
        self.memory_status = QLabel("Memory: Checking...")
        self.disk_status = QLabel("Disk: Checking...")
        sys_h.addWidget(self.dependency_status)
        sys_h.addWidget(self.memory_status)
        sys_h.addWidget(self.disk_status)
        sys_group.setLayout(sys_h)
        layout.addWidget(sys_group)

        # Drop area
        self.drop_area = DropArea()
        self.drop_area.file_dropped.connect(self._file_selected)
        layout.addWidget(self.drop_area)

        # File info
        info_group = QGroupBox("File Information")
        info_v = QVBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_v.addWidget(self.file_label)
        self.pdf_info_label = QLabel()
        self.pdf_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_v.addWidget(self.pdf_info_label)
        info_group.setLayout(info_v)
        layout.addWidget(info_group)

        # Settings
        settings_group = QGroupBox("Compression Settings")
        settings_layout = QVBoxLayout()

        # Presets row
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Presets:"))
        self.preset_combo = QComboBox()
        for name in self.presets.keys():
            self.preset_combo.addItem(name)
        self.preset_combo.currentTextChanged.connect(self._apply_preset_to_ui)
        preset_row.addWidget(self.preset_combo)
        settings_layout.addLayout(preset_row)

        # Toggles row
        row1 = QHBoxLayout()
        self.grayscale_cb = QCheckBox("Convert to grayscale")
        self.force_image_cb = QCheckBox("Force rasterize if needed")
        self.preserve_color_cb = QCheckBox("Preserve color fidelity")
        row1.addWidget(self.grayscale_cb)
        row1.addWidget(self.force_image_cb)
        row1.addWidget(self.preserve_color_cb)
        settings_layout.addLayout(row1)

        # Modes
        mode_h = QHBoxLayout()
        self.mode_group = QButtonGroup()
        self.percentage_radio = QRadioButton("Percentage")
        self.percentage_radio.setChecked(True)
        self.size_radio = QRadioButton("Target Size (MB)")
        self.resolution_radio = QRadioButton("Resolution (DPI)")
        self.reduction_radio = QRadioButton("Size Reduction (%)")
        for r in (self.percentage_radio, self.size_radio, self.resolution_radio, self.reduction_radio):
            self.mode_group.addButton(r)
            mode_h.addWidget(r)
        settings_layout.addLayout(mode_h)

        # Value input
        val_h = QHBoxLayout()
        self.value_input = QDoubleSpinBox()
        self.value_input.setDecimals(0)
        self.value_input.setRange(5, 95)
        self.value_input.setSingleStep(1.0)
        self.value_input.setValue(50.0)
        self.value_input.setSuffix(" %")
        val_h.addWidget(QLabel("Value:"))
        val_h.addWidget(self.value_input)
        settings_layout.addLayout(val_h)

        # Advanced expert controls
        adv_group = QGroupBox("Advanced (expert)")
        adv_h = QHBoxLayout()
        self.min_dpi_spin = QSpinBox()
        self.min_dpi_spin.setRange(50, 300)
        self.min_dpi_spin.setValue(96)
        self.min_dpi_spin.setSuffix(" min DPI")
        self.min_q_spin = QSpinBox()
        self.min_q_spin.setRange(10, 90)
        self.min_q_spin.setValue(30)
        self.min_q_spin.setSuffix(" min JPEG Q")
        adv_h.addWidget(QLabel("Min DPI:"))
        adv_h.addWidget(self.min_dpi_spin)
        adv_h.addWidget(QLabel("Min JPEG quality:"))
        adv_h.addWidget(self.min_q_spin)
        adv_group.setLayout(adv_h)
        settings_layout.addWidget(adv_group)

        self.percentage_radio.toggled.connect(lambda: self._update_value_range('percentage'))
        self.size_radio.toggled.connect(lambda: self._update_value_range('size'))
        self.resolution_radio.toggled.connect(lambda: self._update_value_range('resolution'))
        self.reduction_radio.toggled.connect(lambda: self._update_value_range('reduction'))

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Output path
        out_h = QHBoxLayout()
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("Select output location...")
        out_h.addWidget(self.output_path)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._select_output_path)
        out_h.addWidget(browse_button)
        layout.addLayout(out_h)

        # Buttons
        btn_h = QHBoxLayout()
        self.compress_button = QPushButton("Compress PDF")
        self.compress_button.clicked.connect(self._start_compression)
        self.compress_button.setEnabled(False)
        btn_h.addWidget(self.compress_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel_compression)
        self.cancel_button.setEnabled(False)
        btn_h.addWidget(self.cancel_button)
        layout.addLayout(btn_h)

        # Progress
        prog_group = QGroupBox("Progress")
        prog_v = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        prog_v.addWidget(self.progress_bar)
        self.status_label = QLabel("Ready")
        prog_v.addWidget(self.status_label)
        self.detailed_status = QTextEdit()
        self.detailed_status.setMaximumHeight(160)
        self.detailed_status.setVisible(False)
        prog_v.addWidget(self.detailed_status)
        self.memory_progress = QProgressBar()
        self.memory_progress.setMaximum(100)
        self.memory_progress.setVisible(False)
        self.memory_progress.setFormat("Memory: %p%")
        prog_v.addWidget(self.memory_progress)
        prog_group.setLayout(prog_v)
        layout.addWidget(prog_group)

        main.setLayout(layout)
        self.setCentralWidget(main)

        self._apply_preset_to_ui("Manual / Custom")

    def _check_dependencies(self):
        bits = []
        bits.append("Poppler: OK" if DependencyChecker.check_poppler() else "Poppler: Missing")
        bits.append("PyMuPDF: OK" if DependencyChecker.check_pymupdf() else "PyMuPDF: Missing")
        bits.append("pypdf/PyPDF2: OK" if HAS_PYPDF2 else "pypdf/PyPDF2: Missing")
        bits.append("qpdf: OK" if DependencyChecker.check_qpdf() else "qpdf: Missing")
        bits.append("Ghostscript: OK" if DependencyChecker.check_ghostscript() else "Ghostscript: Missing")
        bits.append("pdfimages: OK" if DependencyChecker.check_pdfimages() else "pdfimages: Missing")
        self.dependency_status.setText(" | ".join(bits))
        self.dependency_status.setStyleSheet("color: green;" if "Missing" not in self.dependency_status.text() else "color: orange;")

    def _show_resources(self):
        try:
            available_memory_gb = DependencyChecker.check_available_memory()
            self.memory_status.setText(f"Memory: {available_memory_gb:.1f} GB available")
            self.memory_status.setStyleSheet("color: green;" if available_memory_gb > 2 else ("color: orange;" if available_memory_gb > 1 else "color: red;"))
        except Exception:
            pass
        out_dir = os.path.dirname(self.output_path.text()) if self.output_path.text() else os.getcwd()
        free_space_gb = DependencyChecker.get_free_disk_space(out_dir)
        self.disk_status.setText(f"Disk: {free_space_gb:.1f} GB free")
        self.disk_status.setStyleSheet("color: green;" if free_space_gb > 5 else ("color: orange;" if free_space_gb > 1 else "color: red;"))

    def _update_value_range(self, mode):
        if mode == 'percentage':
            self.value_input.setDecimals(0)
            self.value_input.setRange(5, 95)
            self.value_input.setSingleStep(1.0)
            self.value_input.setValue(50.0)
            self.value_input.setSuffix(" %")
            self.value_input.setToolTip("Final size as a percentage of original (50% ≈ half the size).")
        elif mode == 'size':
            self.value_input.setDecimals(1)
            self.value_input.setRange(0.1, 2000.0)
            self.value_input.setSingleStep(0.1)
            if self.value_input.value() < 0.1:
                self.value_input.setValue(1.0)
            self.value_input.setSuffix(" MB")
            self.value_input.setToolTip("Target final file size in MB (fractional allowed).")
        elif mode == 'resolution':
            self.value_input.setDecimals(0)
            self.value_input.setRange(72, 600)
            self.value_input.setSingleStep(1.0)
            self.value_input.setValue(150.0)
            self.value_input.setSuffix(" DPI")
            self.value_input.setToolTip("Cap images' effective DPI. No explicit size target.")
        else:
            self.value_input.setDecimals(0)
            self.value_input.setRange(5, 90)
            self.value_input.setSingleStep(1.0)
            self.value_input.setValue(50.0)
            self.value_input.setSuffix(" % smaller")
            self.value_input.setToolTip("Target size reduction relative to original.")

    def _select_output_path(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Compressed PDF", "", "PDF Files (*.pdf)")
        if file_path:
            self.output_path.setText(file_path)

    def _file_selected(self, file_path):
        self.input_file_path = file_path
        filename = os.path.basename(file_path)
        self.file_label.setText(f"Selected: {filename}")

        try:
            self.pdf_info = PDFAnalyzer.get_pdf_info(file_path)
            info_text = []
            if self.pdf_info['pages'] > 0:
                info_text.append(f"Pages: {self.pdf_info['pages']}")
            info_text.append(f"Size: {self.pdf_info['size_mb']:.1f} MB")
            if self.pdf_info['is_encrypted']:
                info_text.append("Encrypted")
            elif self.pdf_info['is_text_based']:
                info_text.append("Text-based")
            else:
                info_text.append("Image/Mixed")
            if self.pdf_info['has_bilevel']:
                info_text.append("B/W scan elements detected")
            self.pdf_info_label.setText(" | ".join(info_text))
        except Exception as e:
            self.pdf_info_label.setText(f"Analysis failed: {str(e)}")
            self.pdf_info = None

        directory = os.path.dirname(file_path) or os.getcwd()
        name, ext = os.path.splitext(filename)
        self.output_path.setText(os.path.join(directory, f"{name}_compressed{ext}"))

        can_compress = (not self.pdf_info or not self.pdf_info.get('is_encrypted', False))
        self.compress_button.setEnabled(can_compress)

    def _start_compression(self):
        if not self.input_file_path:
            QMessageBox.warning(self, "Error", "Please select a PDF file first.")
            return
        if not self.output_path.text():
            QMessageBox.warning(self, "Error", "Please select an output location.")
            return

        if self.percentage_radio.isChecked():
            mode = 'percentage'
        elif self.size_radio.isChecked():
            mode = 'size'
        elif self.resolution_radio.isChecked():
            mode = 'resolution'
        else:
            mode = 'reduction'

        compression_settings = {
            'mode': mode,
            'value': float(self.value_input.value()),
            'grayscale': self.grayscale_cb.isChecked(),
            'force_image_compression': self.force_image_cb.isChecked(),
            'preserve_color_fidelity': self.preserve_color_cb.isChecked(),
            'min_dpi': self.min_dpi_spin.value(),
            'min_jpeg_q': self.min_q_spin.value(),
        }

        self.progress_bar.setVisible(True)
        self.memory_progress.setVisible(True)
        self.detailed_status.setVisible(True)
        self.detailed_status.clear()

        self.status_label.setText("Starting compression...")
        self.compress_button.setEnabled(False)
        self.cancel_button.setEnabled(True)

        self.worker = PDFCompressorWorker(self.input_file_path, self.output_path.text(), compression_settings)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self._compression_finished)
        self.worker.status_update.connect(self._update_status)
        self.worker.memory_usage.connect(self._update_memory_display)
        self.worker.start()

    def _cancel_compression(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.status_label.setText("Cancelling...")
            self.cancel_button.setEnabled(False)

    def _update_status(self, message):
        self.status_label.setText(message)
        self.detailed_status.append(f"{message}")
        cursor = self.detailed_status.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.detailed_status.setTextCursor(cursor)

    def _update_memory_display(self, memory_mb):
        try:
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            memory_percent = (memory_mb / 1024) / max(0.001, total_memory_gb) * 100
            self.memory_progress.setValue(int(memory_percent))
            self.memory_progress.setFormat(f"Memory: {memory_mb:.0f} MB ({memory_percent:.1f}%)")
        except Exception:
            pass

    def _compression_finished(self, success, message):
        self.progress_bar.setVisible(False)
        self.memory_progress.setVisible(False)
        self.detailed_status.setVisible(False)

        self.status_label.setText(message)
        self.compress_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

        if success:
            try:
                original_size = file_size_bytes(self.input_file_path) / (1024**2)
                compressed_size = file_size_bytes(self.output_path.text()) / (1024**2)
                reduction = ((original_size - compressed_size) / max(1e-9, original_size)) * 100
                detailed_message = (f"{message}\n\n"
                                    f"Original size: {original_size:.2f} MB\n"
                                    f"Compressed size: {compressed_size:.2f} MB\n"
                                    f"Size reduction: {reduction:.1f}%")
                QMessageBox.information(self, "Compression Complete", detailed_message)
            except Exception:
                QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Compression Failed", message)

        if self.worker:
            self.worker.deleteLater()
            self.worker = None


# ------------------------- Main -------------------------

def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    app = QApplication(sys.argv)
    window = PDFCompressorApp()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()