"""The app's words in English and Arabic.

    from arabic_ocr.gui.i18n import set_language, t
    set_language("ar")
    t("open")            # -> "فتح…"
    t("page_of", i=2, n=5)

A plain dictionary rather than Qt's .ts/.qm translation files: easy to read and
edit, and no extra build step. To add a language, add a column to every entry
and a name to LANGUAGES. Missing words fall back to English.
"""

from __future__ import annotations

LANGUAGES = {"en": "English", "ar": "العربية"}

#: Languages written right to left: the whole window is mirrored for them.
RTL = {"ar"}

STRINGS: dict[str, dict[str, str]] = {
    "app_name": {"en": "Arabic OCR", "ar": "القارئ العربي"},
    "open": {"en": "Open…", "ar": "فتح…"},
    "open_tip": {"en": "Open images or PDF files", "ar": "فتح صور أو ملفات PDF"},
    "open_folder": {"en": "Open folder…", "ar": "فتح مجلد…"},
    "paste": {"en": "Paste image", "ar": "لصق صورة"},
    "paste_tip": {
        "en": "Read the image on the clipboard (e.g. a screenshot)",
        "ar": "قراءة الصورة الموجودة في الحافظة (مثل لقطة شاشة)",
    },
    "sample": {"en": "Try a sample", "ar": "تجربة مثال"},
    "copy": {"en": "Copy text", "ar": "نسخ النص"},
    "copy_tip": {"en": "Copy all text of this file", "ar": "نسخ كل نص هذا الملف"},
    "export": {"en": "Export…", "ar": "تصدير…"},
    "export_tip": {
        "en": "Save this file's text as TXT, JSON, searchable PDF, hOCR or ALTO",
        "ar": "حفظ نص هذا الملف بصيغة TXT أو JSON أو PDF قابل للبحث أو hOCR أو ALTO",
    },
    "export_all": {"en": "Export all…", "ar": "تصدير الكل…"},
    "export_all_tip": {
        "en": "Save every finished file to the output folder",
        "ar": "حفظ كل الملفات المنتهية في مجلد الحفظ",
    },
    "settings": {"en": "Settings", "ar": "الإعدادات"},
    "about": {"en": "About", "ar": "حول البرنامج"},
    "remove": {"en": "Remove", "ar": "إزالة"},
    "clear": {"en": "Clear list", "ar": "مسح القائمة"},
    "cancel": {"en": "Cancel", "ar": "إلغاء"},
    "files": {"en": "Files", "ar": "الملفات"},
    "fit": {"en": "Fit", "ar": "ملاءمة"},
    "prev_page": {"en": "Previous page", "ar": "الصفحة السابقة"},
    "next_page": {"en": "Next page", "ar": "الصفحة التالية"},
    "page_of": {"en": "Page {i} of {n}", "ar": "صفحة {i} من {n}"},
    "drop_hint": {
        "en": "Drop images or PDFs here, paste a screenshot (Ctrl+V), or click Open.",
        "ar": "اسحب صورًا أو ملفات PDF إلى هنا، أو الصق لقطة شاشة (Ctrl+V)، أو اضغط فتح.",
    },
    "waiting": {"en": "waiting", "ar": "في الانتظار"},
    "reading": {"en": "reading", "ar": "جارٍ القراءة"},
    "done": {"en": "done", "ar": "تم"},
    "failed": {"en": "failed", "ar": "فشل"},
    "cancelled": {"en": "cancelled", "ar": "أُلغي"},
    "edited": {"en": "edited", "ar": "معدَّل"},
    "loading_models": {"en": "Loading models…", "ar": "تحميل النماذج…"},
    "status_reading": {
        "en": "Reading {name}: page {i} of {n} ({left} more in queue)",
        "ar": "قراءة {name}: صفحة {i} من {n} ({left} أخرى في الانتظار)",
    },
    "status_ready": {"en": "Ready", "ar": "جاهز"},
    "status_done": {
        "en": "Done: {name} ({pages} page(s), {lines} line(s), {seconds:.1f} s)",
        "ar": "تم: {name} ({pages} صفحة، {lines} سطر، {seconds:.1f} ث)",
    },
    "copied": {"en": "Text copied", "ar": "تم نسخ النص"},
    "saved": {"en": "Saved {path}", "ar": "تم الحفظ: {path}"},
    "saved_all": {
        "en": "Saved {n} file(s) to {folder}",
        "ar": "تم حفظ {n} ملف في {folder}",
    },
    "no_image": {"en": "The clipboard has no image.", "ar": "لا توجد صورة في الحافظة."},
    "pasted": {"en": "Pasted image {n}", "ar": "صورة ملصقة {n}"},
    "error": {"en": "Error", "ar": "خطأ"},
    "models_missing": {
        "en": "The OCR models could not be loaded:\n{error}",
        "ar": "تعذّر تحميل نماذج التعرف:\n{error}",
    },
    "export_failed": {"en": "Could not save the file:\n{error}", "ar": "تعذّر حفظ الملف:\n{error}"},
    "nothing_to_export": {
        "en": "No finished files to export yet.",
        "ar": "لا توجد ملفات منتهية للتصدير بعد.",
    },
    "unsupported": {
        "en": "Skipped (not an image or PDF): {names}",
        "ar": "تم التجاوز (ليست صورة أو PDF): {names}",
    },
    # Settings dialog
    "settings_title": {"en": "Settings", "ar": "الإعدادات"},
    "language": {"en": "Language", "ar": "اللغة"},
    "keep_diacritics": {"en": "Keep diacritics (tashkeel)", "ar": "إبقاء التشكيل"},
    "keep_tatweel": {"en": "Keep tatweel (ـ)", "ar": "إبقاء التطويل (ـ)"},
    "digits": {"en": "Digits", "ar": "الأرقام"},
    "digits_keep": {"en": "As printed", "ar": "كما هي في الصورة"},
    "digits_western": {"en": "Western (0-9)", "ar": "غربية (0-9)"},
    "digits_arabic": {"en": "Arabic-Indic (٠-٩)", "ar": "عربية هندية (٠-٩)"},
    "output_folder": {"en": "Output folder", "ar": "مجلد الحفظ"},
    "browse": {"en": "Browse…", "ar": "استعراض…"},
    "export_format": {"en": "Format for Export all", "ar": "صيغة تصدير الكل"},
    "options_note": {
        "en": "Options apply to all results at once; nothing is read again.",
        "ar": "تُطبَّق الخيارات على كل النتائج فورًا دون إعادة القراءة.",
    },
    "about_text": {
        "en": "<b>Arabic OCR</b> {version}<br>Reads Arabic, Persian and Urdu text from "
        "images and PDFs. Works offline; nothing leaves this computer.<br>"
        "Apache-2.0 license. Built with Qt (LGPL) and ONNX Runtime.",
        "ar": "<b>القارئ العربي</b> {version}<br>يقرأ النصوص العربية والفارسية والأردية "
        "من الصور وملفات PDF. يعمل دون اتصال بالإنترنت، ولا يغادر أي شيء هذا الجهاز.<br>"
        "رخصة Apache-2.0. مبني بـ Qt (LGPL) و ONNX Runtime.",
    },
}

_language = "en"


def set_language(code: str) -> None:
    global _language
    _language = code if code in LANGUAGES else "en"


def language() -> str:
    return _language


def is_rtl() -> bool:
    return _language in RTL


def t(key: str, **values) -> str:
    """The text for `key` in the current language, with {values} filled in."""
    entry = STRINGS[key]
    text = entry.get(_language) or entry["en"]
    return text.format(**values) if values else text
