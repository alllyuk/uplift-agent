from pathlib import Path
import re
import pdfplumber

from sme_causal.core.config import get_config

# ---------- очистка текста (как в твоём примере, бережно к связным предложениям) ----------
def _clean_text(txt: str) -> str:
    # невидимые символы и переносы
    txt = (txt or "")
    txt = txt.replace("\u00AD", "").replace("\u200b", "").replace("\ufeff", "")
    txt = re.sub(r"([A-Za-zА-Яа-яЁё])-\n([A-Za-zА-Яа-яЁё])", r"\1\2", txt)
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")

    # телефоны, email, одиночные числа и тех.символы
    txt = re.sub(r"\+?\d[\d\s()–-]{6,}", "", txt)
    txt = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", "", txt)
    txt = re.sub(r"\b\d{1,2}[.,/]\d{1,2}[.,/]\d{1,4}\b", "", txt)  # даты/форматы
    txt = re.sub(r"(?<!\w)\d{1,3}(?![\w%])", "", txt)             # одиночные числа
    txt = re.sub(r"[•●▪▫■□◆◇▲▼▶◀※§¤]+", " ", txt)
    txt = re.sub(r"[=+_<>©×°№–—]+", " ", txt)
    txt = re.sub(r"[A-Z]{2,}", "", txt)
    txt = re.sub(r"\s{2,}", " ", txt)

    cleaned_lines = []
    for line in txt.split("\n"):
        s = line.strip()
        if not s:
            cleaned_lines.append("")
            continue
        # доля букв в строке (режем «шум»)
        letters = len(re.findall(r"[A-Za-zА-Яа-яЁё]", s))
        if letters / max(1, len(s)) < 0.35:
            continue
        # подписи/служебные элементы
        if re.match(r"^(Рисунок|Таблица|Источник|Figure|Table|Контакты|СОДЕРЖАНИЕ|АКРА|www\.acra|–c\.|©|Официальный сайт|Банк России|www\.cbr\.ru)",
                    s, re.IGNORECASE):
            continue
        cleaned_lines.append(s)

    clean_text = "\n".join(cleaned_lines)
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text).strip()
    return clean_text


# ---------- извлечение текста из PDF ----------
def _extract_pdf_text(pdf_path: Path) -> str:
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            # x/y_tolerance можно подстроить при необходимости
            t = page.extract_text(x_tolerance=1.5, y_tolerance=2.0) or ""
            pages.append(t)
    # разделяем страницы пустой строкой (сохранит абзацы)
    return "\n\n".join(pages)


# ---------- основная функция: один PDF -> cleaned TXT ----------
def clean_pdf_to_txt(pdf_path: Path, cfg) -> Path:
    """
    Преобразует один PDF в очищенный TXT.
    Вход: pdf_path (файл внутри cfg.raw_documents_dir)
    Выход: путь к сохранённому TXT внутри cfg.cleaned_documents_dir: <stem>_cleaned.txt
    """
    raw_dir = Path(cfg.raw_documents_dir)
    out_dir = Path(cfg.cleaned_documents_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.is_absolute():
        pdf_path = raw_dir / pdf_path

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF не найден: {pdf_path}")

    raw_txt = _extract_pdf_text(pdf_path)
    cleaned = _clean_text(raw_txt)

    out_path = out_dir / f"{pdf_path.stem}_cleaned.txt"
    out_path.write_text(cleaned, encoding="utf-8")
    return out_path


# ---------- батч: обработать все PDF в cfg.raw_documents_dir ----------
def clean_all_pdfs() -> list[Path]:
    cfg = get_config()
    raw_dir = Path(cfg.raw_documents_dir)
    out_dir = Path(cfg.cleaned_documents_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for pdf in sorted(raw_dir.glob("*.pdf")):
        try:
            results.append(clean_pdf_to_txt(pdf, cfg))
        except Exception as e:
            # лог ошибки рядом с целевым .txt
            (out_dir / f"{pdf.stem}_cleaned.err.txt").write_text(str(e), encoding="utf-8")
    return results

# ---------- тестирование функции ----------
if __name__ == "__main__":
    cleaned_files = clean_all_pdfs()
    print(f"Обработано файлов: {len(cleaned_files)}")
