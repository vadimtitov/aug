"""Generate the eval's documents and images — every one invented, nothing real.

Text PDFs get a real text layer (reportlab), scans are images wrapped in a PDF with
no text layer, photos and screenshots are rendered with Pillow.  Rebuilt on every
run, so changing a fixture is changing this file.

    uv run python -m eval.kb.fixtures <out_dir>
"""

import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
]

TEXT_PDFS: dict[str, list[str]] = {
    "klm_eticket.pdf": [
        "KLM Royal Dutch Airlines — Electronic Ticket Receipt",
        "Booking reference: KX7Q2P",
        "Passenger: MR MAX ORLOV",
        "Ticket number: 074 2415 889301",
        "Flight KL1006  London Heathrow (LHR) → Amsterdam Schiphol (AMS)",
        "Date: Friday 14 August 2026   Departure 08:05   Arrival 10:25",
        "Return KL1023  Amsterdam (AMS) → London Heathrow (LHR)",
        "Date: Tuesday 18 August 2026  Departure 19:40   Arrival 20:05",
        "Fare: EUR 338.00   Taxes and charges: EUR 74.80   Total paid: EUR 412.80",
        "Baggage: 1 x 23 kg checked",
    ],
    "tenancy_agreement.pdf": [
        "ASSURED SHORTHOLD TENANCY AGREEMENT",
        "Property: Flat 3, 22 Ferndale Road, London SW9 8BJ",
        "Landlord: Mr Graham Whitlock, 5 Oakfield Close, Guildford GU1 2PX",
        "Tenants: Max Orlov and Lena Orlova",
        "Term: 12 months starting 1 January 2026",
        "Rent: £1,850 per calendar month, payable on the 1st of each month",
        "Deposit: £2,135 held with the Deposit Protection Service",
        "Notice: two months' written notice by either party after month six.",
    ],
    "payslip_2026_09.pdf": [
        "Northwind Fintech Ltd — Payslip",
        "Employee: Max Orlov   Employee no. NW-0457   Pay date: 30 September 2026",
        "Period: September 2026",
        "Gross pay: £6,250.00",
        "Income tax: £1,412.20   National Insurance: £377.25   Pension: £250.00",
        "Net pay: £4,210.55",
    ],
    "hotel_invoice.pdf": [
        "Canal View Hotel — Herengracht 341, 1016 AZ Amsterdam",
        "INVOICE No. CVH-2026-08812",
        "Guest: Max Orlov   Stay: 14 August 2026 – 18 August 2026 (4 nights)",
        "Room: Deluxe Double, canal view",
        "Accommodation 4 × EUR 150.00 = EUR 600.00",
        "City tax: EUR 40.00",
        "Total: EUR 640.00 — PAID by Visa ending 4417 on 18 August 2026",
    ],
    "sofia_vaccinations.pdf": [
        "NHS Personal Child Health Record — Immunisations",
        "Child: Sofia Orlova   Date of birth: 02/02/2026",
        "8 weeks (30/03/2026): 6-in-1, MenB, Rotavirus — given",
        "12 weeks (27/04/2026): 6-in-1, Pneumococcal, Rotavirus — given",
        "3 September 2026: MMR (first dose) — given at Ferndale Medical Centre",
        "Next due: Hib/MenC booster, at 1 year",
    ],
    "car_insurance.pdf": [
        "Brightline Insurance — Motor Policy Schedule",
        "Policy number: AV-55128-K",
        "Policyholder: Max Orlov",
        "Vehicle: Volkswagen Golf 1.5 TSI, blue, registration LK19 XRT",
        "Cover: Comprehensive   Period: 15 June 2026 to 14 June 2027",
        "Annual premium: £612.40   Voluntary excess: £250",
    ],
    "garage_rent_ru.pdf": [
        "ДОГОВОР АРЕНДЫ ГАРАЖА № 17",
        "г. Тверь, ул. Ленина, д. 5, гаражный бокс № 17",
        "Арендодатель: Смирнов Пётр Ильич",
        "Арендатор: Орлов Максим",
        "Арендная плата: 15 000 рублей в месяц, до 10 числа",
        "Срок аренды: с 1 июля 2026 года по 30 июня 2027 года",
    ],
}

SCANS: dict[str, list[str]] = {
    "hardware_receipt_scan.pdf": [
        "FERNDALE HARDWARE",
        "118 Ferndale Rd, London",
        "12/09/2026  14:32",
        "Shelf bracket x4      £11.60",
        "Pine shelf 80cm x2    £19.80",
        "Wall plugs + screws    £6.00",
        "TOTAL                 £37.40",
        "PAID CARD  ****4417",
    ],
    "council_letter_scan.pdf": [
        "Borough of Westbridge — Council Tax",
        "Account reference: 9917-3342",
        "Mr M Orlov, Flat 3, 22 Ferndale Road, London SW9 8BJ",
        "Your property is in Council Tax Band D.",
        "Annual charge 2026/27: £2,104.00, payable in 10 instalments of £210.40",
        "First instalment due 1 November 2026.",
    ],
}

IMAGES: dict[str, list[str]] = {
    "boarding_pass.png": [
        "KLM  BOARDING PASS",
        "ORLOV/MAX MR",
        "FLIGHT KL1006   LHR → AMS",
        "DATE 14AUG   BOARDING 07:25",
        "GATE D7   SEAT 14C   ZONE 3",
        "BOOKING KX7Q2P",
    ],
    "restaurant_receipt.png": [
        "TRATTORIA LUPA",
        "47 Brixton Water Lane",
        "21/09/2026  21:04   Table 6",
        "2x Cacio e pepe     £29.00",
        "1x Branzino         £24.00",
        "Bottle Nero d'Avola £26.00",
        "Service 10%          £7.50",
        "TOTAL               £86.50",
    ],
    "office_wifi.png": [
        "Wi-Fi  >  Northwind Office",
        "Guest network",
        "Network name: NWF-Guest",
        "Password: Ledger#2026",
        "Valid until: 31 Dec 2026",
    ],
    "parking_ticket.png": [
        "PENALTY CHARGE NOTICE",
        "Borough of Westbridge",
        "PCN No. WB48213977",
        "Vehicle: MV23 PLK   Date: 03/10/2026  10:52",
        "Location: Acre Lane, bay 14",
        "Contravention: parked without payment",
        "Charge: £65 (reduced to £32.50 if paid within 14 days)",
    ],
    "atlas_whiteboard.png": [
        "ATLAS LAUNCH — PLAN",
        "Owner: Anna Schmidt",
        "Launch deadline: 30 Nov",
        "1. Payments API freeze — 6 Nov",
        "2. Load test staging — 13 Nov",
        "3. Go/no-go — 27 Nov",
    ],
}


def build(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    font_path = _font()
    pdfmetrics.registerFont(TTFont("EvalSans", font_path))
    paths: dict[str, Path] = {}
    for name, lines in TEXT_PDFS.items():
        paths[name] = _text_pdf(out_dir / name, lines)
    for name, lines in SCANS.items():
        paths[name] = _scan_pdf(out_dir / name, lines, font_path)
    for name, lines in IMAGES.items():
        image = _render(lines, font_path, photo=name != "office_wifi.png")
        image.save(out_dir / name)
        paths[name] = out_dir / name
    return paths


def _font() -> str:
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise SystemExit("No font with Cyrillic glyphs found — install DejaVu Sans")


def _text_pdf(path: Path, lines: list[str]) -> Path:
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFont("EvalSans", 16)
    y = A4[1] - 72
    pdf.drawString(56, y, lines[0])
    pdf.setFont("EvalSans", 11)
    for line in lines[1:]:
        y -= 24
        pdf.drawString(56, y, line)
    pdf.save()
    return path


def _scan_pdf(path: Path, lines: list[str], font_path: str) -> Path:
    image = _render(lines, font_path, photo=True, width=1240)
    pdf = canvas.Canvas(str(path), pagesize=A4)
    w, h = A4
    pdf.drawImage(
        ImageReader(image),
        30,
        h - 30 - image.height * (w - 60) / image.width,
        width=w - 60,
        height=image.height * (w - 60) / image.width,
    )
    pdf.save()
    return path


def _render(lines: list[str], font_path: str, *, photo: bool, width: int = 900) -> Image.Image:
    rng = random.Random(len("".join(lines)))
    font = ImageFont.truetype(font_path, 34)
    height = 90 + 56 * len(lines)
    background = (246, 243, 236) if photo else (255, 255, 255)
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((48, 45 + 56 * i), line, fill=(25, 25, 30), font=font)
    if photo:
        image = image.rotate(rng.uniform(-1.5, 1.5), expand=True, fillcolor=(200, 196, 188))
        image = image.filter(ImageFilter.GaussianBlur(0.6))
    return image


if __name__ == "__main__":
    for name, p in build(
        Path(sys.argv[1] if len(sys.argv) > 1 else "eval-results/kb/fixtures")
    ).items():
        print(name, p)
