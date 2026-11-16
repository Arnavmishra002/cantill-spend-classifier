"""Small synthetic companies that mimic awkward real exports. No real data.

D: B2B product lines with prices in several currencies, one status value ("Received"),
   and a UNSPSC-style tree with Segment, Family and Class name columns.
E: a European export with German headers, semicolons and cp1252 encoding; statuses in
   German, French and Spanish; SAP trailing minus; German, ISO and Excel serial dates;
   gross amounts with a VAT rate and a tax amount; a tree with code and name columns.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

D_TREE = [("Information Technology", "Computer Equipment", "Laptops"),
          ("Information Technology", "Computer Equipment", "Monitors"),
          ("Information Technology", "Software", "Office Software"),
          ("Office Supplies", "Paper Products", "Printer Paper"),
          ("Office Supplies", "Writing Instruments", "Pens"),
          ("Facility Services", "Cleaning", "Janitorial Supplies")]
D_ITEMS = [("Dell Latitude 5440 laptop 16GB", 0), ("HP EliteBook 840 notebook", 0), ("27 inch IPS monitor", 1),
           ("Dell 24in LED display", 1), ("Microsoft 365 Business licence", 2), ("A4 copier paper 80gsm box", 3),
           ("Ballpoint pens blue pack of 50", 4), ("Floor cleaner 5L concentrate", 5)]
D_SELLERS = ["Northwind Traders Ltd", "Contoso Supplies Pty Ltd", "Fabrikam GmbH", "Tailspin Toys Sdn Bhd",
             "Adventure Works Limited", "Wide World Importers"]
D_CCY = {"GBP": 0.78, "EUR": 0.92, "INR": 83.5, "ZAR": 18.3, "NZD": 1.65, "AUD": 1.52, "MYR": 4.7}


def make_company_d(folder: Path, n: int = 600, seed: int = 1) -> Path:
    rnd = random.Random(seed)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    ccys = list(D_CCY)
    for i in range(n):
        text, _ = D_ITEMS[i % len(D_ITEMS)]
        ccy = ccys[i % len(ccys)]
        usd = rnd.uniform(20, 900)
        qty = rnd.randint(1, 5)
        rows.append({"order_id": f"ORD-{100000 + i}", "seller_name": D_SELLERS[i % len(D_SELLERS)],
                     "product_title": text, "quantity": qty,
                     "line_total": f"{usd * qty * D_CCY[ccy]:.2f}", "currency": ccy,
                     "order_date": f"2024-{1 + i % 12:02d}-{1 + i % 27:02d}", "status": "Received"})
    pd.DataFrame(rows).to_csv(folder / "orders.csv", index=False)
    pd.DataFrame(D_TREE, columns=["Segment", "Family", "Class"]).to_csv(folder / "unspsc_tree.csv", index=False)
    return folder


E_TREE = [("10", "Beschaffung IT", "1010", "Hardware", "101010", "Notebooks"),
          ("10", "Beschaffung IT", "1010", "Hardware", "101020", "Bildschirme"),
          ("20", "Büro", "2010", "Büromaterial", "201010", "Papier"),
          ("30", "Dienstleistungen", "3010", "Reinigung", "301010", "Gebäudereinigung")]
E_ITEMS = ["Notebook Lenovo ThinkPad T14", "Monitor 27 Zoll", "Kopierpapier A4 500 Blatt",
           "Unterhaltsreinigung Büro März", "Papier A3 Recycling", "Bildschirm Dell 24"]
E_SUPPLIERS = [("Bürobedarf Müller GmbH", "K1001"), ("Bürobedarf Müller GmbH & Co. KG", "K1001"),
               ("Société Générale de Nettoyage SARL", "K2002"), ("Informática Ibérica S.L.", "K3003"),
               ("Les Services Tremblay Ltée", "K4004")]
# status, expected outcome
E_STATUS = [("Bezahlt", "include"), ("Genehmigt", "include"), ("Payé", "include"), ("Pagado", "include"),
            ("Aprobado", "include"), ("Storniert", "CANCELLED"), ("Abgelehnt", "NOT_APPROVED"),
            ("En attente", "NOT_APPROVED")]


def _de_number(x: float) -> str:
    """1234.5 -> '1.234,50'; negative values use the SAP trailing minus: '1.234,50-'."""
    s = f"{abs(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return s + ("-" if x < 0 else "")


def make_company_e(folder: Path, n: int = 400, seed: int = 2) -> Path:
    rnd = random.Random(seed)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        sup, sid = E_SUPPLIERS[i % len(E_SUPPLIERS)]
        net = round(rnd.uniform(50, 2000), 2) * (-1 if i % 25 == 0 else 1)
        rate = 19 if i % 3 else 7
        tax = round(net * rate / 100, 2)
        day = 1 + i % 27
        style = i % 3
        date = (f"{day:02d}.03.2025" if style == 0 else f"2025-03-{day:02d}" if style == 1
                else str(45716 + day))                     # Excel serial for March 2025
        rows.append({"Zeilen-ID": f"Z{i:05d}", "Belegnummer": f"51000{i // 2:04d}", "Position": 1 + i % 2,
                     "Lieferant": sup, "Kreditor-Nr.": sid, "Bezeichnung": E_ITEMS[i % len(E_ITEMS)],
                     "Rechnungsdatum": date, "Buchungsdatum": f"{min(day + 2, 28):02d}.03.2025",
                     "Bruttobetrag": _de_number(net + tax), "MwSt": str(rate), "Steuerbetrag": _de_number(tax),
                     "Währung": "EUR", "Status": E_STATUS[i % len(E_STATUS)][0]})
    pd.DataFrame(rows).to_csv(folder / "rechnungen_export.csv", index=False, sep=";", encoding="cp1252")
    pd.DataFrame(E_TREE, columns=["L1 Code", "L1 Name", "L2 Code", "L2 Name", "L3 Code", "L3 Name"]).to_csv(
        folder / "kategorien.csv", index=False, sep=";", encoding="cp1252")
    return folder
