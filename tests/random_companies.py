"""Random unseen companies with known answers, for tests. Synthetic data only.

make(seed, work) writes one company's invoice lines and category tree in a random layout and
returns the truth for every record. Layout choices: header language (English, German, French,
Spanish), column order, delimiter, encoding, workbook or text file, title rows, blank rows and
a totals line, number and date formats, net or gross amounts with a tax rate or tax amount,
one or many currencies written as codes or names, with or without a rate column, a status
column in five vocabularies or yes and no flag columns or none, vendor ids with or without
names, and a tree that is wide, coded, parent and child, or one path column.
"""
from __future__ import annotations

import random
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spendcube.clean import load_public_fx  # noqa: E402

FX = load_public_fx("USD")
FX["m"] = FX["month"].dt.strftime("%Y-%m")
RATE = {(r.currency, r.m): r.rate for r in FX.itertuples()}

INDUSTRIES = {
    "hospital": [("Medical", "Consumables", "Gloves and Masks", ["nitrile exam gloves box", "surgical face masks", "latex gloves sterile"]),
                 ("Medical", "Consumables", "Syringes and Needles", ["disposable syringe 5ml", "hypodermic needles 21g"]),
                 ("Medical", "Equipment", "Imaging Equipment", ["ultrasound probe replacement", "x-ray detector panel"]),
                 ("Medical", "Pharmaceuticals", "Generic Drugs", ["paracetamol 500mg tablets", "amoxicillin capsules"]),
                 ("Facilities", "Cleaning", "Janitorial Services", ["ward cleaning service monthly", "floor disinfectant 5l"]),
                 ("Facilities", "Catering", "Patient Meals", ["patient meal service", "kitchen food supplies"]),
                 ("IT", "Software", "Clinical Software", ["patient records software licence", "radiology system subscription"]),
                 ("Professional Services", "Staffing", "Agency Nurses", ["agency nurse shifts", "locum doctor cover"])],
    "construction": [("Materials", "Concrete and Cement", "Ready Mix Concrete", ["ready mix concrete 30mpa", "cement bags 50kg"]),
                     ("Materials", "Steel", "Reinforcing Steel", ["rebar 16mm", "steel mesh sheets"]),
                     ("Materials", "Timber", "Structural Timber", ["pine timber 38x114", "plywood sheets 18mm"]),
                     ("Plant and Equipment", "Hire", "Excavator Hire", ["excavator hire weekly", "20t excavator rental"]),
                     ("Plant and Equipment", "Hire", "Crane Hire", ["mobile crane hire", "tower crane rental month"]),
                     ("Subcontractors", "Electrical", "Electrical Installation", ["electrical wiring subcontract", "db board installation"]),
                     ("Subcontractors", "Plumbing", "Plumbing Installation", ["plumbing rough-in subcontract", "drainage pipe laying"]),
                     ("Site Services", "Security", "Site Security", ["site security guards", "night watch service"])],
    "software": [("Cloud", "Infrastructure", "Compute", ["cloud compute instances", "virtual machines usage"]),
                 ("Cloud", "Infrastructure", "Storage", ["object storage usage", "backup storage gb"]),
                 ("Software", "Licences", "Developer Tools", ["ide licences annual", "code repository seats"]),
                 ("Software", "Licences", "Office Software", ["office suite subscription", "email licences"]),
                 ("Hardware", "End User", "Laptops", ["developer laptop 32gb", "macbook pro 14"]),
                 ("Marketing", "Advertising", "Online Ads", ["search ads campaign", "social media advertising"]),
                 ("Professional Services", "Legal", "Legal Counsel", ["legal fees contract review", "trademark filing counsel"]),
                 ("Facilities", "Office", "Office Rent", ["office rent monthly", "coworking desks rental"])],
    "food": [("Ingredients", "Dairy", "Milk and Cream", ["raw milk bulk litres", "cream 40 percent"]),
             ("Ingredients", "Grains", "Flour", ["wheat flour 25kg", "bread flour bulk"]),
             ("Ingredients", "Sweeteners", "Sugar", ["white sugar 50kg", "glucose syrup drum"]),
             ("Packaging", "Primary", "Bottles and Jars", ["pet bottles 500ml", "glass jars 250g"]),
             ("Packaging", "Secondary", "Cartons", ["corrugated cartons", "shrink wrap rolls"]),
             ("Logistics", "Transport", "Road Freight", ["refrigerated truck delivery", "road freight pallets"]),
             ("Maintenance", "Spares", "Conveyor Spares", ["conveyor belt replacement", "gearbox for filler line"]),
             ("Utilities", "Energy", "Electricity", ["electricity supply month", "power usage charge"])],
    "logistics": [("Fleet", "Fuel", "Diesel", ["diesel bulk delivery", "fuel card diesel"]),
                  ("Fleet", "Maintenance", "Tyres", ["truck tyres 315/80", "tyre fitting service"]),
                  ("Fleet", "Maintenance", "Vehicle Servicing", ["truck service 40000km", "brake pads replacement"]),
                  ("Warehousing", "Equipment", "Forklifts", ["forklift lease monthly", "reach truck rental"]),
                  ("Warehousing", "Consumables", "Pallets and Wrap", ["wooden pallets", "stretch wrap film"]),
                  ("IT", "Telematics", "Vehicle Tracking", ["vehicle tracking subscription", "telematics units"]),
                  ("Insurance", "Fleet Insurance", "Motor Insurance", ["fleet motor insurance premium", "goods in transit cover"]),
                  ("Subcontractors", "Haulage", "Third Party Haulage", ["subcontract haulage loads", "owner driver trips"])],
}
NP_TEXTS = ["Sales Tax", "VAT payable", "GST Input Tax", "Intercompany recharge March", "Bank charges", "Interest expense",
            "Royalty payment", "Withholding tax", "Inter-company charge group services", "Loan repayment", "Payroll"]
STATUS = {
    "en_a": {"Paid": "", "Approved": "", "Cancelled": "CANCELLED", "Rejected": "NOT_APPROVED", "On Hold": "NOT_APPROVED",
             "Pending Approval": "NOT_APPROVED", "Disputed": "NOT_APPROVED"},
    "en_b": {"Posted": "", "Settled": "", "Received": "", "Voided": "CANCELLED", "Draft": "NOT_APPROVED",
             "Under Review": "NOT_APPROVED", "Submitted": "NOT_APPROVED"},
    "de": {"Bezahlt": "", "Genehmigt": "", "Gebucht": "", "Storniert": "CANCELLED", "Abgelehnt": "NOT_APPROVED",
           "Gesperrt": "NOT_APPROVED", "In Prüfung": "NOT_APPROVED"},
    "fr": {"Payé": "", "Approuvé": "", "Comptabilisé": "", "Annulé": "CANCELLED", "Rejeté": "NOT_APPROVED",
           "En attente": "NOT_APPROVED", "Brouillon": "NOT_APPROVED"},
    "es": {"Pagado": "", "Aprobado": "", "Contabilizado": "", "Anulado": "CANCELLED", "Rechazado": "NOT_APPROVED",
           "Pendiente": "NOT_APPROVED", "Borrador": "NOT_APPROVED"},
}
H = {  # header choices per role and language
    "id": {"en": ["Line ID", "line_id", "Record ID", "Invoice Line ID", "Txn ID", "Unique Key", "Line Ref", "Row No"],
           "de": ["Zeilen-ID", "Positions-ID"], "fr": ["ID ligne", "Identifiant ligne"], "es": ["ID línea", "Id de línea"]},
    "inv": {"en": ["Invoice Number", "Invoice No", "Inv #", "Document Number"], "de": ["Rechnungsnummer", "Belegnummer"],
            "fr": ["N° facture", "Numéro de facture"], "es": ["Número de factura", "No factura"]},
    "sup": {"en": ["Supplier Name", "Vendor Name", "Supplier", "Vendor", "Payee", "Creditor Name"], "de": ["Lieferant", "Kreditor Name"],
            "fr": ["Fournisseur", "Nom du fournisseur"], "es": ["Proveedor", "Nombre del proveedor"]},
    "supid": {"en": ["Supplier ID", "Vendor Code", "Vendor No.", "Vendor #"], "de": ["Kreditor-Nr.", "Lieferantennummer"],
              "fr": ["Code fournisseur"], "es": ["Código proveedor"]},
    "desc": {"en": ["Description", "Line Description", "Item Description", "Short Text", "Narration", "Particulars"],
             "de": ["Bezeichnung", "Positionstext"], "fr": ["Libellé", "Désignation"], "es": ["Descripción"]},
    "net": {"en": ["Amount", "Net Amount", "Line Amount", "Net Value", "Invoice Amount", "Total"], "de": ["Nettobetrag", "Betrag"],
            "fr": ["Montant HT", "Montant"], "es": ["Importe neto", "Importe"]},
    "gross": {"en": ["Gross Amount", "Amount incl. VAT", "Total incl. Tax"], "de": ["Bruttobetrag"], "fr": ["Montant TTC"],
              "es": ["Importe IVA incluido"]},
    "taxrate": {"en": ["VAT Rate", "Tax Rate", "VAT %", "Tax %"], "de": ["MwSt", "MwSt-Satz"], "fr": ["Taux TVA", "TVA"],
                "es": ["Tipo IVA", "IVA"]},
    "taxamt": {"en": ["VAT Amount", "Tax Amount"], "de": ["Steuerbetrag"], "fr": ["Montant TVA"], "es": ["Importe IVA"]},
    "cur": {"en": ["Currency", "Curr", "CCY", "Currency Code"], "de": ["Währung"], "fr": ["Devise"], "es": ["Moneda"]},
    "rate": {"en": ["FX Rate", "Exchange Rate", "Currency Rate"], "de": ["Wechselkurs"], "fr": ["Taux de change"],
             "es": ["Tipo de cambio"]},
    "date": {"en": ["Invoice Date", "Document Date", "Doc Date", "Date", "Bill Date"], "de": ["Rechnungsdatum", "Belegdatum"],
             "fr": ["Date facture", "Date de facture"], "es": ["Fecha factura", "Fecha de factura"]},
    "post": {"en": ["Posting Date", "GL Date", "Entry Date"], "de": ["Buchungsdatum"], "fr": ["Date de comptabilisation"],
             "es": ["Fecha de contabilización"]},
    "status": {"en": ["Status", "Invoice Status", "Approval Status", "Doc Status"], "de": ["Status", "Belegstatus"],
               "fr": ["Statut"], "es": ["Estado"]},
    "code": {"en": ["Item Code", "Material Number", "SKU", "Part No", "Product Code"], "de": ["Artikelnummer", "Materialnummer"],
             "fr": ["Référence article"], "es": ["Código de artículo"]},
    "qty": {"en": ["Quantity", "Qty"], "de": ["Menge"], "fr": ["Quantité"], "es": ["Cantidad"]},
}
SUFFIX = [("Ltd", "Limited", "LTD.", "Ltd"), ("Inc", "Incorporated", "INC.", "Inc"), ("LLC", "L.L.C.", "llc", "LLC"),
          ("GmbH", "GMBH", "Gmbh", "GmbH"), ("(Pty) Ltd", "Pty Ltd", "PTY LTD", "(PTY) LTD")]
SYL = ["kel", "mar", "vex", "dun", "pra", "yar", "wex", "lor", "pel", "oak", "tor", "bren", "sil", "hart", "gal", "nor"]
KIND = ["Industrial", "Supplies", "Services", "Trading", "Engineering", "Logistics", "Solutions", "Medical", "Foods", "Systems"]


NEG_STYLE = ["minus"]
SYMBOL = [""]


def fmt_num(x, style):
    neg = x < 0
    a = abs(x)
    if style == "plain":
        s = f"{a:.2f}"
    elif style == "us":
        s = f"{a:,.2f}"
    elif style == "eu":
        s = f"{a:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    else:  # space
        s = f"{a:,.2f}".replace(",", " ").replace(".", ",")
    s = SYMBOL[0] + s
    if not neg:
        return s
    return {"minus": "-" + s, "paren": "(" + s + ")", "trail": s + "-"}[NEG_STYLE[0]]


def make(seed: int, work: Path):
    rnd = random.Random(seed)
    lang = rnd.choice(["en", "en", "en", "de", "fr", "es"])
    ind = rnd.choice(list(INDUSTRIES))
    leaves = INDUSTRIES[ind]
    four = rnd.random() < 0.5
    feats = {"lang": lang, "industry": ind}

    def h(role):
        return rnd.choice(H[role][lang] if rnd.random() < 0.85 else H[role]["en"])

    gross = rnd.random() < 0.35
    multi = rnd.random() < 0.6
    has_rate = multi and rnd.random() < 0.4
    st_mode = rnd.choices(["col", "flags", "none"], [0.75, 0.1, 0.15])[0]
    st_vocab = STATUS[lang if lang != "en" else rnd.choice(["en_a", "en_b"])] if rnd.random() < 0.85 else STATUS[rnd.choice(list(STATUS))]
    num_style = rnd.choice(["plain", "us", "eu", "space"] if lang != "en" else ["plain", "us", "plain"])
    date_style = rnd.choice(["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y", "%d-%b-%Y"] if lang == "en" else ["%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y"])
    taxfmt = rnd.choice(["15", "0.15", "15%"])
    NEG_STYLE[0] = rnd.choice(["minus", "minus", "paren", "trail"])
    SYMBOL[0] = "$" if (not multi and rnd.random() < 0.3) else ""
    cur_style = rnd.choices(["iso", "names", "lower"], [6, 2, 1])[0] if multi else "iso"
    CUR_NAME = {"USD": "US Dollar", "EUR": "Euro", "GBP": "Pound Sterling", "INR": "Indian Rupee", "ZAR": "Rand", "CAD": "Canadian Dollar",
                "AUD": "Australian Dollar", "MXN": "Mexican Peso", "BRL": "Brazilian Real", "JPY": "Japanese Yen", "CHF": "Swiss Franc"}
    stamp = rnd.random() < 0.15
    id_style = rnd.choice(["R", "R", "num", "zeros"])
    name_less = rnd.random() < 0.1
    odd_tree = rnd.random() < 0.2
    status_noise = rnd.random() < 0.2
    feats.update({"neg": NEG_STYLE[0], "symbol": SYMBOL[0], "cur_style": cur_style, "stamp": stamp, "id_style": id_style,
                  "name_less": name_less, "odd_tree": odd_tree, "status_noise": status_noise})
    with_taxamt = gross and rnd.random() < 0.5
    gross_header_neutral = gross and with_taxamt and rnd.random() < 0.3
    with_supid = rnd.random() < 0.6
    cols = {"id": h("id"), "sup": h("sup"), "desc": h("desc"), "date": h("date"),
            "amt": (rnd.choice(["Amount", "Invoice Amount"]) if gross_header_neutral else h("gross")) if gross else h("net")}
    feats.update({"gross": gross, "neutral_gross_header": gross_header_neutral, "multi_ccy": multi, "rate_col": has_rate, "status": st_mode,
                  "num": num_style, "date": date_style, "four_levels": four})
    if rnd.random() < 0.8: cols["inv"] = h("inv")
    if with_supid: cols["supid"] = h("supid")
    if gross and (not with_taxamt or rnd.random() < 0.6): cols["taxrate"] = h("taxrate")
    if with_taxamt: cols["taxamt"] = h("taxamt")
    if multi: cols["cur"] = h("cur")
    if has_rate: cols["rate"] = h("rate")
    if rnd.random() < 0.5: cols["post"] = h("post")
    if st_mode == "col": cols["status"] = h("status")
    if rnd.random() < 0.5: cols["code"] = h("code")
    if rnd.random() < 0.5: cols["qty"] = h("qty")

    # suppliers: entity -> name variants
    ents = []
    firsts = rnd.sample(SYL, rnd.randint(8, 16))        # distinct first syllables: no two firms one letter apart
    for e, first in enumerate(firsts):
        base = (first + rnd.choice(SYL)).title() + " " + rnd.choice(KIND)
        suf = rnd.choice(SUFFIX)
        variants = {f"{base} {suf[0]}", f"{base} {suf[1]}", f"{base.upper()} {suf[2]}", f"{base}, {suf[3]}", base}
        ents.append((f"V{1000 + e}", sorted(variants)))
    ccys = ["USD"] + (rnd.sample(["EUR", "GBP", "INR", "ZAR", "CAD", "AUD", "MXN", "BRL", "JPY", "CHF"], rnd.randint(1, 4)) if multi else [])
    n = rnd.randint(250, 500)
    rows, truth = [], []
    d0 = date(2024, 1, 1)
    for i in range(n):
        leaf = rnd.choice(leaves)
        text = rnd.choice(leaf[3]) + (f" {rnd.randint(1, 40)}" if rnd.random() < 0.3 else "")
        ent, variants = rnd.choice(ents)
        np_line = rnd.random() < 0.04
        if np_line:
            text = rnd.choice(NP_TEXTS)
        d = d0 + timedelta(days=rnd.randint(0, 540))
        cur = rnd.choice(ccys) if rnd.random() < 0.6 else "USD"
        usd = round(rnd.lognormvariate(6, 1.3), 2) * (-1 if rnd.random() < 0.02 else 1)
        table = 1.0 if cur == "USD" else RATE[(cur, d.strftime("%Y-%m"))]
        rate = table * (1 + rnd.uniform(-0.02, 0.02)) if (has_rate and cur != "USD") else table
        net = round(usd * rate, 2)
        usd_true = round(net / rate, 2)
        vat = rnd.choice([0.15, 0.15, 0.2, 0.0]) if gross else 0.0
        tax = round(net * vat, 2)
        st, st_code = (rnd.choices(list(st_vocab.items()), [6, 5] + [1] * (len(st_vocab) - 2))[0] if st_mode == "col" else ("", ""))
        approved, cancelled = "Y", "N"
        if st_mode == "flags":
            approved = "N" if rnd.random() < 0.08 else "Y"
            cancelled = "Y" if rnd.random() < 0.05 else "N"
            st_code = "CANCELLED" if cancelled == "Y" else ("NOT_APPROVED" if approved == "N" else "")
        code = st_code or ("NON_PROCUREMENT" if np_line else "")
        rid = {"R": f"R{seed:03d}-{i:05d}", "num": str(500000 + i), "zeros": f"{i + 1:08d}"}[id_style]
        if rnd.random() < 0.03 and not np_line:
            text = ""                                     # a blank description
        dtxt = d.strftime(date_style) + ((" 00:00:00" if date_style == "%Y-%m-%d" else " 00:00") if stamp else "")
        r = {cols["id"]: rid, cols["sup"]: rnd.choice(variants), cols["desc"]: text,
             cols["date"]: dtxt, cols["amt"]: fmt_num(net + tax if gross else net, num_style)}
        if "inv" in cols: r[cols["inv"]] = f"INV{seed}{i // 2:05d}"
        if "supid" in cols: r[cols["supid"]] = ent
        if "taxrate" in cols: r[cols["taxrate"]] = {"15": f"{vat * 100:g}", "0.15": f"{vat:g}", "15%": f"{vat * 100:g}%"}[taxfmt]
        if "taxamt" in cols: r[cols["taxamt"]] = fmt_num(tax, num_style)
        if "cur" in cols: r[cols["cur"]] = {"iso": cur, "names": CUR_NAME[cur], "lower": cur.lower()}[cur_style]
        if "rate" in cols: r[cols["rate"]] = f"{rate:.6f}" if cur != "USD" else rnd.choice(["1", "1.0", ""])
        if "post" in cols: r[cols["post"]] = (d + timedelta(days=rnd.randint(0, 20))).strftime(date_style)
        if "status" in cols: r[cols["status"]] = (rnd.choice([st.upper(), f" {st} ", st.lower()]) if status_noise else st)
        if st_mode == "flags": r["Approved"], r["Cancelled Flag"] = approved, cancelled
        if "code" in cols: r[cols["code"]] = f"IT{__import__('zlib').crc32(text.encode()) % 9000 + 1000}" if not np_line else ""
        if "qty" in cols: r[cols["qty"]] = str(rnd.randint(1, 50))
        if rnd.random() < 0.4: r["Cost Centre"] = f"CC{rnd.randint(100, 120)}"
        rows.append(r)
        truth.append({"text": text, "record_id": r[cols["id"]], "code": code, "usd": usd_true if not code else 0.0, "ent": ent,
                      "path": leaf[:3] + ((leaf[2] + " - General",) if four else ()), "np": np_line})
    # exact duplicates with a new id
    for j in rnd.sample(range(n), max(1, n // 40)):
        r = dict(rows[j]); r[cols["id"]] = {"R": f"R{seed:03d}-D{j:05d}", "num": str(900000 + j), "zeros": f"9{j + 1:07d}"}[id_style]
        rows.append(r)
        t = dict(truth[j]); t["record_id"] = r[cols["id"]]; t["code"] = t["code"] or "DUPLICATE"; t["usd"] = 0.0
        truth.append(t)
    df = pd.DataFrame(rows).fillna("")
    if name_less and "supid" in cols:
        df = df.drop(columns=[cols["sup"]])               # vendor ids only, as in company B
    else:
        feats["name_less"] = False
    order = list(df.columns); rnd.shuffle(order); df = df[order]

    folder = work / f"c{seed:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    kind = rnd.choices(["csv", "csv;", "tsv", "xlsx", "cp1252", "utf16", "titled"], [4, 2, 1, 2, 1, 0.5, 1])[0]
    feats["file"] = kind
    extras = rnd.random() < 0.25
    feats["blank_and_total_rows"] = extras
    name = rnd.choice(["invoice_lines", "ap_export", "C_lines", "purchases_2025", "rechnungen", "factures"])
    if kind == "xlsx":
        lines_file = f"{name}.xlsx"
        df.to_excel(folder / lines_file, index=False)
    else:
        sep = {"csv": ",", "csv;": ";", "tsv": "\t", "cp1252": ";", "utf16": "\t", "titled": ","}[kind]
        body = df.to_csv(index=False, sep=sep)
        if kind == "titled":
            body = f"AP export{sep * (len(df.columns) - 1)}\n{sep * (len(df.columns) - 1)}\n" + body
        if extras:
            blank = sep * (len(df.columns) - 1) + "\n"
            tot = ["Total"] + [""] * (len(df.columns) - 1)
            body += blank + blank + sep.join(tot) + "\n"
        lines_file = f"{name}.{'txt' if kind in ('tsv', 'utf16') else 'csv'}"
        enc = {"cp1252": "cp1252", "utf16": "utf-16"}.get(kind, "utf-8")
        (folder / lines_file).write_bytes(body.encode(enc, errors="replace"))
    # tree
    paths = sorted({t["path"] for t in truth})
    all_leaves = sorted({(x[0], x[1], x[2]) + ((x[2] + " - General",) if four else ()) for x in leaves})
    if odd_tree:                                           # names with a trailing space or a double space
        fix = lambda nm: nm.replace(" ", "  ", 1) if " " in nm else nm + " "
        all_leaves = [tuple(fix(v) if k == len(x) - 1 else v for k, v in enumerate(x)) for x in all_leaves]
    tfmt = rnd.choices(["wide", "wide_alt", "codes", "parent_child", "path"], [4, 2, 1.5, 1.5, 1])[0]
    feats["tree"] = tfmt
    depth = 4 if four else 3
    if tfmt in ("wide", "wide_alt"):
        names = ([f"category_l{k}" for k in range(1, depth + 1)] if tfmt == "wide" else
                 rnd.choice([[f"Level {k}" for k in range(1, depth + 1)], [f"L{k}" for k in range(1, depth + 1)],
                             ["Segment", "Family", "Class", "Commodity"][:depth]]))
        tree = pd.DataFrame(all_leaves, columns=names)
    elif tfmt == "codes":
        recs = []
        for x in all_leaves:
            rec = {}
            for k in range(depth):
                rec[f"L{k + 1} Code"] = str(__import__('zlib').crc32(" > ".join(x[:k + 1]).encode()) % 90000 + 10000)
                rec[f"L{k + 1} Name"] = x[k]
            recs.append(rec)
        tree = pd.DataFrame(recs)
    elif tfmt == "parent_child":
        ids, recs = {}, []
        for x in all_leaves:
            for k in range(1, depth + 1):
                if x[:k] not in ids:
                    ids[x[:k]] = f"N{len(ids) + 1:03d}"
                    recs.append({"Category ID": ids[x[:k]], "Parent ID": ids.get(x[:k - 1], ""), "Category Name": x[k - 1]})
        tree = pd.DataFrame(recs)
    else:
        tree = pd.DataFrame({"Category Path": [" > ".join(x) for x in all_leaves]})
    tree_file = rnd.choice(["taxonomy.csv", "category_tree.csv", "C_tree.csv"])
    tree.to_csv(folder / tree_file, index=False)
    if rnd.random() < 0.2:
        (folder / "README.txt").write_text("Blind test files for company C.\n")
    # a path column is split on its separator and each part is trimmed
    leaf_paths = {tuple(v.strip() for v in x) for x in all_leaves} if tfmt == "path" else set(all_leaves)
    return folder, lines_file, tree_file, pd.DataFrame(truth), feats, leaf_paths
