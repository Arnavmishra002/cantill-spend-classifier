"""spendcube: config-driven spend cube builder.

Pipeline steps: ingest, clean, spend rules, cross-source matching, de-duplication,
classification, export and validation. Everything company-specific lives in a
YAML config; nothing in this package knows about a particular company, column
name or category name.
"""

__version__ = "1.0.3"
