"""Validate Oracle data without importing SDKs, credentials or model clients."""
from __future__ import annotations
import csv
import hashlib
from pathlib import Path

CATALOG_RELATIVE = 'environment/ObjectList.csv'


class ObjectCatalogError(ValueError):
    pass


def load_object_catalog(path: Path) -> dict[str, dict[str, str]]:
    path = Path(path)
    try:
        with path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            required = {'object_name', 'read', 'simplified_name', 'label'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError('required columns: ' + ', '.join(sorted(required)))
            mapping = {}
            for line, row in enumerate(reader, 2):
                key = (row.get('object_name') or '').strip()
                values = {field: (row.get(field) or '').strip() for field in required - {'object_name'}}
                if not key or key in mapping:
                    raise ValueError(f'line {line}: empty or duplicate object_name')
                if values['read'] not in ('0', '1') or (values['read'] == '1' and not values['simplified_name']):
                    raise ValueError(f'line {line}: invalid read flag or missing readable name')
                mapping[key] = values
            if not mapping or not any(row['read'] == '1' for row in mapping.values()):
                raise ValueError('catalog has no readable objects')
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        raise ObjectCatalogError(f'Invalid Oracle resource {path}: {exc}') from exc
    return mapping


def validate_oracle_resources(project: Path | None = None) -> dict:
    project = Path(project) if project is not None else Path(__file__).resolve().parents[1]
    path = project / CATALOG_RELATIVE
    rows = load_object_catalog(path)
    return dict(file=CATALOG_RELATIVE, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                rows=len(rows), readable_rows=sum(r['read'] == '1' for r in rows.values()))
