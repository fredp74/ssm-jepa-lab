import re
import unicodedata

from csv_environment import render

def transform(rows, operation):
    values = [list(row) for row in rows]

    if operation == "sort_rows":
        return sorted(values)
    if operation == "remove_empty_rows":
        return [row for row in values if any(value != "" for value in row)]
    if operation == "deduplicate_rows":
        result = []
        for row in values:
            if row not in result:
                result.append(row)
        return result
    if operation == "drop_empty_columns":
        keep = [
            i for i in range(len(values[0]))
            if any(row[i] != "" for row in values)
        ]
        return [[row[i] for i in keep] for row in values] if keep else []

    def cell(value):
        if operation == "identity":
            return value
        if operation == "normalize_leading_zeros":
            return str(int(value)) if (
                value.isascii() and value.isdigit()
                and len(value) > 1 and value.startswith("0")
            ) else value
        if operation == "unicode_NFC":
            return unicodedata.normalize("NFC", value)
        if operation == "casefold":
            return value.casefold()
        if operation == "rstrip":
            return value.rstrip()
        if operation == "collapse_spaces":
            return re.sub(r" {2,}", " ", value)
        if operation == "decimal_comma":
            return value.replace(".", ",") if re.fullmatch(
                r"[+-]?[0-9]+\.[0-9]+", value
            ) else value
        if operation == "boolean_normalization":
            return {
                "true": "true", "false": "false",
                "yes": "true", "no": "false",
            }.get(value.lower(), value)
        if operation == "truncate_16":
            return value[:16]
        raise ValueError(f"Operation inconnue : {operation}")

    return [[cell(value) for value in row] for row in values]

def simulate(rows, config, operations):
    values = [list(row) for row in rows]
    for operation in operations:
        if not values:
            break
        values = transform(values, operation)
    return render(values, config) if values else ""
