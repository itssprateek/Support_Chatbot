import requests

def fetch_markdown(url: str) -> str:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.text

def parse_markdown_table(md: str):
    """
    Parses a standard markdown table like:
    | Intent | User Question | Bot Response |
    | --- | --- | --- |
    Returns: list of dicts
    """
    lines = [ln.strip() for ln in md.splitlines() if ln.strip()]

    # Keep only table rows that start with '|'
    table_lines = [ln for ln in lines if ln.startswith("|")]

    if len(table_lines) < 3:
        raise ValueError("No markdown table found in the file.")

    header = [h.strip() for h in table_lines[0].strip("|").split("|")]
    # skip separator row table_lines[1]
    rows = table_lines[2:]

    data = []
    for row in rows:
        cols = [c.strip() for c in row.strip("|").split("|")]
        if len(cols) != len(header):
            continue
        item = dict(zip(header, cols))
        data.append(item)

    return data