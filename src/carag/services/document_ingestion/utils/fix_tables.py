import re


def is_row_empty(row_string: str) -> bool:
    cells = row_string.strip().strip("|").split("|")
    return all(not cell.strip() or cell.strip() == " " for cell in cells)


def _process_table_match(match: re.Match[str]) -> str:
    table_block: str = match.group(0)
    lines = table_block.strip().split("\n")

    sep_index = -1

    for i, line in enumerate(lines):
        if "---" in line and "|" in line:
            sep_index = i
            break

    if sep_index < 1 or sep_index + 1 >= len(lines):
        return table_block

    header_line = lines[sep_index - 1]
    first_data_line = lines[sep_index + 1]

    if is_row_empty(header_line) and not is_row_empty(first_data_line):
        lines[sep_index - 1] = first_data_line

        del lines[sep_index + 1]
        return "\n".join(lines) + "\n"
    return table_block


def fix_markdown_tables(md_text: str) -> str:
    table_pattern = re.compile(r"^(?:\|.*?\n)+", re.MULTILINE)
    return table_pattern.sub(_process_table_match, md_text)


if __name__ == "__main__":
    from markdownify import markdownify as md

    with open("test.md", "r") as f:
        content = f.read()

    md_doc = md(content)
    fx_doc = fix_markdown_tables(md_doc)
    with open("fixed_tables.md", "w") as f:
        f.write(fx_doc)
