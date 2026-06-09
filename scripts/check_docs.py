from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        for key in ("href", "src"):
            if key in values:
                self.links.append(values[key])


root = Path("docs")
parser = LinkParser()
parser.feed((root / "index.html").read_text(encoding="utf-8"))
missing = []
for link in parser.links:
    parsed = urlparse(link)
    if parsed.scheme or link.startswith("#"):
        continue
    target = root / parsed.path
    if not target.exists():
        missing.append(link)
if missing:
    raise SystemExit("Missing docs assets: " + ", ".join(missing))
print(f"Checked {len(parser.links)} documentation links.")

