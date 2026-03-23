import html

import bleach
import markdown

ALLOWED_TAGS = [
    "a",
    "blockquote",
    "br",
    "code",
    "em",
    "hr",
    "li",
    "ol",
    "p",
    "pre",
    "strong",
    "ul",
]
ALLOWED_ATTRIBUTES = {"a": ["href", "title", "target", "rel"]}
ALLOWED_PROTOCOLS = ["http", "https", "mailto"]


def escape_text(text: str) -> str:
    return html.escape(text or "", quote=True)


def markdown_to_safe_html(text: str) -> str:
    raw_html = markdown.markdown(
        text or "",
        extensions=["extra", "fenced_code", "nl2br", "sane_lists"],
        output_format="html5",
    )
    return bleach.clean(
        raw_html,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        protocols=ALLOWED_PROTOCOLS,
        strip=True,
    )
