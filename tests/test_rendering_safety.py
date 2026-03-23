from core.rendering import markdown_to_safe_html


def test_markdown_to_safe_html_strips_script_and_file_links():
    rendered = markdown_to_safe_html(
        '[safe](https://example.com) <script>alert(1)</script> [bad](file:///etc/passwd)'
    )
    assert "<script" not in rendered
    assert "file:///" not in rendered
    assert 'href="https://example.com"' in rendered
