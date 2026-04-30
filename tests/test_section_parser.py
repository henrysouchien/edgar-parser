from edgar_parser import section_parser


def _words(label: str, count: int = 12) -> str:
    return " ".join([label] * count)


def test_toc_anchors_fill_10q_sections_with_short_body_headings():
    html = f"""
    <html><body>
      <table>
        <tr><td><a href="#mda">Management's Discussion and Analysis of Financial Condition and Results of Operations</a></td><td>4</td></tr>
        <tr><td><a href="#fs">Financial Statements</a></td><td>90</td></tr>
        <tr><td><a href="#market">Item 3. Quantitative and Qualitative Disclosures About Market Risk</a></td><td>130</td></tr>
        <tr><td><a href="#controls">Item 4. Controls and Procedures</a></td><td>131</td></tr>
        <tr><td><a href="#legal">Item 1. Legal Proceedings</a></td><td>132</td></tr>
        <tr><td><a href="#risk">Item 1A. Risk Factors</a></td><td>133</td></tr>
      </table>
      <div id="mda">Management's Discussion and Analysis</div><p>{_words("mda")}</p>
      <div id="fs">Financial Statements</div><p>{_words("financial")}</p>
      <div id="market">Item 3. Quantitative and Qualitative Disclosures About Market Risk</div><p>{_words("market")}</p>
      <div id="controls">Item 4. Controls and Procedures</div><p>{_words("controls")}</p>
      <div id="legal">Item 1. Legal Proceedings</div><p>{_words("legal")}</p>
      <div id="risk">Item 1A. Risk Factors</div><p>{_words("risk")}</p>
    </body></html>
    """

    parsed = section_parser.parse_filing_sections(html, "10-Q")

    assert parsed["sections_found"] == [
        "part1_item1",
        "part1_item2",
        "part1_item3",
        "part1_item4",
        "part2_item1",
        "part2_item1a",
    ]
    assert parsed["sections_missing"] == []


def test_toc_anchors_fill_10k_sections_with_short_body_headings():
    html = f"""
    <html><body>
      <table>
        <tr><td><a href="#business">Business</a></td><td>1</td></tr>
        <tr><td><a href="#risk">Risk Factors</a></td><td>11</td></tr>
        <tr><td><a href="#staff">Unresolved Staff Comments</a></td><td>23</td></tr>
        <tr><td><a href="#properties">Properties</a></td><td>24</td></tr>
        <tr><td><a href="#legal">Legal Proceedings</a></td><td>25</td></tr>
        <tr><td><a href="#mda">Management's Discussion and Analysis of Financial Condition and Results of Operations</a></td><td>40</td></tr>
        <tr><td><a href="#market">Quantitative and Qualitative Disclosures About Market Risk</a></td><td>55</td></tr>
        <tr><td><a href="#financials">Financial Statements and Supplementary Data</a></td><td>60</td></tr>
      </table>
      <div id="business">Business</div><p>{_words("business")}</p>
      <div id="risk">Risk Factors</div><p>{_words("risk")}</p>
      <div id="staff">Unresolved Staff Comments</div><p>{_words("staff")}</p>
      <div id="properties">Properties</div><p>{_words("properties")}</p>
      <div id="legal">Legal Proceedings</div><p>{_words("legal")}</p>
      <div id="mda">Management's Discussion and Analysis</div><p>{_words("mda")}</p>
      <div id="market">Quantitative and Qualitative Disclosures About Market Risk</div><p>{_words("market")}</p>
      <div id="financials">Financial Statements and Supplementary Data</div><p>{_words("financials")}</p>
    </body></html>
    """

    parsed = section_parser.parse_filing_sections(html, "10-K")

    assert parsed["sections_found"] == [
        "item_1",
        "item_1a",
        "item_1b",
        "item_2",
        "item_3",
        "item_7",
        "item_7a",
        "item_8",
    ]
    assert parsed["sections_missing"] == []


def test_body_references_do_not_become_section_headers():
    html = f"""
    <html><body>
      <div>Item 3. Quantitative and Qualitative Disclosures About Market Risk</div>
      <p>{_words("market")}</p>
      <div>The other risks and uncertainties detailed in Part I, Item 1A: Risk Factors in the annual report.</div>
      <div>Item 4. Controls and Procedures</div>
      <p>{_words("controls")}</p>
    </body></html>
    """

    parsed = section_parser.parse_filing_sections(html, "10-Q")

    assert parsed["sections_found"] == ["part1_item3", "part1_item4"]
    assert "part2_item1a" in parsed["sections_missing"]
