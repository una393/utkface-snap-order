
from __future__ import annotations
import os
from lxml import etree
import latex2mathml.converter as l2m
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

XSL_PATH = r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"
_XSLT = etree.XSLT(etree.parse(XSL_PATH))


def latex_to_omml_xml(latex: str) -> str:
    mathml = l2m.convert(latex)
    dom = etree.fromstring(mathml.encode("utf-8"))
    out = _XSLT(dom)
    s = etree.tostring(out.getroot(), encoding="unicode")
    if "<m:e/>" in s:
        raise ValueError("empty OMML argument (Word draws dotted placeholder): " + latex)
    return s


def omml_element(latex: str):
    s = latex_to_omml_xml(latex)
    root = etree.fromstring(s.encode("utf-8"))
    tag = etree.QName(root).localname
    ns = etree.QName(root).namespace
    el = parse_xml('<m:%s %s/>' % (tag, nsdecls("m", "w")))
    for child in list(root):
        el.append(child)
    for k, v in root.attrib.items():
        el.set(k, v)
    return el


if __name__ == "__main__":
    print(etree.tostring(omml_element(r"s_i(w)=\frac{\max_j |w_{ij}|}{q_b}"), encoding="unicode")[:400])
