
# -*- coding: utf-8 -*-
"""Check that every table/figure reference resolves and no caption is orphaned.

Usage: python scripts/check_caption_xref.py
Exit code 0 when clean, 1 otherwise.
"""
import io, re, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(ROOT, 'drafts', 'paper2_journal_en.md')
ESM = os.path.join(ROOT, 'drafts', 'paper2_esm_en.md')


def load(p):
    return io.open(p, encoding='utf-8', newline='').read().split('\n')


def captions(lines):
    caps = []
    for i, ln in enumerate(lines, 1):
        m = re.match(r'\*\*Table (S?)(\d+)\.\*\*', ln.strip())
        if m:
            caps.append(('T', m.group(1) + m.group(2), i))
        m = re.match(r'\*\*Figure (S?)(\d+)\.\*\*', ln.strip())
        if m:
            caps.append(('F', m.group(1) + m.group(2), i))
    return caps


def refs(lines):
    out = []
    for i, ln in enumerate(lines, 1):
        for m in re.finditer(r'\bTables?\s*(S?)(\d+)', ln):
            out.append(('T', m.group(1) + m.group(2), i))
        for m in re.finditer(r'\b(?:Figs?\.?|Figures?)\s*(S?)(\d+)', ln):
            out.append(('F', m.group(1) + m.group(2), i))
    return out


def main():
    files = {}
    for p, tag in ((MAIN, 'main'), (ESM, 'esm')):
        L = load(p)
        files[tag] = (captions(L), refs(L))

    problems = []
    for tag, (caps, rs) in files.items():
        other = 'esm' if tag == 'main' else 'main'
        have = {(c[0], c[1]) for c in caps}
        have_other = {(c[0], c[1]) for c in files[other][0]}
        for kind, num, ln in rs:
            if (kind, num) not in have and (kind, num) not in have_other:
                problems.append('%s:%d: %s %s has no caption in either file' % (tag, ln, kind, num))
        used = {(k, n) for k, n, _ in rs}
        for kind, num, ln in caps:
            if (kind, num) not in used:
                problems.append('%s:%d: caption %s %s is never referenced' % (tag, ln, kind, num))

    # Targeted heuristic: "Table S1" is the related-work matrix, "Table S11" the artifact map.
    # Flag any S1 mention that sits next to artifact/mapping vocabulary.
    warns = []
    vocab = ('artifact', 'mapping', 'maps each claim', 'regenerat', 'scripts that', 'produce Table')
    for tag, path in (('main', MAIN), ('esm', ESM)):
        for i, ln in enumerate(load(path), 1):
            for m in re.finditer(r'Table S1(?![0-9])', ln):
                ctx = ln[max(0, m.start() - 90):m.end() + 90].lower()
                if any(v in ctx for v in vocab):
                    warns.append('%s:%d: Table S1 looks like the artifact map (should be S11): ...%s...'
                                 % (tag, i, ln[max(0, m.start() - 60):m.end() + 40]))
    for w in warns:
        print('WARN', w)

    for tag in ('main', 'esm'):
        caps, rs = files[tag]
        print('%s captions T: %s' % (tag, [c[1] for c in caps if c[0] == 'T']))
        print('%s captions F: %s' % (tag, [c[1] for c in caps if c[0] == 'F']))
        print('%s refs     T: %s' % (tag, sorted({n for k, n, _ in rs if k == 'T'},
                                                 key=lambda x: (x.startswith('S'), x))))
        print('%s refs     F: %s' % (tag, sorted({n for k, n, _ in rs if k == 'F'},
                                                 key=lambda x: (x.startswith('S'), x))))
    print('problems:', len(problems))
    print('warnings:', len(warns))
    for p in problems:
        print('  ', p)
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
