"""Label canonicalisation for multi-view semantic voting (data_pipeline / hesai, non-residential scenes).

GPT's per-frame inventories name the same physical class with context variants ("library bookshelf", "library
bookcase", "bookcase"; "interior floor", "library floor"). Voting on raw strings splits the votes so the 0.15 margin
rule turns many instances into 'unknown'. Votes are therefore cast on the canonical class; the most-voted raw name is
kept per instance as `fine_label`. The table only merges explicit synonyms and pure context prefixes; it never merges
distinct functional classes (e.g. 'service counter' stays separate from 'table')."""
import re
from .discovery import ALIASES

EXTRA = {
    'bookcase': 'bookshelf', 'book shelf': 'bookshelf', 'bookshelves': 'bookshelf', 'book cabinet': 'bookshelf',
    'books': 'book', 'reading table': 'table', 'reading desk': 'desk', 'study table': 'table',
    'reading chair': 'chair', 'armless chair': 'chair', 'wall-mounted television': 'television',
    'wall-mounted tv': 'television', 'wall-mounted monitor': 'display screen', 'wall-mounted display screen': 'display screen',
    'monitor screen': 'display screen', 'display monitor': 'display screen', 'digital display screen': 'display screen',
    'fluorescent ceiling light': 'ceiling lamp', 'ceiling light panel': 'ceiling lamp', 'led ceiling light': 'ceiling lamp',
    'ceiling panel light': 'ceiling lamp', 'interior wall surface': 'wall', 'wall surface': 'wall', 'glass wall': 'glass wall',
    'storefront window': 'window', 'glass storefront window': 'window', 'glass window panel': 'window',
    'glass entrance door': 'glass door', 'entrance door': 'door', 'plastic stool': 'stool',
    'exterior building': 'building', 'outdoor pavement': 'pavement', 'outdoor railing': 'railing',
    'literature display rack': 'brochure display rack', 'pamphlet rack': 'brochure display rack',
}
PREFIX = re.compile(r'^(?:library|wooden|wood|interior|indoor|white|metal|plastic|fabric|grey|gray|black|brown) ')
SURFACE = {'floor', 'wall', 'ceiling'}


def canonical(label):
    label = re.sub(r'\s+', ' ', str(label).strip().lower().replace('_', ' '))
    label = ALIASES.get(label, label); label = EXTRA.get(label, label)
    for _ in range(2):                          # 'library wooden table' -> 'table'
        stripped = PREFIX.sub('', label)
        if stripped == label: break
        stripped = ALIASES.get(stripped, stripped); label = EXTRA.get(stripped, stripped)
    for s in SURFACE:
        if label.endswith(' ' + s) and len(label.split()) <= 3 and label.split()[0] in ('tiled', 'tile', 'concrete', 'painted', 'room', 'main'):
            return s
    return label
