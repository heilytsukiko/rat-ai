"""Verified public references for human labeling; no automatic legal classification."""
import json
from pathlib import Path


def load_rules():
    return json.loads((Path(__file__).parent / 'data/rules-rk.json').read_text(encoding='utf-8'))


def rule_source(rule_id):
    catalog = load_rules()
    rule = next((r for r in catalog['rules'] if r['id'] == rule_id), None)
    if rule is None:
        raise ValueError('Unknown verified rule ID')
    return {'title': catalog['document']['title'], 'section': rule['section'], 'url': catalog['document']['url']}
