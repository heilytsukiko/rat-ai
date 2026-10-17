from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rat_ai.rules import load_rules, rule_source


class RuleCatalogTests(unittest.TestCase):
    def test_verified_ids_have_specific_sections_and_official_sources(self):
        catalog = load_rules()
        rules = catalog['rules']
        self.assertEqual(len(rules), len({rule['id'] for rule in rules}))
        for rule in rules:
            source = rule_source(rule['id'])
            self.assertIn('пункт', source['section'])
            self.assertEqual(source['url'], 'https://adilet.zan.kz/rus/docs/V1900019680')
            self.assertTrue(rule['required_observation'])
            self.assertTrue(rule['caution'])
        with self.assertRaises(ValueError):
            rule_source('invented-rule')


if __name__ == '__main__':
    unittest.main()
