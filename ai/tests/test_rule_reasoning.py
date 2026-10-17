import itertools
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from prepare_rule_reasoning import build
from train_rule_reasoning import score_predictions
from rat_ai.evidence import CONDITIONS, assess, parse_facts


class RuleReasoningTests(unittest.TestCase):
    def test_unknown_cannot_be_promoted(self):
        for rule in CONDITIONS:
            for facts in itertools.product(('yes','no','unknown'),repeat=3):
                result = assess(rule,list(facts))
                self.assertEqual(result['status']=='possible_issue', facts==('yes','yes','yes'))
                self.assertFalse(result['legal_verdict'])
                self.assertFalse(result['human_verified'])
                self.assertTrue(result['requires_review'])

    def test_invalid_model_output_fails_closed(self):
        for text in ('{}','{"facts":["yes","yes"]}',
                     '{"facts":["yes","yes","definitely"]}',
                     '{"facts":["yes","yes","yes"],"verdict":true}',
                     '```json\n{"facts":["yes","yes","yes"]}\n```'):
            with self.assertRaises(ValueError):
                parse_facts(text)

    def test_curriculum_and_challenge_are_disjoint(self):
        rows = build()
        train = [r for r in rows if r['split']=='train']
        test = [r for r in rows if r['split']=='challenge']
        self.assertEqual((len(train),len(test)), (324,36))
        for key in ('id','family','observation'):
            self.assertFalse({r[key] for r in train} & {r[key] for r in test})
        self.assertTrue(all(r['synthetic'] for r in rows))
        self.assertEqual({r['expected_status'] for r in test},
                         {'possible_issue','not_supported','insufficient_evidence'})

    def test_benchmark_penalizes_false_accusation_and_invalid_outputs(self):
        rows = [r for r in build() if r['split']=='challenge']
        correct = score_predictions(rows,[json.dumps({'facts':r['facts']}) for r in rows])
        self.assertEqual(correct['exact_facts_accuracy'],1)
        bad = score_predictions(rows,['{"facts":["yes","yes","yes"]}']*len(rows))
        self.assertGreater(bad['unsupported_possible_issues'],0)
        invalid = score_predictions(rows,['wrong']*len(rows))
        self.assertEqual(invalid['status_accuracy'],0)
        self.assertEqual(invalid['invalid_outputs'],len(rows))

    def test_medical_exception_and_incomplete_pack_are_not_positive(self):
        rows = {r['id']:r for r in build()}
        for key in ('NBK231-P12-challenge-1','NBK231-P12-challenge-5','NBK231-P32-challenge-1'):
            self.assertEqual(rows[key]['expected_status'],'not_supported')


if __name__ == '__main__':
    unittest.main()
