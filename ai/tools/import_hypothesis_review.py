"""Import human decisions without changing original labels or validation splits."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rat_ai.hypothesis_review import validate_review


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',type=Path,required=True)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    catalog=json.loads(args.catalog.read_text(encoding='utf-8'))
    rows=validate_review(json.loads(args.review.read_text(encoding='utf-8-sig')),catalog)
    if args.output.exists(): raise ValueError('Fresh output directory required')
    args.output.mkdir(parents=True)
    (args.output/'reviewed-actions-private.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    train=[r for r in rows if r['eligible_for_training']]
    (args.output/'train-actions-human.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in train),encoding='utf-8')
    summary={'decisions':dict(Counter(r['decision'] for r in rows)),'train_action_rows':len(train),
             'confirmed_validation_actions':sum(r['human_verified_action'] and r['proposal']['split']=='validation' for r in rows),
             'violation_labels_created':0,'original_labels_changed':False}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__': main()
