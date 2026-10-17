"""Combine independent machine annotations, export silver TRAIN actions only."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from auto_annotate_gpu import consensus, MODELS

NAMES = {'cash_counting':'Пересчёт наличных','cash_transfer':'Передача наличных',
         'cash_storage':'Размещение наличных','hand_toward_pocket_or_bag':'Рука возле кармана или сумки',
         'phone_handling':'Действия с телефоном','leaving_workstation':'Уход с рабочего места',
         'document_handling':'Действия с документами','computer_use':'Работа за компьютером',
         'no_visible_action':'Видимое действие отсутствует','other':'Другое действие','unclear':'Не определено'}


def merge(dataset, first, second, dataset_sha):
    if first['model'] == second['model'] or {first['model'],second['model']} != set(MODELS.values()):
        raise ValueError('Two distinct configured models are required')
    if any(item.get('dataset_sha256') != dataset_sha for item in (first,second)):
        raise ValueError('Dataset fingerprint mismatch')
    if first.get('prompt_sha256') != second.get('prompt_sha256') or first.get('data_kind') != second.get('data_kind'):
        raise ValueError('Annotation protocols differ')
    expected = set(range(len(dataset['samples'])))
    indexed = []
    for result in (first, second):
        by_index = {row['sample_index']:row for row in result['predictions']}
        if len(by_index) != len(result['predictions']) or set(by_index) != expected:
            raise ValueError('Each pass must cover every sample exactly once')
        indexed.append(by_index)
    annotations = []
    for index, row in enumerate(dataset['samples']):
        a, b = (item[index] for item in indexed)
        annotations.append({'sample_index':index, 'group':row['group'], 'split':row['split'],
                            'center_sec':row['center_sec'], **consensus(a,b),
                            'observations':[a,b], 'rule_id':None})
    # These targets are neutral action codes, never fabricated human/legal labels.
    silver = [{'sample_index':a['sample_index'], 'file':dataset['samples'][a['sample_index']]['file'],
               'group':a['group'], 'action':a['action'], 'label_source':'two_model_auto_agreement',
               'human_verified':False, 'legal_verdict':None} for a in annotations
              if a['split']=='train' and a['status']=='auto_agreement']
    drafts = []
    for group in dataset['groups']:
        targets = group['annotated_event_times_sec'] or [group.get('duration_sec',4)/2]
        for target in targets:
            index = min(group['samples'], key=lambda i:abs(dataset['samples'][i]['center_sec']-target))
            a = annotations[index]
            left,right = a['observations']
            observation = 'Qwen2.5: '+left['observation']+' Qwen3: '+right['observation']
            uncertainty = ('Модели согласились только по категории видимого действия. '
                           if a['status']=='auto_agreement' else 'Модели не дали согласованного ясного ответа. ')
            uncertainty += 'Автоматическая разметка; нарушение и нормативное основание не подтверждены. '
            uncertainty += left['uncertainty']+' '+right['uncertainty']
            drafts.append({'group':group['group'], 'target_sec':target, 'center_sec':a['center_sec'],
                           'review_status':'pending', 'observed_action':observation[:1500],
                           'event_type':NAMES[a['action']], 'uncertainty':uncertainty[:1500]})
    summary = {'schema_version':1, 'windows':len(annotations), 'unique_videos':len(dataset['groups']),
               'models':[first['model'],second['model']], 'data_kind':first['data_kind'],
               'status_counts':dict(Counter(a['status'] for a in annotations)),
               'review_counts':dict(Counter(a['review'] for a in annotations)),
               'action_counts':dict(Counter(a['action'] for a in annotations)),
               'parse_failures':[sum(p.get('parsed') is not True for p in r['predictions']) for r in (first,second)],
               'silver_train_windows':len(silver), 'silver_validation_windows':0,
               'drafts':len(drafts), 'human_verified':False, 'accuracy_measured':False,
               'limitations':['Agreement is not ground truth; models may share errors.',
                              'Four sampled frames per window cannot verify full procedures or offscreen actions.',
                              'Positive training windows were originally selected around supplied points.',
                              'No automatic legal verdict or rule assignment; original binary labels preserved.']}
    return annotations, silver, drafts, summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--first',type=Path,required=True)
    parser.add_argument('--second',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=(args.dataset/'dataset.json').read_bytes()
    digest=hashlib.sha256(source).hexdigest()
    dataset=json.loads(source)
    first=json.loads(args.first.read_text(encoding='utf-8'))
    second=json.loads(args.second.read_text(encoding='utf-8'))
    if first['model']!=MODELS['qwen25-7b']:
        first,second=second,first
    annotations,silver,drafts,summary=merge(dataset,first,second,digest)
    args.output.mkdir(parents=True,exist_ok=True)
    def save(name,value):
        (args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    save('annotations.json',{'dataset_sha256':digest,'human_verified':False,
                            'models':summary['models'],'annotations':annotations})
    save('drafts.json',{'dataset_sha256':digest,'human_review_required':True,'drafts':drafts})
    save('summary.json',summary)
    (args.output/'train-actions-silver.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n'
                                                       for row in silver),encoding='utf-8')
    with (args.output/'annotations.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        fields=['sample_index','group','split','center_sec','action','review','status',
                'observation_qwen25','observation_qwen3','human_verified','rule_id']
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for annotation in annotations:
            row={key:annotation[key] for key in fields if key in annotation}
            for key,observation in zip(('observation_qwen25','observation_qwen3'),annotation['observations']):
                text=observation['observation']
                # Prevent generated text from becoming spreadsheet formulas on export.
                row[key]="'"+text if text.lstrip().startswith(('=','+','-','@')) else text
            writer.writerow(row)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
