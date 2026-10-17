"""Bind human decisions to immutable visible-action proposals, never legal labels."""
import hashlib
import json
from datetime import datetime

DECISIONS=('pending','confirmed','rejected','unclear')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True,separators=(',',':')).encode()).hexdigest()


def make_catalog(proposals):
    rows=[]
    for proposal in proposals:
        if proposal['split'] not in ('train','validation'):
            raise ValueError('Unknown split')
        row=dict(proposal)
        row['proposal_id']=fingerprint(proposal)
        rows.append(row)
    if not rows or len({r['proposal_id'] for r in rows})!=len(rows):
        raise ValueError('Nonempty unique proposals required')
    return {'schema_version':1,'kind':'visible_action_review','catalog_id':fingerprint(rows),'proposals':rows}


def validate_review(payload,catalog):
    if catalog.get('catalog_id')!=fingerprint(catalog['proposals']):
        raise ValueError('Catalog changed')
    for row in catalog['proposals']:
        if row['proposal_id']!=fingerprint({k:v for k,v in row.items() if k!='proposal_id'}):
            raise ValueError('Proposal changed')
    if payload.get('schema_version')!=1 or payload.get('kind')!='visible_action_review' or payload.get('catalog_id')!=catalog['catalog_id']:
        raise ValueError('Review belongs to another proposal catalog')
    indexed={r['proposal_id']:r for r in catalog['proposals']}
    decisions=payload.get('decisions')
    if not isinstance(decisions,list) or len(decisions)!=len(indexed):
        raise ValueError('Complete review state required')
    seen=set()
    result=[]
    for item in decisions:
        if not isinstance(item,dict) or set(item)!={'proposal_id','decision','note','reviewed_at'}:
            raise ValueError('Invalid decision fields')
        key=item['proposal_id']
        if key not in indexed or key in seen or item['decision'] not in DECISIONS:
            raise ValueError('Unknown, duplicate or invalid decision')
        seen.add(key)
        if not isinstance(item['note'],str) or len(item['note'])>2000:
            raise ValueError('Invalid review note')
        timestamp=item['reviewed_at']
        if item['decision']=='pending':
            if timestamp is not None:
                raise ValueError('Pending decision cannot be marked reviewed')
        else:
            if not isinstance(timestamp,str) or len(timestamp)>40:
                raise ValueError('Reviewed time required')
            try:
                parsed=datetime.fromisoformat(timestamp.replace('Z','+00:00'))
                if parsed.tzinfo is None: raise ValueError('Timezone required')
            except ValueError as exc:
                raise ValueError('Invalid reviewed time') from exc
        result.append({'proposal':indexed[key],'decision':item['decision'],'note':item['note'],
                       'reviewed_at':timestamp,'human_verified_action':item['decision']=='confirmed',
                       'human_verified_violation':False,'legal_verdict':None,
                       'eligible_for_training':item['decision']=='confirmed' and indexed[key]['split']=='train'})
    return result
