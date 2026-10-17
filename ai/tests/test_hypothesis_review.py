import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

AI=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(AI))
from rat_ai.hypothesis_review import make_catalog,validate_review


def catalog_fixture():
    return make_catalog([{'split':split,'sample_index':i,'action':'phone_handling',
                          'observation':'Synthetic person holds a phone.'}
                         for i,split in enumerate(('train','validation'))])


def payload_fixture(catalog,decision='confirmed'):
    return {'schema_version':1,'kind':'visible_action_review','catalog_id':catalog['catalog_id'],
            'decisions':[{'proposal_id':r['proposal_id'],'decision':decision,'note':'',
                          'reviewed_at':None if decision=='pending' else '2026-10-18T00:00:00Z'}
                         for r in catalog['proposals']]}


class HypothesisReviewTests(unittest.TestCase):
    def test_confirmation_is_action_only_and_validation_never_enters_training(self):
        catalog=catalog_fixture()
        rows=validate_review(payload_fixture(catalog),catalog)
        self.assertTrue(all(r['human_verified_action'] for r in rows))
        self.assertEqual([r['eligible_for_training'] for r in rows],[True,False])
        self.assertTrue(all(r['legal_verdict'] is None and not r['human_verified_violation'] for r in rows))

    def test_rejection_uncertainty_and_pending_never_create_positive_or_clean_labels(self):
        catalog=catalog_fixture()
        for decision in ('rejected','unclear','pending'):
            rows=validate_review(payload_fixture(catalog,decision),catalog)
            self.assertTrue(all(not r['eligible_for_training'] and not r['human_verified_action'] for r in rows))
            self.assertTrue(all('label' not in r and r['legal_verdict'] is None for r in rows))

    def test_changed_proposal_or_foreign_catalog_cannot_reuse_decisions(self):
        catalog=catalog_fixture(); payload=payload_fixture(catalog)
        changed=copy.deepcopy(catalog); changed['proposals'][0]['observation']='Different action'
        with self.assertRaises(ValueError): validate_review(payload,changed)
        payload['catalog_id']='foreign'
        with self.assertRaises(ValueError): validate_review(payload,catalog)

    def test_duplicate_unknown_or_incomplete_decisions_rejected(self):
        catalog=catalog_fixture()
        for edit in ('duplicate','unknown','incomplete','invented_decision'):
            payload=payload_fixture(catalog)
            if edit=='duplicate': payload['decisions'][1]=payload['decisions'][0]
            if edit=='unknown': payload['decisions'][0]['proposal_id']='foreign'
            if edit=='incomplete': payload['decisions'].pop()
            if edit=='invented_decision': payload['decisions'][0]['decision']='violation'
            with self.assertRaises(ValueError): validate_review(payload,catalog)

    def test_review_requires_time_and_bounded_note(self):
        catalog=catalog_fixture()
        for field,value in (('reviewed_at',None),('reviewed_at','yesterday'),('note','x'*2001)):
            payload=payload_fixture(catalog); payload['decisions'][0][field]=value
            with self.assertRaises(ValueError): validate_review(payload,catalog)

    @unittest.skipUnless(shutil.which('node'),'Node required for offline interactions')
    def test_decision_storage_reload_export_import_roundtrip(self):
        catalog=catalog_fixture()
        scratch=AI/'local/tests'; scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            root=Path(folder)
            (root/'catalog.json').write_text(json.dumps(catalog),encoding='utf-8')
            harness=root/'harness.cjs'
            harness.write_text(r'''
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const catalog=JSON.parse(fs.readFileSync(process.argv[2],'utf8')), script=fs.readFileSync(process.argv[3],'utf8');
const ids=catalog.proposals.map(p=>p.proposal_id), saved=new Map(); let exported;
function element(){return {textContent:'',value:'',listeners:{},attrs:{},addEventListener(k,f){this.listeners[k]=f;},setAttribute(k,v){this.attrs[k]=v;},click(){},focus(){},scrollIntoView(){}};}
function boot(){
 const elements=new Map(), get=id=>{if(!elements.has(id))elements.set(id,element());return elements.get(id);};
 const boxes=ids.map(id=>{const state=element(),note=element(),details={open:false},buttons=['confirmed','rejected','unclear','pending'].map(d=>Object.assign(element(),{dataset:{decision:d}}));
  return {dataset:{proposalId:id},buttons,note,state,details,querySelector:s=>s==='.review-state'?state:s==='textarea'?note:buttons[0],querySelectorAll:()=>buttons,closest:()=>details,scrollIntoView(){}};});
 const document={getElementById:get,querySelectorAll:()=>boxes,createElement:element};
 vm.runInNewContext(script,{document,window:{REVIEW_CATALOG:{catalog_id:catalog.catalog_id,proposal_ids:ids},confirm:()=>true},localStorage:{getItem:k=>saved.get(k),setItem:(k,v)=>saved.set(k,v)},URL:{createObjectURL:b=>{exported=b;return 'blob:test';},revokeObjectURL(){}},Blob,setTimeout:f=>f()});
 return {get,boxes};
}
(async()=>{
 let ui=boot(); assert.match(ui.get('review-progress').textContent,/0 из 2/);
 ui.boxes[0].buttons[0].listeners.click();
 ui.boxes[0].note.listeners.input({target:{value:'human correction <script> remains text'}});
 ui.boxes[1].buttons[1].listeners.click();
 ui.get('review-export').listeners.click(); let payload=JSON.parse(await exported.text());
 assert.equal(payload.decisions[0].decision,'confirmed'); assert.equal(payload.decisions[1].decision,'rejected');
 ui=boot(); assert.match(ui.get('review-progress').textContent,/2 из 2/); assert.equal(ui.boxes[0].note.value,'human correction <script> remains text');
 const before=JSON.stringify(payload), foreign={...payload,catalog_id:'foreign'};
 await ui.get('review-import').listeners.change({target:{files:[{size:100,text:async()=>JSON.stringify(foreign)}],value:'x'}});
 assert.ok(ui.get('review-error').textContent); ui.get('review-export').listeners.click(); assert.equal(await exported.text(),JSON.stringify(JSON.parse(before),null,2));
 payload.decisions[1].decision='unclear';
 await ui.get('review-import').listeners.change({target:{files:[{size:100,text:async()=>JSON.stringify(payload)}],value:'x'}});
 ui.get('review-export').listeners.click(); process.stdout.write(await exported.text());
})().catch(e=>{console.error(e);process.exitCode=1;});
''',encoding='utf-8')
            result=subprocess.run([shutil.which('node'),str(harness),str(root/'catalog.json'),str(AI/'tools/hypothesis_review.js')],capture_output=True,text=True,check=True)
            rows=validate_review(json.loads(result.stdout),catalog)
            self.assertEqual([r['decision'] for r in rows],['confirmed','unclear'])


if __name__=='__main__': unittest.main()
