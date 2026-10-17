import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

AI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI / "tools"))
from prepare_dataset import apply_review
from review_dataset import build_review_page


class ReviewPageTests(unittest.TestCase):
    def setUp(self):
        scratch = AI / "local/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        video = root / "local video.mov"
        video.write_bytes(b"synthetic placeholder; browser rendering not used")
        self.rows = [{"video_id": f"X-{i}", "label": 1 - i, "event_time_sec": 2.0 if not i else None,
                      "video_path": str(video.resolve()), "event_type": "</script><script>bad()</script>"}
                     for i in range(2)]
        self.manifest, self.output = root / "manifest.jsonl", root / "index.html"
        self.manifest.write_text("\n".join(json.dumps(r) for r in self.rows), encoding="utf-8")

    def test_embedded_annotations_cannot_escape_script_and_network_is_disabled(self):
        result = build_review_page(self.manifest, self.output)
        page = self.output.read_text(encoding="utf-8")
        self.assertEqual(result, {"videos": 2, "network_required": False})
        self.assertEqual(page.count("<script>"), 1)
        self.assertEqual(page.count("</script>"), 1)
        self.assertNotIn("__REVIEW_RECORDS__", page)
        self.assertIn("connect-src 'none'", page)
        self.assertIn("media-src file:", page)
        self.assertIn("file:///", page)

    def test_missing_local_video_is_rejected(self):
        self.rows[0]["video_path"] = str(self.output.parent / "absent.mov")
        self.manifest.write_text("\n".join(json.dumps(r) for r in self.rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "existing absolute local"):
            build_review_page(self.manifest, self.output)
        self.assertFalse(self.output.exists())

    def test_ai_draft_is_escaped_and_never_becomes_a_reviewed_label(self):
        build_review_page(self.manifest,self.output)
        old_key=self.output.read_text(encoding='utf-8').split('const storageKey=')[1].split(';')[0]
        path = self.output.parent/'drafts.json'
        action = '</script><script>untrusted()</script>'
        path.write_text(json.dumps({'human_review_required':True,'by_video_id':{
            'X-0':{'observed_action':action,'event_type':'AI suggestion','uncertainty':'Unclear'}}}),encoding='utf-8')
        build_review_page(self.manifest,self.output,suggestions=path)
        page = self.output.read_text(encoding='utf-8')
        self.assertEqual(page.split('const storageKey=')[1].split(';')[0],old_key)
        self.assertNotIn(action,page)
        self.assertEqual(page.count('</script>'),1)
        records = json.loads(page.split('const records=')[1].split(';\n')[0])
        self.assertEqual(records[0]['review_status'],'pending')
        self.assertEqual(records[0]['observed_action'],'')
        self.assertEqual(records[0]['ai_draft']['observed_action'],action)
        self.assertEqual(records[0]['rule_id'],'')

    def test_ai_cannot_set_review_status_or_target_unknown_video(self):
        path = self.output.parent/'drafts.json'
        for drafts in ({'X-0':{'review_status':'reviewed'}},{'unknown':{'observed_action':'test'}}):
            path.write_text(json.dumps({'human_review_required':True,'by_video_id':drafts}),encoding='utf-8')
            with self.assertRaises(ValueError):
                build_review_page(self.manifest,self.output,suggestions=path)

    @unittest.skipUnless(shutil.which("node"), "Node required to test offline page interactions")
    def test_navigation_and_csv_export_roundtrip_into_importer(self):
        build_review_page(self.manifest, self.output)
        # Execute the real embedded JS against a tiny DOM. No browser opens any video.
        harness = self.output.parent / "page-test.cjs"
        harness.write_text(r'''
const fs=require("node:fs"),vm=require("node:vm"),assert=require("node:assert/strict");
const page=fs.readFileSync(process.argv[2],"utf8");
const script=page.split("<script>")[1].split("</script>")[0];
const elements=new Map();let exported;
function element(){return {value:"",textContent:"",listeners:{},append(){},click(){},addEventListener(type,fn){this.listeners[type]=fn;}};}
function get(id){if(!elements.has(id))elements.set(id,element());return elements.get(id);}
const document={getElementById:get,createElement:element};
const saved=new Map();const localStorage={getItem:k=>saved.get(k),setItem:(k,v)=>saved.set(k,v)};
const URL={createObjectURL:blob=>{exported=blob;return "blob:test";},revokeObjectURL(){}};
vm.runInNewContext(script,{document,localStorage,URL,Blob,setTimeout:fn=>fn()});
const change=(id,value)=>{get(id).value=value;get(id).listeners.input();};
change("session_group","scene-1");change("event_type","example");
change("observed_action",'quoted "action"\nsecond line'.replace("\\n","\n"));
change("rule_id","rule-1");change("review_status","reviewed");
get("next").onclick();change("session_group","scene-2");change("review_status","reviewed");
get("previous").onclick();assert.equal(get("event_type").value,"example");
assert.equal(get("session_group").value,"scene-1");
get("export").onclick();assert.ok(exported);exported.text().then(text=>process.stdout.write(text));
''', encoding="utf-8")
        result = subprocess.run([shutil.which("node"), str(harness), str(self.output)],
                                capture_output=True, timeout=15, check=True, encoding="utf-8")
        records = list(csv.DictReader(io.StringIO(result.stdout.lstrip("\ufeff"))))
        self.assertEqual(records[0]["review_status"], "reviewed")
        self.assertEqual(records[1]["session_group"], "scene-2")
        self.assertIn('quoted "action"', records[0]["observed_action"])
        self.assertIn("\nsecond line", records[0]["observed_action"])
        exported = self.output.parent / "event_review.csv"
        exported.write_text(result.stdout, encoding="utf-8-sig")
        imported = apply_review(self.rows, exported)
        self.assertEqual(imported[0]["event_time_sec"], 2.0)
        self.assertEqual(imported[0]["event_type"], "example")


if __name__ == "__main__":
    unittest.main()
