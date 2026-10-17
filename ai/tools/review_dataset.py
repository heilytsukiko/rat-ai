"""Build an offline human-review page; video pixels stay in the local browser."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_dataset import apply_review
from rat_ai.rules import load_rules


def build_review_page(manifest, output, review=None):
    manifest = Path(manifest)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len({row["video_id"] for row in rows}) != len(rows):
        raise ValueError("Review requires a nonempty manifest with unique video IDs")
    if review is not None:
        rows = apply_review(rows, Path(review))
    records = []
    for row in rows:
        source = Path(row["video_path"])
        if not source.is_absolute() or not source.is_file():
            raise ValueError("Video source must be an existing absolute local file")
        record = {key: row.get(key, "") for key in FIELDS}
        record["video_url"] = source.resolve().as_uri()
        record["review_status"] = row.get("review_status") or "pending"
        records.append(record)
    # Escape HTML parser delimiters even inside a script's JSON literal.
    data = json.dumps(records, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    # A newly imported CSV must not be overwritten by an older browser draft.
    identity = hashlib.sha256(manifest.read_bytes() + data.encode()).hexdigest()[:24]
    rules = json.dumps(load_rules(), ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    page = PAGE.replace("__REVIEW_ID__", identity).replace("__RULE_CATALOG__", rules).replace("__REVIEW_RECORDS__", data)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return {"videos": len(records), "network_required": False}


FIELDS = ["video_id", "label", "event_time_sec", "event_type", "observed_action",
          "rule_id", "session_group", "review_status"]

PAGE = r'''<!doctype html>
<html lang="ru"><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; media-src file:; connect-src 'none'; img-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Локальная проверка разметки Rat AI</title>
<style>
body{font:16px system-ui,sans-serif;margin:24px auto;padding:0 16px;max-width:1080px;color:#20202a;background:#f7f7fa}
h1{font-size:26px}p{line-height:1.5}.layout{display:grid;grid-template-columns:1fr 1fr;gap:24px}
video{width:100%;max-height:65vh;background:#111;border-radius:10px}label{display:block;margin:16px 0 6px}
input,textarea,select,button{font:inherit;padding:10px;border:1px solid #bbb;border-radius:6px;box-sizing:border-box}
input,textarea,select{width:100%;background:white}textarea{min-height:76px}button{cursor:pointer;background:#fff;margin:4px}
button.primary{background:#702741;color:white}#status{padding:10px;background:#fff;border-radius:6px}#error{color:#9b122c;min-height:24px}
.hint{color:#555;font-size:14px}nav{display:flex;align-items:center;flex-wrap:wrap;margin-bottom:12px}nav select{width:auto;max-width:70%}
@media(max-width:720px){.layout{grid-template-columns:1fr}}
</style>
<h1>Проверка разметки на ноутбуке</h1>
<p>Ролики открываются с диска. Страница не обращается к сети и не делает выводов о нарушениях.
Запишите только наблюдаемое действие и подтверждённое правило. Для всех связанных дублей
и вариантов одной сцены укажите одинаковую группу съёмки, независимо от метки.</p>
<p class="hint">Метка и исходный таймкод фиксированы. Если они ошибочны, выберите «Нужно уточнить».
По завершении скачайте CSV: это файл, который затем импортируется при подготовке датасета.</p>
<nav><button id="previous">←</button><select id="selector" aria-label="Выбор ролика"></select><button id="next">→</button></nav>
<div class="layout"><div>
<video id="video" controls preload="metadata"></video>
<p id="sourceInfo"></p><button id="seek">К исходному таймкоду</button>
<p class="hint">Если браузер не поддерживает кодек MOV, откройте исходный ролик в локальном
видеоплеере. Отсутствие воспроизведения не означает отсутствие события.</p>
</div><div>
<label for="session_group">Группа съёмки — для каждого ролика</label>
<input id="session_group" placeholder="Например: scene-01">
<label for="event_type">Тип события — для положительных роликов</label><input id="event_type">
<label for="observed_action">Наблюдаемое действие</label><textarea id="observed_action"></textarea>
<label for="rule_id">Правило для проверки</label><input id="rule_id" list="known_rules" placeholder="Выберите пункт № 231 или оставьте пустым">
<datalist id="known_rules"></datalist><p id="rule_hint" class="hint"></p>
<a id="rule_source" target="_blank" rel="noopener noreferrer" hidden>Открыть официальный документ</a>
<label for="review_status">Статус проверки</label><select id="review_status">
<option value="pending">Не проверено</option><option value="needs_review">Нужно уточнить</option><option value="reviewed">Проверено</option></select>
<p class="hint">Статус «Проверено» не подтверждает юридическую корректность ссылки автоматически.
Правило должен проверить участник команды.</p>
</div></div>
<p id="status" role="status"></p><p id="error" role="alert"></p>
<button class="primary" id="export">Скачать event_review.csv</button>
<p class="hint" id="saveHint">Изменения дополнительно сохраняются только в этом браузере.
Чтобы передать их программе, обязательно скачайте CSV.</p>
<script>
"use strict";
const records=__REVIEW_RECORDS__;
const catalog=__RULE_CATALOG__;
const storageKey="rat-ai-review-__REVIEW_ID__";
const editable=["event_type","observed_action","rule_id","session_group","review_status"];
const fields=["video_id","label","event_time_sec",...editable];
const el=id=>document.getElementById(id);
let index=0;
try{
  const saved=JSON.parse(localStorage.getItem(storageKey)||"null");
  if(saved && Array.isArray(saved)){
    const byId=new Map(saved.map(row=>[row.video_id,row]));
    for(const row of records){const old=byId.get(row.video_id);if(old)for(const key of editable)if(typeof old[key]==="string")row[key]=old[key];}
  }
}catch{el("saveHint").textContent="Автосохранение браузера недоступно. Скачивайте CSV, чтобы сохранить изменения.";}
function complete(row){return !!String(row.session_group||"").trim() && (row.label===0 || ["event_type","observed_action","rule_id"].every(key=>String(row[key]||"").trim()));}
function summary(){
  const grouped=records.filter(r=>String(r.session_group||"").trim()).length;
  const reviewed=records.filter(r=>r.review_status==="reviewed" && complete(r)).length;
  el("status").textContent=`Группа указана: ${grouped} из ${records.length}. Проверено: ${reviewed} из ${records.length}.`;
}
function ruleHint(){
  const rule=catalog.rules.find(item=>item.id===el("rule_id").value.trim());
  el("rule_hint").textContent=rule?`${rule.section}: ${rule.summary} Условия: ${rule.required_observation} ${rule.caution}`:"Правило выбирает проверяющий после просмотра. Бинарный детектор его не устанавливает.";
  el("rule_source").hidden=!rule;
  if(rule)el("rule_source").href=catalog.document.url;
}
function save(){
  const row=records[index];
  for(const key of editable)row[key]=el(key).value.trim();
  el("error").textContent=row.review_status==="reviewed" && !complete(row)?"Для статуса «Проверено» заполните группу, а для положительного ролика — тип, действие и правило.":"";
  try{localStorage.setItem(storageKey,JSON.stringify(records.map(({video_url,...rest})=>rest)));}catch{}
  summary();ruleHint();
}
function show(){
  const row=records[index];
  el("selector").value=String(index);
  for(const key of editable)el(key).value=row[key]||"";
  el("video").src=row.video_url;
  el("sourceInfo").textContent=`${row.video_id}: ${row.label?"нарушение":"без нарушения"}; исходный таймкод: ${row.event_time_sec??"нет"}`;
  el("seek").disabled=row.event_time_sec===null || row.event_time_sec==="";
  el("previous").disabled=index===0;el("next").disabled=index===records.length-1;
  el("error").textContent="";summary();ruleHint();
}
for(const rule of catalog.rules){const option=document.createElement("option");option.value=rule.id;option.textContent=`${rule.section} · ${rule.event_type}`;el("known_rules").append(option);}
for(const [i,row] of records.entries()){const option=document.createElement("option");option.value=String(i);option.textContent=row.video_id;el("selector").append(option);}
for(const key of editable)el(key).addEventListener("input",save);
el("rule_id").addEventListener("change",()=>{const rule=catalog.rules.find(item=>item.id===el("rule_id").value.trim());if(rule && !el("event_type").value.trim())el("event_type").value=rule.event_type;save();});
el("selector").addEventListener("change",event=>{save();index=Number(event.target.value);show();});
el("previous").onclick=()=>{save();index--;show();};el("next").onclick=()=>{save();index++;show();};
el("seek").onclick=()=>{el("video").currentTime=Number(records[index].event_time_sec);};
el("export").onclick=()=>{
  save();
  if(records.some(row=>row.review_status==="reviewed" && !complete(row))){el("error").textContent="Есть неполные записи со статусом «Проверено». Заполните поля или измените статус.";return;}
  const quote=value=>'"'+String(value??"").replaceAll('"','""')+'"';
  const csv="\ufeff"+[fields,...records.map(row=>fields.map(key=>row[key]))].map(row=>row.map(quote).join(",")).join("\r\n")+"\r\n";
  const url=URL.createObjectURL(new Blob([csv],{type:"text/csv;charset=utf-8"}));
  const link=document.createElement("a");link.href=url;link.download="event_review.csv";link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
show();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("ai/local/dataset/manifest.jsonl"))
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path, default=Path("ai/local/review/index.html"))
    args = parser.parse_args()
    print(json.dumps(build_review_page(args.manifest, args.output, args.review)))


if __name__ == "__main__":
    main()
