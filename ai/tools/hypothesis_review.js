/* Private offline review: no fetch, telemetry, cloud requests or inferred decisions. */
(() => {
  'use strict';
  const catalog=window.REVIEW_CATALOG;
  const ids=catalog.proposal_ids;
  const allowed=['pending','confirmed','rejected','unclear'];
  const names={pending:'Не проверено',confirmed:'Описание подтверждено',rejected:'Предположение отклонено',unclear:'Недостаточно данных'};
  const el=id=>document.getElementById(id);
  const key='rat-ai-action-review-v1:'+catalog.catalog_id;
  const fresh=()=>ids.map(proposal_id=>({proposal_id,decision:'pending',note:'',reviewed_at:null}));
  let state=fresh(), storageAvailable=true;
  function validate(payload) {
    if (!payload || payload.schema_version!==1 || payload.kind!=='visible_action_review' || payload.catalog_id!==catalog.catalog_id || !Array.isArray(payload.decisions) || payload.decisions.length!==ids.length) throw Error('Этот файл относится к другому набору предположений.');
    const seen=new Set();
    for (const row of payload.decisions) {
      if (!row || Object.keys(row).sort().join(',')!=='decision,note,proposal_id,reviewed_at' || !ids.includes(row.proposal_id) || seen.has(row.proposal_id) || !allowed.includes(row.decision) || typeof row.note!=='string' || row.note.length>2000) throw Error('В файле есть неверные или повторяющиеся решения.');
      if (row.decision==='pending' ? row.reviewed_at!==null : typeof row.reviewed_at!=='string' || row.reviewed_at.length>40 || !Number.isFinite(Date.parse(row.reviewed_at)) || !/(Z|[+-]\d\d:\d\d)$/.test(row.reviewed_at)) throw Error('В файле неверно указано время проверки.');
      seen.add(row.proposal_id);
    }
    return ids.map(id=>({...payload.decisions.find(r=>r.proposal_id===id)}));
  }
  const payload=()=>({schema_version:1,kind:'visible_action_review',catalog_id:catalog.catalog_id,decisions:state});
  function progress() {
    const counts=Object.fromEntries(allowed.map(d=>[d,state.filter(r=>r.decision===d).length]));
    el('review-progress').textContent=`Проверено ${ids.length-counts.pending} из ${ids.length} · подтверждено ${counts.confirmed} · отклонено ${counts.rejected} · неясно ${counts.unclear}`;
    el('review-storage').textContent=storageAvailable ? 'Черновик сохраняется в этом браузере. Скачай решения, чтобы передать их для обработки.' : 'Браузер не разрешил сохранение черновика. Скачай решения до закрытия страницы.';
  }
  function save() {
    try { localStorage.setItem(key,JSON.stringify(payload())); storageAvailable=true; }
    catch { storageAvailable=false; }
    progress();
  }
  const controls=Array.from(document.querySelectorAll('.review-controls'));
  function paint() {
    for (const box of controls) {
      const row=state.find(r=>r.proposal_id===box.dataset.proposalId);
      el('review-error').textContent='';
      box.querySelector('.review-state').textContent=names[row.decision];
      box.querySelector('textarea').value=row.note;
      for (const button of box.querySelectorAll('[data-decision]')) button.setAttribute('aria-pressed',String(button.dataset.decision===row.decision));
    }
    progress();
  }
  try { const saved=localStorage.getItem(key); if (saved) state=validate(JSON.parse(saved)); }
  catch { storageAvailable=false; el('review-error').textContent='Не удалось прочитать черновик. Можно загрузить ранее скачанный файл.'; }
  for (const box of controls) {
    for (const button of box.querySelectorAll('[data-decision]')) button.addEventListener('click',()=>{
      const row=state.find(r=>r.proposal_id===box.dataset.proposalId);
      row.decision=button.dataset.decision;
      row.reviewed_at=row.decision==='pending' ? null : new Date().toISOString();
      save(); paint();
    });
    box.querySelector('textarea').addEventListener('input',event=>{
      const row=state.find(r=>r.proposal_id===box.dataset.proposalId);
      row.note=event.target.value.slice(0,2000);
      if (row.decision!=='pending') row.reviewed_at=new Date().toISOString();
      save();
    });
  }
  el('review-next').addEventListener('click',()=>{
    const next=controls.find(box=>state.find(r=>r.proposal_id===box.dataset.proposalId).decision==='pending');
    if (!next) { el('review-error').textContent='Все предположения проверены. Скачай решения.'; return; }
    next.closest('details').open=true;
    next.scrollIntoView({behavior:'smooth',block:'center'});
    next.querySelector('button').focus({preventScroll:true});
  });
  el('review-export').addEventListener('click',()=>{
    save();
    const url=URL.createObjectURL(new Blob([JSON.stringify(payload(),null,2)],{type:'application/json'}));
    const link=document.createElement('a'); link.href=url; link.download='action-review-'+catalog.catalog_id.slice(0,12)+'.json'; link.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  el('review-import').addEventListener('change',async event=>{
    const file=event.target.files[0]; if (!file) return;
    try {
      if (file.size>5_000_000) throw Error('Файл слишком большой.');
      const incoming=validate(JSON.parse(await file.text()));
      if (state.some(r=>r.decision!=='pending' || r.note) && !window.confirm('Заменить текущие решения содержимым файла? Сначала можно скачать текущую копию.')) return;
      state=incoming; el('review-error').textContent=''; save(); paint();
    } catch (error) { el('review-error').textContent=error.message || 'Не удалось прочитать файл.'; }
    finally { event.target.value=''; }
  });
  paint();
  const first=controls.find(box=>state.find(r=>r.proposal_id===box.dataset.proposalId).decision==='pending') || controls[0];
  if (first) first.closest('details').open=true;
})();
