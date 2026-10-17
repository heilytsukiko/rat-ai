"""Build a private offline comparison page; never print machine observations."""
import argparse
import hashlib
from html import escape
import json
from pathlib import Path

from compare_video_context import MODES, MODELS, nominal_refs, read_frames, tiles, build_messages, summary_for
from merge_auto_annotations import NAMES

MODE_NAMES={'four_frames':'4 кадра','context16':'Больше кадров','context16_tiles':'Больше кадров + увеличенные участки'}
OBJECT_NAMES={'banknotes':'банкноты','papers':'бумаги','phone':'телефон','bag':'сумка',
              'counter':'рабочая поверхность','computer':'компьютер','person':'человек','unclear':'неясно'}
REVIEW_NAMES={'possible_issue':'Модель предлагает проверить эпизод',
              'ordinary_visible_action':'Модель описывает обычное действие',
              'unclear':'Недостаточно ясных данных'}
VISIBILITY_NAMES={'clear':'модель считает действие видимым',
                  'partial':'видно частично', 'unclear':'видимость неясна'}


def render_prediction(row):
    if not row['parsed']:
        return '<p class="uncertain">Ответ модели не прошёл проверку формата.</p>'
    action=escape(NAMES[row['action']])
    objects=escape(', '.join(OBJECT_NAMES[o] for o in row['objects']) or 'не указаны')
    times=escape(', '.join(f'{t:.2f} с' for t in row['evidence_nominal_sec']) or 'не указаны')
    format_note=('<p class="uncertain">Неизвестные категории объектов помечены как неясные; '
                 'такой ответ не принимается как ясная согласованная разметка.</p>'
                 if row.get('unknown_object_codes',0) else '')
    return (f'<p><strong>{action}</strong></p><p>{escape(row["observation"])}</p>'
            f'<p>Видимость: {escape(VISIBILITY_NAMES[row["visibility"]])}.</p>{format_note}'
            f'<p class="uncertain">{escape(row["uncertainty"])}</p>'
            f'<p>Объекты: {objects}.</p><p>Опорные кадры, примерно: {times}.</p>'
            f'<p class="badge">{escape(REVIEW_NAMES[row["review"]])}</p>')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raw=(args.dataset/'dataset.json').read_bytes()
    protocol=json.loads(args.protocol.read_text(encoding='utf-8'))
    if hashlib.sha256(raw).hexdigest()!=protocol['dataset_sha256']:
        raise ValueError('Review dataset differs from model input')
    dataset=json.loads(raw)
    selected=protocol['selected_windows']
    predictions=json.loads(args.predictions.read_text(encoding='utf-8'))['predictions']
    summary_for(predictions,selected)  # Require complete, matched model/mode/window coverage.
    indexed={(r['model_key'],r['mode'],r['sample_index']):r for r in predictions}
    if args.output.exists():
        raise ValueError('Use a fresh review directory')
    args.output.mkdir(parents=True)
    (args.output/'frames').mkdir()
    parts=['''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' data: file:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Сравнение подробности видеоанализа</title>
<style>body{font:16px/1.55 system-ui,sans-serif;background:#111827;color:#e5e7eb;max-width:1500px;margin:auto;padding:24px}
h1,h2,h3{line-height:1.2}summary{cursor:pointer;font-weight:600;padding:16px;background:#1f2937;border-radius:10px}
details{margin:20px 0}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:16px 0}
.card{background:#1f2937;border:1px solid #374151;border-radius:12px;padding:18px;min-width:0;overflow-wrap:anywhere}
.gallery{display:flex;overflow-x:auto;gap:12px;padding:12px 0}figure{margin:0;flex:0 0 220px}img{width:100%;height:auto;border-radius:8px}
figcaption,.uncertain{color:#cbd5e1;font-size:14px}.badge{border-left:3px solid #60a5fa;padding-left:10px}
.notice{padding:16px;background:#26344d;border-radius:12px}hr{border:0;border-top:1px solid #475569;margin:24px 0}
@media(max-width:900px){.grid{grid-template-columns:1fr}body{padding:12px}}</style>
<h1>Сравнение подробности видеоанализа</h1>
<p class="notice">Это машинные описания, а не подтверждённая разметка или вывод о нарушении.
Сравниваются одинаковые моменты видео. Совпадение двух моделей не доказывает правильность.
Времена приблизительные; увеличенные участки не добавляют исходной чёткости.
Страница работает локально и не отправляет кадры в интернет.</p>''']
    for target in selected:
        sample=dataset['samples'][target]
        center=sample['center_sec']
        parts.append(f'<details><summary>Видео {sample["group"]+1} · около {center:.2f} с</summary><div class="grid">')
        for mode in MODES:
            refs=nominal_refs(dataset,target,4 if mode=='four_frames' else 16)
            images=read_frames(args.dataset,dataset,refs)
            parts.append(f'<section class="card"><h2>{escape(MODE_NAMES[mode])}</h2><div class="gallery">')
            for number,(ref,image) in enumerate(zip(refs,images),1):
                name=f'{ref["sample_index"]:05d}-{ref["frame_index"]}.jpg'
                path=args.output/'frames'/name
                if not path.exists():
                    image.save(path,quality=90)
                parts.append(f'<figure><img loading="lazy" src="frames/{name}" alt="Кадр {number}">'
                             f'<figcaption>Кадр {number} · около {ref["nominal_sec"]:.2f} с</figcaption></figure>')
            parts.append('</div>')
            if mode=='context16_tiles':
                _,anchor=build_messages(refs,center,True)
                parts.append(f'<p>Увеличенные участки кадра {anchor+1}, тот же момент:</p><div class="gallery">')
                for number,image in enumerate(tiles(images[anchor]),1):
                    name=f'tile-{target:05d}-{number}.jpg'
                    image.save(args.output/'frames'/name,quality=90)
                    parts.append(f'<figure><img loading="lazy" src="frames/{name}" alt="Участок {number}"></figure>')
                parts.append('</div>')
            for model in MODELS:
                parts.append(f'<h3>{escape(MODELS[model])}</h3>')
                parts.append(render_prediction(indexed[(model,mode,target)]))
            parts.append('</section>')
        parts.append('</div></details>')
    parts.append('</html>')
    (args.output/'index.html').write_text('\n'.join(parts),encoding='utf-8')
    print(json.dumps({'written':True,'windows':len(selected),'observations':len(predictions),'private_offline':True}))


if __name__=='__main__':
    main()
