# Rat AI — контроль кассовых операций

React / TypeScript / Vite, FastAPI / SQLite и локальный CPU-детектор объединены в один сервис. Интерфейс загружает видео, показывает ход обработки, подозрительные фрагменты, таймкоды и кадры-доказательства.

## Запуск всего проекта

Требуются Docker Engine и Docker Compose, Python 3 для создания конфигурации. Из корня репозитория:

```sh
python3 tools/configure_local.py
docker compose up --build -d --wait
```

Откройте http://127.0.0.1:8080. Скрипт создаёт случайный API-ключ в игнорируемом `backend/.env`, существующий файл сохраняется. Если он создан ранее, сравните настройки с `backend/.env.example`: для включённой модели нужны `ANALYSIS_MODE=experimental` и `ANALYZER_COMMAND=["python", "/opt/ai/tools/infer_detector.py"]`.

Если порт 8080 занят, добавьте `FRONTEND_PORT=8081` в корневой `.env` (игнорируется Git) и откройте http://127.0.0.1:8081.

Первая сборка скачивает CPU-зависимости. Веса уже включены в репозиторий; при обработке видео облачные API не вызываются. Backend доступен через Nginx, ключ подставляется на сервере. SQLite и результаты сохраняются в Docker volume при перезапуске.

```sh
docker compose logs --tail=100 backend
docker compose down
```

## Что умеет модель

Включён экспериментальный MobileNetV3-детектор: он выделяет статистически подозрительные фрагменты. Результат требует проверки человеком. Тип нарушения и нормативное основание модель пока не определяет; пустой список кандидатов не доказывает отсутствие нарушения. Интерфейс явно обозначает этот режим.

Это рабочая сквозная интеграция, но не законченный классификатор по контракту жюри. Для итогового `violation` / `no_violation` нужны качественная модель категорий, проверенные нормативные источники и независимая оценка качества. Строгий режим API сохранён для такой модели. GPU/Qwen-эксперименты не входят в запуск Docker.

## Разработка и проверки

Без Docker: [запуск общего локального стека](ai/LOCAL_STACK.md). Требуются Python 3.12, FFmpeg/ffprobe и Node.js 24. Отдельные инструкции: [frontend](frontend/README.md), [backend и контракты](backend/README.md), [детектор](ai/INTEGRATION.md).

```sh
npm --prefix frontend ci
npm --prefix frontend test
npm --prefix frontend run build
# Linux/macOS; команды для Windows — в ai/INTEGRATION.md
python3 -m venv ai/.venv
source ai/.venv/bin/activate
python -m pip install -e "./ai[training]" --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -r backend/requirements.txt pytest httpx
python -m unittest discover -s ai/tests
python -m pytest -c backend/pytest.ini backend/tests -q
python ai/tools/smoke_local_stack.py --url http://127.0.0.1:8080
```

Последняя команда проверяет загрузку синтетического MOV, настоящую модель, опрос задачи и формат результата через Nginx. Это проверка работоспособности, а не качества распознавания. Датасет, ключи и локальные результаты не публикуются в Git.
