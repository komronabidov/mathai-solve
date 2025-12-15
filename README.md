# MathAI Backend

Flask-сервер для математического AI ассистента.

## Быстрый старт

### Установка зависимостей
```bash
pip install -r requirements.txt
```

### Настройка
Файл `.env` уже создан с базовыми настройками:
- `PORT=5503` - порт сервера
- `DEBUG=True` - режим отладки
- `ALLOWED_ORIGINS` - разрешенные origins для CORS

Опционально можно включить LLM-режим (для «текстовых»/олимпиадных задач):
- `LLM_MODE=off|auto|always`
- `LLM_BASE_URL=http://127.0.0.1:1234` (OpenAI-compatible API, будет использоваться `/v1/chat/completions`)
- `LLM_API_KEY=...` (если требуется)
- `LLM_MODEL=...`
- `LLM_TIMEOUT_SECONDS=20`
- `LLM_VERIFY=true|false` (draft→verify→final)

### Запуск сервера

**В Cursor IDE:**
- Откройте `server.py`
- Нажмите кнопку "Run Python File" (▶️) в правом верхнем углу
- Или используйте сочетание клавиш (обычно `Ctrl+F5` или `Cmd+F5`)

**В терминале:**
```bash
python3 server.py
```

или

```bash
./server.py
```

Сервер запустится на `http://127.0.0.1:5503`

## API Endpoints

- `GET /api/health` - проверка работоспособности
- `POST /api/solve` - решение математических задач

## Возможности

- ✅ Решение уравнений
- ✅ Вычисление производных и интегралов
- ✅ Построение графиков
- ✅ OCR распознавание математики из изображений
- ✅ Поддержка LaTeX
- ✅ Статистика и анализ задач
