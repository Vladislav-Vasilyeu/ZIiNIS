# NeuralTextStego

Учебное приложение для исследования текстовой стеганографии. В графическом интерфейсе и CLI можно внедрять сообщения в текст, извлекать их и сравнивать методы. Поддерживаются TXT и DOCX, парольное шифрование и опциональный невидимый маркер.

## Методы

- `classic_synonyms` — замена слов по встроенному словарю.
- `classic_whitespace` — кодирование обычными и неразрывными пробелами.
- `neural_rank` — кодирование по рангу следующего токена.
- `neural_arithmetic` — арифметическое кодирование на основе распределения токенов.
- `neural_sampling` — кодирование выбором группы токенов.

Классические методы выполняются локально. Для нейросетевых методов нужна модель с поддержкой `logprobs` и `top_logprobs`. Для извлечения без маркера используйте те же модель, провайдер, промпт и параметры, что и при внедрении.

## Запуск

Требуется Python 3.10+ и Tk для графического интерфейса.

```powershell
cd NeuralTextStego_V3
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python main.py
```

На macOS и Linux используйте `python3 -m venv .venv`, `.venv/bin/python` и `cp .env.example .env`. Если `.env` уже существует, добавьте нужные параметры в существующий файл.

## Провайдеры

Провайдер и модель в верхней части окна применяются к генерации контейнера, нейросетевому внедрению, извлечению и экспериментам. Нажмите «Загрузить модели» и выберите модель в списке. ID модели также можно ввести вручную. Графический интерфейс запоминает выбор отдельно для каждого провайдера. В CLI укажите `--provider ollama` или `--provider tooken_club` и при необходимости `--model ID`. Начальный выбор провайдера задаёт `DEFAULT_LOGITS_PROVIDER`.

- **Ollama** использует локальный сервер. Укажите адрес в `OLLAMA_BASE_URL` и установленную модель в `OLLAMA_MODEL`.
- **Tooken Club** использует OpenAI-совместимый шлюз `https://tooken.club/v1`. Запишите ключ в `TOOKEN_CLUB_API_KEY` и ID модели в `TOOKEN_CLUB_MODEL`. Доступные ID возвращает `GET /v1/models` с заголовком `Authorization: Bearer <ключ>`. Приложение отправляет текстовые запросы на `POST /v1/chat/completions`. Ключ хранится в локальном `.env`.

Обычный чат Tooken Club подходит для генерации контейнера. Для нейросетевой стеганографии шлюз должен вернуть распределение в `choices[0].logprobs.content[0].top_logprobs`. Проверенные модели, включая `gpt-5.5` по умолчанию, отклонили этот параметр. Для нейросетевых методов используйте Ollama с поддержкой logprobs или укажите совместимую модель Tooken Club.

## Пример CLI

Из каталога `NeuralTextStego_V3` с установленными зависимостями:

```bash
python main.py --cli embed --method classic_whitespace --cover data/sample_cover.txt --secret "hi" --out stego.txt
python main.py --cli extract --method classic_whitespace --stego stego.txt
python -m pytest -q
```

Приложение и тесты находятся в каталоге [NeuralTextStego_V3](NeuralTextStego_V3). Результаты экспериментов сохраняются в `experiments/results/`.
