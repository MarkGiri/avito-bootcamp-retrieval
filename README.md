# Авито. Поиск статей справки по вопросу пользователя

Решение задачи на поиск статей справки по пользовательскому вопросу. Для каждого запроса формируется ранжированный top-10 `article_id`.

Итоговый результат на платформе: **MAP@10 = 0.610612**.

## Запуск

Версии библиотек зафиксированы в `requirements.txt`.

Исходные файлы нужно положить так:

```text
candidate_data/
├── articles.f
├── calibration.f
└── test.f
```

После выполнения всех ячеек появится `answer.csv`.

## Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MarkGiri/avito-bootcamp-retrieval/blob/main/solution.ipynb)

## Подход

HTML очищается через BeautifulSoup. Отдельно сохраняются заголовки, подписи вкладок и спойлеров, элементы таблиц и `alt` изображений. Для текста используются лемматизация, word/char TF-IDF, BM25 и поиск по фрагментам длинных статей.

Финальный score объединяет два канала: supervised-модели по `calibration.f` и прямой retrieval по всем статьям. Качество основных параметров проверялось на repeated 5-fold OOF по MAP@10, дополнительно разбирались ошибки для статей, отсутствующих среди меток train-fold.
