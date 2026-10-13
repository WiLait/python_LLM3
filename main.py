import os
import time
import csv
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
MODEL = os.getenv("LLM_MODEL")
TEMPERATURE = 0.2

SEPARATOR = "-" * 70


# ============== МОДЕЛИ ДАННЫХ ==============

class PromptVersion:
    """Одна версия промпта с описанием."""

    def __init__(self, version_id, description, text):
        self.version_id = version_id
        self.description = description
        self.text = text

    def __repr__(self):
        return f"PromptVersion({self.version_id})"


# ============== КЛИЕНТ LLM ==============

class LLMClient:
    """Общий клиент: создаётся один раз, используется всеми экспериментами."""

    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    def ask(self, system_prompt, user_prompt):
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
            )
            elapsed = time.perf_counter() - started
            return response.choices[0].message.content, elapsed, None
        except Exception as e:
            elapsed = time.perf_counter() - started
            return None, elapsed, str(e)


# ============== БАЗОВЫЙ ЭКСПЕРИМЕНТ ==============

class BaseExperiment:
    name = "base"
    description = "Базовый эксперимент"

    def __init__(self, client):
        self.client = client
        self.results = []

    def run(self):
        raise NotImplementedError

    def print_header(self):
        """Общая информация: печатается один раз для всех экспериментов."""
        print()
        print("=" * 70)
        print(f"ЭКСПЕРИМЕНТ: {self.description}")
        print("=" * 70)
        print(f"  Модель: {MODEL}")
        print(f"  Температура: {TEMPERATURE}")
        print(f"  Базовый URL: {BASE_URL}")
        print("=" * 70)


# ============== ЭКСПЕРИМЕНТ: СРАВНЕНИЕ ВЕРСИЙ ПРОМПТА ==============

class PromptComparisonExperiment(BaseExperiment):
    name = "prompt_comparison"
    description = "Сравнение версий промпта на фиксированном наборе тестов"

    def __init__(self, client, prompt_versions, tests):
        super().__init__(client)
        self.prompt_versions = prompt_versions
        self.tests = tests

    def run(self):
        self.print_header()
        print(f"  Версий промпта: {len(self.prompt_versions)}")
        print(f"  Тестов в наборе: {len(self.tests)}")
        print(f"  Всего запросов: {len(self.prompt_versions) * len(self.tests)}")
        print("=" * 70)

        for prompt in self.prompt_versions:
            print()
            print(f"### ВЕРСИЯ {prompt.version_id}")
            print(SEPARATOR)
            for test in self.tests:
                prefix = f"  [{test['id']}][{test['type']:<12}]"
                print(prefix, end=" ")
                answer, latency, error = self.client.ask(prompt.text, test["input"])
                if error:
                    print(f"ОШИБКА: {error}")
                else:
                    print(f"OK ({latency:.2f} с)")

                self.results.append({
                    "version": prompt.version_id,
                    "prompt_description": prompt.description,
                    "prompt_text": prompt.text,
                    "test_id": test["id"],
                    "test_type": test["type"],
                    "test_input": test["input"],
                    "latency": round(latency, 3),
                    "answer": answer,
                    "error": error,
                })


# ============== ЗАГРУЗКА РЕСУРСОВ ==============

def load_prompts(prompts_dir="prompts"):
    """Загружает все prompt_v*.txt из папки prompts/."""
    versions = []
    if not os.path.isdir(prompts_dir):
        raise FileNotFoundError(f"Папка '{prompts_dir}' не найдена")

    for filename in sorted(os.listdir(prompts_dir)):
        if filename.startswith("prompt_v") and filename.endswith(".txt"):
            version_id = filename.replace(".txt", "")
            path = os.path.join(prompts_dir, filename)
            with open(path, encoding="utf-8") as f:
                text = f.read().strip()

            # Краткое описание по номеру версии
            descriptions = {
                "prompt_v1": "базовый (наивный) запрос",
                "prompt_v2": "структурированный (Role-Task-Context-Constraints-Output)",
                "prompt_v3": "few-shot + JSON + защита от injection",
            }
            desc = descriptions.get(version_id, f"версия {version_id}")
            versions.append(PromptVersion(version_id, desc, text))

    return versions


def load_tests(path="tests.json"):
    """Загружает тестовый набор."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_results(results, filename="results.csv"):
    """Сохраняет результаты в CSV (дописывает в конец)."""
    if not results:
        print("\nНет результатов для сохранения.")
        return

    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 70)
    print(f"Сохранено {len(results)} записей в файл {filename}")
    print("=" * 70)


# ============== ГЛАВНЫЙ ЗАПУСК ==============

def main():
    print("Загрузка ресурсов проекта...")
    prompts = load_prompts()
    tests = load_tests()
    print(f"  Версий промпта: {len(prompts)}")
    print(f"  Тестов: {len(tests)}")

    client = LLMClient()
    experiment = PromptComparisonExperiment(client, prompts, tests)
    experiment.run()
    save_results(experiment.results)


if __name__ == "__main__":
    main()