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


class IndividualTaskRunner:
    """Прогон индивидуального задания: один промпт на фиксированном наборе тестов."""

    def __init__(self, prompt_path, tests_path):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        with open(prompt_path, encoding="utf-8") as f:
            self.system_prompt = f.read().strip()
        with open(tests_path, encoding="utf-8") as f:
            self.tests = json.load(f)
        self.results = []

    def ask(self, user_input):
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_input},
                ],
                temperature=TEMPERATURE,
            )
            elapsed = time.perf_counter() - started
            return response.choices[0].message.content, elapsed, None
        except Exception as e:
            elapsed = time.perf_counter() - started
            return None, elapsed, str(e)

    def run(self):
        print()
        print("=" * 70)
        print("ИНДИВИДУАЛЬНОЕ ЗАДАНИЕ: Классификация обращения клиента")
        print("=" * 70)
        print(f"  Модель: {MODEL}")
        print(f"  Температура: {TEMPERATURE}")
        print(f"  Тестов: {len(self.tests)}")
        print("=" * 70)

        for test in self.tests:
            print()
            print(f"### ТЕСТ {test['id']} ({test['type']})")
            print(SEPARATOR)
            print(f"  Вход: {test['input'][:120]}...")
            answer, latency, error = self.ask(test["input"])

            if error:
                print(f"  ОШИБКА: {error}")
            else:
                print(f"  Время: {latency:.2f} с")
                # Попытка распарсить JSON
                try:
                    parsed = json.loads(answer)
                    print(f"  Категория: {parsed.get('category')}")
                    print(f"  Приоритет: {parsed.get('priority')}")
                    print(f"  Тональность: {parsed.get('sentiment')}")
                    print(f"  Срочно: {parsed.get('requires_immediate_attention')}")
                    print(f"  Injection: {parsed.get('injection_attempt')}")
                    print(f"  Уверенность: {parsed.get('confidence')}")
                    json_valid = True
                except json.JSONDecodeError:
                    print(f"  ⚠ JSON не распарсился, сырой ответ:")
                    print(f"  {answer[:300]}")
                    json_valid = False

            self.results.append({
                "test_id": test["id"],
                "test_type": test["type"],
                "input": test["input"],
                "latency": round(latency, 3),
                "answer": answer,
                "json_valid": json_valid if not error else None,
                "error": error,
            })

        # Сохранение
        filename = "results_individual.csv"
        with open(filename, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)

        print()
        print("=" * 70)
        print(f"Сохранено {len(self.results)} записей в {filename}")
        print("=" * 70)


if __name__ == "__main__":
    runner = IndividualTaskRunner(
        prompt_path="prompts/prompt_v4.txt",   
        tests_path="tests_individual.json",
    )
    runner.run()