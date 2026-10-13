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


# =========================================================================
# Пояснение: в LM Studio часто включён "Override System Prompt", который
# затирает системные инструкции из API. Чтобы обойти это, мы ДУБЛИРУЕМ
# инструкцию и в system, и в user-сообщение. Модель гарантированно её видит.
# =========================================================================


class IndividualTaskRunner:
    def __init__(self, prompt_path, tests_path):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        with open(prompt_path, encoding="utf-8") as f:
            self.system_prompt = f.read().strip()
        with open(tests_path, encoding="utf-8") as f:
            self.tests = json.load(f)
        self.results = []

    def _build_messages(self, user_input):
        """Дублирует инструкцию в user-message для обхода Override System Prompt."""
        combined_user = (
            f"### ИНСТРУКЦИЯ (соблюдай строго):\n{self.system_prompt}\n\n"
            f"### ОБРАЩЕНИЕ КЛИЕНТА:\n{user_input}\n\n"
            f"### ТВОЙ ОТВЕТ (только JSON, без пояснений):"
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": combined_user},
        ]

    def ask(self, user_input):
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=self._build_messages(user_input),
                temperature=TEMPERATURE,
            )
            elapsed = time.perf_counter() - started
            return response.choices[0].message.content, elapsed, None
        except Exception as e:
            elapsed = time.perf_counter() - started
            return None, elapsed, str(e)

    @staticmethod
    def _extract_json(text):
        """Пытается найти JSON в ответе, даже если модель обернула его в текст."""
        if not text:
            return None
        text = text.strip()
        # Убираем markdown-обёртки ```json ... ```
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()
        # Ищем первый { и последний }
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    def run(self):
        print()
        print("=" * 70)
        print("ИНДИВИДУАЛЬНОЕ ЗАДАНИЕ: Классификация обращения клиента")
        print("=" * 70)
        print(f"  Модель:        {MODEL}")
        print(f"  Температура:   {TEMPERATURE}")
        print(f"  Промпт:        {os.path.basename('prompt_v4.txt')}")
        print(f"  Тестов:        {len(self.tests)}")
        print("=" * 70)

        for test in self.tests:
            print()
            print(f"### ТЕСТ {test['id']} ({test['type']})")
            print(SEPARATOR)
            print(f"  Вход: {test['input'][:100]}...")

            answer, latency, error = self.ask(test["input"])

            parsed = None
            json_valid = False

            if error:
                print(f"  ОШИБКА API: {error}")
            else:
                print(f"  Время: {latency:.2f} с")
                parsed = self._extract_json(answer)
                if parsed:
                    json_valid = True
                    print(f"  ✓ JSON распарсен")
                    print(f"    category:    {parsed.get('category')}")
                    print(f"    priority:    {parsed.get('priority')}")
                    print(f"    sentiment:   {parsed.get('sentiment')}")
                    print(f"    attention:   {parsed.get('requires_immediate_attention')}")
                    print(f"    injection:   {parsed.get('injection_attempt')}")
                    print(f"    confidence:  {parsed.get('confidence')}")
                else:
                    print(f"  ✗ JSON не распарсен. Сырой ответ (первые 200 симв.):")
                    print(f"    {answer[:200] if answer else 'None'}")

            self.results.append({
                "test_id": test["id"],
                "test_type": test["type"],
                "input": test["input"],
                "latency": round(latency, 3),
                "answer": answer,
                "json_valid": json_valid,
                "category": parsed.get("category") if parsed else None,
                "priority": parsed.get("priority") if parsed else None,
                "sentiment": parsed.get("sentiment") if parsed else None,
                "injection_attempt": parsed.get("injection_attempt") if parsed else None,
                "requires_immediate_attention": parsed.get("requires_immediate_attention") if parsed else None,
                "error": error,
            })

        # Сохранение
        filename = "results_individual.csv"
        with open(filename, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)

        # Краткая сводка
        total = len(self.results)
        valid = sum(1 for r in self.results if r["json_valid"])
        print()
        print("=" * 70)
        print(f"ИТОГО: {valid}/{total} ответов — валидный JSON")
        print(f"Сохранено в {filename}")
        print("=" * 70)


if __name__ == "__main__":
    runner = IndividualTaskRunner(
        prompt_path="prompts/prompt_v4.txt",
        tests_path="tests_individual.json",
    )
    runner.run()