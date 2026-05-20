# main.py

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor  # [web:2][web:7]
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool  # [web:4]

# 1. Фейковый каталог автозапчастей
AUTO_PARTS_CATALOG = [
    {
        "brand": "Toyota",
        "model": "Corolla",
        "year": 2015,
        "part_name": "Тормозные колодки передние",
        "sku": "TOY-COR-2015-BRK-FR",
        "price": 50,
    },
    {
        "brand": "Toyota",
        "model": "Camry",
        "year": 2018,
        "part_name": "Масляный фильтр",
        "sku": "TOY-CAM-2018-OIL-FLT",
        "price": 15,
    },
    {
        "brand": "BMW",
        "model": "3 Series",
        "year": 2019,
        "part_name": "Воздушный фильтр",
        "sku": "BMW-3S-2019-AIR-FLT",
        "price": 35,
    },
]


# 2. Инструмент для поиска запчасти
@tool
def find_part_in_catalog(brand: str, model: str, year: int, query: str) -> dict:
    """
    Ищет подходящую автозапчасть в каталоге по марке, модели, году выпуска и текстовому описанию.
    Возвращает найденную позицию или сообщение, что ничего не найдено.
    """
    query_lower = query.lower()
    for item in AUTO_PARTS_CATALOG:
        if (
            item["brand"].lower() == brand.lower()
            and item["model"].lower() == model.lower()
            and item["year"] == year
            and any(word in item["part_name"].lower() for word in query_lower.split())
        ):
            return {
                "found": True,
                "brand": item["brand"],
                "model": item["model"],
                "year": item["year"],
                "part_name": item["part_name"],
                "sku": item["sku"],
                "price": item["price"],
            }
    return {
        "found": False,
        "message": "Подходящая запчасть не найдена в каталоге. Попробуйте уточнить запрос."
    }


def build_agent() -> AgentExecutor:
    # 3. Настраиваем LLM (можешь заменить на Ollama-модель)
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # или другая модель
        temperature=0.2,
    )

    tools = [find_part_in_catalog]

    # 4. Промпт агента
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Ты помощник интернет-магазина автозапчастей. "
                "Твоя задача — помогать подбирать запчасти по описанию пользователя. "
                "Всегда сначала уточняй марку, модель и год автомобиля, если они не указаны. "
                "Если запрос понятен, используй доступные инструменты для поиска деталей.",
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # 5. Создаём агента и executor [web:7]
    agent = create_tool_calling_agent(llm, tools, prompt)  # [web:5][web:7]
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # удобно для ДЗ — видно шаги.
    )
    return agent_executor


def main():
    agent_executor = build_agent()

    print("Агент автозапчастей запущен. Напишите свой запрос.")
    while True:
        user_input = input("Пользователь: ")
        if user_input.lower() in {"выход", "exit", "quit"}:
            print("Завершение работы.")
            break

        result = agent_executor.invoke({"input": user_input})
        print("Агент:", result["output"])


if __name__ == "__main__":
    main()