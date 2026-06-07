from src.llm import llm
from src.observability.logging import setup_logging

setup_logging()


def main() -> None:
    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]
    ai_msg = llm.invoke(messages)
    print(ai_msg)


if __name__ == "__main__":
    main()
