import itertools
import sys
import threading
import time
from contextlib import contextmanager

from src import config
from src.ingestion import embedder
from src.rag.engine import RagEngine


@contextmanager
def _spinner(message: str):
    stop = threading.Event()

    def _spin():
        for char in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            if stop.is_set():
                break
            sys.stdout.write(f"\r{char} {message}")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write(f"\r{' ' * (len(message) + 2)}\r")
        sys.stdout.flush()

    t = threading.Thread(target=_spin, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()


def main() -> None:
    engine = RagEngine()
    print(config.CLI_LOADING_TEXT)
    embedder.warm_up()
    history: list[dict] = []
    print(config.CLI_WELCOME)
    while True:
        try:
            question = input(f"\n{config.CLI_USER_PREFIX} : ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{config.CLI_GOODBYE}")
            break
        if question.lower() in config.CLI_EXIT_COMMANDS:
            print(config.CLI_GOODBYE)
            break
        if not question:
            continue
        with _spinner(config.CLI_SPINNER_TEXT):
            result = engine.ask(question, history)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["answer"]})
        print(f"\n{config.CLI_ASSISTANT_PREFIX} : {result['answer']}")
        if result["sources"]:
            print(f"{config.CLI_SOURCES_PREFIX} :")
            for source in result["sources"]:
                print(f"  - {source}")


if __name__ == "__main__":
    main()
