from __future__ import annotations

import os

from redis import Redis
from rq import Queue, Worker


def main() -> None:
    connection = Redis.from_url(
        os.getenv("AUTOEXP_REDIS_URL", "redis://localhost:6379/0")
    )
    Worker([Queue("autoexp", connection=connection)], connection=connection).work(
        with_scheduler=True
    )


if __name__ == "__main__":
    main()
