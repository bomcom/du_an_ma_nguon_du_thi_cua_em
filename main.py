# -*- coding: utf-8 -*-

import os
import sys
import logging

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.chdir(ROOT_DIR)

from orchestrator import HybridSimulationApplication
from graphics.world_view import WorldView

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger("MainLauncher")


def main():

    logger.info("=" * 60)
    logger.info("LHU INTERDISCIPLINARY WORLD ENGINE")
    logger.info("=" * 60)

    app = HybridSimulationApplication()

    app.start()

    view = WorldView(
        width=1400,
        height=900,
        grid_size=16
    )

    try:

        view.run(app)

    except KeyboardInterrupt:

        logger.info("CTRL+C received")

    except Exception:

        logger.exception("Fatal error")

    finally:

        app.stop()

        logger.info("Shutdown complete")

        sys.exit(0)


if __name__ == "__main__":
    main()
    