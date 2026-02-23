from app import mcp

import tools.bars  # noqa: F401
import tools.quotes  # noqa: F401
import tools.trades  # noqa: F401
import tools.snapshots  # noqa: F401
import tools.market_status  # noqa: F401
import tools.assets  # noqa: F401


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
