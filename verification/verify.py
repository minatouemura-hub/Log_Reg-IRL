import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from verification.utils import StaticalVerify  # noqa


def main():
    json_files = [
        "/Users/uemuraminato/Desktop/IRL/plot/M_gr_list.json",
        "/Users/uemuraminato/Desktop/IRL/plot/F_gr_list.json",
    ]
    ver = StaticalVerify(json_files=json_files)
    ver.execute()


if __name__ == "__main__":
    main()
