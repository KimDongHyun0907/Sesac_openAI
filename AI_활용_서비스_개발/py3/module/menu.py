from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import sysInfoTitle

python_title_printer = sysInfoTitle.PythonTitlePrinter()
python_title_printer.sysInfo()