from dotenv import load_dotenv
load_dotenv()

from jp_morgraph.agents.technical import graph
from jp_morgraph.state import AnalystReport

N = 10
valides = 0

for i in range(N):
    try:
        out = graph.invoke({"messages": [("user", "Analyse AAPL au 2026-10-09")]})
        assert isinstance(out["report"], AnalystReport)
        valides += 1
    except Exception as e:
        print(f"Essai {i + 1} : échec ({type(e).__name__}: {e})")

print(f"{valides}/{N} sorties valides")