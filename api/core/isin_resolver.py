import requests
from typing import Optional

YF_SEARCH = "https://query2.finance.yahoo.com/v1/finance/search"

def _search_yahoo(query: str) -> list[dict]:
    try:
        r = requests.get(YF_SEARCH, params={"q": query, "quotesCount": 10, "newsCount": 0}, timeout=10)
        r.raise_for_status()
        js = r.json()
        return js.get("quotes", []) or []
    except Exception:
        return []

def resolve_isin_to_ticker(isin: str, name_hint: Optional[str] = None) -> Optional[str]:
    """
    Essaye de résoudre un ISIN vers un ticker Yahoo.
    1) Cherche par ISIN
    2) Si name_hint, essaye aussi par nom
    Retourne un ticker Yahoo (ex: OR.PA) ou None si introuvable.
    """
    isin = (isin or "").strip().upper()
    if not isin:
        return None

    # 1) recherche directe par ISIN
    quotes = _search_yahoo(isin)
    # 2) si rien, tentative par nom (si fourni)
    if not quotes and name_hint:
        quotes = _search_yahoo(name_hint)

    # Filtrer des résultats pertinents (actions)
    candidates = []
    for q in quotes:
        # exemples de champs: symbol, shortname, longname, exchDisp, quoteType
        typ = (q.get("quoteType") or "").lower()
        sym = q.get("symbol")
        if not sym:
            continue
        if typ in ("equity", "etf", "mutualfund", "index"):
            candidates.append(q)

    if not candidates:
        return None

    # Heuristique simple: prioriser EQUITY puis marchés Europe/US
    def score(q):
        sc = 0
        if (q.get("quoteType") or "").lower() == "equity":
            sc += 10
        sym = q.get("symbol","")
        # bonus si suffixe Euronext (PA, FP, AS, BR, MI, MC, LS)
        if any(sym.endswith(suf) for suf in (".PA",".FP",".AS",".BR",".MI",".MC",".LS",".DE",".F",".BE",".SW",".VI")):
            sc += 3
        # bonus USA
        if "." not in sym:
            sc += 2
        # bonus si le nom contient l'indice name_hint
        if name_hint:
            nm = (q.get("shortname") or q.get("longname") or "").lower()
            if name_hint.lower() in nm:
                sc += 2
        return -sc  # tri ascendant

    candidates.sort(key=score)
    best = candidates[0]
    return best.get("symbol")
