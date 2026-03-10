Pri tvorbe referenčného Excel súboru (Ground Truth) neuplatňujem stratégiu indexovania každého výskytu kľúčového slova (napr. „zmluvná pokuta“). Cieľom je identifikovať výhradne informačné kotvy.

Signál (Relevantný kontext): Odseky obsahujúce meritórne rozhodnutie súdu, konkrétne sumy, percentuálne sadzby a právnu argumentáciu (napr. výrok o priznaní nároku).

Šum (Irelevantný kontext): Procesné vyjadrenia strán, rekapitulácia podaní alebo opakovane spomínané požiadavky žalobcu, ktoré neobsahujú finálnu hodnotu potrebnú pre validáciu JSON výstupu.

Tento prístup zabezpečuje, že model nebude penalizovaný za vynechanie redundantných informácií a zároveň minimalizuje spotrebu tokenov pri zachovaní 100% úspešnosti extrakcie.