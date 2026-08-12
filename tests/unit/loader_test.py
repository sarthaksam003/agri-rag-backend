from carag.services.document_ingestion.mineruloader import MineruLoader

loader = MineruLoader("./data/raw/Harolds_Physics_Formulas_Cheat_Sheet.pdf")
info = loader.load()
print(info)
