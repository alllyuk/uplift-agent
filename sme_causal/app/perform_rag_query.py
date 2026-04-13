from sme_causal.rag.rag_pipeline import RAG

if __name__ == "__main__":
    rag = RAG()

    # Тестовый поиск
    results = rag.perform_query("Как повлияет повышение кредитного лимита для клиента на вероятность роста оборотов?", top_k=3)
    for i, txt in enumerate(results, 1):
        print(f"\n--- TOP {i} ---\n{txt}...")
