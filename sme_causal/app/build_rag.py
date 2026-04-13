from sme_causal.rag.rag_pipeline import RAG

# ===================== Пример использования =====================
if __name__ == "__main__":
    rag = RAG()
    # Полный прогон
    print("start RAG pipeline...")
    rag.run_rag_pipeline(use_metadata=True)

    # Тестовый поиск
    results = rag.perform_query("Как быстро проверить спрос на продукт?", top_k=3)
    for i, txt in enumerate(results, 1):
        print(f"\n--- TOP {i} ---\n{txt[:600]}...")
