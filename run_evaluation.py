from evaluation import evaluate_rag_pipeline

OUTPUT_PATH = "inputs_and_outputs/output.json"
BENCHMARK_PATH = "data/latest_benchmark.json"


def main():
    evaluate_rag_pipeline(OUTPUT_PATH, BENCHMARK_PATH)

if __name__ == "__main__":
    main()
