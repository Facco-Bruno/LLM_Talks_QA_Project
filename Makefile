.PHONY: ingest app dash eval-retrieval eval-llm eval-metrics

ingest:
	python app/ingest.py

app:
	streamlit run app/main.py

dash:
	streamlit run dashboard.py

eval-retrieval:
	python app/evaluate_retrieval.py

eval-llm:
	python app/evaluate_llm.py

eval-metrics:
	python app/evaluate_llm_metrics.py
