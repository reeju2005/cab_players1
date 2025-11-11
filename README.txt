# CAB Prediction Studio (Streamlit)
Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
Place your CSV files as: `CAB - TEAMNAME.csv` in the working directory.
The app auto-detects:
- Player column (by name patterns)
- Season/Year (if present)
- Numeric metrics
Then it lets you:
- Rank **Team of the Year** by top-k composite score (weights are adjustable)
- Predict **next-season** performance (trend per player if season data exists, else baseline)
- List **Players to Watch**
