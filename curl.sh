# curl -X POST http://127.0.0.1:8000/query \
#   -H "Content-Type: application/json" \
#   -d '{"query": "What is the goal of this plan?"}'

# curl -X POST http://127.0.0.1:8000/query   -H "Content-Type: application/json"   -d '{"query": "What is the goal of this plan?"}'

curl -X POST http://127.0.0.1:8000/ingest \
  -H Authorization:"Bearer qwerty-asdf" \
  -F "file=@./PLAN.md" \
  -F "title=Plan"