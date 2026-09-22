#!/bin/zsh
# Generate all four client reports on the current pipeline.
cd "$(dirname "$0")"
bal() { curl -s https://api.deepseek.com/user/balance -H "Authorization: Bearer $(grep -m1 DEEPSEEK_API_KEY .env | cut -d= -f2)" 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['balance_infos'][0]['total_balance'])" 2>/dev/null || echo "?"; }
OUT=${1:-wwrag/runs/final6}
mkdir -p $OUT
run() {
  echo "--- $2 (balance $(bal)) ---"
  .venv-crawl4ai/bin/python wwrag/run.py --resume "inbox/$1" --college usc \
    --out-dir $OUT/$2 --index-root wwrag/index-v3 --skip-index --per-category 24 \
    > $OUT/$2.log 2>&1
  echo "    $2 rc=$?"
}
run "Aashrut Almal CV.pdf"          aashrut_almal  &
run "Aditya Eduworks - Resume .pdf" aditya_khaitan &
wait
run "Aadya Aggarwal x Eduworks.pdf" aadya_aggarwal &
run "Aadya Saha Résumé.pdf"         aadya_saha     &
wait
echo "ALL FOUR COMPLETE, balance $(bal) USD"
ls -la $OUT/*/report.pdf 2>/dev/null
