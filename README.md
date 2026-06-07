# Workflow-CI

Repository untuk kriteria 3: MLflow Project + GitHub Actions CI.

## Struktur
```text
MLProject/
.github/workflows/mlflow-ci.yml
```

## Secrets GitHub
Tambahkan secrets berikut untuk push Docker image:

```text
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
```

## Run lokal
```bash
cd MLProject
mlflow run . -P n_estimators=150 -P max_depth=10
```
