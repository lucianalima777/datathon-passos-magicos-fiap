# Previsão de Risco de Defasagem Escolar - Passos Mágicos

## Visão Geral

Modelo preditivo para estimar o **risco de defasagem escolar** de estudantes da Associação Passos Mágicos, permitindo intervenções pedagógicas preventivas.

**Problema:** Classificação binária - prever se um aluno cairá em defasagem escolar (`RISCO_DEFASAGEM = 1` se `DEFASAGEM < 0`).

**Métrica prioritária:** Recall (evitar falsos negativos mas não deixar alunos em risco passarem despercebidos).

**Resultado:** Logistic Regression com threshold otimizado (0.35) alcançando **recall de 92.9%** no teste, com precision de 69.0%. O sistema também agrupa alunos em **4 perfis pedagógicos** via K-Means e gera relatórios interpretativos via LLM usando a API do DeepSeek.

**Por que o modelo é confiável para produção:**
O threshold foi selecionado via validação cruzada stratificada (5-fold OOF) e avaliado uma única vez no conjunto de teste — garantindo que o resultado não está inflado por overfitting de avaliação. O gap entre holdout de validação (recall 86.9%) e teste (recall 92.9%) é pequeno e favorável, indicando boa capacidade de generalização. A escolha por Logistic Regression é deliberada: com ~410 amostras de treino, modelos mais complexos (Random Forest, Gradient Boosting) apresentaram overfitting expressivo nos experimentos (ver `notebooks/02_selecao_modelos.ipynb`), enquanto o modelo linear generalizou de forma estável. O desbalanceamento de classes (~61% em risco) foi tratado via `class_weight='balanced'`, sem necessidade de oversampling artificial.

---

## Stack Tecnológica

- ML | scikit-learn (Logistic Regression + K-Means), pandas, numpy 
- LLM | DeepSeek API (openai-compatible)
- API | FastAPI, Pydantic v2, uvicorn 
- Monitoring | SQLite (prediction store), loguru (logs), scipy (KS-test + PSI) 
- Frontend | Streamlit 
- Deploy | Docker, Render 
- CI/CD | GitHub Actions (pytest + lint) 
- Testes | pytest, httpx 

## Estrutura do Projeto

O código do backend fica em `backend/src/`, dividido em quatro módulos: `data/` (carga e pré-processamento), `features/` (engenharia de atributos), `models/` (treino, avaliação e clustering) e `monitoring/` (logs, store SQLite e drift). Os scripts de treinamento ficam em `backend/scripts/` e os testes em `backend/tests/`. O frontend Streamlit está em `frontend/app.py`. Artefatos do modelo (`.joblib` e `.json`) são salvos em `backend/data/models/`.

---

## Setup e Execução

### Pré-requisitos

- Python 3.11+
- pip
- (Opcional) Docker 24+
- Dataset em `../dados-datathon/DATATHON/Bases antigas/PEDE_PASSOS_DATASET_FIAP.csv`

### Instalação

```bash
cd passos-magicos-risco-escolar
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Preparar Dados e Treinar Modelos

```bash
python backend/scripts/prepare_data.py
python backend/scripts/train_model.py
python backend/scripts/cluster_students.py
```

### Executar API

```bash
uvicorn backend.src.api.main:app --reload --host 0.0.0.0 --port 8000
# Docs: http://localhost:8000/docs
```

### Executar Frontend

```bash
cd frontend && streamlit run app.py
```

### Testes

```bash
pytest backend/tests/ -v
pytest backend/tests/ --cov=backend/src --cov-report=html
```

### Docker

```bash
docker build -t passos-magicos-api .
docker run -p 8000:8000 passos-magicos-api
```

### Deploy e Artefatos do Modelo

1. Gere os artefatos do modelo:

```bash
python backend/scripts/prepare_data.py
python backend/scripts/train_model.py
```

2. Verifique se existem os arquivos em `backend/data/models/`:
   - `baseline_logreg.joblib` + `baseline_logreg_metrics.json` (classificação)
   - `kmeans_profiles.joblib` (clustering)
3. Para cloud (Render), garanta que esses artefatos estejam presentes no build (recomendado commitar o baseline).
4. Se trocar o modelo de classificação, atualize `MODEL_NAME` em `backend/src/config.py`.
5. Configure a variável de ambiente `DEEPSEEK_API_KEY` para habilitar o assistente pedagógico LLM.

---

## Exemplos de Chamadas à API

### 1) Predição de risco (`POST /predict`)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "INDE_2020": 6.5,
    "IAA_2020": 6.0,
    "IEG_2020": 5.0,
    "IPS_2020": 6.0,
    "IDA_2020": 6.0,
    "IPP_2020": 6.0,
    "IPV_2020": 5.0,
    "IAN_2020": 6.0,
    "IDADE_ALUNO_2020": 12,
    "ANOS_PM_2020": 2,
    "PEDRA_2020": "Agata",
    "PONTO_VIRADA_2020": "Sim",
    "INSTITUICAO_ENSINO_ALUNO_2020": "Publica",
    "FASE_TURMA_2020": "Fase 1 - Turma A"
  }'
```

Exemplo de resposta:

```json
{
  "risco_defasagem": 1,
  "probabilidade": 0.8234,
  "threshold": 0.35,
  "classificacao": "Em Risco"
}
```

### 2) Health check (`GET /health`)

```bash
curl http://localhost:8000/health
```

Exemplo de resposta:

```json
{
  "status": "ok",
  "model_name": "baseline_logreg",
  "model_version": "a1b2c3d4e5f6",
  "total_predictions": 42
}
```

### 3) Histórico de predições (`GET /predictions`)

```bash
curl "http://localhost:8000/predictions?limit=5"
```

### 4) Perfil pedagógico por clustering (`POST /cluster`)

```bash
curl -X POST http://localhost:8000/cluster \
  -H "Content-Type: application/json" \
  -d '{
    "INDE_2020": 6.5,
    "IAA_2020": 6.0,
    "IEG_2020": 5.0,
    "IPS_2020": 6.0,
    "IDA_2020": 6.0,
    "IPP_2020": 6.0,
    "IPV_2020": 5.0,
    "IAN_2020": 6.0,
    "IDADE_ALUNO_2020": 12,
    "ANOS_PM_2020": 2,
    "PEDRA_2020": "Agata",
    "PONTO_VIRADA_2020": "Sim",
    "INSTITUICAO_ENSINO_ALUNO_2020": "Publica",
    "FASE_TURMA_2020": "Fase 1 - Turma A"
  }'
```

```

### 5) Relatório pedagógico via LLM (`POST /explain`)

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{ ... mesmos campos do /predict ... }'
```

Requer `DEEPSEEK_API_KEY` configurada no ambiente.

### 6) Drift detection (`GET /drift`)

```bash
curl "http://localhost:8000/drift?n=50"
```

---

## Pipeline de ML

### 1. Pré-processamento (`preprocessor.py`)
- Conversão de tipos numéricos
- Remoção de linhas sem DEFASAGEM_2021
- Criação do target binário RISCO_DEFASAGEM
- Imputacao de ausentes delegada ao Pipeline sklearn (mediana aprendida apenas no treino)

### 2. Feature Engineering (`engineer.py`)
- **8 indicadores** originais (INDE, IAA, IEG, IPS, IDA, IPP, IPV, IAN)
- **4 agregações** (média, std, min, max dos indicadores)
- **3 ratios** (IDA/IAN, IEG/IPS, INDE-média)
- **4 categóricas normalizadas** (PEDRA, PONTO_VIRADA, INSTITUICAO, FASE_TURMA)
- **4 demográficas** (idade, anos_pm, faixa_etaria, ratio_anos_pm_idade)
- Total: **23 features**

### 3. Treinamento (`trainer.py`)
- Split: 60% treino, 20% validação, 20% teste (estratificado)
- Pipeline sklearn: SimpleImputer + StandardScaler (num) + OneHotEncoder (cat) + Classifier
- Threshold tuning: maximiza recall com min_precision=0.7

### 4. Avaliação (`evaluator.py`)
- Métricas: accuracy, precision, recall, F1, AUC-ROC, AUC-PR
- Threshold otimizado: **0.35** via OOF 5-fold CV (recall=86.9% val, 92.9% teste)

### 5. Clustering (`clusterer.py`)
- K-Means com 4 clusters sobre os 8 indicadores normalizados (StandardScaler)
- Perfis ordenados pela média dos centroides: `risco_alto`, `risco_moderado`, `em_desenvolvimento`, `alto_desempenho`
- Distribuição no dataset de treino (457 amostras): alto_desempenho 174, em_desenvolvimento 157, risco_alto 92, risco_moderado 34
- Artefato: `kmeans_profiles.joblib` (KMeans + StandardScaler + label_map)

### 6. Assistente Pedagógico (DeepSeek LLM)
- **POST /explain** — gera relatório pedagógico individualizado (pontos fortes, áreas de atenção, 3 recomendações)
- Chat de acompanhamento via Streamlit com streaming direto à API DeepSeek
- Contexto do aluno (indicadores + predição de risco + perfil de cluster) injetado no system prompt

### 7. Serving e Monitoring
- **POST /predict** - aceita 13 campos brutos, aplica feature engineering, retorna predição
- **POST /cluster** - classifica o aluno em um dos 4 perfis pedagógicos
- **POST /explain** - gera relatório pedagógico via LLM (requer DEEPSEEK_API_KEY)
- **GET /health** - status da API
- **GET /predictions** - histórico de predições (SQLite)
- **GET /drift** - drift detection via KS-test e PSI (scipy)

---

## Design do Frontend

O frontend foi desenvolvido considerando dois públicos simultâneos:

**Equipe pedagógica da Passos Mágicos (usuário final)**
- Campos com nome completo e sigla entre parênteses (ex.: "Aprendizagem (IDA)") - sem exigir conhecimento prévio dos acrônimos
- Glossário embutido (colapsável) explicando as Pedras, o Ponto de Virada e cada indicador
- Resultado em linguagem direta: "Alto Risco de Defasagem Escolar" com probabilidade em percentual
- Perfil pedagógico do aluno via clustering: `Risco Alto`, `Risco Moderado`, `Em Desenvolvimento` ou `Alto Desempenho`, com descrição interpretável e distâncias para cada centroide
- Relatório pedagógico automático gerado por LLM (pontos fortes, áreas de atenção, recomendações)
- Chat de acompanhamento com assistente pedagógico especializado (contexto do aluno injetado automaticamente)
- Aba "Saúde do Modelo" com explicação acessível sobre monitoramento de drift
- Detalhes técnicos disponíveis via seção expansível, não expostos por padrão

**Avaliador técnico**
- Threshold visível nos detalhes técnicos - evidencia o ajuste de ponto de corte e o tradeoff precision-recall
- Aba de monitoramento preservada - demonstra drift detection em produção via KS-test e PSI
- Probabilidade bruta e parâmetros do modelo acessíveis para auditoria
- Histórico completo de predições consultável com filtro de quantidade

Essa decisão reflete uma prática real de ML Engineering: sistemas em produção precisam ser
interpretáveis tanto para o time de negócio quanto para o time técnico responsável pela manutenção.

---

## Modelo em Produção

### Ciclo de vida e retreinamento

O dataset da Passos Mágicos é anual: os indicadores de um ano são usados para prever a defasagem do ano seguinte. Portanto, o modelo é retreinado uma vez por ano quando os dados da nova coorte ficam disponíveis. O processo é o mesmo pipeline já existente:

```bash
python backend/scripts/prepare_data.py   # processa novo dataset
python backend/scripts/train_model.py    # retreina e salva novos artefatos
python backend/scripts/cluster_students.py
```

O retreinamento também pode ser antecipado se o monitoramento de drift indicar degradação relevante antes do ciclo anual (ver critérios abaixo).

### Contratos do modelo

| Contrato | Valor | Como é garantido |
|---|---|---|
| Recall mínimo | ≥ 80% | Threshold selecionado via OOF 5-fold com `min_precision=0.7`; recall no teste: 92.9% |
| Precisão mínima | ≥ 70% | Restrição explícita no tuning de threshold |
| Threshold em produção | 0.35 | Carregado do artefato `baseline_logreg_metrics.json` — não hardcoded |
| Versão do modelo | hash SHA-256 (12 chars) | Computado no startup da API, exposto em `/health` |
| Auditabilidade | 100% das predições | Todas as chamadas a `/predict` são salvas em SQLite com timestamp, inputs, probabilidade e threshold |

A prioridade por recall reflete o custo assimétrico do problema: um falso negativo (aluno em risco classificado como adequado) tem consequência pedagógica real; um falso positivo implica apenas uma atenção desnecessária.

### Monitoramento de drift e critérios de ação

O endpoint `GET /drift` compara os inputs recentes contra a distribuição de treino nos 8 indicadores numéricos principais usando **KS-test** (p-valor) e **PSI** (Population Stability Index).

| Sinal | Critério | Ação recomendada |
|---|---|---|
| Estável | PSI < 0.1 e KS p-valor ≥ 0.05 em todos os indicadores | Nenhuma |
| Alerta | PSI entre 0.1 e 0.2 em qualquer indicador | Aumentar frequência de monitoramento |
| Crítico | PSI ≥ 0.2 ou `dataset_drift: true` (>50% dos indicadores driftados) | Retreinar o modelo com dados mais recentes |

O drift é calculado a partir de no mínimo 30 predições acumuladas no banco. Abaixo desse volume, o endpoint retorna `status: insufficient_data`.

---

## Entregáveis

- Link Github: https://github.com/lucianalima777/datathon-passos-magicos-fiap
- Link da API: https://passos-magicos-api.onrender.com/
- Link Documentação API: https://passos-magicos-api.onrender.com/docs
- Link do Frontend: https://passos-magicos-frontend.onrender.com/
- Link do vídeo do Youtube - Apresentação: https://youtu.be/2iSA6N8Wea0


---

