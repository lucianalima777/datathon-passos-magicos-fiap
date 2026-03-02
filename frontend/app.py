import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
if not API_URL.startswith("http"):
    API_URL = f"https://{API_URL}"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

st.set_page_config(page_title="Passos Mágicos - Risco Escolar", layout="wide")
st.title("Passos Mágicos - Previsão de Risco de Defasagem Escolar")
st.markdown(
    "Ferramenta de apoio pedagógico para identificação preventiva de alunos em situação de risco. "
    "Preencha os indicadores do aluno referentes ao ano base para obter a avaliação."
)

for key, default in [
    ("prediction_result", None),
    ("prediction_payload", None),
    ("cluster_result", None),
    ("relatorio", None),
    ("chat_messages", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def _build_chat_system_prompt(payload: dict, result: dict, cluster: dict | None = None) -> str:
    PERFIL_LABELS = {
        "risco_alto": "Risco Alto",
        "risco_moderado": "Risco Moderado",
        "em_desenvolvimento": "Em Desenvolvimento",
        "alto_desempenho": "Alto Desempenho",
    }
    cluster_info = ""
    if cluster:
        perfil_label = PERFIL_LABELS.get(cluster.get("perfil", ""), cluster.get("perfil", "N/A"))
        cluster_info = f"- Perfil de grupo (clustering): {perfil_label}\n"

    return (
        "Você é um assistente pedagógico especialista da Associação Passos Mágicos, "
        "uma ONG de transformação social que apoia crianças e jovens em situação de vulnerabilidade "
        "através da educação de qualidade.\n\n"
        "DADOS DO ALUNO AVALIADO:\n"
        f"- INDE (Desenvolvimento Educacional): {payload.get('INDE_2020', 0):.1f}/10\n"
        f"- IAA (Autoavaliação): {payload.get('IAA_2020', 0):.1f}/10\n"
        f"- IEG (Engajamento): {payload.get('IEG_2020', 0):.1f}/10\n"
        f"- IPS (Psicossocial): {payload.get('IPS_2020', 0):.1f}/10\n"
        f"- IDA (Aprendizagem): {payload.get('IDA_2020', 0):.1f}/10\n"
        f"- IPP (Psicopedagógico): {payload.get('IPP_2020', 0):.1f}/10\n"
        f"- IPV (Ponto de Virada): {payload.get('IPV_2020', 0):.1f}/10\n"
        f"- IAN (Adequação ao Nível): {payload.get('IAN_2020', 0):.1f}/10\n"
        f"- Idade: {payload.get('IDADE_ALUNO_2020', 'N/A')} anos | "
        f"Anos na Passos Mágicos: {payload.get('ANOS_PM_2020', 'N/A')}\n"
        f"- Pedra: {payload.get('PEDRA_2020', 'N/A')} | "
        f"Ponto de Virada: {payload.get('PONTO_VIRADA_2020', 'N/A')}\n"
        f"- Instituição: {payload.get('INSTITUICAO_ENSINO_ALUNO_2020', 'N/A')}\n\n"
        "AVALIAÇÃO PREDITIVA:\n"
        f"- Status: {result.get('classificacao', 'N/A')}\n"
        f"- Probabilidade de risco: {result.get('probabilidade', 0):.0%}\n"
        f"{cluster_info}\n"
        "Você está conversando com um professor ou coordenador pedagógico.\n"
        "Seja empático, construtivo e prático. Evite jargões técnicos de machine learning.\n"
        "Foque em recomendações pedagógicas concretas e acionáveis."
    )


def _stream_deepseek(messages: list, api_key: str):
    from openai import OpenAI as OpenAIClient

    client_ai = OpenAIClient(api_key=api_key, base_url="https://api.deepseek.com")
    stream = client_ai.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=True,
        max_tokens=700,
        temperature=0.7,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


tab_predict, tab_history, tab_drift = st.tabs(
    ["Avaliar Aluno", "Consultas Anteriores", "Saúde do Modelo"]
)

with tab_predict:
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Indicadores de Desempenho")
            inde = st.number_input(
                "Índice de Desenvolvimento Educacional (INDE)",
                min_value=0.0, max_value=10.0, value=5.0, step=0.1,
            )
            iaa = st.number_input(
                "Autoavaliação do Aluno (IAA)",
                min_value=0.0, max_value=10.0, value=5.0, step=0.1,
            )
            ieg = st.number_input(
                "Engajamento (IEG)",
                min_value=0.0, max_value=10.0, value=5.0, step=0.1,
            )
            ips = st.number_input(
                "Indicador Psicossocial (IPS)",
                min_value=0.0, max_value=10.0, value=5.0, step=0.1,
            )

        with col2:
            st.subheader("Indicadores de Desempenho (cont.)")
            ida = st.number_input(
                "Aprendizagem (IDA)",
                min_value=0.0, max_value=10.0, value=5.0, step=0.1,
            )
            ipp = st.number_input(
                "Indicador Psicopedagógico (IPP)",
                min_value=0.0, max_value=10.0, value=5.0, step=0.1,
            )
            ipv = st.number_input(
                "Ponto de Virada (IPV)",
                min_value=0.0, max_value=10.0, value=5.0, step=0.1,
            )
            ian = st.number_input(
                "Adequação ao Nível (IAN)",
                min_value=0.0, max_value=10.0, value=5.0, step=0.1,
            )

        with col3:
            st.subheader("Dados do Aluno")
            idade = st.number_input("Idade", min_value=5, max_value=25, value=12)
            anos_pm = st.number_input("Anos na Passos Mágicos", min_value=0, max_value=15, value=2)
            pedra = st.selectbox(
                "Classificação (Pedra)", ["Quartzo", "Ágata", "Ametista", "Topázio"]
            )
            ponto_virada = st.selectbox("Atingiu Ponto de Virada?", ["Sim", "Não"])
            instituicao = st.selectbox(
                "Instituição de Ensino",
                ["Escola Pública", "Rede Decisão", "FIAP", "UNISA", "Estácio"],
            )

        submitted = st.form_submit_button("Analisar Aluno", use_container_width=True)

    with st.expander("Glossário de Termos"):
        st.markdown(
            """
**Classificação por Pedra**
A Passos Mágicos classifica os alunos em quatro níveis de desenvolvimento, do menor para o maior:
- **Quartzo** - fase inicial de desenvolvimento
- **Ágata** - desenvolvimento em progresso
- **Ametista** - bom desenvolvimento, acima da média
- **Topázio** - nível mais avançado de desenvolvimento

**Ponto de Virada**
Momento em que o aluno demonstra uma mudança significativa de postura e engajamento,
indicando que internalizou os valores e o propósito da Passos Mágicos.

**Sobre os Indicadores (escala de 0 a 10)**
- **INDE** - Índice geral de desenvolvimento educacional (indicador sintético)
- **IAA** - Como o aluno avalia seu próprio desempenho e evolução
- **IEG** - Nível de engajamento com as atividades e tarefas
- **IPS** - Aspectos de bem-estar e suporte psicossocial
- **IDA** - Desempenho e evolução na aprendizagem acadêmica
- **IPP** - Avaliação psicopedagógica do aluno
- **IPV** - Indicador relacionado ao Ponto de Virada
- **IAN** - Adequação do aluno ao nível educacional esperado para sua idade
            """
        )

    if submitted:
        payload = {
            "INDE_2020": inde,
            "IAA_2020": iaa,
            "IEG_2020": ieg,
            "IPS_2020": ips,
            "IDA_2020": ida,
            "IPP_2020": ipp,
            "IPV_2020": ipv,
            "IAN_2020": ian,
            "IDADE_ALUNO_2020": idade,
            "ANOS_PM_2020": anos_pm,
            "PEDRA_2020": pedra,
            "PONTO_VIRADA_2020": ponto_virada,
            "INSTITUICAO_ENSINO_ALUNO_2020": instituicao,
        }
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            st.session_state.prediction_result = response.json()
            st.session_state.prediction_payload = payload
            st.session_state.cluster_result = None
            st.session_state.chat_messages = []
            st.session_state.relatorio = None

            try:
                cluster_resp = requests.post(f"{API_URL}/cluster", json=payload, timeout=10)
                if cluster_resp.status_code == 200:
                    st.session_state.cluster_result = cluster_resp.json()
            except Exception:
                pass

            if DEEPSEEK_API_KEY:
                with st.spinner("Gerando análise pedagógica..."):
                    try:
                        explain_resp = requests.post(
                            f"{API_URL}/explain", json=payload, timeout=45
                        )
                        if explain_resp.status_code == 200:
                            st.session_state.relatorio = explain_resp.json().get("relatorio")
                    except Exception:
                        pass

        except requests.exceptions.ConnectionError:
            st.error("Erro: API não disponível. Verifique se o servidor está rodando.")
        except requests.exceptions.RequestException as e:
            st.error(f"Erro na requisição: {e}")

    # Renderiza resultados persistidos no session_state
    if st.session_state.prediction_result:
        result = st.session_state.prediction_result

        st.divider()
        if result["risco_defasagem"] == 1:
            st.error("**Alto Risco de Defasagem Escolar**")
            st.markdown(
                f"Probabilidade estimada de risco: **{result['probabilidade']:.0%}**\n\n"
                "Recomenda-se acompanhamento pedagógico prioritário para este aluno."
            )
        else:
            st.success("**Sem Risco de Defasagem Identificado**")
            st.markdown(
                f"Probabilidade estimada de risco: **{result['probabilidade']:.0%}**\n\n"
                "O aluno não apresenta indicadores de risco para o próximo período."
            )

        with st.expander("Detalhes técnicos"):
            st.markdown(
                f"- **Threshold (ponto de corte):** `{result['threshold']:.2f}` - "
                "o modelo foi calibrado para maximizar a identificação de alunos em risco (alta sensibilidade).\n"
                f"- **Probabilidade bruta:** `{result['probabilidade']:.4f}`"
            )

        # Perfil de cluster
        if st.session_state.cluster_result:
            cluster = st.session_state.cluster_result
            perfil = cluster["perfil"]
            PERFIL_CORES = {
                "risco_alto": "red",
                "risco_moderado": "orange",
                "em_desenvolvimento": "blue",
                "alto_desempenho": "green",
            }
            PERFIL_LABELS = {
                "risco_alto": "Risco Alto",
                "risco_moderado": "Risco Moderado",
                "em_desenvolvimento": "Em Desenvolvimento",
                "alto_desempenho": "Alto Desempenho",
            }
            cor = PERFIL_CORES.get(perfil, "gray")
            label_perfil = PERFIL_LABELS.get(perfil, perfil)
            st.divider()
            st.subheader("Perfil Pedagógico")
            st.markdown(f"**:{cor}[{label_perfil}]**")
            st.caption(cluster["descricao"])
            with st.expander("Distâncias para os perfis"):
                st.markdown(
                    "Distância euclidiana do aluno para cada centroide de perfil "
                    "(menor = mais próximo daquele perfil):"
                )
                dist_sorted = sorted(cluster["distancias"].items(), key=lambda x: x[1])
                for nome, dist in dist_sorted:
                    nome_label = PERFIL_LABELS.get(nome, nome)
                    st.markdown(f"- **{nome_label}**: `{dist:.2f}`")

        # Relatório pedagógico
        if st.session_state.relatorio:
            st.divider()
            st.subheader("Análise Pedagógica")
            st.info(st.session_state.relatorio)

with tab_history:
    st.subheader("Consultas Anteriores")
    st.markdown("Registro de todas as avaliações realizadas pela ferramenta.")
    limit = st.slider("Quantidade de registros", 10, 200, 50)
    try:
        resp = requests.get(f"{API_URL}/predictions", params={"limit": limit}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Nenhuma avaliação registrada ainda.")
    except requests.exceptions.RequestException:
        st.warning("API não disponível.")

with tab_drift:
    st.subheader("Saúde do Modelo")
    st.markdown(
        "Esta análise compara o perfil dos alunos avaliados recentemente com os dados "
        "utilizados no treinamento do modelo. Mudanças significativas no perfil dos alunos "
        "podem indicar que o modelo precisa ser atualizado para continuar operando com precisão."
    )
    if st.button("Verificar Saúde do Modelo"):
        try:
            resp = requests.get(f"{API_URL}/drift", params={"n": 50}, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            if result.get("status") == "insufficient_data":
                st.info(
                    "Ainda não há avaliações suficientes para esta análise. "
                    "Realize pelo menos 30 avaliações para habilitar o monitoramento."
                )
            else:
                if result.get("dataset_drift"):
                    st.warning(
                        "**Atenção: o perfil dos alunos recentes difere do histórico de treinamento.**\n\n"
                        "Recomenda-se revisar o modelo com dados mais atualizados antes de continuar utilizando."
                    )
                else:
                    st.success(
                        "**Modelo operando normalmente.** "
                        "O perfil dos alunos recentes é consistente com o histórico de treinamento."
                    )
                with st.expander("Detalhes técnicos (KS-test + PSI)"):
                    st.markdown(
                        f"- Proporção de features com drift detectado: **{result['drift_share']:.1%}**\n"
                        f"- Colunas analisadas: {result.get('n_features', 'N/A')}\n"
                        f"- Colunas com drift (KS): {result.get('n_drifted_columns', 0)}"
                    )
                    st.json(result["column_details"])
        except requests.exceptions.RequestException:
            st.warning("API não disponível.")


# Chat do Assistente Pedagógico
if st.session_state.prediction_result and DEEPSEEK_API_KEY:
    result = st.session_state.prediction_result
    st.divider()
    st.subheader("Assistente Pedagógico")
    st.caption(
        "Converse com o assistente para aprofundar a análise deste aluno. "
        "As perguntas e respostas ficam vinculadas ao aluno atual."
    )

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Pergunte sobre este aluno..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        system_prompt = _build_chat_system_prompt(
            st.session_state.prediction_payload,
            result,
            st.session_state.cluster_result,
        )
        messages_for_ai = [{"role": "system", "content": system_prompt}] + [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_messages
        ]

        with st.chat_message("assistant"):
            try:
                placeholder = st.empty()
                ai_response = ""
                for chunk in _stream_deepseek(messages_for_ai, DEEPSEEK_API_KEY):
                    ai_response += chunk
                    placeholder.markdown(ai_response)
            except Exception as e:
                ai_response = f"Erro ao conectar ao assistente: {e}"
                st.error(ai_response)

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": ai_response}
        )
