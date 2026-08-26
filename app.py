import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import trafilatura
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


APP_VERSION = "2026-08-26-v3-jsonfix"

st.set_page_config(
    page_title="AI Financial Research Assistant",
    page_icon="📊",
    layout="wide",
)

st.set_option("client.toolbarMode", "viewer")

st.title("📊 AI Financial Research Assistant")
st.caption(
    "DART 재무정보와 최신 뉴스를 기반으로 "
    "기업금융 RM · PB 관점의 기업 리서치를 제공합니다."
)
st.caption(f"App version: {APP_VERSION}")

DART_API_KEY = st.secrets["DART_API_KEY"]
NAVER_CLIENT_ID = st.secrets["NAVER_CLIENT_ID"]
NAVER_CLIENT_SECRET = st.secrets["NAVER_CLIENT_SECRET"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

NAVER_HEADERS = {
    "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
    "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET,
}

GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

AI_CLIENT = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        timeout=300000,
        retry_options=types.HttpRetryOptions(attempts=1),
    ),
)

REQUEST_TIMEOUT = 30
ARTICLE_TIMEOUT = 10

COMMON_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


class FinancialReport(BaseModel):
    financial_summary: List[str]
    rm_view: List[str]
    pb_view: List[str]
    additional_checks: List[str]


class NewsPick(BaseModel):
    issue_name: str
    category: str
    importance: int = Field(ge=1, le=5)
    representative_index: int
    reason: str


class NewsSelection(BaseModel):
    issues: List[NewsPick]


class NewsDeepItem(BaseModel):
    issue_name: str
    category: str
    importance: int = Field(ge=1, le=5)
    article_index: int
    summary: List[str]
    rm_view: List[str]
    pb_view: List[str]
    comparison_reference: List[str]


class NewsDeepReport(BaseModel):
    issues: List[NewsDeepItem]


def safe_float(value):
    if value is None:
        return np.nan

    text = str(value).strip().replace(",", "")

    if text in ("", "-", "None"):
        return np.nan

    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]

    try:
        return float(text)
    except (TypeError, ValueError):
        return np.nan


def dart_get_json(endpoint, **params):
    try:
        response = requests.get(
            f"https://opendart.fss.or.kr/api/{endpoint}",
            params={"crtfc_key": DART_API_KEY, **params},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException:
        raise RuntimeError("DART_NETWORK_ERROR") from None

    if response.status_code != 200:
        raise RuntimeError("DART_HTTP_ERROR")

    try:
        data = response.json()
    except ValueError:
        raise RuntimeError("DART_RESPONSE_ERROR") from None

    if data.get("status") not in (None, "000", "013"):
        raise RuntimeError("DART_API_ERROR")

    return data


@st.cache_data(ttl=86400, show_spinner=False)
def load_corp_df():
    csv_path = Path(__file__).resolve().parent / "corp_codes.csv"

    if not csv_path.exists():
        raise FileNotFoundError("CORP_CODES_FILE_MISSING")

    df = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    required = {"회사명", "DART코드", "종목코드"}
    if not required.issubset(df.columns):
        raise ValueError("CORP_CODES_FORMAT_ERROR")

    return df


def find_company(corp_df, company_query):
    query = str(company_query).strip()

    if not query:
        raise ValueError("EMPTY_COMPANY_NAME")

    exact = corp_df[
        corp_df["회사명"].astype(str).str.strip() == query
    ]

    if not exact.empty:
        row = exact.iloc[0]
    else:
        matches = corp_df[
            corp_df["회사명"]
            .astype(str)
            .str.contains(
                re.escape(query),
                case=False,
                na=False,
                regex=True,
            )
        ]

        if matches.empty:
            raise ValueError("COMPANY_NOT_FOUND")

        row = matches.iloc[0]

    return {
        "회사명": str(row["회사명"]).strip(),
        "DART코드": str(row["DART코드"]).strip(),
        "종목코드": str(row["종목코드"]).strip(),
    }


def annual_financials(corp_code, years):
    aliases = {
        "매출액": ["매출액", "수익(매출액)", "영업수익"],
        "영업이익": ["영업이익", "영업이익(손실)"],
        "당기순이익": [
            "당기순이익",
            "당기순이익(손실)",
            "연결당기순이익",
        ],
        "자산총계": ["자산총계"],
        "부채총계": ["부채총계"],
        "자본총계": ["자본총계"],
    }

    data_map = {
        metric: {year: np.nan for year in years}
        for metric in [*aliases.keys(), "영업현금흐름"]
    }

    for year in years:
        data = dart_get_json(
            "fnlttSinglAcnt.json",
            corp_code=corp_code,
            bsns_year=str(year),
            reprt_code="11011",
        )

        if data.get("status") == "000":
            rows = data.get("list", [])
            cfs_rows = [r for r in rows if r.get("fs_div") == "CFS"]
            if cfs_rows:
                rows = cfs_rows

            for metric, names in aliases.items():
                for name in names:
                    hit = next(
                        (
                            r
                            for r in rows
                            if str(r.get("account_nm", "")).strip() == name
                        ),
                        None,
                    )

                    if hit is None:
                        continue

                    value = safe_float(hit.get("thstrm_amount"))
                    if pd.notna(value):
                        data_map[metric][year] = value / 100000000
                        break

        for fs_div in ("CFS", "OFS"):
            cash = dart_get_json(
                "fnlttSinglAcntAll.json",
                corp_code=corp_code,
                bsns_year=str(year),
                reprt_code="11011",
                fs_div=fs_div,
            )

            if cash.get("status") != "000":
                continue

            ocf_names = {
                "영업활동으로인한현금흐름",
                "영업활동현금흐름",
                "영업활동으로부터의현금흐름",
            }

            for row in cash.get("list", []):
                account_name = (
                    str(row.get("account_nm", ""))
                    .replace(" ", "")
                    .strip()
                )

                if account_name not in ocf_names:
                    continue

                value = safe_float(row.get("thstrm_amount"))
                if pd.notna(value):
                    data_map["영업현금흐름"][year] = value / 100000000
                    break

            if pd.notna(data_map["영업현금흐름"][year]):
                break

    return pd.DataFrame(data_map).T[years]


def general_ratios(df_fin, years):
    ratio_df = pd.DataFrame(
        index=[
            "영업이익률 (%)",
            "순이익률 (%)",
            "부채비율 (%)",
            "영업현금흐름률 (%)",
            "매출성장률 (%)",
        ],
        columns=years,
        dtype=float,
    )

    for year in years:
        sales = df_fin.loc["매출액", year]
        operating = df_fin.loc["영업이익", year]
        net_income = df_fin.loc["당기순이익", year]
        debt = df_fin.loc["부채총계", year]
        equity = df_fin.loc["자본총계", year]
        ocf = df_fin.loc["영업현금흐름", year]

        if pd.notna(sales) and sales != 0:
            if pd.notna(operating):
                ratio_df.loc["영업이익률 (%)", year] = operating / sales * 100
            if pd.notna(net_income):
                ratio_df.loc["순이익률 (%)", year] = net_income / sales * 100
            if pd.notna(ocf):
                ratio_df.loc["영업현금흐름률 (%)", year] = ocf / sales * 100

        if pd.notna(debt) and pd.notna(equity) and equity != 0:
            ratio_df.loc["부채비율 (%)", year] = debt / equity * 100

    for i in range(1, len(years)):
        prev_year = years[i - 1]
        year = years[i]
        prev_sales = df_fin.loc["매출액", prev_year]
        sales = df_fin.loc["매출액", year]

        if (
            pd.notna(prev_sales)
            and pd.notna(sales)
            and prev_sales != 0
        ):
            ratio_df.loc["매출성장률 (%)", year] = (
                sales / prev_sales - 1
            ) * 100

    return ratio_df.round(2)


def naver_news_search(company_name, display=30):
    try:
        response = requests.get(
            "https://naverapihub.apigw.ntruss.com/search/v1/news",
            headers=NAVER_HEADERS,
            params={
                "query": company_name,
                "display": display,
                "start": 1,
                "sort": "date",
                "format": "json",
            },
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException:
        raise RuntimeError("NAVER_NETWORK_ERROR") from None

    if response.status_code != 200:
        raise RuntimeError("NAVER_HTTP_ERROR")

    try:
        payload = response.json()
    except ValueError:
        raise RuntimeError("NAVER_RESPONSE_ERROR") from None

    news = []

    for item in payload.get("items", []):
        title = BeautifulSoup(
            item.get("title", ""),
            "html.parser",
        ).get_text(" ", strip=True)

        description = BeautifulSoup(
            item.get("description", ""),
            "html.parser",
        ).get_text(" ", strip=True)

        news.append(
            {
                "제목": title,
                "설명": description,
                "원문링크": item.get("originallink") or "",
                "네이버링크": item.get("link") or "",
                "발행일": item.get("pubDate") or "",
            }
        )

    return news


def extract_article_body(news_item, max_chars=7000):
    urls = [
        news_item.get("원문링크", ""),
        news_item.get("네이버링크", ""),
    ]

    for url in urls:
        if not url:
            continue

        try:
            response = requests.get(
                url,
                headers=COMMON_HEADERS,
                timeout=ARTICLE_TIMEOUT,
                allow_redirects=True,
            )

            if response.status_code != 200:
                continue

            body = trafilatura.extract(
                response.text,
                include_comments=False,
                include_tables=False,
            )

            if body and len(body.strip()) >= 200:
                return body.strip()[:max_chars]

        except Exception:
            continue

    return ""


def extract_selected_articles(news_list, selected_issues):
    selected = []

    for issue in selected_issues:
        idx = issue.representative_index

        if not (1 <= idx <= len(news_list)):
            continue

        article = news_list[idx - 1].copy()
        article["이슈명"] = issue.issue_name
        article["카테고리"] = issue.category
        article["중요도"] = issue.importance
        article["선정이유"] = issue.reason
        article["원본인덱스"] = idx
        selected.append(article)

    if not selected:
        return []

    with ThreadPoolExecutor(max_workers=min(5, len(selected))) as executor:
        future_map = {
            executor.submit(extract_article_body, item): pos
            for pos, item in enumerate(selected)
        }

        for future in as_completed(future_map):
            pos = future_map[future]
            try:
                body = future.result()
            except Exception:
                body = ""

            selected[pos]["본문"] = body

    return selected


def gemini_structured(prompt, schema):
    """한 모델이 한도/서버 문제로 실패하면 다음 Gemini 모델로 자동 전환한다."""

    schema_text = json.dumps(
        schema.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )

    final_prompt = f"""{prompt}

출력 형식 규칙:
- 설명문, 마크다운, 코드블록 없이 JSON 객체 하나만 출력합니다.
- 아래 JSON Schema의 필드명과 자료형을 정확히 지킵니다.
- 모든 필수 필드를 반드시 포함합니다.

JSON Schema:
{schema_text}
"""

    blocked_models = st.session_state.get(
        "gemini_blocked_models",
        []
    )

    last_error = "GEMINI_ERROR"

    for model_name in GEMINI_MODELS:

        if model_name in blocked_models:
            continue

        try:
            response = AI_CLIENT.models.generate_content(
                model=model_name,
                contents=final_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=8192,
                ),
            )

        except Exception as e:
            message = str(e).lower()

            # 사용량 한도 초과
            if (
                "429" in message
                or "resource_exhausted" in message
                or "quota" in message
            ):
                last_error = "RATE_LIMIT"

                if model_name not in blocked_models:
                    blocked_models.append(model_name)

                    st.session_state[
                        "gemini_blocked_models"
                    ] = blocked_models

                continue

            # 서버 혼잡
            if (
                "503" in message
                or "unavailable" in message
            ):
                last_error = "SERVICE_BUSY"
                continue

            # 시간 초과
            if (
                "504" in message
                or "deadline" in message
                or "timeout" in message
            ):
                last_error = "TIMEOUT"
                continue

            # 프로젝트에서 해당 모델 사용 불가
            if (
                "404" in message
                or "not found" in message
                or "403" in message
                or "permission_denied" in message
            ):
                last_error = "MODEL_UNAVAILABLE"

                if model_name not in blocked_models:
                    blocked_models.append(model_name)

                    st.session_state[
                        "gemini_blocked_models"
                    ] = blocked_models

                continue

            last_error = "GEMINI_ERROR"
            continue

        try:
            text = (
                response.text
                or ""
            ).strip()

            if not text:
                last_error = "EMPTY_RESPONSE"
                continue

            # 혹시 ```json 코드블록이 섞여도 제거
            text = re.sub(
                r"^```(?:json)?\s*",
                "",
                text,
                flags=re.IGNORECASE,
            )

            text = re.sub(
                r"\s*```$",
                "",
                text,
            )

            # JSON 앞뒤 잡문 제거
            start = text.find("{")
            end = text.rfind("}")

            if (
                start >= 0
                and end > start
            ):
                text = text[
                    start:end + 1
                ]

            payload = json.loads(text)

            parsed = schema.model_validate(
                payload
            )

            # 어떤 모델이 성공했는지 내부 기록
            st.session_state[
                "last_gemini_model"
            ] = model_name

            return parsed, None

        except (
            json.JSONDecodeError,
            ValueError,
            TypeError,
        ):
            last_error = "JSON_FORMAT_ERROR"
            continue

        except Exception:
            last_error = "GEMINI_ERROR"
            continue

    return None, last_error


def run_financial_ai(company_name, analysis_mode, df_fin, ratio_df):
    ratio_text = (
        ratio_df.to_string()
        if not ratio_df.empty
        else "금융회사이므로 일반기업 재무비율 미적용"
    )

    prompt = f"""
당신은 기업금융 RM과 PB를 위한 재무 분석가입니다.

분석 대상 기업:
{company_name}

분석모드:
{analysis_mode}

[DART 최근 3개년 재무정보 / 억원]
{df_fin.to_string()}

[재무비율]
{ratio_text}

규칙:
1. 위 DART 숫자만 사용합니다.
2. 외부 사실이나 추측을 추가하지 않습니다.
3. 업종 평균이나 비교기업 자료가 없으면
   높다, 낮다, 우수하다, 양호하다 같은 절대평가를 하지 않습니다.
4. 영업현금흐름만으로 채무상환능력을 단정하지 않습니다.
5. 실제 숫자 변화 중심으로 설명합니다.
6. 근거가 부족하면 '추가 확인 필요'라고 적습니다.
7. 금융회사라면 일반기업용 비율을 억지로 해석하지 말고
   추가로 필요한 금융회사 전용 지표를 additional_checks에 적습니다.
8. RM View는 현금창출, 부채, 상환재원, 자금수요, 사업 리스크 관점입니다.
9. PB View는 성장성, 수익성, 이익의 질, 실적 모멘텀,
   주주가치 관련 확인 포인트 중심입니다.
"""

    return gemini_structured(prompt, FinancialReport)


def run_news_selection_ai(company_name, news_list):
    news_text = "\n\n".join(
        (
            f"[{i}]\n"
            f"제목: {item['제목']}\n"
            f"설명: {item['설명']}\n"
            f"발행일: {item['발행일']}"
        )
        for i, item in enumerate(news_list, 1)
    )

    prompt = f"""
분석 대상 기업은 '{company_name}'입니다.

아래 최신 뉴스 30개에서 기업 분석 가치가 높은 핵심 이슈를 선정하세요.

뉴스:
{news_text}

규칙:
1. 분석 대상 기업이 실제 핵심 주체인 기사만 우선합니다.
2. 단순 회사명 언급, 일반 증시 수급 기사, 테마성 기사,
   다른 회사가 핵심 주체인 기사는 제외합니다.
3. 실적, 재무, 투자·M&A, 수주·계약, 제품·기술,
   산업·규제, 수요·시장, 원가·환율, 노무,
   주주환원, 경영전략 등 기업 본질과 직접 관련된 이슈를 우선합니다.
4. 같은 사건을 다룬 기사는 하나의 이슈로 묶고
   그 이슈를 가장 직접적으로 다룬 기사를 representative_index로 고릅니다.
5. representative_index는 반드시 위 뉴스 번호를 사용합니다.
6. 중요도 3 이상만 issues에 포함합니다.
7. 최대 5개만 선택합니다.
8. 대표기사는 가능하면 제목에 '{company_name}'이 직접 등장하고,
   해당 이슈를 구체적으로 다루는 기사를 우선합니다.
9. 단기 주가 등락이나 기관/외국인 순매수만 다룬 기사는
   원칙적으로 핵심 이슈에서 제외합니다.
"""

    report, error = gemini_structured(prompt, NewsSelection)

    if report:
        report.issues = [
            issue
            for issue in report.issues
            if issue.importance >= 3
            and 1 <= issue.representative_index <= len(news_list)
        ][:5]

    return report, error


def run_news_deep_ai(company_name, selected_articles):
    if not selected_articles:
        return None, "NO_SELECTED_NEWS"

    chunks = []

    for i, article in enumerate(selected_articles, 1):
        body = article.get("본문", "").strip()

        if body:
            source_text = body
            source_type = "실제 기사본문"
        else:
            source_text = article.get("설명", "")
            source_type = "기사본문 확보 실패 - 검색 설명문"

        chunks.append(
            f"""
[선정 이슈 {i}]
이슈명: {article['이슈명']}
카테고리: {article['카테고리']}
중요도: {article['중요도']}
원본 뉴스 번호: {article['원본인덱스']}
제목: {article['제목']}
발행일: {article['발행일']}
자료유형: {source_type}

자료:
{source_text}
"""
        )

    article_text = "\n".join(chunks)

    prompt = f"""
당신은 기업금융 RM과 PB를 위한 뉴스 분석가입니다.

분석 대상 기업:
{company_name}

아래는 2차 선별된 핵심 이슈와 기사 자료입니다.

{article_text}

규칙:
1. 제공된 기사 자료에 있는 사실만 사용합니다.
2. 분석 대상 기업 중심으로 작성합니다.
3. 다른 기업 정보는 comparison_reference에만 작성합니다.
4. 기사에 없는 숫자를 만들거나 추정하지 않습니다.
5. 매수·매도 추천을 하지 않습니다.
6. 실제 기사본문이 아니라 검색 설명문만 제공된 경우,
   그 범위를 넘어서는 단정은 하지 말고 필요하면 '추가 확인 필요'라고 적습니다.
7. 각 이슈의 importance는 입력된 중요도를 유지합니다.
8. article_index는 위 '선정 이슈 번호'를 사용합니다.
9. RM View는 현금창출, 부채, 상환재원, 자금수요,
   사업 리스크에 미치는 영향 중심입니다.
10. PB View는 성장성, 수익성, 이익의 질,
    실적 모멘텀, 주주가치 관련 확인 포인트 중심입니다.
"""

    return gemini_structured(prompt, NewsDeepReport)


def analyze_company(company_query):
    corp_df = load_corp_df()
    company = find_company(corp_df, company_query)

    company_name = company["회사명"]
    corp_code = company["DART코드"]

    info = dart_get_json("company.json", corp_code=corp_code)
    industry_code = str(info.get("induty_code", "")).strip()

    analysis_mode = (
        "금융회사"
        if industry_code.startswith(("64", "65", "66"))
        else "일반기업"
    )

    current_year = datetime.now().year
    years = list(range(current_year - 3, current_year))

    df_fin = annual_financials(corp_code, years)

    ratio_df = (
        general_ratios(df_fin, years)
        if analysis_mode == "일반기업"
        else pd.DataFrame()
    )

    financial_report, financial_error = run_financial_ai(
        company_name,
        analysis_mode,
        df_fin,
        ratio_df,
    )

    news_list = naver_news_search(company_name, display=30)

    selection_report, selection_error = run_news_selection_ai(
        company_name,
        news_list,
    )

    selected_articles = []

    if selection_report and selection_report.issues:
        selected_articles = extract_selected_articles(
            news_list,
            selection_report.issues,
        )

    news_report = None
    news_deep_error = None

    if selected_articles:
        news_report, news_deep_error = run_news_deep_ai(
            company_name,
            selected_articles,
        )

    return {
        "company": company,
        "industry_code": industry_code,
        "analysis_mode": analysis_mode,
        "financial": df_fin,
        "ratios": ratio_df,
        "financial_report": financial_report,
        "financial_error": financial_error,
        "news_list": news_list,
        "selection_report": selection_report,
        "selection_error": selection_error,
        "selected_articles": selected_articles,
        "news_report": news_report,
        "news_deep_error": news_deep_error,
    }


def render_safe_error(code):
    messages = {
        "EMPTY_COMPANY_NAME": "회사명을 입력해주세요.",
        "COMPANY_NOT_FOUND": "DART 기업목록에서 해당 회사를 찾지 못했습니다.",
        "CORP_CODES_FILE_MISSING": "기업목록 파일을 불러오지 못했습니다.",
        "CORP_CODES_FORMAT_ERROR": "기업목록 파일 형식에 문제가 있습니다.",
        "DART_NETWORK_ERROR": "DART 연결에 실패했습니다. 잠시 후 다시 시도해주세요.",
        "DART_HTTP_ERROR": "DART 서버 응답에 문제가 있습니다.",
        "DART_RESPONSE_ERROR": "DART 응답을 처리하지 못했습니다.",
        "DART_API_ERROR": "DART API 요청을 처리하지 못했습니다.",
        "NAVER_NETWORK_ERROR": "NAVER 뉴스 연결에 실패했습니다.",
        "NAVER_HTTP_ERROR": "NAVER 뉴스 서버 응답에 문제가 있습니다.",
        "NAVER_RESPONSE_ERROR": "NAVER 뉴스 응답을 처리하지 못했습니다.",
    }

    return messages.get(
        code,
        "분석 중 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
    )


def render_result(result):
    company = result["company"]

    st.success(f"{company['회사명']} 분석이 완료되었습니다.")

    col1, col2, col3 = st.columns(3)
    col1.metric("회사명", company["회사명"])
    col2.metric("종목코드", company["종목코드"] or "-")
    col3.metric("분석모드", result["analysis_mode"])

    if result["analysis_mode"] == "금융회사":
        st.warning(
            "현재 금융회사 전용 BIS · NIM · 건전성 지표 모듈은 개발 중입니다. "
            "일반기업 비율은 적용하지 않습니다."
        )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 재무", "🏦 RM View", "📈 PB View", "📰 핵심 뉴스"]
    )

    financial_report = result["financial_report"]
    news_report = result["news_report"]

    with tab1:
        st.subheader("최근 3개년 재무정보")
        st.caption("단위: 억원")
        st.dataframe(
            result["financial"].round(1),
            use_container_width=True,
        )

        if not result["ratios"].empty:
            st.subheader("핵심 재무비율")
            st.dataframe(
                result["ratios"],
                use_container_width=True,
            )

        if financial_report:
            st.subheader("AI 핵심 재무 흐름")
            for item in financial_report.financial_summary:
                st.write(f"• {item}")

            st.subheader("추가 확인사항")
            for item in financial_report.additional_checks:
                st.write(f"• {item}")
        else:
            st.warning(
                "재무 AI 해석을 완료하지 못했습니다. "
                "위 DART 재무수치는 정상적으로 수집되었습니다."
            )

    with tab2:
        if financial_report:
            st.subheader("재무 기반 RM View")
            for item in financial_report.rm_view:
                st.write(f"• {item}")
        else:
            st.info("재무 기반 RM View를 생성하지 못했습니다.")

        if news_report and news_report.issues:
            st.subheader("뉴스 기반 RM View")
            for issue in news_report.issues:
                st.markdown(f"**{issue.issue_name}**")
                for item in issue.rm_view:
                    st.write(f"• {item}")

    with tab3:
        if financial_report:
            st.subheader("재무 기반 PB View")
            for item in financial_report.pb_view:
                st.write(f"• {item}")
        else:
            st.info("재무 기반 PB View를 생성하지 못했습니다.")

        if news_report and news_report.issues:
            st.subheader("뉴스 기반 PB View")
            for issue in news_report.issues:
                st.markdown(f"**{issue.issue_name}**")
                for item in issue.pb_view:
                    st.write(f"• {item}")

    with tab4:
        selection_report = result["selection_report"]

        if not selection_report or not selection_report.issues:
            st.info(
                "중요도 3 이상의 핵심 뉴스를 선별하지 못했거나 "
                "뉴스 AI 선별 단계가 일시적으로 실패했습니다."
            )
            return

        if news_report and news_report.issues:
            for number, issue in enumerate(news_report.issues[:5], 1):
                st.markdown(f"### {number}. {issue.issue_name}")
                st.caption(
                    f"{issue.category} · 중요도 {issue.importance}/5"
                )

                for item in issue.summary:
                    st.write(f"• {item}")

                st.markdown("**🏦 뉴스 RM View**")
                for item in issue.rm_view:
                    st.write(f"• {item}")

                st.markdown("**📈 뉴스 PB View**")
                for item in issue.pb_view:
                    st.write(f"• {item}")

                if issue.comparison_reference:
                    st.markdown("**🔎 비교 참고**")
                    for item in issue.comparison_reference:
                        st.write(f"• {item}")

                if (
                    1
                    <= issue.article_index
                    <= len(result["selected_articles"])
                ):
                    source = result["selected_articles"][
                        issue.article_index - 1
                    ]

                    article_url = (
                        source["원문링크"]
                        or source["네이버링크"]
                    )

                    st.markdown(
                        f"**대표기사:** {source['제목']}"
                    )

                    if article_url:
                        st.link_button(
                            "🔗 기사 원문 보기",
                            article_url,
                        )

                    if not source.get("본문", ""):
                        st.caption(
                            "※ 이 기사는 본문 추출에 실패해 "
                            "검색 제목·설명 범위에서만 분석했습니다."
                        )

                st.divider()

        else:
            st.warning(
                "핵심 뉴스는 선별했지만 심층 뉴스 AI 분석을 완료하지 못했습니다."
            )

            for number, issue in enumerate(selection_report.issues[:5], 1):
                idx = issue.representative_index
                st.markdown(f"### {number}. {issue.issue_name}")
                st.caption(
                    f"{issue.category} · 중요도 {issue.importance}/5"
                )
                st.write(f"• 선정 이유: {issue.reason}")

                if 1 <= idx <= len(result["news_list"]):
                    source = result["news_list"][idx - 1]
                    article_url = source["원문링크"] or source["네이버링크"]

                    st.markdown(f"**대표기사:** {source['제목']}")
                    if article_url:
                        st.link_button("🔗 기사 원문 보기", article_url)


company_query = st.text_input(
    "분석할 회사명을 입력하세요",
    placeholder="예: 팬오션, 삼성전자, 대한항공, LG생활건강",
)

st.caption(
    "AI 호출은 기업 1회 분석당 최대 3회입니다. "
    "연속해서 여러 기업을 빠르게 분석하면 모델 사용량 제한에 걸릴 수 있습니다."
)

analyze_button = st.button(
    "🔍 기업 분석 시작",
    type="primary",
    use_container_width=True,
)

if analyze_button:
    query = company_query.strip()

    if not query:
        st.warning("회사명을 입력해주세요.")
        st.stop()

    # 앱 버전을 키에 포함해 이전 버전의 실패 캐시를 절대 재사용하지 않음
    session_key = f"{APP_VERSION}::analysis::{query}"

    try:
        if session_key in st.session_state:
            result = st.session_state[session_key]
            st.info("같은 세션의 정상 완료 분석 결과를 다시 표시합니다.")
        else:
            with st.spinner(
                "재무 분석 → 뉴스 선별 → 핵심 뉴스 심층분석을 진행하고 있습니다..."
            ):
                result = analyze_company(query)

            # 세 단계가 모두 정상 완료된 경우에만 캐시
            financial_ok = result.get("financial_report") is not None
            selection_ok = result.get("selection_report") is not None
            selected_count = len(result.get("selected_articles") or [])
            deep_ok = (
                selected_count == 0
                or result.get("news_report") is not None
            )

            if financial_ok and selection_ok and deep_ok:
                st.session_state[session_key] = result

        render_result(result)

    except (ValueError, RuntimeError, FileNotFoundError) as e:
        st.error(render_safe_error(str(e)))

    except Exception:
        st.error(
            "분석 중 일시적인 오류가 발생했습니다. "
            "잠시 후 다시 시도해주세요."
        )

st.divider()
st.caption(
    "본 서비스는 기업 리서치 보조 도구이며 투자 권유 또는 "
    "매수·매도 추천을 제공하지 않습니다. 중요한 의사결정 전에는 "
    "DART 공시와 기사 원문을 직접 확인하세요."
)
