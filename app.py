import re
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import trafilatura

from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="AI Financial Research Assistant",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI Financial Research Assistant")

st.caption(
    "DART 재무정보와 최신 뉴스를 기반으로 "
    "RM · PB 관점의 기업 분석을 제공합니다."
)


# ============================================================
# SECRETS
# ============================================================

DART_API_KEY = st.secrets["DART_API_KEY"]
NAVER_CLIENT_ID = st.secrets["NAVER_CLIENT_ID"]
NAVER_CLIENT_SECRET = st.secrets["NAVER_CLIENT_SECRET"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

NAVER_HEADERS = {
    "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
    "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET,
}

GEMINI_MODEL = "gemini-3.5-flash"

AI_CLIENT = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        timeout=300000,
        retry_options=types.HttpRetryOptions(
            attempts=1
        ),
    ),
)

REQUEST_TIMEOUT = 30
ARTICLE_TIMEOUT = 12

COMMON_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


# ============================================================
# AI OUTPUT
# ============================================================

class NewsIssue(BaseModel):
    issue_name: str
    category: str
    importance: int = Field(ge=1, le=5)
    article_index: int
    summary: List[str]
    rm_view: List[str]
    pb_view: List[str]
    comparison_reference: List[str]


class FinalCompanyReport(BaseModel):
    financial_summary: List[str]
    rm_view: List[str]
    pb_view: List[str]
    additional_checks: List[str]
    news_issues: List[NewsIssue]


# ============================================================
# BASIC FUNCTIONS
# ============================================================

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
            params={
                "crtfc_key": DART_API_KEY,
                **params,
            },
            timeout=REQUEST_TIMEOUT,
        )

    except requests.RequestException:
        raise RuntimeError(
            "DART_NETWORK_ERROR"
        ) from None

    if response.status_code != 200:
        raise RuntimeError(
            "DART_HTTP_ERROR"
        )

    try:
        data = response.json()

    except ValueError:
        raise RuntimeError(
            "DART_RESPONSE_ERROR"
        ) from None

    if data.get("status") not in (
        None,
        "000",
        "013",
    ):
        raise RuntimeError(
            "DART_API_ERROR"
        )

    return data


# ============================================================
# COMPANY LIST
# GitHub corp_codes.csv 사용
# DART 전체목록 다운로드 없음
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False,
)
def load_corp_df():

    csv_path = (
        Path(__file__).resolve().parent
        / "corp_codes.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            "CORP_CODES_FILE_MISSING"
        )

    df = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    required_columns = {
        "회사명",
        "DART코드",
        "종목코드",
    }

    if not required_columns.issubset(
        df.columns
    ):
        raise ValueError(
            "CORP_CODES_FORMAT_ERROR"
        )

    return df


def find_company(
    corp_df,
    company_query,
):

    query = str(
        company_query
    ).strip()

    if not query:
        raise ValueError(
            "EMPTY_COMPANY_NAME"
        )

    exact = corp_df[
        corp_df["회사명"]
        .astype(str)
        .str.strip()
        == query
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
            raise ValueError(
                "COMPANY_NOT_FOUND"
            )

        row = matches.iloc[0]

    return {
        "회사명":
            str(
                row["회사명"]
            ).strip(),

        "DART코드":
            str(
                row["DART코드"]
            ).strip(),

        "종목코드":
            str(
                row["종목코드"]
            ).strip(),
    }


# ============================================================
# FINANCIAL DATA
# ============================================================

def annual_financials(
    corp_code,
    years,
):

    aliases = {

        "매출액": [
            "매출액",
            "수익(매출액)",
            "영업수익",
        ],

        "영업이익": [
            "영업이익",
            "영업이익(손실)",
        ],

        "당기순이익": [
            "당기순이익",
            "당기순이익(손실)",
            "연결당기순이익",
        ],

        "자산총계": [
            "자산총계"
        ],

        "부채총계": [
            "부채총계"
        ],

        "자본총계": [
            "자본총계"
        ],
    }

    data_map = {
        metric: {
            year: np.nan
            for year in years
        }
        for metric in [
            *aliases.keys(),
            "영업현금흐름",
        ]
    }

    for year in years:

        data = dart_get_json(
            "fnlttSinglAcnt.json",
            corp_code=corp_code,
            bsns_year=str(year),
            reprt_code="11011",
        )

        if data.get("status") == "000":

            rows = data.get(
                "list",
                []
            )

            cfs_rows = [
                row
                for row in rows
                if row.get(
                    "fs_div"
                ) == "CFS"
            ]

            if cfs_rows:
                rows = cfs_rows

            for metric, names in aliases.items():

                for name in names:

                    hit = next(
                        (
                            row
                            for row in rows
                            if str(
                                row.get(
                                    "account_nm",
                                    ""
                                )
                            ).strip()
                            == name
                        ),
                        None,
                    )

                    if hit is None:
                        continue

                    value = safe_float(
                        hit.get(
                            "thstrm_amount"
                        )
                    )

                    if pd.notna(value):

                        data_map[
                            metric
                        ][year] = (
                            value
                            / 100000000
                        )

                        break

        for fs_div in (
            "CFS",
            "OFS",
        ):

            cash = dart_get_json(
                "fnlttSinglAcntAll.json",
                corp_code=corp_code,
                bsns_year=str(year),
                reprt_code="11011",
                fs_div=fs_div,
            )

            if cash.get(
                "status"
            ) != "000":
                continue

            ocf_names = {
                "영업활동으로인한현금흐름",
                "영업활동현금흐름",
                "영업활동으로부터의현금흐름",
            }

            for row in cash.get(
                "list",
                []
            ):

                account_name = str(
                    row.get(
                        "account_nm",
                        ""
                    )
                ).replace(
                    " ",
                    ""
                ).strip()

                if account_name not in ocf_names:
                    continue

                value = safe_float(
                    row.get(
                        "thstrm_amount"
                    )
                )

                if pd.notna(value):

                    data_map[
                        "영업현금흐름"
                    ][year] = (
                        value
                        / 100000000
                    )

                    break

            if pd.notna(
                data_map[
                    "영업현금흐름"
                ][year]
            ):
                break

    return pd.DataFrame(
        data_map
    ).T[years]


def general_ratios(
    df_fin,
    years,
):

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

        sales = df_fin.loc[
            "매출액",
            year
        ]

        operating = df_fin.loc[
            "영업이익",
            year
        ]

        net_income = df_fin.loc[
            "당기순이익",
            year
        ]

        debt = df_fin.loc[
            "부채총계",
            year
        ]

        equity = df_fin.loc[
            "자본총계",
            year
        ]

        ocf = df_fin.loc[
            "영업현금흐름",
            year
        ]

        if (
            pd.notna(sales)
            and sales != 0
        ):

            if pd.notna(operating):

                ratio_df.loc[
                    "영업이익률 (%)",
                    year
                ] = (
                    operating
                    / sales
                    * 100
                )

            if pd.notna(net_income):

                ratio_df.loc[
                    "순이익률 (%)",
                    year
                ] = (
                    net_income
                    / sales
                    * 100
                )

            if pd.notna(ocf):

                ratio_df.loc[
                    "영업현금흐름률 (%)",
                    year
                ] = (
                    ocf
                    / sales
                    * 100
                )

        if (
            pd.notna(debt)
            and pd.notna(equity)
            and equity != 0
        ):

            ratio_df.loc[
                "부채비율 (%)",
                year
            ] = (
                debt
                / equity
                * 100
            )

    for i in range(
        1,
        len(years)
    ):

        previous_year = years[
            i - 1
        ]

        year = years[i]

        previous_sales = df_fin.loc[
            "매출액",
            previous_year
        ]

        sales = df_fin.loc[
            "매출액",
            year
        ]

        if (
            pd.notna(previous_sales)
            and pd.notna(sales)
            and previous_sales != 0
        ):

            ratio_df.loc[
                "매출성장률 (%)",
                year
            ] = (
                sales
                / previous_sales
                - 1
            ) * 100

    return ratio_df.round(2)


# ============================================================
# NAVER NEWS
# ============================================================

def naver_news_search(
    company_name,
    display=30,
):

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
        raise RuntimeError(
            "NAVER_NETWORK_ERROR"
        ) from None

    if response.status_code != 200:
        raise RuntimeError(
            "NAVER_HTTP_ERROR"
        )

    try:
        payload = response.json()

    except ValueError:
        raise RuntimeError(
            "NAVER_RESPONSE_ERROR"
        ) from None

    news = []

    for item in payload.get(
        "items",
        []
    ):

        title = BeautifulSoup(
            item.get(
                "title",
                ""
            ),
            "html.parser",
        ).get_text(
            " ",
            strip=True,
        )

        description = BeautifulSoup(
            item.get(
                "description",
                ""
            ),
            "html.parser",
        ).get_text(
            " ",
            strip=True,
        )

        news.append({
            "제목": title,

            "설명": description,

            "원문링크":
                item.get(
                    "originallink"
                )
                or "",

            "네이버링크":
                item.get(
                    "link"
                )
                or "",

            "발행일":
                item.get(
                    "pubDate"
                )
                or "",
        })

    return news


def extract_article_body(
    news_item,
    max_chars=6000,
):

    candidate_urls = [
        news_item.get(
            "원문링크",
            ""
        ),
        news_item.get(
            "네이버링크",
            ""
        ),
    ]

    for url in candidate_urls:

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

            if (
                body
                and len(
                    body.strip()
                ) >= 200
            ):

                return (
                    body
                    .strip()[
                        :max_chars
                    ]
                )

        except Exception:
            continue

    return ""


def normalized_title(title):

    text = re.sub(
        r"\[[^\]]*\]|\([^)]*\)",
        " ",
        str(title),
    )

    text = re.sub(
        r"[^0-9A-Za-z가-힣]+",
        "",
        text,
    )

    return text.lower()


def article_candidates(
    news_list,
    target_name,
    max_candidates=10,
):

    ranked = []

    for position, news in enumerate(
        news_list
    ):

        title = news.get(
            "제목",
            ""
        )

        description = news.get(
            "설명",
            ""
        )

        score = 0

        if target_name in title:
            score += 3

        if target_name in description:
            score += 1

        ranked.append(
            (
                -score,
                position,
                news,
            )
        )

    ranked.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    candidates = []
    seen_titles = set()

    for _, _, news in ranked:

        key = normalized_title(
            news.get(
                "제목",
                ""
            )
        )

        if not key:
            continue

        if key in seen_titles:
            continue

        body = extract_article_body(
            news
        )

        if not body:
            continue

        seen_titles.add(
            key
        )

        item = news.copy()

        item["본문"] = body

        candidates.append(
            item
        )

        if (
            len(candidates)
            >= max_candidates
        ):
            break

    return candidates


# ============================================================
# MAIN ANALYSIS
# 같은 회사 결과 15분 캐시
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False,
)
def analyze_company(
    company_query,
):

    corp_df = load_corp_df()

    company = find_company(
        corp_df,
        company_query,
    )

    company_name = company[
        "회사명"
    ]

    corp_code = company[
        "DART코드"
    ]

    info = dart_get_json(
        "company.json",
        corp_code=corp_code,
    )

    industry_code = str(
        info.get(
            "induty_code",
            ""
        )
    ).strip()

    if industry_code.startswith(
        (
            "64",
            "65",
            "66",
        )
    ):
        analysis_mode = "금융회사"

    else:
        analysis_mode = "일반기업"

    current_year = (
        datetime.now().year
    )

    years = list(
        range(
            current_year - 3,
            current_year,
        )
    )

    df_fin = annual_financials(
        corp_code,
        years,
    )

    if (
        analysis_mode
        == "일반기업"
    ):

        ratio_df = general_ratios(
            df_fin,
            years,
        )

    else:

        ratio_df = pd.DataFrame()

    news_list = naver_news_search(
        company_name,
        display=30,
    )

    candidates = article_candidates(
        news_list,
        company_name,
        max_candidates=10,
    )

    if not candidates:

        return {
            "company": company,
            "industry_code":
                industry_code,
            "analysis_mode":
                analysis_mode,
            "financial":
                df_fin,
            "ratios":
                ratio_df,
            "news":
                news_list,
            "candidates":
                [],
            "report":
                None,
            "ai_error":
                "NO_ARTICLE_BODY",
        }

    article_parts = []

    for i, article in enumerate(
        candidates,
        1,
    ):

        article_parts.append(
            f"""
[기사 {i}]

제목:
{article['제목']}

발행일:
{article['발행일']}

설명:
{article['설명']}

본문:
{article['본문']}
"""
        )

    article_text = "\n".join(
        article_parts
    )

    if ratio_df.empty:

        ratio_text = (
            "금융회사이므로 "
            "일반기업 재무비율 미적용"
        )

    else:

        ratio_text = (
            ratio_df.to_string()
        )

    prompt = f"""
당신은 기업금융 RM과 PB를 위한
기업 리서치 분석가입니다.

분석 대상:
{company_name}

분석모드:
{analysis_mode}

업종코드:
{industry_code}

[DART 최근 3개년 재무정보 / 억원]

{df_fin.to_string()}

[재무비율]

{ratio_text}

[최신 실제 기사본문]

{article_text}

규칙:

1. 제공된 자료만 사용합니다.
2. 재무분석은 DART 숫자에만 근거합니다.
3. 외부 사실이나 숫자를 임의로 만들지 않습니다.
4. 비교기업 또는 업종평균이 제공되지 않았다면
   높다, 낮다, 우수하다, 양호하다 등의
   절대평가를 하지 않습니다.
5. 영업현금흐름만으로 채무상환능력을 단정하지 않습니다.
6. 뉴스분석은 제공된 실제 기사본문에만 근거합니다.
7. 분석 대상 회사가 기사의 핵심 주체가 아닌
   일반 시장기사는 제외합니다.
8. 같은 사건을 다룬 기사는 하나의 이슈로 묶습니다.
9. 최종 뉴스 이슈는 중요도 3 이상만 포함합니다.
10. 최종 뉴스 이슈는 최대 5개입니다.
11. article_index는 제공된 기사 번호만 사용합니다.
12. 다른 기업 정보는 comparison_reference에만 작성합니다.
13. 기사에 없는 숫자를 추정하거나 계산하지 않습니다.
14. 매수·매도 추천을 하지 않습니다.
15. 판단 근거가 부족하면 추가 확인 필요라고 명시합니다.
16. RM View는 현금창출, 부채, 상환재원,
    자금수요, 사업리스크 중심입니다.
17. PB View는 성장성, 수익성, 이익의 질,
    실적 모멘텀, 주주가치 관련 확인 포인트 중심입니다.
18. 중요도 1 또는 2인 뉴스는
    news_issues에 절대 포함하지 않습니다.
"""

    try:

        response = (
            AI_CLIENT
            .models
            .generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type=
                        "application/json",
                    response_schema=
                        FinalCompanyReport,
                ),
            )
        )

        report = (
            response.parsed
        )

    except Exception:

        return {
            "company": company,
            "industry_code":
                industry_code,
            "analysis_mode":
                analysis_mode,
            "financial":
                df_fin,
            "ratios":
                ratio_df,
            "news":
                news_list,
            "candidates":
                candidates,
            "report":
                None,
            "ai_error":
                "GEMINI_ERROR",
        }

    report.news_issues = [
        issue
        for issue
        in report.news_issues
        if issue.importance >= 3
    ][:5]

    return {
        "company": company,
        "industry_code":
            industry_code,
        "analysis_mode":
            analysis_mode,
        "financial":
            df_fin,
        "ratios":
            ratio_df,
        "news":
            news_list,
        "candidates":
            candidates,
        "report":
            report,
        "ai_error":
            None,
    }


# ============================================================
# SAFE ERROR MESSAGES
# API 키 / URL / 내부예외 절대 표시하지 않음
# ============================================================

def render_safe_error(
    error_code
):

    messages = {

        "EMPTY_COMPANY_NAME":
            "회사명을 입력해주세요.",

        "COMPANY_NOT_FOUND":
            "DART 기업목록에서 "
            "해당 회사를 찾지 못했습니다.",

        "CORP_CODES_FILE_MISSING":
            "기업목록 파일을 "
            "불러오지 못했습니다.",

        "CORP_CODES_FORMAT_ERROR":
            "기업목록 파일 형식에 "
            "문제가 있습니다.",

        "DART_NETWORK_ERROR":
            "DART 연결에 실패했습니다. "
            "잠시 후 다시 시도해주세요.",

        "DART_HTTP_ERROR":
            "DART 서버 응답에 문제가 있습니다. "
            "잠시 후 다시 시도해주세요.",

        "DART_RESPONSE_ERROR":
            "DART 응답을 "
            "처리하지 못했습니다.",

        "DART_API_ERROR":
            "DART API 요청을 "
            "처리하지 못했습니다.",

        "NAVER_NETWORK_ERROR":
            "NAVER 뉴스 연결에 "
            "실패했습니다.",

        "NAVER_HTTP_ERROR":
            "NAVER 뉴스 서버 응답에 "
            "문제가 있습니다.",

        "NAVER_RESPONSE_ERROR":
            "NAVER 뉴스 응답을 "
            "처리하지 못했습니다.",
    }

    return messages.get(
        error_code,
        "분석 중 일시적인 오류가 발생했습니다. "
        "잠시 후 다시 시도해주세요.",
    )


# ============================================================
# UI
# ============================================================

company_query = st.text_input(
    "분석할 회사명을 입력하세요",
    placeholder=
        "예: 팬오션, 삼성전자, 대한항공",
)

analyze_button = st.button(
    "🔍 기업 분석 시작",
    type="primary",
    use_container_width=True,
)


if analyze_button:

    if not company_query.strip():

        st.warning(
            "회사명을 입력해주세요."
        )

        st.stop()

    try:

        with st.spinner(
            "DART 재무정보와 "
            "최신 뉴스를 분석하고 있습니다..."
        ):

            result = analyze_company(
                company_query.strip()
            )

    except (
        ValueError,
        RuntimeError,
        FileNotFoundError,
    ) as e:

        st.error(
            render_safe_error(
                str(e)
            )
        )

        st.stop()

    except Exception:

        st.error(
            "분석 중 일시적인 오류가 발생했습니다. "
            "잠시 후 다시 시도해주세요."
        )

        st.stop()

    company = result[
        "company"
    ]

    st.success(
        f"{company['회사명']} "
        "분석이 완료되었습니다."
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    col1.metric(
        "회사명",
        company["회사명"],
    )

    col2.metric(
        "종목코드",
        company["종목코드"]
        or "-",
    )

    col3.metric(
        "분석모드",
        result["analysis_mode"],
    )

    if (
        result["analysis_mode"]
        == "금융회사"
    ):

        st.warning(
            "현재 금융회사 전용 "
            "BIS · NIM · 건전성 지표 모듈은 "
            "개발 중입니다. "
            "일반기업 비율은 적용하지 않습니다."
        )

    tab1, tab2, tab3, tab4 = (
        st.tabs([
            "📊 재무",
            "🏦 RM View",
            "📈 PB View",
            "📰 핵심 뉴스",
        ])
    )

    report = result[
        "report"
    ]

    with tab1:

        st.subheader(
            "최근 3개년 재무정보"
        )

        st.caption(
            "단위: 억원"
        )

        st.dataframe(
            result[
                "financial"
            ].round(1),
            use_container_width=True,
        )

        if not result[
            "ratios"
        ].empty:

            st.subheader(
                "핵심 재무비율"
            )

            st.dataframe(
                result[
                    "ratios"
                ],
                use_container_width=True,
            )

        if report:

            st.subheader(
                "AI 핵심 재무 흐름"
            )

            for item in (
                report
                .financial_summary
            ):

                st.write(
                    f"• {item}"
                )

            st.subheader(
                "추가 확인사항"
            )

            for item in (
                report
                .additional_checks
            ):

                st.write(
                    f"• {item}"
                )

    with tab2:

        if report:

            st.subheader(
                "기업금융 RM View"
            )

            for item in (
                report.rm_view
            ):

                st.write(
                    f"• {item}"
                )

        else:

            st.info(
                "AI 분석 결과가 없습니다."
            )

    with tab3:

        if report:

            st.subheader(
                "PB View"
            )

            for item in (
                report.pb_view
            ):

                st.write(
                    f"• {item}"
                )

        else:

            st.info(
                "AI 분석 결과가 없습니다."
            )

    with tab4:

        if report:

            issues = [
                issue
                for issue
                in report.news_issues
                if issue.importance >= 3
            ][:5]

            if not issues:

                st.info(
                    "중요도 3 이상의 "
                    "핵심 뉴스가 없습니다."
                )

            for number, issue in enumerate(
                issues,
                1,
            ):

                st.markdown(
                    f"### {number}. "
                    f"{issue.issue_name}"
                )

                st.caption(
                    f"{issue.category} · "
                    f"중요도 "
                    f"{issue.importance}/5"
                )

                for item in (
                    issue.summary
                ):

                    st.write(
                        f"• {item}"
                    )

                st.markdown(
                    "**🏦 뉴스 RM View**"
                )

                for item in (
                    issue.rm_view
                ):

                    st.write(
                        f"• {item}"
                    )

                st.markdown(
                    "**📈 뉴스 PB View**"
                )

                for item in (
                    issue.pb_view
                ):

                    st.write(
                        f"• {item}"
                    )

                if (
                    issue
                    .comparison_reference
                ):

                    st.markdown(
                        "**🔎 비교 참고**"
                    )

                    for item in (
                        issue
                        .comparison_reference
                    ):

                        st.write(
                            f"• {item}"
                        )

                if (
                    1
                    <= issue.article_index
                    <= len(
                        result[
                            "candidates"
                        ]
                    )
                ):

                    source = result[
                        "candidates"
                    ][
                        issue.article_index
                        - 1
                    ]

                    article_url = (
                        source[
                            "원문링크"
                        ]
                        or source[
                            "네이버링크"
                        ]
                    )

                    st.markdown(
                        f"**대표기사:** "
                        f"{source['제목']}"
                    )

                    if article_url:

                        st.link_button(
                            "🔗 기사 원문 보기",
                            article_url,
                        )

                st.divider()

        else:

            st.error(
                "Gemini AI 분석을 "
                "완료하지 못했습니다."
            )

            if (
                result["ai_error"]
                == "NO_ARTICLE_BODY"
            ):

                st.caption(
                    "분석 가능한 기사본문을 "
                    "충분히 확보하지 못했습니다."
                )

            else:

                st.caption(
                    "Gemini 응답이 일시적으로 "
                    "실패했습니다. "
                    "DART 재무정보와 뉴스 수집 결과에는 "
                    "영향을 주지 않습니다."
                )


st.divider()

st.caption(
    "본 서비스는 기업 리서치 보조 도구이며 "
    "투자 권유 또는 매수·매도 추천을 제공하지 않습니다. "
    "중요한 의사결정 전에는 DART 공시와 기사 원문을 "
    "직접 확인하세요."
)
