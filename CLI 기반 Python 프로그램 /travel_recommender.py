import os
import sys
import re
import json
import argparse
from datetime import datetime

import requests
from google import genai
from google.genai import types as genai_types

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
KAKAO_KEYWORD_SEARCH_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
RESULTS_DIR = "results"
RESTAURANT_DISPLAY_COUNT = 5
REQUEST_TIMEOUT = 30

def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gemini API + 카카오맵 API를 이용한 국내 여행 추천 프로그램"
    )
    parser.add_argument(
        "-date",
        required=True,
        help='여행 날짜, "YYYY-MM-DD" 형식 (예: 2026-10-05)',
    )
    return parser.parse_args()


def validate_date(date_str: str) -> str:
    """YYYY-MM-DD 형식인지 검증한다. 아니면 사용법을 출력하고 종료한다."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except ValueError:
        print("[오류] 날짜 형식이 올바르지 않습니다.")
        print('사용법: python travel_recommender.py -date "YYYY-MM-DD"')
        print('예시:   python travel_recommender.py -date "2026-10-05"')
        sys.exit(1)

def load_api_keys() -> dict:
    keys = {
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY"),
        "KAKAO_REST_API_KEY": os.environ.get("KAKAO_REST_API_KEY"),
    }

    missing = [k for k, v in keys.items() if not v]
    if missing:
        print("[오류] 다음 환경변수가 설정되어 있지 않습니다:", ", ".join(missing))
        print()
        print("설정 방법 안내")
        print("-" * 50)
        print("1) 프로젝트 루트에 .env 파일을 만들고 아래처럼 작성하세요.")
        print("   GEMINI_API_KEY=your_gemini_api_key")
        print("   KAKAO_REST_API_KEY=your_kakao_rest_api_key")
        print()
        print("   또는 터미널에서 직접 환경변수로 설정할 수 있습니다.")
        print('   [macOS/Linux]  export GEMINI_API_KEY="YOUR_KEY"')
        print('   [Windows PS]   $env:GEMINI_API_KEY="YOUR_KEY"')
        print("-" * 50)
        print("참고: 카카오 REST API 키는 Kakao Developers 앱 설정에서")
        print("      '카카오맵' 사용 설정을 ON으로 해야 로컬 API를 호출할 수 있습니다.")
        sys.exit(1)

    return keys

def build_genai_client(api_key: str) -> genai.Client:
    """요청마다 새로 만들지 않도록 main()에서 한 번만 생성해 재사용한다."""
    return genai.Client(
        api_key=api_key,
        http_options=genai_types.HttpOptions(timeout=REQUEST_TIMEOUT * 1000),
    )


def call_gemini(client: genai.Client, prompt: str) -> str:
    """Gemini API를 호출해 생성된 텍스트를 반환한다. 실패 시 예외를 던진다."""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    if not response.text:
        raise ValueError("Gemini 응답 텍스트가 비어 있습니다(안전 필터 등으로 후보가 없을 수 있음).")
    return response.text


def extract_json_block(text: str) -> dict:
    """LLM 응답에서 코드펜스/잡텍스트를 제거하고 JSON 객체를 파싱한다."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned.strip(), flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned.strip()).strip()
    
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise json.JSONDecodeError("JSON 객체를 찾을 수 없습니다.", cleaned, 0)

    json_str = cleaned[start : end + 1]
    return json.loads(json_str)

REQUIRED_KEYS_SCHEMA = {
    "recommended_city": str,
    "weather": str,
    "events": list,
    "reason": str,
}

def build_first_prompt(date_str: str, retry: bool = False) -> str:
    base_instruction = f"""
당신은 국내 여행 추천 도우미입니다. 사용자가 입력한 날짜 "{date_str}"를 기준으로,
해당 시기에 여행하기 좋은 국내(대한민국) 도시 한 곳을 추천하세요.

반드시 아래 JSON 스키마와 동일한 키를 가진 JSON 객체 "하나만" 출력하세요.
설명 문장, 코드펜스(```), 그 외 부가 텍스트를 절대 포함하지 마세요.

스키마:
{{
  "recommended_city": "string, 예: 제주",
  "weather": "string, 해당 시기 일반적인 날씨 요약",
  "events": ["string", "1~3개의 행사/축제 후보"],
  "reason": "string, 추천 근거 2~4문장"
}}
"""
    if retry:
        base_instruction += (
            "\n\n이전 응답이 올바른 JSON으로 파싱되지 않았습니다. "
            "위 스키마의 필수 키만 포함한 순수 JSON 객체만 다시 출력하세요."
        )
    return base_instruction.strip()


def validate_schema(data: dict) -> bool:
    for key, expected_type in REQUIRED_KEYS_SCHEMA.items():
        if key not in data:
            return False
        if not isinstance(data[key], expected_type):
            return False
    return True


def get_first_recommendation(date_str: str, client: genai.Client, errors: list) -> dict:
    log("1단계: Gemini API로 날씨/행사 기반 1차 추천을 요청합니다...")

    for attempt in range(2):
        retry = attempt == 1
        if retry:
            log("  - JSON 파싱 실패로 재시도합니다 (1회 한정)...")
        try:
            prompt = build_first_prompt(date_str, retry=retry)
            text = call_gemini(client, prompt)
            parsed = extract_json_block(text)
            if not validate_schema(parsed):
                raise ValueError("필수 키/타입이 스키마와 일치하지 않습니다.")
            log(f"  - 추천 도시: {parsed['recommended_city']}")
            return parsed
        except Exception as e:
            msg = f"1차 추천 생성 오류 (시도 {attempt + 1}/2, {type(e).__name__}): {e}"
            log(f"  - {msg}")
            errors.append(msg)
            
    fallback = {
        "recommended_city": "정보 없음",
        "weather": "정보 없음",
        "events": [],
        "reason": "1차 추천 생성에 실패하여 기본값으로 대체되었습니다.",
    }
    log("  - 1차 추천 생성에 최종 실패. 기본값으로 대체 후 다음 단계로 진행합니다.")
    return fallback

def search_restaurants(city: str, kakao_rest_api_key: str, errors: list) -> list:
    log(f"2단계: 카카오맵 API로 '{city}' 맛집을 검색합니다...")

    if not city or city == "정보 없음":
        msg = "추천 도시 정보가 없어 맛집 검색을 건너뜁니다."
        log(f"  - {msg}")
        errors.append(msg)
        return []

    headers = {"Authorization": f"KakaoAK {kakao_rest_api_key}"}
    params = {
        "query": f"{city} 맛집",
        "size": RESTAURANT_DISPLAY_COUNT,
        "page": 1,
        "sort": "accuracy",
    }

    try:
        response = requests.get(
            KAKAO_KEYWORD_SEARCH_URL,
            headers=headers,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        
        response.raise_for_status()
        data = response.json()
        documents = data.get("documents", [])

        restaurants = []
        for item in documents:
            restaurants.append(
                {
                    "name": item.get("place_name", ""),
                    "address": item.get("road_address_name") or item.get("address_name", ""),
                    "category": item.get("category_name", ""),
                    "url": item.get("place_url", ""),
                    "x": item.get("x", ""),
                    "y": item.get("y", ""),
                }
            )

        if not restaurants:
            log("  - 검색 결과 0건. '데이터 없음'으로 다음 단계로 진행합니다.")
        else:
            log(f"  - 맛집 {len(restaurants)}건 검색 완료.")
        return restaurants

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "unknown"
        detail = ""
        try:
            detail = e.response.json().get("message", "") if e.response is not None else ""
        except ValueError:
            pass
        msg = (
            f"카카오맵 API 인증/요청 오류 (status={status}) {detail}: "
            "KAKAO_REST_API_KEY 값과 앱의 '카카오맵' 사용 설정(ON) 여부를 확인하세요."
        )
        log(f"  - {msg}")
        errors.append(msg)
        return []
    except (requests.RequestException, ValueError) as e:
        msg = f"카카오맵 API 호출 실패 ({type(e).__name__}): {e}"
        log(f"  - {msg}")
        errors.append(msg)
        return []

def build_report_prompt(date_str: str, first_json: dict, restaurants: list, errors: list) -> str:
    restaurants_text = (
        json.dumps(restaurants, ensure_ascii=False, indent=2) if restaurants else "[]"
    )
    errors_text = json.dumps(errors, ensure_ascii=False, indent=2) if errors else "[]"

    return f"""
당신은 국내 여행 리포트 작성 도우미입니다. 아래 데이터를 바탕으로 여행 날짜 "{date_str}"에 대한
최종 여행 리포트를 Markdown 형식으로 작성하세요.

[1차 추천 데이터]
{json.dumps(first_json, ensure_ascii=False, indent=2)}

[맛집 검색 결과] (빈 배열이면 "데이터 없음"으로 표기)
{restaurants_text}

[오류 목록] (참고용, 빈 배열이면 리포트에 언급하지 않아도 됨)
{errors_text}

리포트에는 반드시 아래 항목을 Markdown 섹션(##)으로 포함하세요:
1. 추천 지역 및 추천 이유 요약
2. 날씨 요약
3. 행사/축제 목록 (없으면 "행사 정보 없음"으로 표기)
4. 맛집 리스트 (이름/주소/카테고리/링크 포함, 0건이면 "데이터 없음"으로 표기)
5. 1일 일정 제안 (오전 / 오후 / 저녁 구성)

Markdown 텍스트만 출력하고, 다른 설명이나 코드펜스는 포함하지 마세요.
""".strip()

def generate_final_report(
    date_str: str, first_json: dict, restaurants: list, errors: list, client: genai.Client
) -> str:
    log("3단계: Gemini API로 최종 여행 리포트를 생성합니다...")
    prompt = build_report_prompt(date_str, first_json, restaurants, errors)

    try:
        text = call_gemini(client, prompt)
        cleaned = re.sub(r"^```(markdown|md)?", "", text.strip(), flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned.strip()).strip()
        log("  - 리포트 생성 완료.")
        return cleaned
    except Exception as e:
        msg = f"최종 리포트 생성 오류 ({type(e).__name__}): {e}"
        log(f"  - {msg}")
        errors.append(msg)
        return build_fallback_report(date_str, first_json, restaurants, errors)

def build_fallback_report(date_str: str, first_json: dict, restaurants: list, errors: list) -> str:
    lines = [f"# {date_str} 국내 여행 추천 리포트 (자동 생성 실패로 인한 대체 리포트)", ""]
    lines.append("## 추천 지역 및 추천 이유")
    lines.append(f"- 추천 지역: {first_json.get('recommended_city', '정보 없음')}")
    lines.append(f"- 추천 이유: {first_json.get('reason', '정보 없음')}")
    lines.append("")
    lines.append("## 날씨 요약")
    lines.append(first_json.get("weather", "정보 없음"))
    lines.append("")
    lines.append("## 행사/축제 목록")
    events = first_json.get("events", [])
    if events:
        for ev in events:
            lines.append(f"- {ev}")
    else:
        lines.append("행사 정보 없음")
    lines.append("")
    lines.append("## 맛집 리스트")
    if restaurants:
        for r in restaurants:
            lines.append(f"- {r['name']} ({r.get('category', '')}) - {r.get('address', '')}")
    else:
        lines.append("데이터 없음")
    lines.append("")
    lines.append("## 1일 일정 제안")
    lines.append("- 오전: 추천 지역 주요 명소 방문")
    lines.append("- 오후: 자유 여행 또는 인근 관광지 탐방")
    lines.append("- 저녁: 맛집 리스트 참고하여 식사")
    lines.append("")
    if errors:
        lines.append("## errors")
        for e in errors:
            lines.append(f"- {e}")
    return "\n".join(lines)

def save_results(
    date_str: str, first_json: dict, restaurants: list, errors: list, report_md: str
) -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    raw_data = {
        "date": date_str,
        "generated_at": datetime.now().isoformat(),
        "first_recommendation": first_json,
        "restaurants": restaurants,
        "errors": errors,
    }

    json_path = os.path.join(RESULTS_DIR, f"{date_str}_raw.json")
    md_path = os.path.join(RESULTS_DIR, f"{date_str}_report.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    return {"json_path": json_path, "md_path": md_path}

def main() -> None:
    args = parse_args()
    date_str = validate_date(args.date)

    log(f"국내 여행 추천 프로그램을 시작합니다. (날짜: {date_str})")

    keys = load_api_keys()
    errors: list = []

    client = build_genai_client(keys["GEMINI_API_KEY"])

    first_json = get_first_recommendation(date_str, client, errors)

    restaurants = search_restaurants(
        first_json.get("recommended_city", ""),
        keys["KAKAO_REST_API_KEY"],
        errors,
    )

    report_md = generate_final_report(date_str, first_json, restaurants, errors, client)

    log("결과를 저장합니다...")
    paths = save_results(date_str, first_json, restaurants, errors, report_md)

    print()
    print("=" * 60)
    print("완료되었습니다! 결과물이 아래 경로에 저장되었습니다.")
    print(f"  - 원본 데이터(JSON): {os.path.abspath(paths['json_path'])}")
    print(f"  - 최종 리포트(Markdown): {os.path.abspath(paths['md_path'])}")
    if errors:
        print(f"  - 처리 중 {len(errors)}건의 오류가 발생했습니다 (리포트/JSON의 errors 참고).")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[중단] 사용자에 의해 프로그램이 중단되었습니다.")
        sys.exit(1)
