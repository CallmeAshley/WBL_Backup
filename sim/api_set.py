import os
import openai
from openai import OpenAI
import getpass
from dotenv import load_dotenv

load_dotenv()


def setup_api_key():
    api_key = os.getenv("UPSTAGE_API_KEY")

    if not api_key:
        print("Upstage API 키가 환경변수에 설정되지 않았습니다.")
        print("API 키를 직접 입력하거나, 환경변수 UPSTAGE_API_KEY에 설정해주세요.")
        print("\n옵션 1: 직접 입력 (현재 세션에서만 유효)")
        api_key = getpass.getpass("Upstage API 키를 입력하세요: ")

    if api_key:
        os.environ["UPSTAGE_API_KEY"] = api_key
        print("API 키가 설정되었습니다!")
        return api_key
    else:
        print("API 키가 설정되지 않았습니다.")
        return None


# API 키 설정
api_key = setup_api_key()

# OpenAI 클라이언트 설정 (Upstage API 엔드포인트 사용)
if api_key:
    client = OpenAI(api_key=api_key, base_url="https://api.upstage.ai/v1/solar")
    print("solar-pro2 API가 준비되었습니다!")
else:
    print("API 클라이언트를 설정할 수 없습니다. API 키를 확인해주세요.")
    
