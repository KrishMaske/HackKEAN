import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings


HELLO_WORLD_VERSION = "5c7d5dc6dd8bf75c1acaa8565735e7986bc5b66206b55cca93cb72c9bf15ccaa"


def print_response_error(response: requests.Response) -> None:
    try:
        data = response.json()
        title = data.get("title") or "Replicate error"
        detail = data.get("detail") or data.get("error") or response.text
        print(f"{title}: {detail}")
    except ValueError:
        print(response.text[:1000])


def check_auth() -> bool:
    if not settings.replicate_api_token:
        print("REPLICATE_API_TOKEN is missing from backend/.env")
        return False

    response = requests.get(
        "https://api.replicate.com/v1/predictions",
        headers={"Authorization": f"Bearer {settings.replicate_api_token}"},
        timeout=30,
    )
    print(f"Auth check status: {response.status_code}")

    if response.status_code != 200:
        print_response_error(response)
        return False

    print("Token is valid. Replicate API is reachable.")
    return True


def check_prediction() -> bool:
    payload = {
        "version": HELLO_WORLD_VERSION,
        "input": {"text": "SceneShift"},
    }
    response = requests.post(
        "https://api.replicate.com/v1/predictions",
        headers={
            "Authorization": f"Bearer {settings.replicate_api_token}",
            "Content-Type": "application/json",
            "Prefer": "wait=10",
        },
        json=payload,
        timeout=30,
    )
    print(f"Prediction check status: {response.status_code}")

    if response.status_code != 201:
        print_response_error(response)
        return False

    data = response.json()
    print(f"Prediction status: {data.get('status')}")
    print(f"Prediction output: {data.get('output')}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Replicate token and optional prediction access.")
    parser.add_argument(
        "--prediction-test",
        action="store_true",
        help="Create a tiny hello-world prediction. This can require account credit.",
    )
    args = parser.parse_args()

    if not check_auth():
        return 1

    if args.prediction_test and not check_prediction():
        return 1

    if not args.prediction_test:
        print("Skipped prediction test. Run with --prediction-test after adding billing credit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
