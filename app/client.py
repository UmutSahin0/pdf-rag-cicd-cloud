import requests

def ask_question(question: str):
    url = "http://127.0.0.1:8000/ask"  # FastAPI sunucun burada çalışıyor olmalı
    payload = {"question": question}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        answer = response.json().get("answer", "No answer returned.")
        print("✅ Cevap:", answer)
    except requests.exceptions.RequestException as e:
        print("❌ Hata oluştu:", e)


if __name__ == "__main__":
    # Buraya istediğin soruyu yazabilirsin
    ask_question("Umut Şahin kız arkadaşının ismi ne?")
