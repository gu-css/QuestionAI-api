import requests
import concurrent.futures
import time

URL = "https://clickbait-api.onrender.com/predict"
CONCURRENCY = 30
TOTAL_REQUESTS = 300
HEADLINE = "Descubra o segredo chocante que vai mudar sua vida"

def send_request(req_id):
    start = time.time()
    try:
        response = requests.post(URL, params={"headline": f"{HEADLINE} {req_id}"}, timeout=20)
        elapsed = time.time() - start
        return {"status": response.status_code, "time": elapsed, "id": req_id}
    except Exception as e:
        elapsed = time.time() - start
        return {"status": "ERROR", "time": elapsed, "error": str(e), "id": req_id}

def main():
    print(f"Iniciando Teste de Estresse PESADO na API...")
    print(f"URL: {URL}")
    print(f"Requisições Totais: {TOTAL_REQUESTS}")
    print(f"Concorrência Máxima: {CONCURRENCY}")
    
    start_total = time.time()
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [executor.submit(send_request, i) for i in range(TOTAL_REQUESTS)]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(future.result())
            if i % 50 == 0:
                print(f"Completadas: {i}/{TOTAL_REQUESTS}...")
                
    total_time = time.time() - start_total
    
    success = [r for r in results if r["status"] == 200]
    errors = [r for r in results if r["status"] != 200]
    
    times = [r["time"] for r in success]
    avg_time = sum(times) / len(times) if times else 0
    max_time = max(times) if times else 0
    min_time = min(times) if times else 0
    
    print("\n--- RESULTADO DO TESTE DE ESTRESSE PESADO ---")
    print(f"Requisições Totais: {TOTAL_REQUESTS}")
    print(f"Sucesso (HTTP 200): {len(success)}")
    print(f"Falhas / Erros:  {len(errors)}")
    print(f"Tempo Total: {total_time:.2f} segundos")
    if times:
        print(f"Tempo Médio (Sucesso): {avg_time:.3f} segundos")
        print(f"Tempo Mais Rápido: {min_time:.3f} segundos")
        print(f"Tempo Mais Lento: {max_time:.3f} segundos")
    print("---------------------------------------------")
    
    with open("docs/resultado_teste_de_estresse_pesado.txt", "w") as f:
        f.write(f"--- RESULTADO DO TESTE DE ESTRESSE PESADO ---\n")
        f.write(f"Requisições Totais: {TOTAL_REQUESTS}\n")
        f.write(f"Concorrência Máxima: {CONCURRENCY}\n")
        f.write(f"Sucesso (HTTP 200): {len(success)}\n")
        f.write(f"Falhas / Erros:  {len(errors)}\n")
        f.write(f"Tempo Total: {total_time:.2f} segundos\n")
        if times:
            f.write(f"Tempo Médio (Sucesso): {avg_time:.3f} segundos\n")
            f.write(f"Tempo Mais Rápido: {min_time:.3f} segundos\n")
            f.write(f"Tempo Mais Lento: {max_time:.3f} segundos\n")
        f.write("---------------------------------------------\n")

if __name__ == "__main__":
    main()
