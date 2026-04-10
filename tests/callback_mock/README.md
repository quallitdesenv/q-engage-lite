# Callback Mock Server

Servidor HTTP simples para testar callbacks do tracker.

## Uso

1. Iniciar o servidor mock:
```bash
python -m tests.callback_mock
```

2. Configurar o endpoint no `settings.json`:
```json
{
  "event": {
    "endpoint": "http://localhost:8080"
  }
}
```

3. Executar o tracker:
```bash
python -m tracker
```

4. Os payloads recebidos serão salvos em `./results/`

## Formato dos arquivos

- Nome: `detections_YYYYMMDD_HHMMSS_microseconds.json`
- Localização: `tests/callback_mock/results/`
