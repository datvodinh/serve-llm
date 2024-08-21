# Serve LLM with high throughput and scalable using Ray and vLLM

## Install

```bash
cd serve
make
```

## Serve

- Locally
  
```bash
make serve
```

- Docker

```bash
docker run --gpus all -v .:/code ghcr.io/datvodinh/serve:latest
```

- Kubernetes

```bash
kubectl apply -f k8s/ray-service.yaml
```
