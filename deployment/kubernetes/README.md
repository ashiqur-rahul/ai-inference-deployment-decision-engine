# Kubernetes Deployment Simulation

Apply manifests:

```bash
kubectl apply -f dashboard-deployment.yaml
kubectl apply -f api-deployment.yaml
kubectl apply -f hpa.yaml
```

Port-forward dashboard:

```bash
kubectl port-forward svc/ai-deployment-dashboard-service 8501:8501
```

Port-forward API:

```bash
kubectl port-forward svc/ai-deployment-api-service 8000:8000
```
