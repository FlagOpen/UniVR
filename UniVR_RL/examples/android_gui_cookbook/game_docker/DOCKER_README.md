# Number Selection Game - Docker Image

## 📦 Image Information

**Image name**: `number-game-rl`  
**Current version**: `v1.4`  
**Image registry**: `ccr.ccs.tencentyun.com/yuehuazhang/number-game-rl`  
**Architecture**: `linux/amd64`  
**Size**: ~124MB
**Base image**: `python:3.11-slim`

## 🚀 Usage

### 1. Docker Deployment

```bash
# Pull and run from Tencent Cloud image registry
docker run -d \
  --name number-game \
  -p 8000:8000 \
  ccr.ccs.tencentyun.com/yuehuazhang/number-game-rl:v1.4

# Custom port (e.g. map to 9000)
docker run -d \
  --name number-game \
  -p 9000:8000 \
  ccr.ccs.tencentyun.com/yuehuazhang/number-game-rl:v1.4
```

### 2. Access the Game

Open your browser and visit:
```
http://localhost:8000/number_game.html
```

### 3. Kubernetes Deployment (Recommended)

Deploy using the provided `game.yaml` configuration file:

```bash
# Deploy to Kubernetes cluster
kubectl apply -f game.yaml
```

**game.yaml configuration description:**

```yaml
# Deployment configuration
apiVersion: apps/v1
kind: Deployment
metadata:
  name: number-game
spec:
  replicas: 1                                    # Number of replicas
  template:
    spec:
      containers:
      - name: number-game
        image: ccr.ccs.tencentyun.com/yuehuazhang/number-game-rl:v1.4
        imagePullPolicy: IfNotPresent            # Image pull policy
        ports:
        - containerPort: 8000
        resources:
          limits:
            cpu: "2"                             # CPU limit: 2 cores
            memory: 4Gi                          # Memory limit: 4GB
          requests:
            cpu: "2"                             # CPU request: 2 cores
            memory: 4Gi                          # Memory request: 4GB

---
# Service configuration (LoadBalancer type)
apiVersion: v1
kind: Service
metadata:
  name: number-game
  annotations:
    service.cloud.tencent.com/direct-access: "true"  # Tencent Cloud direct access
spec:
  type: LoadBalancer                             # Use load balancer
  allocateLoadBalancerNodePorts: false           # Do not allocate node ports
  ports:
  - name: 8000-8000-tcp
    port: 8000
    targetPort: 8000
    protocol: TCP
  selector:
    k8s-app: number-game
```

**Access after deployment:**

```bash
# Check service status
kubectl get svc number-game

# Get LoadBalancer external IP
kubectl get svc number-game -o jsonpath='{.status.loadBalancer.ingress[0].ip}'

# Access the game (replace with the actual external IP)
# http://<EXTERNAL-IP>:8000/number_game.html
```

**Scaling:**

```bash
# Scale replicas
kubectl scale deployment number-game --replicas=3

# Check Pod status
kubectl get pods -l k8s-app=number-game
```

**Delete deployment:**

```bash
kubectl delete -f game.yaml
```

## 🎮 Game Description

This is a **conditional-reversal number selection game** for reinforcement learning training.

### Game Rules

1. **Observe the indicator lights** (3 circles at the top of the screen):
   - 🟢 Green light: select the **largest** number
   - 🔴 Red light: select the **smallest** number
   - 🟡 Yellow light: select the **middle** number

2. **Scoring rules**:
   - Correct selection: +10 points
   - Incorrect selection: -10 points

3. **Game objective**: Complete 10 rounds and achieve the highest score

### Supported Resolutions

- Optimized for: 720x1280 (Android devices)
- Compatible with: desktop browsers, tablets, mobile phones

## 🔧 Image Contents

```
/app/
  └── number_game.html  # Game HTML file (includes CSS and JavaScript)
```

## 📝 Environment Variables

No environment variables need to be configured; works out of the box.

## 🐛 Troubleshooting

### Container fails to start
```bash
docker logs number-game
```

### Port conflict
```bash
# Use a different port
docker run -d --name number-game -p 9000:8000 number-game-rl:v1.0
```

### Check container status
```bash
docker ps -a | grep number-game
```
