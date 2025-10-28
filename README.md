# IsaacLab 도커 사용시 주의사항
./docker/container.py start ros2 실행시 

## 마운트 설정
docker-compose.yaml 파일을 보면 다음과 같이 bind 마운트가 설정되어 있습니다:
- type: bind

  source: ../source

  target: ${DOCKER_ISAACLAB_PATH}/source

- type: bind

  source: ../scripts

  target: ${DOCKER_ISAACLAB_PATH}/scripts

- type: bind

  source: ../docs

  target: ${DOCKER_ISAACLAB_PATH}/docs

- type: bind

  source: ../tools

  target: ${DOCKER_ISAACLAB_PATH}/tools

## 실제 경로 매핑
.env.base에서 DOCKER_ISAACLAB_PATH=/workspace/isaaclab로 정의되어 있으므로:

- 컨테이너: /workspace/isaaclab/source ↔ 로컬: /home/kimkh/IsaacLab/source
- 컨테이너: /workspace/isaaclab/scripts ↔ 로컬: /home/kimkh/IsaacLab/scripts
- 컨테이너: /workspace/isaaclab/docs ↔ 로컬: /home/kimkh/IsaacLab/docs
- 컨테이너: /workspace/isaaclab/tools ↔ 로컬: /home/kimkh/IsaacLab/tools

결론: 컨테이너 내부에서 /workspace/isaaclab 디렉토리의 하위 폴더들(source, scripts, docs, tools)은 현재 로컬의 /home/kimkh/IsaacLab 디렉토리와 실시간으로 동기화됩니다. 즉, 컨테이너에서 파일을 수정하면 로컬에도 반영되고, 로컬에서 수정해도 컨테이너에 즉시 반영됩니다.

# 현재 모델 설계 
## F1TENTH 레이싱 환경 (f1tenth_env.py)
### 관측 공간 (Observation):
- LiDAR 센서: 1080개 ray (270도 스캔, 0.25도 해상도)
- 차량 상태: 속도(speed) + 조향각(steering)
- 총 차원: 1082차원 (1080 + 2)
### 행동 공간 (Action):
- 조향 속도 (steering velocity): [-3.2, 3.2] rad/s
- 가속도 (acceleration): [-9.51, 9.51] m/s²
- 총 차원: 2차원 (연속 행동 공간)
### 물리 시뮬레이션:
- 차량 동역학 모델 (f1tenth_gym 기반)
- VESC 스타일 속도 제어
- 실제 F1TENTH 하드웨어 파라미터 사용
### 보상 함수:
- 진행 보상: 전진 거리 × 20 (높은 가중치)
- 속도 보상: 최대 +1.2 (5m/s 이상)
- **저속 페널티 (연속형)**: Inverse 함수, 1m/s에서 0, 정지 시 -38
- **벽 근접 페널티 (연속형)**: 지수 함수, 25cm부터 시작, 5cm에서 -11.9
- 충돌 페널티: 제거 (연속 페널티로 대체)

---

# SAC 학습 사용법

## 개요
CNN과 MLP를 동시에 학습하여 LiDAR 기반 자율주행을 학습합니다.

### 학습 전략
CNN (LiDAR 특징 추출) + MLP (제어 정책) 동시 학습

---

## 학습 시작

### 실행 명령어
```bash
cd /home/kimkh/IsaacLab

# 학습 시작 (헤드리스 모드)
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-F1tenth-Direct-v0 \
    --algorithm SAC \
    --num_envs 32 \
    --headless

# 시각화 모드로 학습 (학습 과정 확인)
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-F1tenth-Direct-v0 \
    --algorithm SAC \
    --num_envs 8
```

### 학습 결과
- **체크포인트 저장 위치**: `logs/skrl/f1tenth_sac/[날짜_시간]/checkpoints/best_agent.pt`
- **주요 지표**: 랩 완주율, 평균 속도, 충돌률

---

## 학습률 설정 (자동)

CNN과 MLP 동시 학습:
- Policy CNN: 1e-4
- Policy MLP: 1e-4
- Critic CNN: 5e-4
- Critic MLP: 5e-4

---

## 네트워크 아키텍처

### CNN (LiDAR Feature Extractor)
```
Input: (batch, 1, 1080)
Conv1D: 1 → 32 channels (kernel=5, stride=2)
Conv1D: 32 → 64 channels (kernel=3, stride=2)
Conv1D: 64 → 128 channels (kernel=3, stride=2)
Flatten + FC: 17280 → 128 features
```

### Policy MLP
```
Input: 128 (LiDAR) + 2 (vehicle state) = 130
Hidden: 130 → 128 → 128 → 64
Output: 2 actions (steering velocity, acceleration)
```

### Critic MLP
```
Input: 128 (LiDAR) + 2 (vehicle state) + 2 (actions) = 132
Hidden: 132 → 256 → 256 → 128
Output: 1 (Q-value)
```

---

## 학습 모니터링

### TensorBoard 사용
```bash
# 학습 중 실시간 모니터링
tensorboard --logdir logs/skrl/f1tenth_sac
```

### 주요 메트릭
- `Reward/episode`: 에피소드당 누적 보상
- `Episode_length`: 에피소드 길이 (생존 시간)
- `Collision_rate`: 충돌률
- `Lap_time`: 랩타임
- `Average_speed`: 평균 속도

---

## 체크포인트 관리

### 자동 저장
- `best_agent.pt`: 최고 성능 모델 (Reward 기준)
- `agent_10000.pt`: 10000 스텝마다 주기적 저장

### 학습 재개
```bash
# 중단된 학습 이어서 하기
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-F1tenth-Direct-v0 \
    --algorithm SAC \
    --num_envs 32 \
    --checkpoint logs/skrl/f1tenth_sac/[날짜_시간]/checkpoints/agent_10000.pt \
    --headless
```

---

## 예상 학습 시간 (단일 GPU)

| 학습 목표 | 스텝 수 | 예상 시간 |
|----------|--------|-----------|
| 기본 주행 능력 | ~100,000 | 4-6시간 |
| 충돌 회피 개선 | ~250,000 | 10-15시간 |
| 최종 모델 (랩타임 최적화) | ~500,000 | 20-30시간 |

---

## 트러블슈팅

### 문제 1: 체크포인트 로딩 실패
```
해결: 체크포인트 경로가 정확한지 확인
ls logs/skrl/f1tenth_sac/*/checkpoints/
```

### 문제 2: GPU 메모리 부족
```
해결: --num_envs를 줄이기 (32 → 16 → 8)
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-F1tenth-Direct-v0 \
    --algorithm SAC \
    --num_envs 16 \
    --headless
```

### 문제 3: 학습이 너무 느림
```
해결: 헤드리스 모드 사용 (--headless)
GPU 사용 확인: nvidia-smi
```

### 문제 4: 학습이 수렴하지 않음
```
해결: 학습률 조정 또는 네트워크 크기 확인
CNN과 MLP가 동시에 학습되므로 초반에는 불안정할 수 있음
충분한 스텝(100,000+)을 학습하여 안정화 대기
```
