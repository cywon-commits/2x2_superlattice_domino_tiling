# 작업 지시: 8겹 타일링 QMC 검증 (남은 계산)

Claude Code에서 이 폴더를 열고 아래 순서대로 실행한다. 모든 계는 이분 그래프라 부호 문제가 없다.

## 0. 준비

```
pip install numpy scipy numba
cd sse && python validate.py      # 1–2분, 마지막 줄이 ALL PASS여야 한다
cd ..
```

ALL PASS가 아니면 여기서 멈추고 출력 전체를 보고한다.

## 1. 남은 작업 실행

```
python run_queue.py -j <코어 수>
```

큐는 끝난 작업을 건너뛰므로 중단돼도 같은 명령으로 이어서 돌린다. `runs/`에는 이미 끝난 5개(시드 1)가 들어 있다.

| 묶음 | 계 | β | 단일 코어 시간(참고) | 목적 |
| --- | --- | --- | --- | --- |
| A | facet_1band_N656, facet_2band_N656 | 300 | 각 약 50–60분 | Casimir 성분 분리(가장 중요) |
| A | facet_1band_N164, facet_2band_N164 | 600 | 각 약 12분 | β 수렴 확인 |
| D | ribbon_L27_k9_N783, ribbon_L27_k1_N783 | 300 | 각 약 23분 | 꺾임 ε_t, J₁ |
| B | z5crystal_L6_N540, dice_L6_N432 | 300 | 각 약 15–20분 | 결정 ε의 크기 수렴 |

코어가 4개 이상이면 `-j 4`로 1–1.5시간 안에 끝난다. 시간이 남으면 `python run_queue.py -j <코어 수> --only A --extra-seeds 2`로 A의 통계를 늘린다.

## 2. 분석

```
python analyze.py > results.md
```

## 3. 보고

`results.md` 전체와 다음 판정을 보고한다.

- **A.** N = 164의 β = 300과 β = 600 결과가 오차 안에서 같은지(β 수렴). N = 164와 656에서 Casimir를 분리한 ε₅의 QMC/LSWT 비율. 현재 N = 164 비율 0.73 ± 0.07이 Casimir 차이 때문인지가 핵심 질문이다.
- **B.** L = 4와 L = 6의 마름모당 ε가 같은지. 현재 비율: z5 결정 1.29, 다이스 1.25.
- **D.** ε_t와 J₁의 QMC/LSWT 비율.
- 재빈닝 오차(`bins_E`)가 빈 크기에 따라 커지지 않는지. 커지면 sweep을 늘린다.

## 파일

- `sse/sse_det.py`: 결정론적 루프 SSE. 루프 길이 상한이 없고, sweep 비용은 연산자 수에 비례한다. `--fixM`은 고정 S^z 섹터 표본추출 옵션이다.
- `sse/validate.py`, `sse/ed_sector.py`: ED 검증.
- `qmc_inputs/`: 19개 그래프(JSON과 결합 목록), README, manifest.
- `runs/`: 결과(그래프·β·시드별 JSON).
- `run_queue.py`, `analyze.py`.
