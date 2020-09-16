#나만의 스타크래프트 만들기 실습 코드

1. 2020-08-21
- 구현 내용
    - racepack으로 각 종족별 모듈 작성
    - Protoss 코드 작성
    - 파일런생성->게이트웨이생성->질럿생성->공격
    
- 미구현 내용
    - 프로브 추가 생성 -> 로직이 꼬임
    - idle worker 선택 후 미네랄 채취 -> 미네랄 위치를 하드코딩 해야 하는지?
    - 파일런 및 게이트웨이 추가 -> 위치 지정을 하드코딩 해야하는지

2. 2020-08-31
- 구현 내용
    - probe, zealot 생산
    - pylon, gateway 건설
        - 건설시 pylon/nexus위치 기준으로 범위 지정하여 생산
    - 건물 생산 점수 및 do nothing panalty 부여
        - panalty 부과시 학습이 더 빨리 진행되는 것이 체감됨

3. 2020-09-07
- 구현 내용
    - assimilator, cybernetics core, forge 건설 추가
    - stalker 생산 추가
    - sparce reward 변경
- 미구현 내용
    - worker들이 assimilator에 제대로 배속되지 않음
    - 건물 위치가 계속 찾을 수 없음으로 나오는 경우가 많음
    
4. 2020-09-15 Final
- 구현 내용
    - ProtossDQN.py 실행 필요
    - stargate, roboticsfacility, roboticsbay 건설 추가
    - mothershipcore, mothership, colossus, immortal, tempest, voidray 생산 추가
    - 앞마당 멀티 1개소 
- 결과
    - 별도의 트레이닝 과정이 없이도 easy모드는 쉽게 이김
    - 난이도를 높여 학습중
