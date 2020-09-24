# "스타크래프트2로 배우는 강화학습 - 2020 SEASON 1"  Final Project Tournament 결과

1. 참가자 리스트(총 8명)
- 09360 조용준 테란 : 고급 유닛에 대한 실험과 난이도에 따른 실험 결과까지 잘 수행함.
- 05026 박상원 테란 : 기본적으로 제공되는 테란 베이스라인 코드에 자신만의 State 와 Action을 정의하고 공격에 필요한 supply depot, barrack을 최대한 많이 만들어서 Marine을 빠르게 생산할 수 있도록 수정하고 한번의 action으로 최대한 많은 marine들이 공격할 수 있도록 하고, 멀티지역을 고려하여 공격 범위를 확장함. 추가적으로 NN의 Depth를 조정하여 실험함.
- 09287 서대웅 프로토스 :  베이스 확장, 앞마당 멀티 구현 및 프로토스의 AirCraft 등 고급유닛 사용 등 많은 실험을 수행.
- 10336 김명환 테란 : 베이스 확장, 테란의 AirCraft 등 고급유닛 사용 등 다양한 실험을 수행.
- 10071 오동훈 테란 : NN 구조, 하이퍼파라미터 수정, Screen/Minimap Image 입력사용 CNN 활용 등을 수행함.
- 10395 이현호 프로토스 : 베이스라인 코드에 없는 Protoss 베이스라인 코드를 새로 구현함. 'Zealot' 유닛을 생성하고 공격하는 코드를 코드를 이미 구현되어 있는 DQN 알고리즘을 적용함.
- 10073 오필훈 프로토스 : 베이스라인 코드에 없는 Protoss 베이스라인 코드를 새로 구현함. 'Zealot' 유닛을 생성하고 공격하는 코드를 코드를 이미 구현되어 있는 VanillaDQN 알고리즘을 적용하여 학습하도록 수정. 난이도 에 따른 시험을 수행하여 학습함.
- 10274 최지은 테란 : Vanilla DQN외에도 추가적으로 Dualing DQN을 구현하시고 테란 종족의 여러 공격 유닛, 건물 유닛과 관련된 다양한 Action을 실험함.
- ~~10472 오수은 저그 : 유일하게 저그종족을 구현하였고 전통적 강화학습 Q-Learning 알고리즘을 구현함.~~ (오수은 님도 수고 많이 하셨는데...아래와 같은 문제로 Final Project 공유회 때 토너먼트에서 대전이 불가했습니다.)


##### Feature Action 과 Raw Action 동시사용할 경우 문제 : 스타크래프트2 Action은 크게 Feature Action 과 Raw Action을 사용하는데…오수은 님은 Feature Action 을 사용하셨고, 다른 참가자 분들은 모두 STEP4에서 설명드린 Raw Action을 사용했습니다. 각가의 에이전트를 돌릴때는 문제가 없지만, 대전시에는 Feature Action 과 Raw Action을 모두사용할 때 “available_actions” 라는 Observation을 사용할 수 없습니다. “available_actions”를 사용할 경우 대전시 오류가 나고 우회할 수 있는 방법을 찾아보고 있는데 PySC2 자체의 문제라서 쉽지 않을 것 같습니다.

2. Tournament 대진표 : https://challonge.com/xarsyno5

3. Tournament 대진 진행 프로젝트 Github : https://github.com/parksurk/skcc-drl-sc2-course-2020_1st

4. Tournament 결과
- ROUND 1 승자 : 박상원(테란), 서대웅(프로토스), 이현호(프로토스), 오필훈(프로토스)
- SEMIFINAL 승자 : 서대웅(프로토스), 오필훈(프로토스)
- FINAL 승자 : 오필훈(프로토스)

5. 주의사항
- 오필훈(프로토스)님 우승의 원인은 ‘프로토스 초반 러쉬 전략’ 을 방어할수 있는 상대가 없었던 걸로 판단됩니다.(물론 토너먼트이기 때믄에 모든 상대와 겨루지 않았기 때문에 대진표가 다를 경우 완전히 다른 결과가 나올 수 있습니다.)
- 상세한 내용은 첨부한 replays 폴더 화위에 리플래이 파일(*.SC2Replay)을 참고바랍니다.
- 리플레이 파일은 스타크래프트2가 로컬PC에 설치되어있다는 가정하에서 *.SC2Replay 파일을 더블클릭하시고 스타크래프트2가 실행되어 BattleNet 계정으로 로그인을 거쳐야만 보실 수 있습니다.(사내보안/유해사이트 차단 등으로 BattleNet 등의 게임사이트가 네트워크 차단되었는 경우 에러가 발생하니 유의바랍니다!!!)

## "스타크래프트2로 배우는 강화학습" Course 소개

- Course 수행 기간 : 2020.08.12 ~ 2020.09.24 (about 7 weeks)
- Course 수행 방법 :O2O Project-based Course
- Google Classroom : '현업에서 활용하는 나만의 StarCraft2 강화학습 Agent 만들기'
- Course Officer : SK 주식회사 C&C , Tech Training Group 이유종 선임
- Course Coach : SK 주식회사 C&C , Tech Training Group 박석 수석
- Course Schedule reference : https://www.notion.so/parksurk/2-3f69488a5320462392baf47107919872
- Course Main Github reference : https://github.com/parksurk/dmarl-sc2
