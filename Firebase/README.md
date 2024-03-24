https://www.youtube.com/watch?v=1QGWCzYNulQ&list=PL9mWH7l2hU3NN1W5pOyKTI9XV7Tes9RkI&index=2

1.Flutter 개발환경 마련(SDK, 안드로이드스튜디오 등등)

2.가상기기 설정

3.dependency, 플러그인 
command : flutter pub get 

3.Firebase 시작
프로젝트 생성 > Authentication+Cloud Firestore+Storage (각각 세부설정 필요 이메일인증/국가seoul/테스트모드 등등)

4.Firbase 서비스 rule 설정
firestore & storage 두개 설정 필요

command : allow read write ~~ 2024.02.21

위의 부분을

command : allow read, write:

command :   if request.auth != null;

로 바꿔야함

5. Flutter프로젝트에 Firebase 연동하기

! Node 설치 -안정적 버전

! Git 설치

**!! 플러터 아이콘 선택 후 순서 따르기**중요

5-2.안드로이드 스튜디오 돌아와서

! >> npm~~ 명령어 터미널에 실행


! >> firebase login(로그인)


! FlutterFire Cli 관련 명령어 2개 터미널에 실행
android기기에서만 실행


! >> firebase init
이후 엔터
