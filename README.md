# hktoss_MidProject

## branch 사용법
* branch name : <이름>_<현재 진행중인 작업>
    * ex HongGilDong_preprocessing
* main 에서 작업하면 충돌나기 때문에, 반드시 branch 생성 또는 이동 후 작업
    * 현재 branch 확인 방법 : git branch
    * branch 이동 방법 : git checkout <branch_name>
    * branch 에서 작업 후, main branch 로 pull request 를 보냅니다
    * 해당 pull request 를 github 관리자가 검토 후, main branch 와 병합합니다.
* 자기 branch 에서 작업한 후 변경사항을 저장하는 방법
    * 자기 branch 에 우선 올리기
    => git add . > git commit -m "commit message" > git push
    * 자기 branch 에 저장 했으면, main 에 변경사항 올리기
    => vscode 또는 github 에서 pull request 보내기 (본인 branch 에서 main 으로!)