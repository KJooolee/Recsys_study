import sys
# main.py 파일에서 메인 파이프라인 함수를 가져옵니다.
from main import run_pipeline

def get_user_choice():
    """터미널에서 사용자의 입력을 받는 CLI 함수"""
    print("="*50)
    print(" 🚀 당근마켓 JD 타겟팅 추천시스템 파이프라인 🚀 ")
    print("="*50)
    
    # 1. 데이터셋 선택
    print("\n[1] 데이터셋을 선택하세요.")
    print("  1. Clothing")
    print("  2. Electronics")
    print("  3. Video Games")
    dataset_choice = input("입력 (1/2/3): ").strip()
    
    dataset_map = {'1': 'Clothing', '2': 'Electronics', '3': 'Video_Games'}
    # 잘못된 입력이 들어오면 기본값으로 Electronics를 선택하도록 안전장치 마련
    source = dataset_map.get(dataset_choice, 'Electronics') 
    
    # 2. 모델 선택
    print("\n[2] 학습할 모델을 선택하세요.")
    print("  1. BPR-MF (Baseline)")
    print("  2. LightGCN (Graph - 예정)")
    print("  3. SASRec (Sequence - 예정)")
    model_choice = input("입력 (1/2/3): ").strip()
    
    # 3. 실행 모드 선택
    print("\n[3] 실행 모드를 선택하세요.")
    print("  1. 새로 학습하고 평가하기 (Train & Eval)")
    print("  2. 저장된 체크포인트 불러와서 평가만 하기 (Test Only)")
    mode_choice = input("입력 (1/2): ").strip()
    
    return source, model_choice, mode_choice

if __name__ == "__main__":
    try:
        # 1. 터미널에서 사용자 입력 받기
        source, model_choice, mode_choice = get_user_choice()
        
        # 2. 메인 엔진(main.py)으로 변수를 전달하여 실제 학습/평가 파이프라인 가동
        run_pipeline(source, model_choice, mode_choice)
        
    except KeyboardInterrupt:
        # 사용자가 도중에 Ctrl+C를 눌러 끌 때 에러 메시지 없이 깔끔하게 종료되도록 처리
        print("\n\n프로그램을 강제 종료합니다. 수고하셨습니다!")
        sys.exit(0)