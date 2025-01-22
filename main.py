import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import os
import json
from streamlit_gsheets import GSheetsConnection
import math
import pandas as pd
import firebase_admin
from firebase_admin import firestore, credentials, db


@st.cache_data(ttl=60)  # 60초마다 캐시 갱신
def get_data():
    # Create a connection object.
    conn = st.connection("gsheets", type=GSheetsConnection)
    
    # 매번 최신 데이터를 읽어오도록 설정
    df = conn.read(clear_cache=True)  # 캐시 비우고 새로 읽어오기

    sub_TF = 0
    sub_sub_TF = 0
    
    # 데이터 가공
    tabs_data = {}
    sub_sub = {}
    
    for row in df.itertuples():
        tab_name = row.탭
        
        if tab_name not in tabs_data:
            tabs_data[tab_name] = {"passage": row.지문, "questions": [], "correct_answers": []}
    
        # 선지 리스트로 변환
        choices = [getattr(row, f'선지{i}') for i in range(1, 6)]

        # 하위 문제 (지문 평가, 문제 평가) 받아오기
        if sub_TF <= 1 and row.지문평가1:
            if sub_TF == 0:
                sub_sub["sub_questions_passage"] = [row.지문평가1, row.지문평가2, row.지문평가3, row.지문평가4]
                sub_sub["sub_questions_problems"] = [row.문제평가1, row.문제평가2, row.문제평가3, row.문제평가4]
            sub_TF += 1

        # 하위 문제 부연설명 받아오기
        if sub_sub_TF == 0 and sub_TF == 2 and row.지문평가1:
            sub_sub["sub_sub_questions_passage"] = [row.지문평가1, row.지문평가2, row.지문평가3, row.지문평가4]
            sub_sub["sub_sub_questions_problems"] = [row.문제평가1, row.문제평가2, row.문제평가3, row.문제평가4]
            sub_sub_TF += 1
        
    
        # 질문 추가
        tabs_data[tab_name]["questions"].append({
            "question": row.질문,
            "choices": choices,
            "sub_questions_passage": sub_sub["sub_questions_passage"],  # 하위 질문 추가 - 지문
            "sub_questions_problems": sub_sub["sub_questions_problems"]  # 하위 질문 추가 - 문제
        })

    
        # 정답 추가 (정답이 문자열일 경우 정수로 변환)
        # tabs_data[tab_name]["correct_answers"].append(int(row.정답))

        # 정답이 NaN이 아니면 정수로 변환
        correct_answer = row.정답
        if not math.isnan(correct_answer):  # NaN이 아닌지 확인
            tabs_data[tab_name]["correct_answers"].append(int(correct_answer))
    
    return tabs_data, sub_sub


def initialize_firestore():
    """Firestore 초기화 함수."""
    if not firebase_admin._apps:  # 이미 초기화된 경우 방지
        cred = credentials.Certificate({
            "type": st.secrets["firebase"]["type"],
            "project_id": st.secrets["firebase"]["project_id"],
            "private_key_id": st.secrets["firebase"]["private_key_id"],
            "private_key": st.secrets["firebase"]["private_key"].replace("\\n", "\n"),
            "client_email": st.secrets["firebase"]["client_email"],
            "client_id": st.secrets["firebase"]["client_id"],
            "auth_uri": st.secrets["firebase"]["auth_uri"],
            "token_uri": st.secrets["firebase"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"]
        })
        firebase_admin.initialize_app(cred)

    # Firestore 초기화
    return firestore.client()



def save_data_to_firestore():
    """Firestore에 데이터를 저장하는 함수."""
    db = initialize_firestore()

    # 전화번호 뒷자리, 나이 가져오기
    phone_number = st.session_state.get("phone_number")
    age = st.session_state.get("age")

    # 현재 탭 정보 가져오기
    tab_idx = st.session_state.current_tab
    tab_key = f"answers_tab{tab_idx}"
    passage_key = f"subquestions_passage_tab{tab_idx}"
    problems_key = f"subquestions_problems_tab{tab_idx}"

    # 세션 상태에서 사용자 데이터 가져오기
    user_answers = st.session_state.get(tab_key, [])
    passage_eval = st.session_state.get(passage_key, {})
    problems_eval = st.session_state.get(problems_key, {})

    # 피드백 내용 가져오기
    feedback_key = f"feedback_tab{tab_idx}"
    feedback = st.session_state.get(feedback_key, "")

    # 저장할 데이터 생성
    data_to_save = {
        "전화번호 뒷자리": phone_number,
        "나이": age,
        "탭": tab_idx,
        "사용자 답안": user_answers,
        "지문 평가": {f"passage_q{i+1}": passage_eval.get(f"passage_q{i+1}", "") for i in range(len(user_answers))},
        "문제 평가": {f"problems_q{i+1}": problems_eval.get(f"problems_q{i+1}", "") for i in range(len(user_answers))},
        "피드백": feedback,  # 피드백 추가
        "제출 시간": firestore.SERVER_TIMESTAMP,  # Firestore 서버 시간
    }

    # 전화번호를 기준으로 해당 탭의 콜렉션에 저장
    collection_name = f"tab_{tab_idx}"  # 각 탭마다 별도의 콜렉션
    doc_id = f"user_{phone_number}_{age}"
    doc_ref = db.collection(collection_name).document(doc_id)  # phone_number를 기준으로 문서 생성
    doc_ref.set(data_to_save)  # Firestore의 set() 메서드로 데이터 저장

    # 기존에 데이터가 있으면 배열에 누적 (배열 방식)
    # doc_ref.set(data_to_save, merge=True)  # `merge=True`로 덮어쓰지 않고 기존 문서에 추가

    # st.success(f"Tab {tab_idx} 데이터가 Firestore의 {collection_name} 콜렉션에 저장되었습니다!")
    st.success(f"입력해주신 답변이 성공적으로 제출되었습니다. 감사합니다!")




##########################################################
##########################################################


# 첫 번째 페이지: 전화번호 뒷자리 4자리를 입력 받는 페이지
def first_page():
    st.markdown("# [ 전교 일등의 비밀 ]")
    st.markdown("#### LLM-as-a-Judge를 이용한 수능 문제 출제 및 평가")
    st.title("  ")
    
    st.markdown("##### 전화번호 뒷자리 입력")
    phone_number = st.text_input("전화번호 뒷자리 4자리를 입력해주세요:", max_chars=4)

    st.subheader("  ")

    st.markdown("##### 나이")
    age = st.selectbox("만 나이를 선택해주세요:", range(15, 71), index=None, placeholder="만 나이를 선택해 주세요...")
    
    st.subheader("  ")

    if st.button("다음"):
        if phone_number and phone_number.isdigit() and len(str(phone_number)) == 4 and isinstance(age, int):
            st.session_state.phone_number = phone_number
            st.session_state.age = age
            st.session_state.page = "second"  # 두 번째 페이지로 이동
            st.rerun()  # 페이지 강제 새로고침
        else:
            st.error("전화번호 뒷자리 4자리를 정확하게 입력하세요.")


def reset_radio_values_for_all_tabs():
    # 모든 탭의 세션 상태 초기화
    for tab_key in tabs_data.keys():
        # 지문 관련 문제 초기화
        passage_key = f"subquestions_passage_tab{tab_key}"
        if passage_key in st.session_state:
            for problems_q_key in st.session_state[passage_key]:
                st.session_state[passage_key][problems_q_key] = [None] * len(st.session_state[passage_key][problems_q_key])

        # 문제 관련 문제 초기화
        problems_key = f"subquestions_problems_tab{tab_key}"
        if problems_key in st.session_state:
            for problems_q_key in st.session_state[problems_key]:
                st.session_state[problems_key][problems_q_key] = [None] * len(st.session_state[problems_key][problems_q_key])


# 두 번째 페이지: 문제 풀이 페이지
def second_page():
    st.set_page_config(layout="wide")
    st.markdown(
        """
        <style>
        .horizontal-scroll {
            display: flex;
            overflow-x: auto;
            white-space: nowrap;
        }
        .horizontal-scroll > div {
            flex: 0 0 auto;
            margin-right: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    tabs_data, sub_sub = get_data()
    
    # 탭 세션 관리
    tabs = list(tabs_data.keys())

    if "current_tab" not in st.session_state:
        st.session_state.current_tab = tabs[0]

    # if "completed_tabs" not in st.session_state:
        # st.session_state.completed_tabs = set()

    # 사용자 답안을 저장할 리스트 (Session State 사용) - 탭 개수만큼 답안 세션 관리
    # tab_count = 5  # 예시로 7개의 탭이 있다고 가정
    # for tab_idx in range(1, tab_count + 1):

    tabs = tabs[:5]
    
    for tab_idx in tabs:
        tab_key = f"answers_tab{tab_idx}"
        if tab_key not in st.session_state:
            # 각 탭에 대해 문제의 개수에 맞는 사용자 답안을 저장할 리스트 초기화
            # st.session_state[tab_key] = [None] * len(questions[tab_idx - 1])  # questions[tab_idx - 1]는 탭에 맞는 문제 리스트
            st.session_state[tab_key] = [None] * len(tabs_data[tab_idx]["questions"])  # 탭에 맞는 문제 리스트

        submitted_key = f"submitted_tab{tab_idx}"
        if submitted_key not in st.session_state:
            # 각 탭에 대해 제출 여부 상태 초기화
            st.session_state[submitted_key] = False

        submitted_key_2 = f"submitted_tab{tab_idx}_2"
        if submitted_key_2 not in st.session_state:
            # 각 탭에 대해 제출 여부 상태 초기화
            st.session_state[submitted_key_2] = 0

        # 완료된 탭을 저장할 리스트 초기화
        if "completed_tabs" not in st.session_state:
            st.session_state["completed_tabs"] = []

        correct_status_key = f"correct_status_tab{tab_idx}"
        if correct_status_key not in st.session_state:
            st.session_state[correct_status_key] = [None] * len(tabs_data[tab_idx]["questions"])  # 각 문제별 정답 여부 초기화

        # 하위 문제 답안 세션 초기화
        passage_key = f"subquestions_passage_tab{tab_idx}"
        problems_key = f"subquestions_problems_tab{tab_idx}"
        
        # 지문 평가 세션 초기화
        if passage_key not in st.session_state:
            st.session_state[passage_key] = {
                f"passage_q{q_idx+1}": [None] * len(question["sub_questions_passage"])
                for q_idx, question in enumerate(tabs_data[tab_idx]["questions"])
                if "sub_questions_passage" in question  # 지문 평가 질문이 있는 경우만 처리
            }
        
        # 문제 평가 세션 초기화
        if problems_key not in st.session_state:
            st.session_state[problems_key] = {
                f"problems_q{q_idx+1}": [None] * len(question["sub_questions_problems"])
                for q_idx, question in enumerate(tabs_data[tab_idx]["questions"])
                if "sub_questions_problems" in question  # 문제 평가 질문이 있는 경우만 처리
            }

    
    # 평가 기준 설명
    st.markdown("#### [ 전교 일등의 비밀 ]")
    st.markdown("##### LLM-as-a-Judge를 이용한 수능 문제 출제 및 평가")


    # st.markdown("##### 다음 평가 기준에 따라 4개의 수능 국어 지문/문제를 평가해주세요")
    # st.subheader("아래 4개의 수능 국어 문제를 풀고, 다음 평가 기준에 따라 평가해주세요")

    st.write("안녕하세요. 이 사이트는 **LLM이 만든 수능 지문 및 문제의 품질 평가**를 위해 만들어졌습니다. 프로젝트를 개선해나가기 위해 여러분의 피드백을 받고자 합니다. 지문과 문제의 유창성, 난이도, 객관성 등 다양한 요소에 대한 평가를 부탁드립니다.")
    st.caption("")
    
    st.markdown("###### 이용 방법")
    st.write("5개 분야의 지문과 문제가 있습니다. 문제를 풀어본 후, 각각의 지문과 문제에 대해 평가해주세요. **4개의 [지문] 평가**, **4개의 [문제] 평가**에 대해 점수를 매긴 후 '평가 제출하기' 버튼을 눌러주시면 됩니다! 줄글 피드백에는 하고 싶으신 말씀을 적어주시면 되지만, 이는 필수가 아니므로 부담 없이 넘어가주셔도 됩니다. 또한, 문제를 풀고 답안을 제출하면 정답/오답 여부를 확인하실 수 있습니다.")
    st.caption("")
    
    # # 평가 기준 설명
    # left_col_eval, right_col_eval = st.columns([3, 4])

    # # 왼쪽 열: 지문 평가 기준
    # with left_col_eval:
    #     st.write("[지문 평가 기준] ")
    #     st.write("・  ")
    #     st.write("・  ")

    # # 오른쪽 열: 문제 평가 기준
    # with right_col_eval:
    #     st.write("[문제 평가 기준] ")
    #     st.write("・  ")
    #     st.write("・  ")

    st.markdown("  ")

    
    # 버튼 표시
    cols = st.columns(len(tabs))
    for i, tab in enumerate(tabs):

        submitted_key = f"submitted_tab{tab}"
        
        # 선택된 탭과 푼 탭을 구분하기 위한 스타일 설정
        if tab == st.session_state.current_tab:
            style = "border: 2px solid #f04848; background-color: lightgreen; font-weight: bold;"  # 선택된 탭
            # style = "border: 2px solid #1E90FF; background-color: lightgreen; font-weight: bold;"  # 선택된 탭
        elif tab in st.session_state["completed_tabs"]:  # 푼 탭을 체크
            style = "border: 2px solid #f7abab; background-color: lightblue;"  # 푼 탭
        else:
            style = ""  # 기본 스타일

        if isinstance(tab, str):
            button_clicked = cols[i].button(tab, key=f"tab_button_{i}", use_container_width=True)
            if button_clicked:
                st.session_state.current_tab = tab  # 선택된 탭을 저장
                st.rerun()  # 페이지 강제 새로고침
            
            # if cols[i].button(tab, key=f"tab_button_{i}"):
            #     st.session_state.current_tab = tab

            # 선택된 탭에 스타일을 적용
            cols[i].markdown(f'<div style="{style}"></div>', unsafe_allow_html=True)


        
    current_data = tabs_data[st.session_state.current_tab]
    passage = current_data["passage"]
    questions = current_data["questions"]

    # 가로 선 추가
    st.divider()

    # 타이틀
    st.markdown("  ")
    st.title("국어 영역")
    st.title("  ")



    
    with st.container():
        # 컬럼 비율: 지문(3) + 문제(2) + 하위 문제들(1씩)
        cols = st.columns([3, 2, 1, 1, 1, 1])
    
        # 1️⃣ 지문 (첫 번째 컬럼)
        with cols[0]:
            st.header("지문")
            st.write(passage)
    
        # 2️⃣ 문제 (두 번째 컬럼)
        questions = tabs_data[st.session_state.current_tab]["questions"]
    
        # 답안 세션 키 생성
        tab_key = f"answers_tab{st.session_state.current_tab}"
    
        # 딕셔너리로 초기화
        # if tab_key not in st.session_state:
        #     st.session_state[tab_key] = {}

        with cols[1]:
            st.markdown(f"<h4>문제</h4>", unsafe_allow_html=True)
            
        for idx, q in enumerate(questions):
            with cols[1]:
                # st.markdown(f"<h5>문제 {idx + 1}</h5>", unsafe_allow_html=True)
                st.write(q["question"])
    
                # 문제 답변 선택
                main_key = f"main_question_{st.session_state.current_tab}_{idx}"
                selected_main = st.radio(
                    label=" ",
                    options = q["choices"],
                    index=None,
                    key=main_key
                )
    
                # 답안 저장
                st.session_state[tab_key][idx] = (
                    q["choices"].index(selected_main) + 1 if selected_main else None
                )


            # 3️⃣ 하위 문제 - 지문 관련
            if idx == 0:  # 첫 번째 문제에서만 지문 관련 질문을 출력
                sub_questions_passage = q.get("sub_questions_passage", [])
                sub_col_start = 2  # 하위 질문이 시작하는 컬럼 인덱스
                sub_col_end = len(cols)  # 컬럼의 끝 인덱스
                
                for sub_idx, sub_q in enumerate(sub_questions_passage):
                    col_position = sub_col_start + sub_idx  # 순서대로 컬럼 인덱스를 계산
                
                    if col_position < sub_col_end:  # 컬럼 범위를 초과하지 않도록 제한
                        with cols[col_position]:
                            
                            # 배경색을 추가하기 위한 스타일 적용
                            st.markdown("""
                                <style>
                                    .sub-question-container-1 {
                                        background-color: #d6d6d6;  /* 배경색 설정 */
                                        padding: 10px;              /* 내부 여백 */
                                        border-radius: 5px;         /* 둥근 모서리 */
                                        margin-bottom: 10px;        /* 하단 여백 */
                                    }
                                    .sub-question-container-1 h6 {
                                        margin-bottom: 10px;        /* 제목과 내용 간의 간격 */
                                    }
                                </style>
                            """, unsafe_allow_html=True)
                
                            # 하위 문제 영역을 div로 감싸서 배경색 적용
                            st.markdown('<div class="sub-question-container-1">', unsafe_allow_html=True)
                            
                            # 문제 제목
                            st.markdown(f"<h6>[지문] 평가 {sub_idx + 1}.</h6>", unsafe_allow_html=True)
                            st.markdown(f"<h6>{sub_q}</h6>", unsafe_allow_html=True)

                            passage_key = f"subquestions_passage_tab{st.session_state.current_tab}"
                
                            # 고유한 key 생성 및 라디오 버튼으로 점수 선택
                            for problems_q_key, sub_answers in st.session_state[passage_key].items():
                                sub_key = f"sub_question_{st.session_state.current_tab}_{idx + 1}_{problems_q_key}_sub{sub_idx+1}"
                
                                # NaN 또는 None 확인 및 처리
                                sub_passage_value = sub_sub["sub_sub_questions_passage"][sub_idx]
                                if sub_passage_value is not None and not (isinstance(sub_passage_value, float) and math.isnan(sub_passage_value)):
                                    # sub_question = f"({sub_passage_value})"
                                    sub_question = sub_passage_value.split("\n")  # 줄 바꿈 기준으로 나누기
                                else:
                                    sub_question = " "  # 빈 문자열로 대체

                                for line in sub_question:
                                    st.caption(line)
                
                                selected_passage = st.radio(
                                    # sub_question,
                                    "",
                                    options=[5, 4, 3, 2, 1],  # 1부터 5까지 선택 가능
                                    index=None,  # 기본값 없음
                                    key=sub_key
                                )

                                st.session_state[passage_key][problems_q_key][sub_idx] = selected_passage if selected_passage else None
                                
                                break



            # 3️⃣ 하위 문제 - 문제 관련 (세 번째 컬럼부터 순서대로 배치)
            sub_questions_problems = q.get("sub_questions_problems", [])
            sub_col_start = 2  # 하위 질문이 시작하는 컬럼 인덱스
            sub_col_end = len(cols)  # 컬럼의 끝 인덱스
            
            for sub_idx, sub_q in enumerate(sub_questions_problems):
                col_position = sub_col_start + sub_idx  # 순서대로 컬럼 인덱스를 계산
            
                if col_position < sub_col_end:  # 컬럼 범위를 초과하지 않도록 제한
                    with cols[col_position]:
                        
                        # 배경색을 추가하기 위한 스타일 적용
                        st.markdown("""
                            <style>
                                .sub-question-container-2 {
                                    background-color: #e7e7e7;  /* 배경색 설정 */
                                    padding: 10px;              /* 내부 여백 */
                                    border-radius: 5px;         /* 둥근 모서리 */
                                    margin-bottom: 10px;        /* 하단 여백 */
                                }
                                .sub-question-container-2 h6 {
                                    margin-bottom: 10px;        /* 제목과 내용 간의 간격 */
                                }
                            </style>
                        """, unsafe_allow_html=True)
            
                        # 하위 문제 영역을 div로 감싸서 배경색 적용
                        st.markdown('<div class="sub-question-container-2">', unsafe_allow_html=True)
                        
                        # 문제 제목
                        st.markdown(f"<h6>[문제] 평가 {idx + 1}-{sub_idx + 1}.</h6>", unsafe_allow_html=True)
                        st.markdown(f"<h6>{sub_q}</h6>", unsafe_allow_html=True)

                        problems_key = f"subquestions_problems_tab{st.session_state.current_tab}"
            
                        # 고유한 key 생성 및 라디오 버튼으로 점수 선택
                        for problems_q_key, sub_answers in st.session_state[problems_key].items():
                            sub_key = f"sub_question_{st.session_state.current_tab}_{idx + 1}_{problems_q_key}_sub{sub_idx+1}"
            
                            # NaN 또는 None 확인 및 처리
                            sub_problem_value = sub_sub["sub_sub_questions_problems"][sub_idx]
                            if sub_problem_value is not None and not (isinstance(sub_problem_value, float) and math.isnan(sub_problem_value)):
                                sub_question = sub_problem_value.split("\n")  # 줄 바꿈 기준으로 나누기
                                # sub_question = f"({sub_problem_value})"
                            else:
                                sub_question = " "  # 빈 문자열로 대체

                            for line in sub_question:
                                st.caption(line)
            
                            selected_problems = st.radio(
                                # "    ".join(sub_question),
                                "",
                                options=[5, 4, 3, 2, 1],  # 1부터 5까지 선택 가능
                                index=None,  # 기본값 없음
                                key=sub_key
                            )

                            st.session_state[problems_key][problems_q_key][sub_idx] = selected_problems if selected_passage else None

                            break

                        # 배경색 div 종료
                        st.markdown('</div>', unsafe_allow_html=True)


    with st.container():
        cols = st.columns([5, 4])

        with cols[0]:
            # 제출 버튼
            if st.button("답안 제출하기"):
                submitted_key = f"submitted_tab{st.session_state.current_tab}"
                st.session_state[submitted_key] = True
                submitted_key_2 = f"submitted_tab{st.session_state.current_tab}_2"
                st.session_state[submitted_key_2] += 1
        
            if st.session_state.get(f"submitted_tab{st.session_state.current_tab}", False):
                st.success("답안이 제출되었습니다!")
        
                # 지문과 문제를 두 열로 나누기 (너비 비율 조정)
                left_col_2, right_col_2 = st.columns([3, 4])
        
                with left_col_2:
                    # 선택한 답안 표시
                    # st.subheader("선택한 답안:")
                    tab_key = f"answers_tab{st.session_state.current_tab}"
        
                    for idx, answer in enumerate(st.session_state[tab_key]):
                        selected_choice = answer if answer is not None else "선택 안 함"
                        # st.write(f"문제 {idx + 1}: {selected_choice}")
                    st.write("   ")
        
                with right_col_2:
                    # 답안 비교 및 정답 확인
                    # st.subheader("정답 확인:")
        
                    # 탭에 해당하는 correct_answers 가져오기
                    correct_answers = tabs_data[st.session_state.current_tab]["correct_answers"]
                    tab_key = f"answers_tab{st.session_state.current_tab}"
                    correct_status_key = f"correct_status_tab{st.session_state.current_tab}"
        
                    # 제출한 답안과 정답 비교
                    result = []
                    for idx, answer in enumerate(st.session_state[tab_key]):
                        # 정답 여부를 correct_status에 저장
                        is_correct = answer == correct_answers[idx]
                        st.session_state[correct_status_key][idx] = is_correct
        
                        correct = "맞았습니다" if is_correct else "틀렸습니다"
                        # st.write(f"문제 {idx + 1}: {correct}")
                        result.append(f"문제 {idx + 1}: {correct}")
        
                    st.session_state[submitted_key] = False
                    
        
            if st.session_state[submitted_key] == False and st.session_state[f"submitted_tab{st.session_state.current_tab}_2"] > 0:
                # st.info를 사용하여 결과를 표시
                # st.info("\n".join(result))  # 정답 여부를 한 번에 보여줌
                # st.info(st.session_state[f'correct_status_tab{st.session_state.current_tab}'])
                for idx, each_result in enumerate(st.session_state[f'correct_status_tab{st.session_state.current_tab}']):
                    if each_result == False:
                        st.info(f"문제 {idx + 1}: 틀렸습니다.")
                    else:
                        st.info(f"문제 {idx + 1}: 맞았습니다.")


        with cols[1]:                
            # '평가 제출하기' 버튼 추가
            if st.button("평가 제출하기"):
                all_selected = True  # 기본적으로 모두 선택된 것으로 가정

                # 현재 탭에 대한 지문 및 문제 평가 완료 여부 확인용
                tabs = list(tabs_data.keys())  # 탭 리스트 가져오기
            
                # 현재 탭에 대한 지문 관련 문제와 문제 관련 문제 체크
                tab_idx = st.session_state.current_tab  # 현재 탭 가져오기
                passage_key = f"subquestions_passage_tab{tab_idx}"
                problems_key = f"subquestions_problems_tab{tab_idx}"
            
                # 1️⃣ 지문 관련 문제 체크 (한 탭에 4개)
                # for problems_q_key, sub_answers in st.session_state.get(passage_key, {}).items():
                for problems_q_key, sub_answers in st.session_state[passage_key].items():
                    for sub_idx, value in enumerate(sub_answers):
                        # 4개 항목 확인: 값이 None 또는 1, 2, 3, 4, 5가 아닌 값이 들어가면 안됨
                        if value not in [1, 2, 3, 4, 5]:
                            all_selected = False
                            # st.error(f"선택되지 않은 항목: {problems_q_key} - 지문 관련 문제 {sub_idx + 1}")  # 선택되지 않은 항목
                            st.error(f"[지문 평가]에 선택되지 않은 항목이 있습니다.")  # 선택되지 않은 항목
                            break
            
                # 2️⃣ 문제 관련 문제 체크 (각 문제마다 4개)
                if all_selected:  # 지문 관련 문제가 다 선택되었다면, 문제 관련 문제 체크
                    for idx, q in enumerate(tabs_data[tab_idx]["questions"]):
                        problem_sub_answers = st.session_state.get(f"subquestions_problems_tab{tab_idx}", {}).get(f"problems_q{idx+1}", [])
                        for sub_idx, value in enumerate(problem_sub_answers):
                            # 각 문제마다 4개 항목 확인: 값이 None 또는 1, 2, 3, 4, 5가 아닌 값이 들어가면 안됨
                            if value not in [1, 2, 3, 4, 5]:
                                all_selected = False
                                # st.error(f"선택되지 않은 항목: 문제 {idx+1} - 문제 관련 문제 {sub_idx + 1}")  # 선택되지 않은 항목
                                st.error(f"[문제 평가]에 선택되지 않은 항목이 있습니다.")  # 선택되지 않은 항목
                                break


            
                # 평가 완료 여부에 대한 메시지 출력
                if all_selected:
                    
                    # 완료된 탭 리스트에 추가
                    if tab_idx not in st.session_state["completed_tabs"]:
                        st.session_state["completed_tabs"].append(tab_idx)
                        
                    st.success("지문 및 문제 평가를 완료하였습니다!")

                    # **구글 시트에 데이터 저장 호출**
                    # save_data_to_gsheet()

                    # firebase에 데이터 저장
                    # save_data_to_firebase()

                    save_data_to_firestore()  # Firestore에 데이터 저장
                    # st.success("평가 데이터가 성공적으로 제출되었습니다!")

                    # 모든 탭 상태 초기화
                    reset_radio_values_for_all_tabs()

                    st.rerun()

                    
                else:
                    st.error("모든 문제에 대해 평가를 선택해주세요.")

    

    st.write("   ")


    # 피드백 제출 처리 함수
    def submit_feedback():
        feedback_input = st.session_state[f"feedback_input_{st.session_state.current_tab}"].strip()
        if feedback_input:
            # 현재 탭에 맞는 피드백과 제출 상태를 저장
            feedback_key = f"feedback_tab{st.session_state.current_tab}"
            feedback_submitted_key = f"feedback_submitted_tab{st.session_state.current_tab}"

            st.session_state[feedback_key] = feedback_input  # 입력 내용 저장
            st.session_state[feedback_submitted_key] = True  # 제출 상태 업데이트
                
            # 피드백이 제출되었을 때 Firestore에 데이터 저장
            save_data_to_firestore()  # 평가 제출 시와 동일한 방식으로 데이터 저장
            
        else:
            st.warning("피드백을 입력해주세요.")

    with st.container():
        cols = st.columns([1])
    
        # 그냥 컬럼 하나 뿐
        with cols[0]:

            # 각 탭에 대한 피드백 세션 상태 초기화
            if f'feedback_submitted_tab{st.session_state.current_tab}' not in st.session_state:
                st.session_state[f'feedback_submitted_tab{st.session_state.current_tab}'] = False
        
            if f'feedback_tab{st.session_state.current_tab}' not in st.session_state:
                st.session_state[f'feedback_tab{st.session_state.current_tab}'] = ""
            
            # 가로 선 추가
            st.divider()
        
            # 피드백 입력 필드 (제출 후에도 유지)
            st.write("   ")
            st.write("   ")
            st.write("   ")
            st.subheader(f"[{st.session_state.current_tab}] 지문 및 문제에 대한 전반적인 피드백을 남겨주세요.")  # 현재 탭에 맞는 제목
        
            # 각 탭별로 고유한 key를 설정하여 피드백 입력 창 만들기
            feedback_input_key = f"feedback_input_{st.session_state.current_tab}"
            st.text_area("의견을 남겨주세요:", key=feedback_input_key)
        
            # 피드백 제출 버튼 (콜백 함수 연결)
            st.button("피드백 제출하기", on_click=submit_feedback)
        
            # 피드백 제출 완료 메시지 및 입력한 내용 표시
            if st.session_state[f'feedback_submitted_tab{st.session_state.current_tab}']:
                # st.success("피드백이 제출되었습니다!")
                st.write("남겨주신 피드백:")
                st.info(st.session_state[f'feedback_tab{st.session_state.current_tab}'])  # 저장된 피드백 출력





# 페이지 관리
if 'page' not in st.session_state:
    st.session_state.page = "first"  # 첫 번째 페이지로 시작

# 페이지 전환
if st.session_state.page == "first":
    first_page()
elif st.session_state.page == "second":
    second_page()
