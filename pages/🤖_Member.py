import streamlit as st

st.set_page_config(
    page_title="Thành viên",
    page_icon="🤖",
)

st.sidebar.success("Select a demo above.")

# Chèn CSS vào ứng dụng Streamlit
st.markdown("""
    <style>
    img::selection {
    background: none;
    }

    *::selection {
    background: #001730;
    color: #fff;
    }
    .title {
        text-align: center;
    }
    .card {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        margin: 10px;     
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }
    .card__header{
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: #333;
        text-align: center;
    }
    .card__header img{
        width: 150px;
    }
    .card__info{
        width: auto;      
        color: #777;
        text-align: left;
    }
    .line{
        width: 80%;
        height:1px;
        background-color:#777;
    }
    </style>
    """, unsafe_allow_html=True)

# Thông tin cá nhân của học sinh 1
student1 = {
    'name': 'Nguyễn Trường An',
    'student_id': '21110117',
    'email': '21110117@student.hcmute.edu.vn',
    'phone': '0396779183',
}

# Thông tin cá nhân của học sinh 2
student2 = {
    'name': 'Mai Anh Khoa',
    'student_id': '21110836',
    'email': '21110836@student.hcmute.edu.vn',
    'phone': '0869628507',
}

def display_student_info(student):
    card_html = f"""
    <div class='card'>
        <div class='card__header'>
            <img src='https://lutiband.com/wp-content/uploads/2020/08/Who_Icon.png' alt='{student['name']}'>
            <h2>{student['name']}</h2>
        </div>
        <div class='line'>
        </div>
        <div class='card__info'>
            <p><strong>Mã số sinh viên:</strong> {student['student_id']}</p>
            <p><strong>Email:</strong> {student['email']}</p>
            <p><strong>Số điện thoại:</strong> {student['phone']}</p>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

st.markdown("<h1 class='title'>🎉 Thành Viên Nhóm 🎉</h1>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader('🤖Thành viên 1:')
    display_student_info(student1)

with col2:
    st.subheader('🤖Thành viên 2:')
    display_student_info(student2)

st.markdown("<br><br>", unsafe_allow_html=True)
