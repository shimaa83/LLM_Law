import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# إعداد مفتاح Google
GOOGLE_API_KEY = "AIzaSyA5g3q_6QTHfAmjKffMnpwZNiIvIlOaJAE"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# مسار ملف القانون
LAW_FILE_PATH = "law4.txt"

# تكوين Streamlit
st.set_page_config(
    page_title="المستشار القانوني للأحوال الشخصية",
    page_icon="⚖️",
    layout="wide",
)

# تخصيص CSS للواجهة
st.markdown("""
<style>
    .main-header {
        color: #1a5276;
        font-family: 'Arial', 'Tajawal', sans-serif;
    }
    .legal-box {
        background-color: #f8f9fa;
        border-left: 5px solid #2c3e50;
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .stTextInput label, .stButton>button {
        font-family: 'Arial', 'Tajawal', sans-serif;
    }
    .stButton>button {
        background-color: #1a5276;
        color: white;
        font-weight: bold;
    }
    .disclaimer {
        font-size: 12px;
        color: #7f8c8d;
        font-style: italic;
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
    }
    .user-message {
        background-color: #e8f4f8;
        margin-left: 40px;
    }
    .assistant-message {
        background-color: #f0f7ee;
        margin-right: 40px;
    }
    .message-content {
        margin-left: 12px;
        margin-right: 12px;
        text-align: right;
        direction: rtl;
    }
    .avatar {
        min-width: 40px;
        font-size: 24px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ---------- قراءة محتوى الملف القانوني ---------- #
@st.cache_data
def load_law_file(file_path=LAW_FILE_PATH):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            return content
    except FileNotFoundError:
        st.error(f"⚠️ لم يتم العثور على الملف في: {file_path}")
        return None
    except Exception as e:
        st.error(f"⚠️ حدث خطأ أثناء قراءة الملف: {e}")
        return None

# ---------- إعداد نظام CAG باستخدام Gemini ---------- #
@st.cache_resource
def setup_cag_system(law_content):
    # تقسيم المستند إلى أجزاء
    docs = [Document(page_content=law_content)]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
        keep_separator=True
    )
    split_data = text_splitter.split_documents(docs)

    # استخدام نموذج التضمين (التصحيح هنا)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    # إنشاء قاعدة بيانات المتجهات
    vectorstore = Chroma.from_documents(
        documents=split_data, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    # إعداد الاسترجاع
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8}
    )
    
    # استخدام نموذج Gemini (التصحيح هنا)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=1,
        google_api_key=GOOGLE_API_KEY
    )

    # إنشاء قالب المحادثة
    system_prompt = """
    أنت مساعد قانوني متخصص في الإجابة على الأسئلة المتعلقة بقانون الأحوال الشخصية المصري بناءً على النص المقدم.
    استخدم الأجزاء التالية من السياق المسترجع للإجابة على السؤال بدقة.
    أجب باللغة العربية فقط.
    اجعل إجابتك فيها كل التفاصيل المتاحة فى النص المقدم.
    احرص على الاستشهاد بالمواد القانونية ذات الصلة إن وجدت.
    إذا كانت الإجابة غير موجودة في النص المقدم، فقل بوضوح "المعلومات المطلوبة غير متوفرة في النص المقدم".

    السياق:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # إنشاء سلاسل المعالجة
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# ---------- معالجة الأسئلة ---------- #
def process_question(law_content, question):
    try:
        if not law_content:
            return "⚠️ لا يمكن معالجة السؤال لأن محتوى القانون غير متوفر"
            
        cag_system = setup_cag_system(law_content)
        response = cag_system.invoke({"input": question})
        return response['answer']
        
    except Exception as e:
        error_msg = f"⚠️ حدث خطأ أثناء معالجة السؤال: {str(e)}"
        st.error(error_msg)
        return error_msg

# ---------- الواجهة الرئيسية ---------- #
def main():
    # عنوان التطبيق
    st.markdown('<h1 class="main-header">⚖️ المستشار القانوني للأحوال الشخصية</h1>', unsafe_allow_html=True)
    
    # وصف التطبيق
    with st.markdown('<div class="legal-box">', unsafe_allow_html=True):
        st.markdown("""
        <p style="direction: rtl; text-align: right;">
        مرحباً بك في المستشار القانوني للأحوال الشخصية. يمكنك طرح أسئلتك المتعلقة بقانون الأحوال الشخصية المصري 
        وسأحاول الإجابة عليها استناداً إلى النصوص القانونية المتاحة.
        </p>
        """, unsafe_allow_html=True)
    
    # تحميل ملف القانون
    law_content = load_law_file()
    
    if law_content is None:
        st.warning("⚠️ لم يتم العثور على ملف القانون. يرجى التأكد من وجود الملف في المسار الصحيح.")
        return
    
    # إدخال السؤال
    question = st.text_input("❓ اكتب سؤالك حول قانون الأحوال الشخصية:", key="question_input")
    
    # زر البحث
    if st.button("🔍 ابحث عن الإجابة", type="primary"):
        if question:
            with st.spinner("⏳ جاري البحث في نصوص القانون..."):
                answer = process_question(law_content, question)
                
                # عرض السؤال والإجابة
                st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="avatar">👤</div>
                    <div class="message-content">{question}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <div class="avatar">⚖️</div>
                    <div class="message-content">{answer}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ يرجى إدخال سؤال أولاً")

if __name__ == "__main__":
    main()
