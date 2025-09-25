import streamlit as st
import joblib
import easyocr
import pandas as pd
import requests

# --- Load model ---
model = joblib.load("diabetes_health_indicators_classifier_v1.joblib")

# --- Language dictionary for multilingual support ---
LANGUAGES = {
    "English": {
        "title": "🩺 Diabetes Risk Prediction",
        "description": "📝 Fill the form to predict diabetes risk based on patient health indicators.",
        "submit": "🔍 Predict",
        "HighBP": "💓 High Blood Pressure",
        "HighChol": "🩸 High Cholesterol",
        "CholCheck": "✅ Cholesterol Check (last 5 years)",
        "BMI": "⚖️ Body Mass Index (BMI)",
        "Smoker": "🚬 Smoker (≥5 packs)",
        "Stroke": "🧠 History of Stroke",
        "HeartDiseaseorAttack": "❤️ Coronary Heart Disease / Heart Attack",
        "PhysActivity": "🏃 Physical Activity (past 30 days)",
        "Fruits": "🍎 Consumes Fruits Daily",
        "Veggies": "🥦 Consumes Vegetables Daily",
        "HvyAlcoholConsump": "🍺 Heavy Alcohol Consumption",
        "AnyHealthcare": "🏥 Healthcare Coverage",
        "NoDocbcCost": "💰 Difficulty Visiting Doctor (Cost)",
        "GenHlth": "💚 General Health (1=Excellent,5=Poor)",
        "MentHlth": "🧠 Days Mental Health Not Good (30 days)",
        "PhysHlth": "💪 Days Physical Health Not Good (30 days)",
        "DiffWalk": "🚶 Difficulty Walking",
        "Sex": "👫 Sex (0=Female,1=Male)",
        "Age": "🎂 Age Category (1=18-24 ... 13=80+)",
        "Education": "🎓 Education Level (1=No school ... 6=College 4+ yrs)",
        "Income": "💵 Income Level (1=<$10k ... 8=$75k+)",
        "high_risk": "⚠️ High Risk of Diabetes",
        "low_risk": "✅ Low Risk of Diabetes",
        "risk_low": "✅ Low",
        "risk_medium": "⚠️ Medium",
        "risk_high": "❌ High",
        "risk_text": "Your Diabetes Risk",
        "prob_text": "Probability of Diabetes",
        "nutrition_title": "🥗 AI Personal Nutrition",
        "nutrition_description": "Enter ingredients and get a healthy low-sugar recipe for diabetics.",
        "nutrition_input": "🍴 Ingredients (comma separated)",
        "nutrition_submit": "📨 Generate Recipe",
        "recipe": "### 🥗 Recipe\n",
        "system_prompt": "You are a helpful AI nutritionist. Provide recipes that are low in sugar and suitable for diabetics. Use clear instructions.",
        "chatbot_title": "💬 Health Chatbot",
        "chatbot_description": "You can chat or vent about your health concerns here.",
        "chatbot_input": "✏️ Type your message here...",
        "chatbot_submit": "📨 Send",
        "chatbot_warning": "Please enter a message.",
        "chatbot_system_prompt": "You are a helpful and empathetic health assistant. Respond politely and provide guidance on general health and wellness.",
        "ocr_title": "📸 OCR Food Composition",
        "ocr_description": "Upload a photo of the product composition. The OCR text will be summarized by AI for easy understanding (multilingual).",
        "ocr_upload": "📷 Upload Product Image",
        "ocr_button": "🔍 Extract & Summarize",
        "ocr_result": "### 📄 OCR Extracted Text",
        "ocr_summary": "### 🤖 AI Summary",
        "ocr_system_prompt": "You are an AI nutritionist, summarize the product composition and state if it is safe for people with diabetes.",
        "ocr_spinner_extract": "🔎 Extracting text...",
        "ocr_spinner_summarize": "🤖 Summarizing with AI...",
        "ocr_no_text": "No text detected in the image."
    },
    "Bahasa": {
        "title": "🩺 Prediksi Risiko Diabetes",
        "description": "📝 Isi formulir berikut untuk memprediksi risiko diabetes berdasarkan indikator kesehatan pasien.",
        "submit": "🔍 Prediksi",
        "HighBP": "💓 Tekanan Darah Tinggi",
        "HighChol": "🩸 Kolesterol Tinggi",
        "CholCheck": "✅ Pemeriksaan Kolesterol (5 tahun terakhir)",
        "BMI": "⚖️ Indeks Massa Tubuh (BMI)",
        "Smoker": "🚬 Perokok (≥5 bungkus)",
        "Stroke": "🧠 Riwayat Stroke",
        "HeartDiseaseorAttack": "❤️ Penyakit Jantung Koroner / Serangan Jantung",
        "PhysActivity": "🏃 Aktivitas Fisik (30 hari terakhir)",
        "Fruits": "🍎 Mengonsumsi Buah Setiap Hari",
        "Veggies": "🥦 Mengonsumsi Sayur Setiap Hari",
        "HvyAlcoholConsump": "🍺 Konsumsi Alkohol Berat",
        "AnyHealthcare": "🏥 Memiliki Asuransi Kesehatan",
        "NoDocbcCost": "💰 Kesulitan ke Dokter karena Biaya",
        "GenHlth": "💚 Kesehatan Umum (1=Sangat Baik,5=Buruk)",
        "MentHlth": "🧠 Hari Kesehatan Mental Buruk (30 hari terakhir)",
        "PhysHlth": "💪 Hari Kesehatan Fisik Buruk (30 hari terakhir)",
        "DiffWalk": "🚶 Kesulitan Berjalan",
        "Sex": "👫 Jenis Kelamin (0=Perempuan,1=Laki-laki)",
        "Age": "🎂 Kategori Usia (1=18-24 ... 13=80+)",
        "Education": "🎓 Tingkat Pendidikan (1=Tidak sekolah ... 6=College 4+ tahun)",
        "Income": "💵 Tingkat Pendapatan (1=<$10k ... 8=$75k+)",
        "high_risk": "⚠️ Risiko Diabetes Tinggi",
        "low_risk": "✅ Risiko Diabetes Rendah",
        "risk_low": "✅ Rendah",
        "risk_medium": "⚠️ Sedang",
        "risk_high": "❌ Tinggi",
        "risk_text": "Risiko Diabetes Anda",
        "prob_text": "Probabilitas Diabetes",
        "nutrition_title": "🥗 AI Nutrisi Personal",
        "nutrition_description": "Masukkan bahan makanan untuk mendapatkan resep sehat rendah gula untuk penderita diabetes.",
        "nutrition_input": "🍴 Bahan makanan (pisahkan dengan koma)",
        "nutrition_submit": "📨 Buat Resep",
        "recipe": "### 🥗 Resep\n",
        "system_prompt": "Anda adalah AI ahli nutrisi. Berikan resep rendah gula dan sesuai untuk penderita diabetes. Gunakan instruksi yang jelas.",
        "chatbot_title": "💬 Chatbot Kesehatan",
        "chatbot_description": "Silakan curhat atau tanyakan masalah kesehatan Anda.",
        "chatbot_input": "✏️ Ketik pesan Anda di sini...",
        "chatbot_submit": "📨 Kirim",
        "chatbot_warning": "Silakan masukkan pesan.",
        "chatbot_system_prompt": "Anda adalah asisten kesehatan yang ramah dan empatik. Jawablah dengan sopan dan berikan panduan terkait kesehatan secara umum.",
        "ocr_title": "📸 OCR Komposisi Makanan",
        "ocr_description": "Unggah foto komposisi produk. Teks hasil OCR akan dirangkum oleh AI agar mudah dipahami (multilingual).",
        "ocr_upload": "📷 Unggah Gambar Produk",
        "ocr_button": "🔍 Ekstrak & Rangkum",
        "ocr_result": "### 📄 Teks Hasil Ekstraksi OCR",
        "ocr_summary": "### 🤖 Rangkuman AI",
        "ocr_system_prompt": "Anda adalah AI nutrisionis, ringkas komposisi produk dan sebutkan apakah aman untuk penderita diabetes.",
        "ocr_spinner_extract": "🔎 Mengekstrak teks...",
        "ocr_spinner_summarize": "🤖 Merangkum dengan AI...",
        "ocr_no_text": "Tidak ada teks yang terdeteksi pada gambar."
    }
}

# --- Sidebar navigation ---
st.set_page_config(page_title="🩺 DiabetMate", layout="wide")
lang = st.sidebar.selectbox("🌐 Pilih Bahasa / Select Language", options=["Bahasa", "English"])
L = LANGUAGES[lang]

page = st.sidebar.selectbox("📌 Pilih Halaman / Select Page", [L["title"], L["nutrition_title"], L["chatbot_title"], L["ocr_title"]])

# --- DIABETES PREDICTION PAGE ---
if page == L["title"]:
    st.title(L["title"])
    st.write(L["description"])

    with st.form("diabetes_form"):
        col1, col2 = st.columns(2)

        with col1:
            HighBP = st.selectbox(L["HighBP"], [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            HighChol = st.selectbox(L["HighChol"], [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            CholCheck = st.selectbox(L["CholCheck"], [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            BMI = st.number_input(L["BMI"], min_value=10, max_value=60, value=25)
            Smoker = st.selectbox(L["Smoker"], [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            Stroke = st.selectbox(L["Stroke"], [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            HeartDiseaseorAttack = st.selectbox(L["HeartDiseaseorAttack"], [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            PhysActivity = st.selectbox(L["PhysActivity"], [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            Fruits = st.selectbox(L["Fruits"], [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            Veggies = st.selectbox(L["Veggies"], [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            HvyAlcoholConsump = st.selectbox(L["HvyAlcoholConsump"], [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        with col2:
            AnyHealthcare = st.selectbox(L["AnyHealthcare"], [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            NoDocbcCost = st.selectbox(L["NoDocbcCost"], [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            GenHlth = st.slider(L["GenHlth"], 1, 5, 3)
            MentHlth = st.slider(L["MentHlth"], 0, 30, 5)
            PhysHlth = st.slider(L["PhysHlth"], 0, 30, 5)
            DiffWalk = st.selectbox(L["DiffWalk"], [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            Sex = st.selectbox(L["Sex"], [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
            Age = st.slider(L["Age"], 1, 13, 5)
            Education = st.slider(L["Education"], 1, 6, 4)
            Income = st.slider(L["Income"], 1, 8, 4)

        submitted = st.form_submit_button(L["submit"])

    if submitted:
        input_df = pd.DataFrame([{
            "HighBP": HighBP,
            "HighChol": HighChol,
            "CholCheck": CholCheck,
            "BMI": BMI,
            "Smoker": Smoker,
            "Stroke": Stroke,
            "HeartDiseaseorAttack": HeartDiseaseorAttack,
            "PhysActivity": PhysActivity,
            "Fruits": Fruits,
            "Veggies": Veggies,
            "HvyAlcoholConsump": HvyAlcoholConsump,
            "AnyHealthcare": AnyHealthcare,
            "NoDocbcCost": NoDocbcCost,
            "GenHlth": GenHlth,
            "MentHlth": MentHlth,
            "PhysHlth": PhysHlth,
            "DiffWalk": DiffWalk,
            "Sex": Sex,
            "Age": Age,
            "Education": Education,
            "Income": Income
        }])

        prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

        st.subheader("📊 Hasil Prediksi / Prediction Result")

        if prob < 0.3:
            risk_label = L["risk_low"]
            st.success(f"{L['risk_text']}: {risk_label}")
        elif prob < 0.6:
            risk_label = L["risk_medium"]
            st.warning(f"{L['risk_text']}: {risk_label}")
        else:
            risk_label = L["risk_high"]
            st.error(f"{L['risk_text']}: {risk_label}")

        st.write(f"**{L['prob_text']}:** {prob:.1%}")

# --- AI PERSONAL NUTRITION PAGE ---
elif page == L["nutrition_title"]:
    st.title(L["nutrition_title"])
    st.write(L["nutrition_description"])

    ingredients = st.text_area(L["nutrition_input"])
    if st.button(L["nutrition_submit"]):
        if not ingredients.strip():
            st.warning("Please enter ingredients.")
        else:
            GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
            if not GEMINI_API_KEY:
                st.error("API key not found!")
            else:
                url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
                headers = {
                    "Content-Type": "application/json",
                    "X-goog-api-key": GEMINI_API_KEY
                }
                payload = {
                    "contents": [
                        {"parts": [{"text": f"{ingredients}"}]}
                    ],
                    "systemInstruction": {
                        "parts": [{"text": L["system_prompt"]}]
                    }
                }
                # --- Loading spinner ---
                with st.spinner("Loading..."):
                    response = requests.post(url, headers=headers, json=payload)

                if response.status_code == 200:
                    data = response.json()
                    try:
                        recipe_text = data["candidates"][0]["content"]["parts"][0]["text"]
                        st.markdown(f"{L['recipe']}{recipe_text}")
                    except Exception:
                        st.error("Failed to parse response from API.")
                else:
                    st.error(f"Error: {response.status_code} {response.text}")

# --- INTERACTIVE HEALTH CHATBOT PAGE ---
elif page == L["chatbot_title"]:
    st.title(L["chatbot_title"])
    st.write(L["chatbot_description"])

    # Initialize & Migrate chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for msg in st.session_state.chat_history:
        if msg["role"] == "bot":
            msg["role"] = "assistant"

    # Display existing chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["message"])

    # Use st.chat_input for "sticky" input
    if user_message := st.chat_input(L["chatbot_input"]):
        # Add user message to history and display it
        st.session_state.chat_history.append({"role": "user", "message": user_message})
        with st.chat_message("user"):
            st.markdown(user_message)

        # Placeholder for bot's response
        placeholder = st.empty()

        # Request to Gemini API
        with st.spinner("Loading..."):
            GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
            if not GEMINI_API_KEY:
                placeholder.error("API key not found!")
            else:
                url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
                headers = {
                    "Content-Type": "application/json",
                    "X-goog-api-key": GEMINI_API_KEY
                }

                # Build chat context from history
                context_history = [
                    {"role": "user", "parts": [{"text": c["message"]}]} if c["role"] == "user"
                    else {"role": "model", "parts": [{"text": c["message"]}]}
                    for c in st.session_state.chat_history[:-1]  # all but the last message
                ]

                payload = {
                    "contents": context_history + [{"role": "user", "parts": [{"text": user_message}]}],
                    "systemInstruction": {"parts": [{"text": L["chatbot_system_prompt"]}]}
                }

                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=120)
                    response.raise_for_status()
                    data = response.json()
                    reply_text = data["candidates"][0]["content"]["parts"][0]["text"]
                except requests.exceptions.RequestException as e:
                    reply_text = f"Network Error: {e}"
                except (KeyError, IndexError) as e:
                    reply_text = f"Failed to parse response from API. Detail: {e}"

        # Display bot's response in the placeholder
        with placeholder.chat_message("assistant"):
            st.markdown(reply_text)

        # Add to history
        st.session_state.chat_history.append({"role": "assistant", "message": reply_text})

# --- OCR FOOD COMPOSITION PAGE ---
elif page == L["ocr_title"]:
    st.title(L["ocr_title"])
    st.write(L["ocr_description"])

    uploaded_file = st.file_uploader(L["ocr_upload"], type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        if st.button(L["ocr_button"]):
            with st.spinner(L["ocr_spinner_extract"]):
                # EasyOCR reader (multilingual support, default en+id)
                reader = easyocr.Reader(['en', 'id'])
                result = reader.readtext(uploaded_file.read(), detail=0)
                extracted_text = "\n".join(result)

            if extracted_text.strip():
                st.markdown(f"{L['ocr_result']}\n\n{extracted_text}")

                # Send to Gemini for explanation
                GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
                if not GEMINI_API_KEY:
                    st.error("API key not found!!")
                else:
                    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
                    headers = {
                        "Content-Type": "application/json",
                        "X-goog-api-key": GEMINI_API_KEY
                    }
                    
                    payload = {
                        "contents": [
                            {"parts": [{"text": extracted_text}]}
                        ],
                        "systemInstruction": {
                            "parts": [{"text": L["ocr_system_prompt"]}]
                        }
                    }

                    with st.spinner(L["ocr_spinner_summarize"]):
                        response = requests.post(url, headers=headers, json=payload)

                    if response.status_code == 200:
                        try:
                            ai_summary = response.json()["candidates"][0]["content"]["parts"][0]["text"]
                            st.markdown(f"{L['ocr_summary']}\n\n{ai_summary}")
                        except Exception:
                            st.error("Failed to parse AI response.")
                    else:
                        st.error(f"Error: {response.status_code} {response.text}")
            else:
                st.warning(L["ocr_no_text"])
