import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler


st.title("Căsătoriile aranjate: Tradiție vs. Alegere personală – O confruntare între iubire și aranjamente")

st.header("Setul de date pentru proiectul la PSW")
st.markdown(
    """
    <style>
    .custom-title {
        color: #77DD77;
        font-size: 40px;
        text-align: center;
        color: #77DD77 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="custom-title">RĂDUCU Elena-Nicole și RĂDOI Valentina Anamaria</h1>', unsafe_allow_html=True)

# Bara laterală pentru navigare între secțiuni
section = st.sidebar.radio("Navigați la:",
                           ["Subiectul setului de date",
                            "De ce am ales acest subiect?",
                            "Setul de date",
                            "Setul de date din alta perpesctiva",
                            "Fișier CSV - prezentare date",
                            "Curățarea datelor",
                            "Transformarea datelor - Metode de codificare",
                            "Metode de scalare",
                            "Curățarea datelor dintr-un alt set de date"])

# ---------------------------
# Secțiunea: Subiectul setului de date
# ---------------------------
if section == "Subiectul setului de date":
    st.header("Căsătoria în INDIA: iubire VS aranjamente")
    st.markdown(r"""
        ### Cum a evoluat tradiția căsătoriilor aranjate din India în epoca în care avem Tinder și OkCupid.

        Tradiția căsătoriilor aranjate în India a trecut prin transformări semnificative odată cu apariția platformelor moderne de dating precum Tinder și OkCupid, dar nu a dispărut complet. În schimb, s-a adaptat la noile tehnologii și la schimbările sociale.

        - **De la aranjamente familiale la „aranjamente digitale”** 
        
        Dacă în trecut părinții și rudele extinse jucau rolul principal în selectarea partenerului, astăzi acest proces este mediat tot mai mult de platforme online dedicate căsătoriilor, precum Shaadi.com, Jeevansathi.com sau BharatMatrimony. Aceste platforme funcționează ca un „Tinder pentru căsătorii”, permițând părinților și tinerilor să caute potriviri compatibile bazate pe educație, castă, religie și statut economic.
        
        - **Hibridizarea între căsătorii aranjate și dating modern**
        
        Mulți tineri indieni din clasele urbane și mijlocii folosesc simultan atât aplicații de dating casual (Tinder, Bumble, Hinge), cât și site-uri matrimoniale tradiționale. Deși dating-ul modern le oferă libertatea de a explora relații fără presiune, în multe cazuri, decizia finală privind căsătoria implică și familia.
        
        - **Creșterea „căsătoriilor semi-aranjate”**
        
        Un fenomen tot mai frecvent este cel al căsătoriilor „semi-aranjate”, unde tinerii își aleg partenerii, dar părinții încă au un cuvânt de spus în validarea relației. Astfel, tinerii își pot cunoaște partenerul pe Tinder sau la un eveniment social, dar oficializarea căsătoriei implică acceptul familiei.
        
        - **Schimbarea normelor sociale și căsătoriile inter-castă**
        
        Tehnologia a facilitat relațiile între caste și religii diferite, lucru care era considerat tabu în trecut. Deși rezistența din partea părinților și a comunității persistă, tinerii indieni sunt din ce în ce mai dispuși să își urmeze propriile alegeri.
        
        - **Presiunea socială și influența familiei rămân puternice**
        
        Chiar și cu accesul la aplicații de dating, mulți tineri indieni sunt presați să accepte căsătorii aranjate din motive culturale, familiale sau economice. Pentru mulți, căsătoria este în continuare văzută ca o alianță între familii, nu doar ca o alegere individuală.
        
        """, unsafe_allow_html=True)

# ---------------------------
# Secțiunea: De ce am ales acest subiect?
# ---------------------------
elif section == "De ce am ales acest subiect?":
    st.header("Motivarea alegerii")
    st.write("""
   - Relevanța culturală și socială – Căsătoriile aranjate sunt încă o practică prezentă în multe culturi, iar compararea acestora cu căsătoriile bazate pe alegerea personală poate oferi o perspectivă mai clară asupra impactului lor asupra individului și societății.

 - Confruntarea dintre tradiție și modernitate – Într-o lume în continuă schimbare, unde valorile tradiționale sunt puse față în față cu libertatea de alegere, acest subiect este important pentru a înțelege cum se adaptează căsătoria la noile realități.

- Impactul asupra fericirii și relațiilor – Este interesant să analizăm dacă iubirea și compatibilitatea se pot dezvolta într-o căsătorie aranjată sau dacă alegerea personală este cheia unei relații de succes.

- Curiozitate și dezbatere – Subiectul generează discuții interesante, întrucât există argumente pro și contra pentru ambele tipuri de căsătorii, iar fiecare societate sau individ poate avea o perspectivă diferită.


    """)

# ---------------------------
# Secțiunea: Setul de date
# ---------------------------
elif section == "Setul de date":
    st.header("Setul de date")

    # Citirea fișierului CSV
    file_path = "../marriage_data_india.csv"
    df = pd.read_csv(file_path)

    # Afișarea datelor
    st.write("Afisarea datelor in DataFrame:")
    st.dataframe(df)

    st.subheader("Filtrarea rândurilor după condiții")
    st.write("Filtrarea rândurilelor unde coloana 'Age_at_Marriage' conține valori mai mari sau egale cu 30:")
    st.code("df_sample[df_sample['Age_at_Marriage'] >= 30]", language="python")
    st.write(df[df['Age_at_Marriage'] >= 30])


    st.write("Filtrare și selectarea coloanelor specifice:")
    filtered_df = df.loc[df['Age_at_Marriage'] < 20, ['Children_Count', 'Years_Since_Marriage']]
    st.code("""
    filtered_df = df_sample.loc[df_sample['Age_at_Marriage'] < 20, ['Children_Count', 'Years_Since_Marriage']]
    print(filtered_df)
        """, language="python")
    st.write("Rânduri unde Age_at_Marriage este mai mic de 20 (afișând Children_Count și Years_Since_Marriage):")
    st.write(filtered_df)

# ---------------------------
# Secțiunea: Fișier CSV - prezentare date
# ---------------------------
elif section == "Fișier CSV - prezentare date":
    st.header("Prezentarea datelor dintr-un fișier CSV încărcat")

    # incarcarea fișierului CSV
    uploaded_file = st.file_uploader("Încarcă fișierul CSV", type=["csv"])

    if uploaded_file is not None:
        # citirea datelor
        df = pd.read_csv(uploaded_file)
        st.write(df)

        # Verificarea primelelor 5 rânduri
        st.write("Primele 5 rânduri din date:", df.head())

        # Verificarea exisței unor coloanele necesare
        required_columns = {'Age_at_Marriage', 'Children_Count'}
        if not required_columns.issubset(df.columns):
            st.error(f"Fișierul trebuie să conțină coloanele: {required_columns}")
        else:
            # Creearea categoriilor de vârstă
            bins = [18, 25, 30, 35, 40, 45, 50]
            labels = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-50']
            df['grupa_varsta'] = pd.cut(df['Age_at_Marriage'], bins=bins, labels=labels, include_lowest=True)

            # Agregarea datelelor
            df_grouped = df.groupby('grupa_varsta').agg({
                'Children_Count': ['mean', 'sum'],  # Media și suma copiilor
                'Age_at_Marriage': 'count'  # Numărul de persoane în fiecare categorie
            })

            # Redenumirea coloanelelor pentru claritate
            df_grouped.columns = ['numar_mediu_copii', 'total_copii', 'numar_indivizi']

            # Evitarea valorilor NaN
            df_grouped = df_grouped.fillna(0)

            # Afișarea datelelor agregate
            st.write("Datele agregate:", df_grouped)

            # Grafic de bare - Distribuția numărului de indivizi în funcție de vârstă
            st.write("Grafic de bare - Distribuția numărului de indivizi în funcție de vârstă:")
            st.bar_chart(df_grouped['numar_indivizi'])

            # Grafic de linie - Media copiilor pe categorii de vârstă
            st.write("Grafic de linie - Media copiilor pe categorii de vârstă:")
            st.line_chart(df_grouped['numar_mediu_copii'])

# ---------------------------
# Secțiunea: Curățarea datelor
# ---------------------------
elif section == "Curățarea datelor":
    st.header("Pregătirea setului de date prin tratarea valorilor lipsa si a valorilor extreme")

    file_path = "../marriage_data_india.csv"
    df = pd.read_csv(file_path)

    # Tratarea valorilor lipsă
    st.subheader("Tratarea valorilor lipsă")

    # Analiza valorilor lipsă
    st.write("Numărul de valori lipsă per coloană este:")
    missing_vals_column = df.isnull().sum()
    st.write(missing_vals_column)

    st.write("Procentul de valori lipsă per coloană este:")
    missing_vals_column_precent = (missing_vals_column / len(df)) * 100
    st.write(missing_vals_column_precent)

    # Vizualizarea valorilor lipsa
    missing_df = pd.DataFrame({
        'Valori lipsa': missing_vals_column,
        'Procentajul valorilor lipsa': missing_vals_column_precent
    })
    missing_df = missing_df[missing_df['Valori lipsa'] > 0].sort_values('Procentajul valorilor lipsa', ascending=False)
    st.write(missing_df)

    # Tratarea valorilor extreme folosind IQR
    st.subheader("Tratarea valorilor extreme")

    # Identificarea valorilor extreme folosind IQR
    Q1 = df['Age_at_Marriage'].quantile(0.25)
    Q3 = df['Age_at_Marriage'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Eliminăm valorile extreme din coloana 'Age_at_Marriage'
    df_cleaned = df[(df['Age_at_Marriage'] >= lower_bound) & (df['Age_at_Marriage'] <= upper_bound)]

    # Afișăm setul de date curățat de valori extreme
    st.write("Setul de date fără valori extreme:")
    st.dataframe(df_cleaned)

# ---------------------------
# Secțiunea: Curățarea datelor dintr-un alt set de date
# ---------------------------
elif section == "Curățarea datelor dintr-un alt set de date":
    st.header("Pregătirea setului de date prin tratarea valorilor lipsa si a valorilor extreme")

    file_path = "../2015_16_Statewise_Secondary.csv"
    df = pd.read_csv(file_path)

    # Tratarea valorilor lipsă
    st.subheader("Tratarea valorilor lipsă")

    # Analiza valorilor lipsă
    st.write("Numărul de valori lipsă per coloană este:")
    missing_vals_column = df.isnull().sum()
    st.write(missing_vals_column)

    st.write("Procentul de valori lipsă per coloană este:")
    missing_vals_column_precent = (missing_vals_column / len(df)) * 100
    st.write(missing_vals_column_precent)

    # Vizualizarea valorilor lipsa
    missing_df = pd.DataFrame({
        'Valori lipsa': missing_vals_column,
        'Procentajul valorilor lipsa': missing_vals_column_precent
    })
    missing_df = missing_df[missing_df['Valori lipsa'] > 0].sort_values('Procentajul valorilor lipsa', ascending=False)
    st.write(missing_df)

    # Verificăm valorile lipsă
    missing_values = df.isnull().sum()
    st.write(f"Valori lipsă în fiecare coloană:\n{missing_values}")

    # Opțiuni pentru tratarea valorilor lipsă
    st.write("Alege cum să tratezi valorile lipsă:")
    treat_option = st.radio("Opțiuni", ["Împune valoare fixă (ex: mediană)", "Șterge rândurile cu valori lipsă"])

    if treat_option == "Împune valoare fixă (ex: mediană)":
        # Împunem media pentru valorile numerice
        # Aplică mediană doar pe coloanele numerice
        numeric_columns = df.select_dtypes(include=['number']).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        st.write("Valori lipsă au fost înlocuite cu medianele corespunzătoare.")
    elif treat_option == "Șterge rândurile cu valori lipsă":
        # Ștergem rândurile care conțin valori lipsă
        df.dropna(inplace=True)
        st.write("Rândurile cu valori lipsă au fost șterse.")

    # Afișăm setul de date curățat
    st.write("Setul de date curățat:")
    st.dataframe(df)

# ---------------------------
# Secțiunea: Transformarea datelor - Metode de codificare
# ---------------------------
elif section == "Transformarea datelor - Metode de codificare":
    st.header("Metode de codificare")

    file_path = "../marriage_data_india.csv"
    df = pd.read_csv(file_path)

    categorical_columns = ['Marriage_Type', 'Gender', 'Caste_Match', 'Urban_Rural']

    st.subheader("Codificare binară personalizată: Male=0, Female=1 | Arranged=0, Love=1")
    gender_map = {'Male': 0, 'Female': 1}
    marriage_type_map = {'Arranged': 0, 'Love': 1}

    df_encoded = df[['ID', 'Gender', 'Marriage_Type']].copy()
    df_encoded['Gender'] = df_encoded['Gender'].map(gender_map)
    df_encoded['Marriage_Type'] = df_encoded['Marriage_Type'].map(marriage_type_map)

    st.write(df_encoded)

    # Metoda One-Hot Encoding pentru un alt exemplu
    st.write("Codificare One-Hot Encoding pentru 'Gender':")
    df_encoded = pd.get_dummies(df, columns=categorical_columns)
    st.write(df_encoded)

    st.subheader("Label Encoding")
    st.subheader("Codificarea datelor prin utilizarea valorilor 0, 1 si 2")
    df_columns = [
        'Marriage_Type', 'Gender', 'Caste_Match', 'Religion',
        'Parental_Approval', 'Urban_Rural', 'Dowry_Exchanged',
        'Divorce_Status', 'Spouse_Working', 'Inter-Caste', 'Inter-Religion'
    ]
    df_one_hot_encoded = pd.get_dummies(df, columns=df_columns, drop_first=True)
    st.write("Coloane codificate cu check box:")
    st.write(df_one_hot_encoded)

    st.write("Codificare pentru nivelul de educatie")
    st.write(
        "Am transformat valorile 'School', 'Graduate', 'Postgraduate' în 0, 1, 2 pentru a reflecta progresia logică a nivelului de educație.")
    education_level_map = {
        'School': 0,
        'Graduate': 1,
        'Postgraduate': 2
    }
    df_education_level = df[['Education_Level']].copy()
    df_education_level['Education_Level'] = df_education_level['Education_Level'].map(education_level_map)
    st.write(df_education_level)

    st.write("Codificare pentru nivelul de venit")
    st.write(
        "Valorile 'Low', 'Middle', 'High' au fost codificate ca 0, 1, 2 pentru a reflecta ordinea crescătoare a veniturilor.")
    income_level_map = {
        'Low': 0,
        'Middle': 1,
        'High': 2
    }
    df_income_level = df[['Income_Level']].copy()
    df_income_level['Income_Level'] = df_income_level['Income_Level'].map(income_level_map)
    st.write(df_income_level)

    st.write("Codificare pentru satisfactia conjugala")
    st.write(
        "Valorile 'Low', 'Medium', 'High' au fost codificate în 0, 1, 2 pentru a exprima nivelul de satisfacție într-un mod interpretabil numeric.")
    marital_satisfaction_map = {
        'Low': 0,
        'Medium': 1,
        'High': 2
    }
    df_marital_satisfaction = df[['Marital_Satisfaction']].copy()
    df_marital_satisfaction['Marital_Satisfaction'] = df_marital_satisfaction['Marital_Satisfaction'].map(
        marital_satisfaction_map)
    st.write(df_marital_satisfaction)

    st.write("Tabel cu cele trei coloane codificate")
    df_encoding = df[['ID', 'Education_Level', 'Income_Level', 'Marital_Satisfaction']].copy()
    df_encoding['Education_Level'] = df_encoding['Education_Level'].map(education_level_map)
    df_encoding['Income_Level'] = df_encoding['Income_Level'].map(income_level_map)
    df_encoding['Marital_Satisfaction'] = df_encoding['Marital_Satisfaction'].map(marital_satisfaction_map)
    st.write(df_encoding)

    st.subheader("Codificarea datelor prin Label Encoding")

    # Codificare prin Label Encoding
    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])

    # Afișăm datele codificate
    st.write("Datele codificate:")
    st.dataframe(df)

# ---------------------------
# Secțiunea: Metode de scalare
# ---------------------------
elif section == "Metode de scalare":
    st.header("Metode de scalare")

    file_path = "../marriage_data_india.csv"
    df = pd.read_csv(file_path)

    # declarare lista cu coloanele numerice
    numeric_cols = ['Age_at_Marriage', 'Children_Count', 'Years_Since_Marriage']
    df_numeric = df[numeric_cols].copy()

    st.subheader("Date originale (ne-scalate)")
    st.write(df_numeric)

    minmax_scaler = MinMaxScaler()
    df_minmax = pd.DataFrame(minmax_scaler.fit_transform(df_numeric),
                             columns=[col + '_minmax' for col in numeric_cols])
    st.subheader("Min-Max Scaling (valori între 0 și 1)")
    st.write(df_minmax)

    # st.write("Valorile minime și maxime după Min-Max Scaling:")
    # for col in df_minmax.columns:
        # st.write(f"{col}: min={df_minmax[col].min():.2f}, max={df_minmax[col].max():.2f}")

    standard_scaler = StandardScaler()
    df_standard = pd.DataFrame(standard_scaler.fit_transform(df_numeric),
                               columns=[col + '_standard' for col in numeric_cols])

    st.subheader("Standard Scaling (media 0, deviație standard 1)")
    st.write(df_standard)

    # st.write("Media și deviația standard după Standard Scaling:")
    # for col in df_standard.columns:
        # st.write(f"{col}: mean={df_standard[col].mean():.2f}, std={df_standard[col].std():.2f}")

# ---------------------------
# Secțiunea: Setul de date din alta perpesctiva
# ---------------------------
elif section == "Setul de date din alta perpesctiva":
    st.header("Setul de date din alta perpesctiva")

    # Citirea fișierului CSV
    file_path = "../marriage_data_india.csv"
    df = pd.read_csv(file_path)

    # st.subheader("Informatii")
    # st.write(df.shape)
    # st.write(df.info())
    # st.write(df.head())

    st.write("Statistici descriptive pentru coloanele numerice:")
    st.write(df.describe())

    st.subheader("Media vârstei la căsătorie pe gen")
    age_by_gender = df.groupby('Gender')['Age_at_Marriage'].mean()
    st.write(age_by_gender)
    st.bar_chart(age_by_gender)

    st.subheader("Media copiilor în funcție de gen și tipul de căsătorie")
    children_stats = df.groupby(['Gender', 'Marriage_Type'])['Children_Count'].mean().reset_index()
    st.write(children_stats)