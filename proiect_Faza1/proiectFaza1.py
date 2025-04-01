import streamlit as st
import pandas as pd
import numpy as np

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
                            "Setul de date"])

# ---------------------------
# Secțiunea: Noțiuni de bază Streamlit
# ---------------------------
if section == "Subiectul setului de date":
    st.header("Căsătoria în INDIA: iubire VS aranjamente")
    st.markdown(r"""
        ### Cum a evoluat tradiția căsătoriilor aranjate din India în epoca în care avem Tinder și OkCupid.

        Tradiția căsătoriilor aranjate în India a trecut prin transformări semnificative odată cu apariția platformelor moderne de dating precum Tinder și OkCupid, dar nu a dispărut complet. În schimb, s-a adaptat la noile tehnologii și la schimbările sociale.

        - *De la aranjamente familiale la „aranjamente digitale”* 

        Dacă în trecut părinții și rudele extinse jucau rolul principal în selectarea partenerului, astăzi acest proces este mediat tot mai mult de platforme online dedicate căsătoriilor, precum Shaadi.com, Jeevansathi.com sau BharatMatrimony. Aceste platforme funcționează ca un „Tinder pentru căsătorii”, permițând părinților și tinerilor să caute potriviri compatibile bazate pe educație, castă, religie și statut economic.

        - *Hibridizarea între căsătorii aranjate și dating modern*

        Mulți tineri indieni din clasele urbane și mijlocii folosesc simultan atât aplicații de dating casual (Tinder, Bumble, Hinge), cât și site-uri matrimoniale tradiționale. Deși dating-ul modern le oferă libertatea de a explora relații fără presiune, în multe cazuri, decizia finală privind căsătoria implică și familia.

        - *Creșterea „căsătoriilor semi-aranjate”*

        Un fenomen tot mai frecvent este cel al căsătoriilor „semi-aranjate”, unde tinerii își aleg partenerii, dar părinții încă au un cuvânt de spus în validarea relației. Astfel, tinerii își pot cunoaște partenerul pe Tinder sau la un eveniment social, dar oficializarea căsătoriei implică acceptul familiei.

        - *Schimbarea normelor sociale și căsătoriile inter-castă*

        Tehnologia a facilitat relațiile între caste și religii diferite, lucru care era considerat tabu în trecut. Deși rezistența din partea părinților și a comunității persistă, tinerii indieni sunt din ce în ce mai dispuși să își urmeze propriile alegeri.

        - *Presiunea socială și influența familiei rămân puternice*

        Chiar și cu accesul la aplicații de dating, mulți tineri indieni sunt presați să accepte căsătorii aranjate din motive culturale, familiale sau economice. Pentru mulți, căsătoria este în continuare văzută ca o alianță între familii, nu doar ca o alegere individuală.

        """, unsafe_allow_html=True)

# ---------------------------
# Secțiunea: Introducere în Pandas
# ---------------------------
elif section == "De ce am ales acest subiect?":
    st.header("Motivarea alegerii")
    st.write("""
   - Relevanța culturală și socială – Căsătoriile aranjate sunt încă o practică prezentă în multe culturi, iar compararea acestora cu căsătoriile bazate pe alegerea personală poate oferi o perspectivă mai clară asupra impactului lor asupra individului și societății.

 - Confruntarea dintre tradiție și modernitate – Într-o lume în continuă schimbare, unde valorile tradiționale sunt puse față în față cu libertatea de alegere, acest subiect este important pentru a înțelege cum se adaptează căsătoria la noile realități.

- Impactul asupra fericirii și relațiilor – Este interesant să analizăm dacă iubirea și compatibilitatea se pot dezvolta într-o căsătorie aranjată sau dacă alegerea personală este cheia unei relații de succes.

- Curiozitate și dezbatere – Subiectul generează discuții interesante, întrucât există argumente pro și contra pentru ambele tipuri de căsătorii, iar fiecare societate sau individ poate avea o perspectivă diferită.


    """)

elif section == "Setul de date":
    st.header("Setul de date")

    # Citirea fișierului CSV (asigură-te că fișierul este în același director sau specifică calea completă)
    file_path = "marriage_data_india.csv"  # Schimbă cu calea reală a fișierului tău CSV
    df = pd.read_csv(file_path)

    # Afișarea datelor
    st.write("Afisarea datelor in DataFrame:")
    st.dataframe(df)

    st.subheader("Filtrarea rândurilor după condiții")
    st.write("Filtrați rândurile unde 'Age_at_Marriage' este mai mare sau egală cu 30:")
    st.code("df_sample[df_sample['Age_at_Marriage'] >= 30]", language="python")
    st.write(df[df['Age_at_Marriage'] >= 30])

    st.write("1. *Filtrare și Selectarea Coloanelor Specifice:*")
    filtered_df = df.loc[df['Age_at_Marriage'] < 20, ['Children_Count', 'Years_Since_Marriage']]
    st.code("""
    filtered_df = df_sample.loc[df_sample['Age_at_Marriage'] < 20, ['Children_Count', 'Years_Since_Marriage']]
    print(filtered_df)
        """, language="python")
    st.write("Rânduri unde Age_at_Marriage este mai mic de 20 (afișând Children_Count și Years_Since_Marriage):")
    st.write(filtered_df)

    # Încarcă fișierul CSV
    uploaded_file = st.file_uploader("Încarcă fișierul CSV", type=["csv"])

    if uploaded_file is not None:
        # Citim datele
        df = pd.read_csv(uploaded_file)

        # Verificăm primele rânduri
        st.write("Primele 5 rânduri din date:", df.head())

        # Verificăm dacă există coloanele necesare
        required_columns = {'Age_at_Marriage', 'Children_Count'}
        if not required_columns.issubset(df.columns):
            st.error(f"Fișierul trebuie să conțină coloanele: {required_columns}")
        else:
            # Creăm categorii de vârstă
            bins = [18, 25, 30, 35, 40, 45, 50]
            labels = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-50']
            df['grupa_varsta'] = pd.cut(df['Age_at_Marriage'], bins=bins, labels=labels, include_lowest=True)

            # Agregăm datele
            df_grouped = df.groupby('grupa_varsta').agg({
                'Children_Count': ['mean', 'sum'],  # Media și suma copiilor
                'Age_at_Marriage': 'count'  # Numărul de persoane în fiecare categorie
            })

            # Renumim coloanele pentru claritate
            df_grouped.columns = ['numar_mediu_copii', 'total_copii', 'numar_indivizi']

            # Evităm valori NaN
            df_grouped = df_grouped.fillna(0)

            # Afișăm datele agregate
            st.write("Datele agregate:", df_grouped)

            st.write("Grafic de bare - Distribuția numărului de indivizi în funcție de vârstă:")
            # Grafic de bare - Distribuția numărului de indivizi în funcție de vârstă
            st.bar_chart(df_grouped['numar_indivizi'])

            st.write("Grafic de linie - Media copiilor pe categorii de vârstă:")
            # Grafic de linie - Media copiilor pe categorii de vârstă
            st.line_chart(df_grouped['numar_mediu_copii'])