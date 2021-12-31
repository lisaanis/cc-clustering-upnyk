import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
import streamlit as st

sns.set_style()
plt.style.use('fivethirtyeight')

## Membaca data dari file csv
CreditCard_df = pd.read_csv('CreditCard_Data.csv')

# Mengubah nama kolom agar mudah dipahami
CreditCard_df.rename(columns={'CUST_ID': 'Id Cust', 'BALANCE': 'Saldo', 'BALANCE_FREQUENCY': 'Frek Saldo', 'PURCHASES': 'Pembelian', 'ONEOFF_PURCHASES': 'Pembelian Sekali', 'INSTALLMENTS_PURCHASES': 'Pembelian Mencicil', 'CASH_ADVANCE': 'Penarikan Tunai', 'PURCHASES_FREQUENCY': 'Frek Pembelian', 'ONEOFF_PURCHASES_FREQUENCY': 'Frek Pembelian Sekali', 'PURCHASES_INSTALLMENTS_FREQUENCY': 'Frek Pembelian Mencicil', 'CASH_ADVANCE_FREQUENCY': 'Frek Pembelian UangMuka', 'CASH_ADVANCE_TRX': 'Jml Transaksi UangMuka', 'PURCHASES_TRX': 'Jml Transaksi Pembelian', 'CREDIT_LIMIT': 'Batas Kredit', 'PAYMENTS': 'Total Pembayaran', 'MINIMUM_PAYMENTS': 'Total Pembayaran Min', 'PRC_FULL_PAYMENT': 'Prs Pembayaran Penuh', 'TENURE': 'Jangka Waktu'}, inplace=True)

## Terdapat 2 variabel yang memiliki nilai kosong. Mengisi missing value (nilai kosong) dengan median/nilai tengah dari kolom tersebut
# Credit Limit/Batas Kredit
CreditCard_df.loc[(CreditCard_df['Batas Kredit'].isnull() == True), 'Batas Kredit'] = CreditCard_df['Batas Kredit'].median()
# Min Payment/Total Pembayaran Min
CreditCard_df.loc[(CreditCard_df['Total Pembayaran Min'].isnull() == True), 'Total Pembayaran Min'] = CreditCard_df['Total Pembayaran Min'].median()

##UI
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Segmentasi Pelanggan Kartu Kredit")

st.sidebar.title("Model Credit Card Clustering")

st.sidebar.info("""
        Ini merupakan sebuah projek machine learning,
        Dimana dilakukan analisis dan pengelompokkan pelanggan
        berdasarkan perilaku penggunaan kartu kredit
        dengan menggunakan metode K-Means Clustering.
        """)

st.markdown("""
        ## Tentang Data
        Sumber data kartu kredit merupakan data real case yang diperoleh dari Kaggle yang disediakan oleh Arjun Bhasin,
        seorang Data Engineer: https://www.kaggle.com/arjunbhasin2013/ccdata. Data berasal dari sebuah Bank di New York,
        data yang digunakan yaitu data dari 8950 pengguna aktif kartu kredit selama 6 bulan terakhir yang mencakup 18 variabel perilaku pelanggan.
        Data ini mencakup pembelian dan kebiasaan pembayaran pelanggan, seperti seberapa sering pelanggan melakukan pembelian satu kali atau dengan cicilan,
        atau seberapa sering pelanggan melakukan penarikan tunai, dan seberapa banyak pembayaran yang dilakukan oleh pelanggan, serta perilaku lainnya.
        """)

st.markdown("""
        ## Permasalahan Bisnis
        Adanya kartu kredit membuat pihak bank memperoleh bunga yang harus dibayarkan nasabah, namun juga akan berdampak buruk bagi bank
        jika nasabah tidak membayar tagihan kartu kreditnya tepat waktu sehingga akan mengalami kerugian.
        Untuk melakukan antisipasi agar pihak bank tidak mengalami kerugian dan untuk menganalisis karakteristik nasabah tersebut maka perlu dilakukan segmentasi pelanggan
        berdasarkan perilaku penggunaan kartu kredit nasabah sehingga dapat ditentukan strategi pemasaran yang efektif.
        """)


st.markdown("""
        ## Target
        Berdasarkan permasalahan bisnis di atas, target dari proyek penelitian ini adalah: <br>
        1. Berhasil melakukan segmentasi pelanggan menggunakan Algoritma Machine Learning (KMeans Clustering) dengan Python. <br>
        2. Mendapatkan jumlah cluster terbaik dan optimal serta mampu menunjukkan karakteristik yang berbeda dari setiap cluster.
        """, True)


st.cache(persist = True)
st.sidebar.markdown("""
        ## Customer Data
        """)

if st.sidebar.checkbox("Raw Data"):
        st.subheader("Raw Data/Data mentah")
        st.write(CreditCard_df)

## DATA PREPARATION
## Menentukan informasi baru dari variabel yang ada

#1. Pembelian bulanan
CreditCard_df['Pembelian Bulanan']=CreditCard_df['Pembelian']/CreditCard_df['Jangka Waktu']
#2. Penarikan Tunai bulanan
CreditCard_df['Penarikan Bulanan']=CreditCard_df['Penarikan Tunai']/CreditCard_df['Jangka Waktu']

## Diketahui bahwa terdapat 4 tipe pelanggan dalam melakukan pembelian yaitu
#1. Pelanggan yang hanya melakukan pembelian satu kali
#2. Pelanggan yang hanya melakukan pembelian cicilan
#3. Pelanggan yang melakukan keduanya (pembelian cicilan dan satu kali pembayaran)
#4. Pelanggan yang tidak melakukan pembelian apapun

#Maka dilakukan pengkategorian berdasarkan perilaku pembelian pelanggan

def pembelian(CreditCard_df):   
    if (CreditCard_df['Pembelian Sekali']==0) & (CreditCard_df['Pembelian Mencicil']==0):
        return 'Tidak ada pembelian'
    if (CreditCard_df['Pembelian Sekali']==0) & (CreditCard_df['Pembelian Mencicil']>0):
        return 'Pembayaran Mencicil'
    if (CreditCard_df['Pembelian Sekali']>0) & (CreditCard_df['Pembelian Mencicil']==0):
        return 'Pembayaran Satukali'
    if (CreditCard_df['Pembelian Sekali']>0) & (CreditCard_df['Pembelian Mencicil']>0):
         return 'Keduanya'

CreditCard_df['Tipe Pembelian']=CreditCard_df.apply(pembelian,axis=1)

#4. Batas penggunaan (rasio saldo yang tersisa dengan jumlah batas kredit) untuk memperkirakan rasio saldo terhadap batas untuk setiap pelanggan
# Tingkat pemanfaatan yang lebih tinggi menunjukkan adanya risiko kredit. Sehingga, tingkat pemanfaatan yang lebih rendah lebih baik
CreditCard_df['Batas Penggunaan']=CreditCard_df['Saldo']/CreditCard_df['Batas Kredit']

#5. Rasio pembayaran (pembayaran total dengan minimum pembayaran)
CreditCard_df['Rasio Pembayaran']=CreditCard_df['Total Pembayaran']/CreditCard_df['Total Pembayaran Min']

# Melakukan log tranformation untuk mengurangi kemiringan pada data yang tidak simetris dan menghilangkan outliers
credit_log=CreditCard_df.drop(['Id Cust','Tipe Pembelian'],axis=1).applymap(lambda x: np.log(x+1))

## Drop original variabel yang telah digunakan untuk membuat variabel baru. Apabila variabel dibawah ini dikorelasikan dengan variabel turunan akan meningkatkan redudansi data
col=['Saldo','Pembelian','Penarikan Tunai','Jangka Waktu','Total Pembayaran','Total Pembayaran Min','Prs Pembayaran Penuh','Batas Kredit']
credit_pre=credit_log[[x for x in credit_log.columns if x not in col]]

## Mengekspolarasi data untuk mendapatkan wawasan tentang profil pelanggan dengan menggunakan fitur 'Tipe Pembelian' agar didapatkan perilaku pelanggan

##1. Tingkat penggunaan/Batas Penggunaan over Tipe Pembelian
    #2a Menghitung rata-rata tingkat penggunaan kartu kredit untuk setiap tipe pembelian
c1 = CreditCard_df.groupby(['Tipe Pembelian'])['Batas Penggunaan'].mean().sort_values(ascending = True).reset_index()

##2. Rata-rata Pembelian Bulanan over Tipe Pembelian
    #3a Menghitung rata-rata Pembelian Rata-Rata Bulanan untuk setiap tipe pembelian
c2 = CreditCard_df.groupby(by=['Tipe Pembelian'])['Pembelian Bulanan'].mean().sort_values(ascending=False) 

##3. Rata-rata Penarikan Tunai Bulanan over Tipe Pembelian
    #4a Menghitung rata-rata Penarikan Rata-Rata Bulanan untuk setiap tipe pembelian
c3 = CreditCard_df.groupby(['Tipe Pembelian'])['Penarikan Bulanan'].mean().sort_values(ascending=False).reset_index()

# original dataset dengan kolom kategori tipe pembelian dikonversi ke tipe angka untuk proses clustering nantinya
credit_original=pd.concat([CreditCard_df,pd.get_dummies(CreditCard_df['Tipe Pembelian'])],axis=1)

## Data Preparation untuk proses Machine Learning

# Membuat dummy variabel untuk kategori tipe pembelian
credit_pre['Tipe Pembelian']=CreditCard_df.loc[:,'Tipe Pembelian']
#pd.get_dummies(credit_pre['Tipe Pembelian']).head()

# Sekarang menggabungkannya dengan dataframe asli
credit_dummy=pd.concat([credit_pre,pd.get_dummies(credit_pre['Tipe Pembelian'])],axis=1)

tipe=['Tipe Pembelian']
credit_dummy=credit_dummy.drop(tipe,axis=1)

## Standardrizing data

# Menskalakan variabel numerik
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
credit_scaled=ss.fit_transform(credit_dummy)

## Applying PCA 

#untuk mereduksi dimensi/fitur 
from sklearn.decomposition import PCA

credit_ratio={}
for n in range(1,18):
    pc=PCA(n_components=n)
    credit_pca=pc.fit(credit_scaled)
    credit_ratio[n]=sum(credit_pca.explained_variance_ratio_)
    
#Karena 5 components menjelaskan sekitar 87% varians jadi kami memilih 5 components

pc_fixed=PCA(n_components=5).fit(credit_scaled)
reduced_credit=pc_fixed.fit_transform(credit_scaled)

cc_new=pd.DataFrame(data = reduced_credit, columns=['pc1', 'pc2', 'pc3', 'pc4', 'pc5'])

column_cc=credit_dummy.columns
pd.DataFrame(pc_fixed.components_.T, columns=['PC_' +str(i) for i in range(5)],index=column_cc)

# Analisis Faktor : varians dijelaskan oleh masing-masing components
pd.Series(pc_fixed.explained_variance_ratio_,index=['PC_'+ str(i) for i in range(5)])

## UI
st.sidebar.markdown("""
        ## Data Analysis
        """)

st.markdown("""
        ## Data Analysis
        Data adalah salah satu fitur penting dari setiap organisasi/institut/lembaga
        karena membantu para pemimpin bisnis untuk membuat keputusan berdasarkan fakta,
        angka statistik, dan tren. Karena ruang lingkup data yang berkembang ini,
        ilmu data muncul sebagai bidang multidisiplin. Ini menggunakan pendekatan ilmiah,
        prosedur, algoritma, dan kerangka kerja untuk mengekstrak pengetahuan dan wawasan
        dari sejumlah besar data.
        """)

if st.sidebar.checkbox("Data Analysis"):
        st.subheader("Distribusi Pelanggan Berdasarkan Tipe Pembelian")
        labels = ['Keduanya', 'Pembayaran Mencicil', 'Tidak ada pembelian', 'Pembayaran Satukali']
        size = CreditCard_df['Tipe Pembelian'].value_counts()
        colors = ['lightgreen', 'orange', 'lightblue', 'yellow']
        explode = [0.1, 0, 0, 0]
        plt.rcParams['figure.figsize'] = (12, 12)
        plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')
        plt.title('Tipe Pembelian', fontsize = 23)
        plt.axis('off')
        plt.legend(loc='upper left')
        st.pyplot()
        st.text("")

        st.subheader("Tingkat Utilisasi Rata-rata berdasarkan Tipe Pembelian")
        plt.figure(figsize = (10,7))
        sns.barplot(c1['Tipe Pembelian'], c1['Batas Penggunaan'], color='lightcoral')
        st.pyplot()
        st.caption("Pelanggan yang melakukan pembelian cicilan memiliki tingkat utilisasi/nilai yang paling rendah yang menunjukkan nilai kredit yang baik dan risiko kredit yang rendah.")

        st.subheader("Rata-rata Pembelian Bulanan untuk setiap Tipe Pembelian")
        plt.figure(figsize = (10,7))
        c2.plot(kind='bar',color='darkgoldenrod')
        plt.xticks(rotation=0)
        st.pyplot()
        st.caption("Pelanggan dengan tipe pembelian keduanya baik pembelian sekali pembayaran atau pembayaran cicilan membuat total jumlah pembelian rata-rata tertinggi selama 12 bulan terakhir.")

        st.subheader("Rata-rata Penarikan Tunai Bulanan untuk setiap Tipe Pembelian")
        plt.figure(figsize = (10,7))
        sns.barplot(c3['Tipe Pembelian'], c3['Penarikan Bulanan'], color='darkolivegreen')
        st.pyplot()
        st.caption("Pelanggan yang tidak melakukan pembelian baik pembelian sekali pembayaran atau pembelian mencicil membuat penarikan tunai rata-rata bulanan tertinggi.")

## UI
st.sidebar.markdown("""
        ## Jumlah K Optimal
        """)

st.markdown("""
        ## Menentukan K Optimal pada K-Means
        Untuk mendapatkan jumlah cluster yang optimal dan terbaik, kami melakukan pengukuran performansi
        menggunakan  metode Elbow Method, Calinski-Harabasz Index (C-H Index), dan Silhouette Coefficient.
        Target keberhasilan yang direncanakan yaitu mendapatkan jumlah cluster yang terbaik
        dan mampu menunjukkan karakteristik yang berbeda dari setiap cluster.
        """)

if st.sidebar.checkbox("K Optimal pada K-Means"):
        st.subheader("Menentukan Nilai K Menggunakan Elbow Method")
        from sklearn.cluster import KMeans
        sse = {}
        for k in range(2,11):
            kmeans =  KMeans(n_clusters=k, max_iter=1000).fit(cc_new)
            sse[k] = kmeans.inertia_
        plt.figure(figsize = (10,7))
        plt.plot(list(sse.keys()), list(sse.values()))
        plt.xlabel("Nilai Cluster")
        plt.ylabel("SSE")
        st.pyplot()
        st.caption("Dapat dilihat menggunakan Elbow Method, pada garis distorsi, bagian garis yang membelok sehingga terlihat seperti siku lengan pada titik k=4")

        st.subheader("Menentukan Nilai K Menggunakan Silhouette Score")
        from sklearn.metrics import silhouette_score
        silhouette_scores = []
        for k in range(2,11):
            silhouette_scores.append(
                silhouette_score(cc_new, KMeans(n_clusters = k).fit_predict(cc_new)))
            keys = [2, 3, 4, 5, 6, 7, 8, 9, 10]    
        plt.figure(figsize = (10,7))
        plt.bar(keys, silhouette_scores) 
        plt.xlabel("Nilai Cluster")
        plt.ylabel("Silhouette Score")
        st.pyplot()
        st.caption("Dapat dilihat pula menggunakan Silhouette Score bahwa score tertinggi pada titik k=4 yaitu 0.459")

        st.subheader("Menentukan Nilai K Menggunakan Calinski Harabasz Score")
        from sklearn.metrics import calinski_harabasz_score
        calinski_score={}
        for k in range(2,11):
            km_score=KMeans(n_clusters=k)
            km_score.fit(cc_new)
            calinski_score[k]=calinski_harabasz_score(cc_new,km_score.labels_)
        plt.figure(figsize = (10,7))
        pd.Series(calinski_score).plot()
        plt.xlabel("Nilai Cluster")
        plt.ylabel("Calinski Harabasz Score")
        st.pyplot()
        st.caption("Dapat dilihat pula menggunakan Calinski Harabasz Score bahwa score tertinggi pada titik k=4 yaitu 6174")


## UI
st.sidebar.markdown("""
        ## Clustering
        """)

st.markdown("""
        ## Clustering K-Means
        Clustering merupakan metode yang digunakan untuk membagi rangkaian data menjadi beberapa grup berdasarkan kesamaan-kesamaan yang telah ditentukan sebelumnya.
        Dengan menggunakan clustering kita dapat mengelompokkan pelanggan berdasarkan kemiripan penggunaan kartu kredit dan memisahkan pelanggan yang tidak mirip sejauh mungkin.
        Algoritma K-Means dipilih sebagai metode dalam melakukan pengelompokkan nasabah karena mampu melakukan cluster dengan jumlah data yang besar dan data outliers
        (data menyimpang terlalu jauh dari data-data lainnya) dengan waktu yang cepat. 
        """)

if st.sidebar.checkbox("Clustering K-Means"):
        st.subheader("Clustering menggunakan K-Means dengan k=4")
        # Clustering menggunakan Algoritma K-Means dengan k=4
        from sklearn.cluster import KMeans
        kmeansCC_4=KMeans(n_clusters=4)
        kmeansCC_4.fit(cc_new)
        labels = kmeansCC_4.labels_

        final_df = pd.concat([cc_new, pd.DataFrame({'cluster':labels})], axis = 1)
        final_df.head()

        ## Visualisasi
        plt.figure(figsize=(15,10))
        sns.scatterplot(x="pc1", y="pc2", hue="cluster", data=final_df,palette=['red','green','blue','yellow'])
        plt.scatter(kmeansCC_4.cluster_centers_[:, 0], kmeansCC_4.cluster_centers_[:, 1], c = 'black', marker ='*', label='centroid')
        plt.legend()
        st.pyplot()

        st.subheader("Clusters Pairplot")
        cc_pairplot=pd.DataFrame(reduced_credit,columns=['PC_' +str(i) for i in range(5)])
        cc_pairplot['Cluster']=kmeansCC_4.labels_
        #hubungan antar component pada data
        sns.pairplot(cc_pairplot,hue='Cluster', palette= 'Set1', diag_kind='kde',height=2)
        st.pyplot()

        st.subheader("Label Cluster")
        # Pemilihan variabel yang akan digunakan untuk menentukan informasi baru
        kolom_info=['Jml Transaksi Pembelian','Pembelian Bulanan','Penarikan Bulanan','Batas Penggunaan','Jml Transaksi UangMuka',
         'Rasio Pembayaran','Keduanya','Pembayaran Mencicil','Pembayaran Satukali','Tidak ada pembelian','Batas Kredit']
        # Menggabungkan label cluster dari proses Clustering K-Means dengan data
        CC_4=pd.concat([credit_original[kolom_info],pd.Series(kmeansCC_4.labels_,name='Cluster')],axis=1)
        st.write(CC_4)

        st.subheader("Distribusi Data Setiap Cluster")
        # Nilai Mean memberikan indikasi yang baik tentang distribusi data. Sehingga dicari nilai rata-rata untuk setiap variabel untuk setiap cluster 0,1,2,3
        CC_k4=CC_4.groupby('Cluster')\
               .apply(lambda x: x[kolom_info].mean()).T
        st.write(CC_k4)

        st.subheader("Visualisasi")
        #Visualisasi
        fig,ax=plt.subplots(figsize=(20,15))
        index=np.arange(len(CC_k4.columns))

        c1=np.log(CC_k4.loc['Penarikan Bulanan',:].values)
        c2=CC_k4.loc['Batas Penggunaan',:].values
        c3=np.log(CC_k4.loc['Pembelian Bulanan',:].values)
        c4=CC_k4.loc['Rasio Pembayaran',:].values
        c5=CC_k4.loc['Pembayaran Mencicil',:].values
        c6=CC_k4.loc['Pembayaran Satukali',:].values

        bar_width=.12
        cc1=plt.bar(index,c1,color='r',label='Penarikan Bulanan',width=bar_width)
        cc2=plt.bar(index+bar_width,c2,color='g',label='Batas Penggunaan',width=bar_width)
        cc3=plt.bar(index+2*bar_width,c3,color='b',label='Pembelian Bulanan',width=bar_width)
        cc4=plt.bar(index+3*bar_width,c4,color='y',label='Rasio Pembayaran',width=bar_width)
        cc5=plt.bar(index+4*bar_width,c5,color='m',label='Pembayaran Mencicil',width=bar_width)
        cc6=plt.bar(index+5*bar_width,c6,color='k',label='Pembayaran Satukali',width=bar_width)

        plt.xlabel("Cluster")
        plt.title("Informasi")
        plt.xticks(index + bar_width, ('Cluster-0', 'Cluster-1', 'Cluster-2', 'Cluster-3'))
        plt.legend()
        st.pyplot()

        st.subheader("Distribusi Cluster Pelanggan")
        # Persentase tiap cluster dari semua pelanggan
        jml=CC_4.groupby('Cluster').apply(lambda x: x['Cluster'].value_counts())
        persen=pd.Series((jml.values.astype('float')/ CC_4.shape[0])*100,name='Persentase')

        labels = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']
        size = CC_4['Cluster'].value_counts().sort_index()
        colors = ['lightcoral', 'lightseagreen', 'royalblue', 'gold']
        
        plt.rcParams['figure.figsize'] = (12, 12)
        plt.pie(size, colors = colors, labels = labels, shadow = True, autopct = '%.2f%%')
        plt.title('Distribusi Pelanggan Berdasarkan Cluster', fontsize = 23)
        plt.axis('off')
        plt.legend(loc='upper left')
        st.pyplot()

        st.subheader("Tabel Cluster Pelanggan")
        CC_type4 =CC_4.drop(['Jml Transaksi Pembelian','Pembelian Bulanan', 'Penarikan Bulanan', 'Batas Penggunaan', 'Jml Transaksi UangMuka','Rasio Pembayaran', 'Keduanya', 'Pembayaran Mencicil', 'Pembayaran Satukali', 'Tidak ada pembelian', 'Batas Kredit'],axis=1)

        def type_cluster(CC_type4):
                if (CC_type4['Cluster']==0): return 'Customer Berisiko'
                if (CC_type4['Cluster']==1): return 'Customer Biasa'
                if (CC_type4['Cluster']==2): return 'Customer Target'
                if (CC_type4['Cluster']==3): return 'Customer Prioritas'

        CreditCard_df['Tipe Cluster']=CC_type4.apply(type_cluster, axis=1)

        def karakteristik(CC_type4):
                if (CC_type4['Cluster']==0): return 'Kelompok pelanggan yang melakukan transaksi pembelian sekali bayar terbanyak dan memiliki rasio pembayaran paling sedikit di antara semua cluster.'
                if (CC_type4['Cluster']==1): return 'Kelompok pelanggan yang memiliki penarikan tunai bulanan tertinggi dan melakukan pembelian cicilan maupun sekali beli, memiliki skor pembelian rata-rata yang buruk.'
                if (CC_type4['Cluster']==2): return 'Kelompok pelanggan yang memiliki pembelian Rata-rata maksimum dan penarikan tunai bulanan yang baik, serta pelanggan Cluster ini melakukan pembelian cicilan maupun satu kali pembayaran.'
                if (CC_type4['Cluster']==3): return 'Kelompok pelanggan yang melakukan pembayaran cicilan terbanyak dan tertinggi, memiliki rasio pembayaran maksimal dan tidak melakukan pembelian sekali pembayaran, serta merupakan kelompokan dengan risiko minimum'

        CreditCard_df['Karakteristik']=CC_type4.apply(karakteristik, axis=1)
        
        df = CreditCard_df.drop(CreditCard_df.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]], axis=1)
        st.write(df)

        st.subheader("Pencarian Customer")
        id_cust = st.text_input('Searching', 'C10001', max_chars=6)

        if(df['Id Cust'] == id_cust).any():
                index = df.index
                new_val = df['Id Cust'] == id_cust
                b = index[new_val]
                new_output = b.tolist()
                tc = df['Tipe Cluster'].iloc[new_output].values[0]
                karakter = df['Karakteristik'].iloc[new_output].values[0]
                st.write('ID', id_cust, 'dengan tipe cluster', tc, 'merupakan ', karakter)
        else:
                st.write('Tidak ada customer dengan ID', id_cust)

        
