import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from joblib import dump, load
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

def read_Clusterfile(moving,duration):
    dic = {1:'01',5:"05",10:'10',15:"15",30:'30',40:'40',50:"50",60:"60"}
    mov = dic[moving]
    dur = dic[duration]
    data_path = 'C:/0_Academy/VScode/Data/Stat_Var/'
    file_name = 'Stat_{}_{}.csv'.format(mov,dur)
    df   = pd.read_csv(data_path + file_name)
    observations = df.copy()
    observations['Date'] = pd.to_datetime(observations['Date'])
    print("Import Done")
    return observations

def read_Monthly():
    data_path = 'C:/0_Academy/VScode/Data/Stat_Var/'
    file_name = 'Stat_Monthly.csv'
    df   = pd.read_csv(data_path + file_name)
    observations = df.copy()
    observations['Date'] = pd.to_datetime(observations['Date'])
    print("Import Done")
    return observations

#clf = load('KM_Cluster.joblib') 
#scaler = load('KM_Cluster_scale.joblib') pip install --upgrade pip
#centroid = clf.cluster_centers_
#Centroid = scaler.inverse_transform(centroid)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
def K_means_Clustering(data,n_clusters=3):
    df01 = data[['sin','cos','WS95','R_bar']].copy()
    df01.dropna(how='any',inplace=True)
    df01.reset_index(inplace=True,drop=True)

    scaler = StandardScaler()
    scaler.fit(df01)
    x_train = scaler.transform(df01)
    #接下來匯入KMeans函式庫
    n_clusters = 3
    clf = KMeans(n_clusters)
    clf.fit(x_train) #開始訓練！
    return clf,scaler

def Sort_Label(data,clf,scaler,n_clusters=3):
    df01 = data[['sin','cos','WS95','R_bar']].copy()
    df01.fillna(method='ffill',inplace=True)
    df01.reset_index(inplace=True,drop=True)
    x_train = scaler.transform(df01)
    data['label'] = clf.predict(x_train)
    #data['label'] = clf.labels_
    centro = clf.cluster_centers_
    Centro = scaler.inverse_transform(centro)
    array  = Centro[:,2] # sort with WS
    temp   = array.argsort()
    ranks  = np.empty_like(temp)
    ranks[temp]   = np.arange(len(array))
    lab_chg       = {i:ranks[i] for i in range(n_clusters)}
    data['label'] = data['label'].apply(lambda x: lab_chg[x])
    return data

def Centro_Info(data,clf,scaler):
    df01 = data[['sin','cos','WS95','R_bar']].copy()
    centro = clf.cluster_centers_
    Centro = scaler.inverse_transform(centro)
    array  = Centro[:,1] # sort with R_bar
    temp   = array.argsort()
    ranks  = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    Cen    = Centro[temp]
    df_cen = pd.DataFrame(Cen)
    df_cen.columns = df01.columns.copy()
    df_cen['WD95'] = np.arctan2(df_cen['sin'],df_cen['cos'])*180/np.pi
    return df_cen


# Clustering
# 計算並繪製輪廓分析的結果
# 因下列為迴圈寫法, 無法再分拆為更小執行區塊, 請見諒
def Silhoutte_Score(data,range_n_clusters=[2,3,4,5]):
    df01 = data[['WS95','R_bar','sin','cos',]].copy()
    df01.dropna(how='any',inplace=True)
    df01.reset_index(inplace=True,drop=True)
    scaler = StandardScaler()
    scaler.fit(df01)
    X = scaler.transform(df01)

    SSE=[]
    Sil_score=[]
    for n_clusters in range_n_clusters:
        # 設定小圖排版為 1 row 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        # 左圖為輪廓分析(Silhouette analysis), 雖然輪廓係數範圍在(-1,1)區間, 但範例中都為正值, 因此我們把顯示範圍定在(-0.1,1)之間
        ax1.set_xlim([-0.1, 1])
        # (n_clusters+1)*10 這部分是用來在不同輪廓圖間塞入空白, 讓圖形看起來更清楚
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
        # 宣告 KMean 分群器, 對 X 訓練並預測
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        # 計算 Sum of Square Error 
        SSE.append(clusterer.inertia_)
        # 計算所有點的 silhouette_score 平均
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg,
          "SSE is:", clusterer.inertia_)
        Sil_score.append(silhouette_avg)
        # 計算所有樣本的 The silhouette_score
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        y_lower = 10
        for i in range(n_clusters):
            # 收集集群 i 樣本的輪廓分數，並對它們進行排序
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
            # 在每個集群中間標上 i 的數值
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            # 計算下一個 y_lower 的位置
            y_lower = y_upper + 10
            
        ax1.set_title("The silhouette plot for the various clusters.",fontsize=20)
        ax1.set_xlabel("The silhouette coefficient values",fontsize=16)
        ax1.set_ylabel("Cluster label",fontsize=16)

        # 將 silhouette_score 平均所在位置, 畫上一條垂直線
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])  # 清空 y 軸的格線
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        # 右圖我們用來畫上每個樣本點的分群狀態, 從另一個角度觀察分群是否洽當
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 2], X[:, 3], marker='.', s=80, lw=0, alpha=0.7,c=colors, edgecolor='k')
        # 在右圖每一群的中心處, 畫上一個圓圈並標註對應的編號
        centers = clusterer.cluster_centers_
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',c="white", alpha=1, s=500, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,s=200, edgecolor='k')
            ax2.set_title("The visualization of the clustered data.",fontsize=20)
            ax2.set_xlabel("Feature space for the 1st feature",fontsize=16)
            ax2.set_ylabel("Feature space for the 2nd feature",fontsize=16)
            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n_clusters),fontsize=22, fontweight='bold')
        plt.show()
    return SSE, Sil_score, scaler

#Elbow_Methods
def Elbow_Method(Sil_score,SSE,k_clusters):
    fig, ax1 = plt.subplots(figsize=(12,6))
    color1 = 'tab:red'
    color2 = 'tab:blue'
    ax1.plot(k_clusters,Sil_score, color=color1) 
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(k_clusters,SSE, color=color2)
    ax1.tick_params(axis='x',labelsize=20)
    ax1.tick_params(axis='y', labelcolor=color1,labelsize=20)
    ax1.set_title('Validation Score',fontsize=25)
    ax1.set_xlabel('k_cluster',fontsize=25)
    ax1.set_ylabel('Silhoutte Score', color=color1,fontsize=25)
    ax2.set_ylabel('Sum of Square Distance', color=color2,fontsize=25)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color2,labelsize=20)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
