# -*- coding: utf-8 -*-

import pandas as pd
import pickle
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
import numpy as np
import sknetwork
from sknetwork.gnn.gnn_classifier import GNNClassifier
from sknetwork.classification import get_accuracy_score

df = pd.read_pickle('soja_sentences_nodes_ready.pkl').reset_index()
# Dataset: https://www.sciencedirect.com/science/article/pii/S2352340924005122?via%3Dihub
# Download: https://data.mendeley.com/datasets/f8fdmpp6yh/2

df.head()

nbb     = [[8, 2], [16,2]]
rtt     = [0.0001, 0.001, 0.01]
tpp     = ['Sage', 'Conv']
ptt     = [50, 100]


"""
--------------------------------------------------------------------------
Paper: Section 3.3 - Graph Modeling
--------------------------------------------------------------------------

"""

import networkx as nx

def create_graph(txt_embed='Paraph_MiniLM', txt_clus=None, txt_clus_pip=None, ts_close=None, ts_vol=None):

    Gp = nx.Graph()

    for idx, row in df.iterrows():

        news_node = 'news:' + str(idx)
        Gp.add_node(news_node, layer='txt_embeddings', label=int(row['label']), features = row[txt_embed], date= str(row['Date']))

        if txt_clus != None:
            cluster_news_node = 'cluster_text:' + str(row[txt_clus])
            Gp.add_node(cluster_news_node, layer='txt_clusters', label=-2, features = [0] * 384)
            Gp.add_edge(news_node, cluster_news_node)

        if (txt_clus_pip != None) and (int(row[txt_clus_pip]) >= 0):
            cluster_txt_pip_node = 'cluster_text_pip:' + str(row[txt_clus_pip])
            Gp.add_node(cluster_txt_pip_node, layer='pip_text', label=-2, features = [0] * 384)
            Gp.add_edge(news_node, cluster_txt_pip_node)

        if (ts_close != None) and (int(row[ts_close]) >= 0):
            cluster_close_node = 'cluster_close:' + str(row[ts_close])
            Gp.add_node(cluster_close_node, layer='ts_close', label=-2, features = [0] * 384)
            Gp.add_edge(news_node, cluster_close_node)

        if (ts_vol != None) and (int(row[ts_vol]) >= 0):
            cluster_vol_node = 'cluster_volume:' + str(row[ts_vol])
            Gp.add_node(cluster_vol_node, layer='ts_volume', label=-2, features = [0] * 384)
            Gp.add_edge(news_node, cluster_vol_node)

    return Gp

"""
--------------------------------------------------------------------------
Paper: 	Section 3.4 - Graph Neural Networks; and
	Section 4 - Experimental Evaluation
--------------------------------------------------------------------------

"""

def set_test(graph, date, knn_graph): # Função para esconder o rotulo dos nós de teste

  test_mask = []
  for node in graph.nodes:
    if knn_graph:
      if graph.nodes[node]['date'] > date:
        test_mask.append(True)
      else:
        test_mask.append(False)
    else:
      if 'news' in node and graph.nodes[node]['date'] > date:
        test_mask.append(True)
      else:
        test_mask.append(False)

  return test_mask

def train_gnn(graph, tpp, nbb, rtt, ptc, knn_graph):

    adjacency = sparse.csr_matrix(nx.adjacency_matrix(graph, dtype=np.bool_))     # Matriz de Adjacencia
      
    features = np.array([graph.nodes[node]['features'] for node in graph.nodes])  # Representação inicial
      
    labels_true = np.array([graph.nodes[node]['label'] for node in graph.nodes])  # Rótulos
      
    test_mask = set_test(graph, '2021-01-01', knn_graph)                          # Mascara para teste
      
    labels = labels_true.copy()                                                   # Mascara para teste
      
    labels[test_mask] = -1                                                        # Mascara para teste
      
    gnn = GNNClassifier(layer_types=tpp, dims=nbb, learning_rate= rtt, early_stopping=True, patience=ptc, verbose=False) # definição da GNN
      
    labels_pred = gnn.fit_predict(adjacency, features, labels, validation = 0.1, n_epochs=1000, random_state=81, history=True) # treino e predição da GNN
      
    return get_accuracy_score(labels_true[test_mask], labels_pred[test_mask]), gnn.history_

"""
--------------------------------------------------------------------------
Paper: Section 4 - Experimental Evaluation
--------------------------------------------------------------------------

"""

# Saving the results

def saveGraph(file, lrpr, ltpp, lnbb, lrtt, lptt, lval):

    df_res = pd.DataFrame()

    df_res['Representação'] = lrpr
    df_res['Tipo']          = ltpp
    df_res['Dimensão']      = lnbb
    df_res['Taxa Aprend.']  = lrtt
    df_res['Paciencia']     = lptt
    df_res['Validação']     = lval

    df_res.to_csv(f'{file}.csv')

# BruteForce with all parameters

def bruteForce(Grp, rep_mod):

    lrp, ltp, lnb, lrt, lpt, lva = [], [], [], [], [], []

    for rt in rtt:
        for pt in ptt:
            for nb in nbb:
                for tp in tpp:

                    acc, his = train_gnn(Grp, tp, nb, rt, pt, False)
                    val = round(float(his['val_accuracy'][-1]), 4)
                    print(f"Type: {tp} \t dimensions: {nb} \t learning rate: {rt} \t patience: {pt} \t Val: {val}")
                    lrp.append(rep_mod)
                    ltp.append(tp)
                    lnb.append(nb)
                    lrt.append(rt)
                    lpt.append(pt)
                    lva.append(val)

    return lrp, ltp, lnb, lrt, lpt, lva



"""
--------------------------------------------------------------------------
Paper: Section 4.4 Results
--------------------------------------------------------------------------

"""

# Graph1: Text and Time Series daily
# Table 3 - Daily

def graphs1():

    tx_clus =    ['Paraph_MiniLM_clus_8', 'Paraph_MiniLM_clus_16']
    ts_close =   ['Close_clus_8', 'Close_clus_16']
    ts_volume =  ['Volume_clus_8', 'Volume_clus_16']

    # Fig. 3 - C : News Node, Text Cluster, and Closing Cluster 
    ltpp, lnbb, lrtt, lval, lrpr, lptt = [], [], [], [], [], []
    for txc in tx_clus:
        for tsc in ts_close:
            rep_mod = f"{txc}__{tsc}"
            print(rep_mod)
            G_txtsc = create_graph(txt_clus=txc, ts_close=tsc, ts_vol=None)

            rp, tp, nb, rt, pt, va = bruteForce(G_txtsc, rep_mod)
            lrpr.extend(rp)
            ltpp.extend(tp)
            lnbb.extend(nb)
            lrtt.extend(rt)
            lptt.extend(pt)
            lval.extend(va)

    saveGraph('01_tx_tsclose', lrpr, ltpp, lnbb, lrtt, lptt, lval)

    # Fig. 3 - V : News Node, Text Cluster, and Volume Cluster 
    ltpp, lnbb, lrtt, lval, lrpr, lptt = [], [], [], [], [], []
    for txc in tx_clus:
        for tsv in ts_volume:
            rep_mod = f"{txc} + {tsv}"
            print(rep_mod)
            G_tx_tsv = create_graph(txt_clus=txc, ts_close=None, ts_vol=tsv)

            rp, tp, nb, rt, pt, va = bruteForce(G_tx_tsv, rep_mod)
            lrpr.extend(rp)
            ltpp.extend(tp)
            lnbb.extend(nb)
            lrtt.extend(rt)
            lptt.extend(pt)
            lval.extend(va)

    saveGraph('01_tx_tsvolume', lrpr, ltpp, lnbb, lrtt, lptt, lval)

    # Fig. 3 - C/V : News Node, Text Cluster, Closing Cluster, and Volume Cluster 
    ltpp, lnbb, lrtt, lval, lrpr, lptt = [], [], [], [], [], []
    for txc in tx_clus:
        for tsc in ts_close:
            for tsv in ts_volume:
                rep_mod = f"{txc} + {tsc} + {tsv}"
                print(rep_mod)
                G_tx_tscv = create_graph(txt_clus=txc, ts_close=tsc, ts_vol=tsv)

                rp, tp, nb, rt, pt, va = bruteForce(G_tx_tscv, rep_mod)
                lrpr.extend(rp)
                ltpp.extend(tp)
                lnbb.extend(nb)
                lrtt.extend(rt)
                lptt.extend(pt)
                lval.extend(va)

    saveGraph('01_tx_tsclose_tsvol', lrpr, ltpp, lnbb, lrtt, lptt, lval)


    # Fig. 3 - k-nn. 
    ltpp, lnbb, lrtt, lval, lrpr, lptt = [], [], [], [], [], []
    knn_g = kneighbors_graph(df['Paraph_MiniLM'].to_list(), 2, mode='connectivity', include_self=False)
    G_knn = nx.Graph(knn_g)

    for idx, row in df.iterrows():
        G_knn.nodes[idx]['label'] = int(row['label'])
        G_knn.nodes[idx]['features'] = row['Paraph_MiniLM']
        G_knn.nodes[idx]['date'] = str(row['Date'])

    rp, tp, nb, rt, pt, va = bruteForce(G_knn, 'Knn_connectivity')
    lrpr.extend(rp)
    ltpp.extend(tp)
    lnbb.extend(nb)
    lrtt.extend(rt)
    lptt.extend(pt)
    lval.extend(va)

    saveGraph('01_tx_knn', lrpr, ltpp, lnbb, lrtt, lptt, lval)


#Graph: PIP time series and Text
#Paper: Table 3 - PIP

def graphs2():

    tx_clus =       ['Paraph_MiniLM_clus_8', 'Paraph_MiniLM_clus_16']
    tx_pip_clus =   ['Paraph_MiniLM_clus_pip_8', 'Paraph_MiniLM_clus_pip_16']
    ts_pip_close =  ['Close_clus_pip_8', 'Close_clus_pip_16']
    ts_pip_volume = ['Volume_clus_pip_8', 'Volume_clus_pip_16']
    
    tx_clus =       ['Paraph_MiniLM_clus_16']
    tx_pip_clus =   ['Paraph_MiniLM_clus_pip_16']
    ts_pip_close =  ['Close_clus_pip_16']
    ts_pip_volume = ['Volume_clus_pip_16']
    

    # PIP - T: News Node, Text Cluster, Text PIP Cluster
    ltpp, lnbb, lrtt, lval, lrpr, lptt = [], [], [], [], [], []
    for ltc in tx_clus:
        for ltp in tx_pip_clus:

            rep_mod = f"{ltc}__{ltp}"
            print(rep_mod)
            G_ttp = create_graph(txt_clus=ltc, txt_clus_pip=ltp, ts_close=None, ts_vol=None)

            rp, tp, nb, rt, pt, va = bruteForce(G_ttp, rep_mod)
            lrpr.extend(rp)
            ltpp.extend(tp)
            lnbb.extend(nb)
            lrtt.extend(rt)
            lptt.extend(pt)
            lval.extend(va)

    file = f"02_1_{rep_mod}"
    saveGraph(file, lrpr, ltpp, lnbb, lrtt, lptt, lval)
    
    # PIP - C: News Node, Text Cluster, Closing PIP
    ltpp, lnbb, lrtt, lval, lrpr, lptt = [], [], [], [], [], []
    for ltc in tx_clus:
        for lcp in ts_pip_close:
            rep_mod = f"{ltc}__{lcp}"
            print(rep_mod)

            G_tcp = create_graph(txt_clus=ltc, txt_clus_pip=None, ts_close=lcp, ts_vol=None)

            rp, tp, nb, rt, pt, va = bruteForce(G_tcp, rep_mod)
            lrpr.extend(rp)
            ltpp.extend(tp)
            lnbb.extend(nb)
            lrtt.extend(rt)
            lptt.extend(pt)
            lval.extend(va)

    file = f"02_2_{rep_mod}"
    saveGraph(file, lrpr, ltpp, lnbb, lrtt, lptt, lval)

    # PIP - T / C: News Node, Text Cluster, Text Cluster PIP, Closing Cluster PIP
    ltpp, lnbb, lrtt, lval, lrpr, lptt = [], [], [], [], [], []
    for ltc in tx_clus:
        for ltp in tx_pip_clus:
            for lcp in ts_pip_close:
                rep_mod = f"{ltc}_{ltp}_{lcp}"
                print(rep_mod)

                G_tptc = create_graph(txt_clus=ltc, txt_clus_pip=ltp, ts_close=lcp, ts_vol=None)

                rp, tp, nb, rt, pt, va = bruteForce(G_tptc, rep_mod)
                lrpr.extend(rp)
                ltpp.extend(tp)
                lnbb.extend(nb)
                lrtt.extend(rt)
                lptt.extend(pt)
                lval.extend(va)

    file = f"02_3_{rep_mod}"
    saveGraph(file, lrpr, ltpp, lnbb, lrtt, lptt, lval)
    
    # PIP - V: News Node, Text Cluster, Volume Cluster PIP
    ltpp, lnbb, lrtt, lval, lrpr, lptt = [], [], [], [], [], []
    for ltc in tx_clus:
        for lvp in ts_pip_volume:
            rep_mod = f"{ltc}__{lvp}"
            print(rep_mod)

            G_tvp = create_graph(txt_clus=ltc, txt_clus_pip=None, ts_close=None, ts_vol=lvp)

            rp, tp, nb, rt, pt, va = bruteForce(G_tvp, rep_mod)
            lrpr.extend(rp)
            ltpp.extend(tp)
            lnbb.extend(nb)
            lrtt.extend(rt)
            lptt.extend(pt)
            lval.extend(va)

    file = f"02_4_{rep_mod}"
    saveGraph(file, lrpr, ltpp, lnbb, lrtt, lptt, lval)
    
    # PIP - T/V: News Node, Text Cluster, Text Cluster PIP, Volume Cluster PIP
    ltpp, lnbb, lrtt, lval, lrpr, lptt = [], [], [], [], [], []
    for ltc in tx_clus:
        for ltp in tx_pip_clus:
            for lvp in ts_pip_volume:
                rep_mod = f"{ltc}_{ltp}_{lvp}"
                print(rep_mod)

                G_tptv = create_graph(txt_clus=ltc, txt_clus_pip=ltp, ts_close=None, ts_vol=lvp)

                rp, tp, nb, rt, pt, va = bruteForce(G_tptv, rep_mod)
                lrpr.extend(rp)
                ltpp.extend(tp)
                lnbb.extend(nb)
                lrtt.extend(rt)
                lptt.extend(pt)
                lval.extend(va)

    file = f"02_5_{rep_mod}"
    saveGraph(file, lrpr, ltpp, lnbb, lrtt, lptt, lval)

    # PIP - T/C/V : News Node, Text Cluster, Text Cluster PIP, Closing Cluster PIP, Volume Cluster PIP, 
    ltpp, lnbb, lrtt, lval, lrpr, lptt = [], [], [], [], [], []
    for ltc in tx_clus:
        for ltp in tx_pip_clus:
            for lcp in ts_pip_close:
                for lvp in ts_pip_volume:
                    
                    rep_mod = f"{ltc}_{ltp}_{lcp}_{lvp}"
                    print(rep_mod)

                    G_tptcv = create_graph(txt_clus=ltc, txt_clus_pip=ltp, ts_close=lcp, ts_vol=lvp)

                    rp, tp, nb, rt, pt, va = bruteForce(G_tptcv, rep_mod)
                    lrpr.extend(rp)
                    ltpp.extend(tp)
                    lnbb.extend(nb)
                    lrtt.extend(rt)
                    lptt.extend(pt)
                    lval.extend(va)

    file = f"02_6_{rep_mod}"
    saveGraph(file, lrpr, ltpp, lnbb, lrtt, lptt, lval)

# Graph3: Top Label (TL) with time series and Text
# Paper: Table 3 - TL

def graphs3():

    tx_clus =       ['Paraph_MiniLM_clus_8', 'Paraph_MiniLM_clus_16']
    tx_top_clus =   ['top_text_clus_8', 'top_text_clus_16']
    ts_top_close =  ['top_ts_Close_5_clus_8', 'top_ts_Close_5_clus_16']
    ts_top_volume = ['top_ts_Volume_5_clus_8', 'top_ts_Volume_5_clus_16']


    # TL - T: News Node, Text Cluster, Text TL Cluster
    ltpp, lnbb, lrtt, lval, lrpr, lptt = [], [], [], [], [], []
    for ltc in tx_clus:
        for ltp in tx_top_clus:

            rep_mod = f"{ltc}__{ltp}"
            print(rep_mod)
            G_ttp = create_graph(txt_clus=ltc, txt_clus_pip=ltp, ts_close=None, ts_vol=None)

            rp, tp, nb, rt, pt, va = bruteForce(G_ttp, rep_mod)
            lrpr.extend(rp)
            ltpp.extend(tp)
            lnbb.extend(nb)
            lrtt.extend(rt)
            lptt.extend(pt)
            lval.extend(va)

    file = f"03_1_{rep_mod}"
    saveGraph(file, lrpr, ltpp, lnbb, lrtt, lptt, lval)
    
    # TL - C: News Node, Text Cluster, Closing TL Cluster
    ltpp, lnbb, lrtt, lval, lrpr, lptt = [], [], [], [], [], []
    for ltc in tx_clus:
        for lcp in ts_top_close:

            rep_mod = f"{ltc}__{lcp}"
            print(rep_mod)

            G_tcp = create_graph(txt_clus=ltc, txt_clus_pip=None, ts_close=lcp, ts_vol=None)

            rp, tp, nb, rt, pt, va = bruteForce(G_tcp, rep_mod)
            lrpr.extend(rp)
            ltpp.extend(tp)
            lnbb.extend(nb)
            lrtt.extend(rt)
            lptt.extend(pt)
            lval.extend(va)

    file = f"03_2_{rep_mod}"
    saveGraph(file, lrpr, ltpp, lnbb, lrtt, lptt, lval)

    # TL - T/C: News Node, Text Cluster, Text TL Cluster, Closing TL Cluster
    ltpp, lnbb, lrtt, lval, lrpr, lptt = [], [], [], [], [], []
    for ltc in tx_clus:
        for ltp in tx_top_clus:
            for lcp in ts_top_close:
                rep_mod = f"{ltc}__{ltp}__{lcp}"
                print(rep_mod)

                G_tptc = create_graph(txt_clus=ltc, txt_clus_pip=ltp, ts_close=lcp, ts_vol=None)

                rp, tp, nb, rt, pt, va = bruteForce(G_tptc, rep_mod)
                lrpr.extend(rp)
                ltpp.extend(tp)
                lnbb.extend(nb)
                lrtt.extend(rt)
                lptt.extend(pt)
                lval.extend(va)

    file = f"03_3_{rep_mod}"
    saveGraph(file, lrpr, ltpp, lnbb, lrtt, lptt, lval)

    # TL - V: News Node, Text Cluster, Volume TL Cluster
    ltpp, lnbb, lrtt, lval, lrpr, lptt = [], [], [], [], [], []
    for ltc in tx_clus:
        for lvp in ts_top_volume:
            rep_mod = f"{ltc}__{lvp}"
            print(rep_mod)

            G_tvp = create_graph(txt_clus=ltc, txt_clus_pip=None, ts_close=None, ts_vol=lvp)

            rp, tp, nb, rt, pt, va = bruteForce(G_tvp, rep_mod)
            lrpr.extend(rp)
            ltpp.extend(tp)
            lnbb.extend(nb)
            lrtt.extend(rt)
            lptt.extend(pt)
            lval.extend(va)

    file = f"03_4_{rep_mod}"
    saveGraph(file, lrpr, ltpp, lnbb, lrtt, lptt, lval)

    # TL - T/V: News Node, Text Cluster, Text TL Cluster, Volume TL Cluster
    ltpp, lnbb, lrtt, lval, lrpr, lptt = [], [], [], [], [], []
    for ltc in tx_clus:
        for ltp in tx_top_clus:
            for lvp in ts_top_volume:
                rep_mod = f"{ltc}__{ltp}__{lvp}"
                print(rep_mod)

                G_tptv = create_graph(txt_clus=ltc, txt_clus_pip=ltp, ts_close=None, ts_vol=lvp)

                rp, tp, nb, rt, pt, va = bruteForce(G_tptv, rep_mod)
                lrpr.extend(rp)
                ltpp.extend(tp)
                lnbb.extend(nb)
                lrtt.extend(rt)
                lptt.extend(pt)
                lval.extend(va)

    file = f"03_5_{rep_mod}"
    saveGraph(file, lrpr, ltpp, lnbb, lrtt, lptt, lval)

    # TL - T/C/V: News Node, Text Cluster, Text TL Cluster, Closing TL Cluster, Volume TL Cluster
    ltpp, lnbb, lrtt, lval, lrpr, lptt = [], [], [], [], [], []
    for ltc in tx_clus:
        for ltp in tx_top_clus:
            for lcp in ts_top_close:
                for lvp in ts_top_volume:
                    rep_mod = f"{ltc}_{ltp}_{lcp}_{lvp}"
                    print(rep_mod)

                    G_tptcv = create_graph(txt_clus=ltc, txt_clus_pip=ltp, ts_close=lcp, ts_vol=lvp)

                    rp, tp, nb, rt, pt, va = bruteForce(G_tptcv, rep_mod)
                    lrpr.extend(rp)
                    ltpp.extend(tp)
                    lnbb.extend(nb)
                    lrtt.extend(rt)
                    lptt.extend(pt)
                    lval.extend(va)

    file = f"03_6_{rep_mod}"
    saveGraph(file, lrpr, ltpp, lnbb, lrtt, lptt, lval)


graphs1() # Daily
graphs2() # PIP
graphs3() # TL

