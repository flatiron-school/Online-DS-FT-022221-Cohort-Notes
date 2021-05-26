# !pip install -U fsds
from fsds.imports import *
import plotly.express as px
from ipywidgets import interact
## Lets functionzie the above process
## Lets make a preprocessing pipeline with SimpleImputer and StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics

def get_kmeans_results(X_pca,k,random_state=42):
    
    ## Fit model and get preds
    model = KMeans(k,random_state=random_state)
    model.fit(X_pca) 
    k_preds = pd.Series(model.predict(X_pca))
    
    ## Save metrics
    results = {}
    results["k"] = k
    results['Calinski-Harahasz'] = metrics.calinski_harabasz_score(X_pca,k_preds)
    results['Inertia'] = model.inertia_
    return results


def get_clusters_plot_df(X_pca,k,random_state=42,col_name ='cluster'):
    ## Fit model and get preds
    model = KMeans(k,random_state=random_state)
    model.fit(X_pca) 
    k_preds = pd.Series(model.predict(X_pca))
    
    plot_df = X_pca.copy()
    plot_df[col_name] = k_preds
    return plot_df


def plotly_pca(plot_df,x="PC1",y="PC2",z="PC3",symbol=None,color=None):
    
    ## Plotly Figure
    fig  = px.scatter_3d(plot_df, x=x, y=y, z=z, symbol=symbol,color=color,
                         template='plotly_dark')
    
    ## Update marker size
    fig.update_traces(marker={'size':2})
 
    ## Fix the colorbar size
    layout = fig.layout
    layout['coloraxis']['colorbar'].update(lenmode="pixels", len=100) 
    fig.update_layout(layout)   
    
    ## Set config and show 
    config = dict({'scrollZoom': False})
    fig.show(config=config)

