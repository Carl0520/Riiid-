from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import KernelPCA

class Tags_encoder():
    def __init__(self, n):
        self.n = n
        self.vectorizer = CountVectorizer()
        self.transformer = TfidfTransformer()
        self.kpca = KernelPCA(n_components= self.n, kernel='linear')
    
    def fit_transform(self, x):
        
        x = self.vectorizer.fit_transform(x)
        x = self.transformer.fit_transform(x)
        x = self.kpca.fit_transform(x.toarray())
        x = pd.DataFrame(x, columns=['PCA_' + str(i) for i in range(self.n)])
        return x
    
    
#     def transform(self,x):
        
#         x = self.vectorizer.transform(x)
#         x = self.transformer.transform(x)
#         x = self.kpca.transform(x.toarray())
#         return pd.DataFrame(x,columns=['PCA_' + str(i) for i in range(self.n)])
        

# en = Tags_encoder(5)
# en.fit_transform(questions['tags'])
# en.transform(questions['tags'])



import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

class Tbins_qauc_encode():
    def __init__(self, n = 50):
        self.disc = KBinsDiscretizer(n_bins=n, encode='ordinal', strategy='quantile')
        
        
    def fit_transform(self, input_df):
        self.tmp = input_df[['timestamp','answered_correctly']].copy()
        self.disc.fit(self.tmp[['timestamp']])
        self.tmp['ts_bin'] = self.disc.transform(self.tmp[['timestamp']])
        self.tmp = self.tmp.drop(columns = ['timestamp'])
        self.tmp = self.tmp[self.tmp.answered_correctly != -1].groupby(['ts_bin'], as_index=False).mean()
        self.tmp.columns = ['ts_bin', 'tbins_qauc_encode']
        self.dict = self.tmp.set_index('ts_bin').to_dict()['tbins_qauc_encode']
        return self.tmp['tbins_qauc_encode']
    
    # input dataframe , columns [[timestamp]]   
    def transform(self, x):
        x = self.disc.transform(x[['timestamp']].copy())
        x = pd.DataFrame(x, columns = ['ts_bin'])
        x.iloc[:,0] = x.iloc[:,0].apply(lambda z : self.dict[z])
        return x.iloc[:,0]
        

# trans = Tbins_qauc_encode(train)
# trans.fit_transform()
# x = train[['timestamp']]
# x = trans.disc.transform(x)