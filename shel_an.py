import pandas as pd
import re
from sklearn import preprocessing
df=pd.read_csv("trainan.csv")


df["Name"].fillna(value=0, inplace=True)
for i in df.index.values:
	if df["Name"][i]!=0:
		df["Name"][i]=1

df.drop("AnimalID", axis=1)
df["Namec"]=df["Name"].astype("category")
df["Namec"]=df["Namec"].cat.rename_categories(range(1,3))
df=df.drop("Name", axis=1)

df["AnimalType"].fillna(value=0, inplace=True)
df["AnimalTypec"]=df["AnimalType"].astype("category")
df["AnimalTypec"]=df["AnimalTypec"].cat.rename_categories(range(1,3))
df=df.drop("AnimalType", axis=1)

df["SexuponOutcome"].fillna(value=0, inplace=True)
df["SexuponOutcomec"]=df["SexuponOutcome"].astype("category")
df["SexuponOutcomec"]=df["SexuponOutcomec"].cat.rename_categories(range(1,7))
df=df.drop("SexuponOutcome", axis=1)




df["Breed"].fillna(value=0, inplace=True)
df["Breedc"]=df["Breed"].astype("category")
df["Breedc"]=df["Breedc"].cat.rename_categories(range(1,1679))
df=df.drop("Breed", axis=1)

df["Color"].fillna(value=0, inplace=True)
df["Colorc"]=df["Color"].astype("category")
df["Colorc"]=df["Colorc"].cat.rename_categories(range(1,412))
df=df.drop("Color", axis=1)


df["OutcomeSubtype"].fillna(value=0, inplace=True)
df["OutcomeSubtypec"]=df["OutcomeSubtype"].astype("category")
df["OutcomeSubtypec"]=df["OutcomeSubtypec"].cat.rename_categories(range(1,18))
df=df.drop("OutcomeSubtype", axis=1)




p=[]
for i in df.index.values:
	p=p+[df["DateTime"][i].rsplit('-')[0]]

df["Year"]=pd.Series(p).astype("category")
df["Year"]=df["Year"].cat.rename_categories(range(1,5))


df=df.drop("DateTime", axis=1)

df["AgeuponOutcome"].fillna(value='1 year', inplace=True)
q=[]
b=[]
for j in df.index.values:
	a=str(df["AgeuponOutcome"][j])
	b=[int(s) for s in a.split() if s.isdigit()]

	if a.split()[1] in ('year','years'):
		b[0]=b[0]*52
	if a.split()[1] in ('month','months'):
		b[0]=b[0]*4
	
	q=q+[b[0]]


_Q_=pd.Series(q).astype("object")

df=df.drop("AgeuponOutcome", axis=1)

_Q_ = preprocessing.scale(_Q_)

df["Age"]=_Q_
df_train=df[df.OutcomeType.notnull()]
df_test=df[df.OutcomeType.isnull()]

y_train=df_train["OutcomeType"].astype("category")
df=df.drop("OutcomeType", axis=1)


from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder()
Xtrsp=pd.DataFrame(enc.fit_transform(df_train[["Namec","AnimalTypec","SexuponOutcomec","Breedc","Colorc","OutcomeSubtypec","Year"]]).toarray())
Xtstsp=pd.DataFrame(enc.transform(df_test[["Namec","AnimalTypec","SexuponOutcomec","Breedc","Colorc","OutcomeSubtypec","Year"]]).toarray())
Xtrf=df_train["Age"]
Xtstf=df_test["Age"]


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
sel = SelectKBest(chi2, k=600)
Xtrsp=pd.DataFrame(sel.fit_transform(Xtrsp,y_train))
Xtstsp=pd.DataFrame(sel.transform(Xtstsp))



Xtrf.index=Xtrsp.index
X_train=pd.concat([Xtrsp,Xtrf], axis=1)
Xtstf.index=Xtstsp.index
X_test=pd.concat([Xtstsp,Xtstf], axis=1)







from sklearn.linear_model import LogisticRegression 
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import RandomForestClassifier 



#from sklearn.metrics import log_loss
#from sklearn.cross_validation import train_test_split
#from sklearn.grid_search import GridSearchCV



#Xtr,Xts,ytr,yts=train_test_split(X_train,y_train, test_size=0.05)
#model=RandomForestClassifier(n_estimators=600)
#prmtr={'C': [0.01,0.1,1,10,100,1000], 'solver': ['lbfgs','newton-cg'], 'tol': [0.000001,0.00001,0.0001,0.001,0.01] }
#model=GridSearchCV(LogisticRegression(multi_class='multinomial'), prmtr) 
#model.fit(X_train, y_train)
#print model
#ypr=model.predict_proba(Xts)
#print ypr
#print log_loss(yts,ypr)

model=LogisticRegression(solver='lbfgs', multi_class='multinomial')
model.fit(X_train,y_train)
Y_pred=model.predict_proba(X_test)
#print Y_pred

out=pd.DataFrame()
out['ID']=pd.Series(range(1,11457))
out['Adoption']=Y_pred[:,0]
out['Died']=Y_pred[:,1]
out['Euthanasia']=Y_pred[:,2]
out['Return_to_owner']=Y_pred[:,3]
out['Transfer']=Y_pred[:,4]
out.to_csv('outan.csv', index=False)

