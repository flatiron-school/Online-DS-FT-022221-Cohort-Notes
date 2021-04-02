from sklearn.datasets import make_regression

X, y, c = make_regression(n_features=2, n_samples=500, n_informative=1,
                       coef=True, random_state=42)

df = pd.DataFrame(np.hstack((X, y.reshape(-1, 1))), columns=['info', 'noninfo', 'target'])

A1 = df[df['noninfo'] <= 0]['target']
B1 = df[df['noninfo'] > 0]['target']

A2 = df[df['info'] <= 0]['target']
B2 = df[df['info'] > 0]['target']

A1.to_csv('a1.csv')
B1.to_csv('b1.csv')
A2.to_csv('a2.csv')
B2.to_csv('b2.csv')