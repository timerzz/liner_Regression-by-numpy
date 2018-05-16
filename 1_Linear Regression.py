import numpy as np

def liner_Regression(data_x,data_y,learningRate,Loopnum):
    Weight=np.ones(shape=(1,data_x.shape[1]))
    baise=np.array([[1]])

    for num in range(Loopnum):
        WXPlusB = np.dot(data_x, Weight.T) + baise

        loss=np.dot((data_y-WXPlusB).T,data_y-WXPlusB)/data_y.shape[0]
        w_gradient = -(2/data_x.shape[0])*np.dot((data_y-WXPlusB).T,data_x)
        baise_gradient = -2*np.dot((data_y-WXPlusB).T,np.ones(shape=[data_x.shape[0],1]))/data_x.shape[0]

        Weight=Weight-learningRate*w_gradient
        baise=baise-learningRate*baise_gradient
        if num%50==0:
            print(loss)
    return (Weight,baise)

if __name__== "__main__":
    data_x=np.random.normal(0,10,[5,3])
    Weights=np.array([[3,4,6]])
    noise=np.random.normal(0,0.05,[5,1])
    data_y=np.dot(data_x,Weights.T)+5+noise

    res=liner_Regression(data_x,data_y,learningRate=0.003,Loopnum=10000)
    print(res[0],res[1])


