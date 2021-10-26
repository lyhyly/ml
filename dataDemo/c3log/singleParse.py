import numpy as np

fileName = "C4NoExt.csv"
timeOut = 2000

timeStamp,elapsed,label,responseCode,responseMessage,threadName,dataType,success,failureMessage,bytes,sentBytes,grpThreads,allThreads,URL,Latency,IdleTime,Connect = np.loadtxt(fileName,dtype=str,delimiter=",",skiprows=1,unpack=True)
X = np.column_stack((np.float64(timeStamp),np.float64(elapsed),label,responseCode,responseMessage,threadName,dataType,success,failureMessage,bytes,sentBytes,grpThreads,allThreads,URL,Latency,IdleTime,Connect))

C3_timeout = X[np.float64(X[:,1]) > timeOut]
C3_total = X.shape[0]

C3_percent = C3_timeout.shape[0] * 100 / C3_total
print("\n%s 大于 %d ms: %d/%d (%.2f%%)" %
      (fileName, timeOut, C3_timeout.shape[0], C3_total, C3_percent))