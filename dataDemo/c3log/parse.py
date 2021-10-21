import numpy as np

timeStamp,elapsed,label,responseCode,responseMessage,threadName,dataType,success,failureMessage,bytes,sentBytes,grpThreads,allThreads,URL,Latency,IdleTime,Connect = np.loadtxt("C3.csv",dtype=str,delimiter=",",skiprows=1,unpack=True)
X = np.column_stack((np.float64(timeStamp),np.float64(elapsed),label,responseCode,responseMessage,threadName,dataType,success,failureMessage,bytes,sentBytes,grpThreads,allThreads,URL,Latency,IdleTime,Connect))

# np.float64(X[:,1])

rowNum = np.where(np.logical_and(np.float64(timeStamp) < 1634705885618,success != "true"))

Xt = np.delete(X,rowNum,axis=0)

C3 = Xt[Xt[:,2] == "C3-in接口测试"]
C3_timeout = C3[np.float64(C3[:,1]) > 3000]
C3_total = C3.shape[0]

NEO = Xt[Xt[:,2] == "NEO-in接口测试"]
NEO_timeout = NEO[np.float64(NEO[:,1]) > 3000]
NEO_total = NEO.shape[0]

C3_percent = C3_timeout.shape[0] * 100 / C3_total
NEO_percent = NEO_timeout.shape[0] * 100 / NEO_total
print("\nC3: %d/%d (%.2f%%)" %
      (C3_timeout.shape[0], C3_total, C3_percent))
print("\nNEO: %d/%d (%.2f%%)" %
      (NEO_timeout.shape[0], NEO_total, NEO_percent))