import requests as req
import numpy as np
import demjson as js

fileName = "sn.list"

sn,ol,materialCode = np.loadtxt(fileName,skiprows=3,dtype=str,unpack=True)

# json = {"sn":"3Q142N000027","materialCode":"9010K9D40005","remark":"电池盖翘起嫌疑品"}

header = {
    "cookie": "xmuuid=XMGUEST-88D45E70-4907-11EB-81F1-6516504BF02F; mstuid=1609158227628_1145; userId=1; OUTFOX_SEARCH_USER_ID_NCOO=1032008172.9277618; factory=1; XM_agreement_sure=1; versionCode=1.10.5; Hm_lvt_c3e3e8b3ea48955284516b186acf0f4e=; _aegis_cas=eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJuYmYiOjE2MzUxMzIzODUsImlhdCI6MTYzNTEzMjM4NSwiaXNzIjoiTUktSU5GT1NFQyIsImF1ZCI6ImZzLmJlLm1pLmNvbSIsImMiOjEsImV4cCI6MTYzNTM5NTE4NSwiaWQiOiJjMmIxMDA2YjAzYmVkYmM2ZjgzMjNlNWE4MzYwYjQ2NCIsInR5cCI6ImNhcyIsInN1YiI6ImxpdXlhbmc2NSIsImRldGFpbCI6Io5G3LR5fVx1MDAwZonGn1av0XnEUlx1MDAxNPH5JJ7WXHUwMDFjLmZcdTAwMDJcdTAwMGJoUtW8rVx1MDAwZdJtXHUwMDdmqciEO1aTMvp0kcWyd1x0W_0yVTuq06yj05BQiFlev1Dd89FtZrQr-dmlWNZ6oHP62Iwk_K21pU3tKk75QlGZXHUwMDA0-Ys-cjDuWN9cdTAwMDVr07qb21x1MDAxZSw8vqv1wl6Mtm1cbi7osPzKrTi-doBR4Fx1MDAwYi6Eylx1MDAxY66nyt_-TaulYpaT-O_U4Gz8dcuzcPw1yEXtjvn0fiVGXGb846BnmHpcL7tfLG8nv59HpEg7WF07v1on3r1cdTAwMGW39IKr6Vx1MDAxYtVbUV-490P8XGbLlZM73b2ZeVx1MDAxNtaiTPaSnJk3saJcXHqOlTBcdTAwMTZcdTAwMGZcdTAwMTHIYd5cdTAwMWR1I8OPx1DvXHUwMDFk28qaRM9cdTAwMDVcdTAwMDbmzrDTXHUwMDBimWVcdTAwMGZiNccg3O_8hpGMXGZ4orYlbY_ARlx1MDAwMfjWU4Ay115l5lvhgtiTsOdcdTAwMGKxfU1b9eLek8tW9-HTKv_2QN6TWoZJe88qWnyZXHUwMDAyfpLUULSSLVx1MDAxMovZhuP6ZTyAWc9uuDlUPz3rNJPxZjAxXHUwMDEw8V5vbJl6XHUwMDFjdaH7p5KacnmVXVx1MDAxZlNcdTAwMWGgXHUwMDE0XHUwMDAyJkFARmxcdTAwMWJI8F2LvWD2_J81r0JcdTAwMTUmzWVA5CCB_NnldSEzXGZvWsFcXMtDc1x1MDAwZpJBtVx1MDAwYjCihLhcdTAwMWJlcG_oNVx1MDAwZa_3dGdcdTAwMTO5TPhcdTAwMTTruFx1MDAwNZpG5W_RQdxcdFTNQay1Qlwvq72FXHUwMDAwJVZcdTAwMGVcdTAwMDQudtG-Uynt7EEr3Ng02N5cdTAwMWJcdTAwMTWPaPpcdTAwMWXPZbgsu67I5b61XHUwMDA22UMgfLCmr7s1kVvqPSExTZDpmVx1MDAwMDT9JPpcdTAwMDeAJdJnuCVyXHUwMDFjl-fx3ETsX8QoZKX7wdbEhk4yii5ZS1x1MDAxMobCLabA6Vx1MDAxN-nmXHUwMDA3XHUwMDFkaChVtj5LNttm5K1UqYhcdTAwMDBcdTAwMTFcdOcqIMRcdTAwMTU2UFwvTT9cblx1MDAxYvlcdTAwMGYyQ55CwUBUkFx1MDAxNDhcL9fPWNYzpsam6ehcdTAwMWLSh-YgRlvBgE7g1DKJ0DKbz7mNn92FwfXYtTjtrlx1MDAwM6vb-Fx1MDAwYqhotCbM6th-c1OAQ6Mx9-2IP1x1MDAwMiJ9.2Bq3B9Vba_DBrJVYsspPUVBFzOqG0o3wnYIRaVSyG2Xm09_pbkPal-djl4B2xMEbOgyy7Mij0mlCuuxet5LWOQ; language=zh-CN; CAS_TOKEN=7c9577f6-da4b-4f7d-9d60-29df8e98ba1c",
    "content-type": "application/json",
    "Host" : "fs.be.mi.com"
}

for i in range(sn.size):
    json = {"sn": sn[i], "materialCode": materialCode[i], "remark": "电池盖翘起嫌疑品"}
    res = req.post("http://5-c3.miwms.neo.b2c.srv/mes/restful/banList/material",json=json,headers=header)
    resObj = js.decode(res.text)
    print(res.text)
    if resObj.get("code") != 200:
        print("json:%s ,i:%d" %(json,i))
        break
print("done")



