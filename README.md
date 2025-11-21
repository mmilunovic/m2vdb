vector db built by someone with no idea how to build a vector db


### SIFT1M (1,000,000 vectors, 128D)

| Index                    | Build (ms) | Index (MB) | Bytes/Vec | QPS | p50 (ms) | p99 (ms) | Recall@k |
| ------------------------ | ---------: | ---------: | --------: | --: | -------: | -------: | -------: |
| PyBruteForce-euclidean   |        746 |      649.0 |       681 |   7 |   140.47 |   204.02 |    1.000 |
| RustBruteForce-euclidean |        698 |        0.0 |         0 |  34 |    28.74 |    40.31 |    1.000 |
| PQ(m=8,k=256)-euclidean  |    425167* |      191.5 |       201 |  26 |    38.47 |    51.56 |    0.332 |


---

### FASTTEXT (1,000,000 vectors, 300D)

| Index                 | Build (ms) | Index (MB) | Bytes/Vec | QPS | p50 (ms) | p99 (ms) | Recall@k |
| --------------------- | ---------: | ---------: | --------: | --: | -------: | -------: | -------: |
| PyBruteForce-cosine   |        707 |     1305.1 |      1369 |   5 |   183.27 |   310.86 |    1.000 |
| RustBruteForce-cosine |       1074 |        0.0 |         0 |   9 |   115.97 |   128.29 |    1.000 |
| PQ(m=10,k=256)-cosine |    559221* |      199.5 |       209 |  22 |    45.67 |    56.49 |    0.283 |
