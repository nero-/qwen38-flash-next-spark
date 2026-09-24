# TP2 deployment

The direct-connected `sparks-2` cluster is being brought up under
`~/builds/qwen-tp2` on both nodes. See [the current profile and operations](tp2/README.md)
and [the pinned launcher](tp2/launch.py).

The old source-based cluster instructions have been superseded. Both ranks
use the same pinned Docker image and the ordinary-CUDA b12x loader. PLE is
resident in RAM; no disk PLE setting is used. There is no SparkRing dependency.

Mac controller: `~/Agent/Builds/spark-ctl.sh`.

```bash
./spark-ctl.sh start
./spark-ctl.sh wait
./spark-ctl.sh status
./spark-ctl.sh logs
./spark-ctl.sh logs-r1
./spark-ctl.sh stop
```

Do not start the historical TP1 container at the same time. The Mac controller now manages both ranks.
