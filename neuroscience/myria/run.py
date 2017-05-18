from myria import *
import numpy
import json
from myria.connection import MyriaConnection
from myria.relation import MyriaRelation
from myria.udf import MyriaPythonFunction
from raco.types import STRING_TYPE, BOOLEAN_TYPE, LONG_TYPE, BLOB_TYPE

connection = MyriaConnection(rest_url='http://localhost:8753',execution_url='http://localhost:8080')
py = myria.udf.FunctionTypes.PYTHON
outType= "BLOB_TYPE"


###queries
# json query for building and broadcasting mask

q = MyriaQuery.submit("""
uda dti(subjid,imgid, img) {
   [EMPTY_BLOB as tm];
   [fit_model(imgid ,flatmapid, img )];
   [tm];
};

T1 = scan(raw);
T2 = scan(bmask);
imgs = [from T1, T2
        where T1.subjid = T2.subjid,
        emit T1.subjid, T1.imgid, T1.img, T2.mask];
t = [from t
     emit t.imgid, t.subjid, denoise(t.img, t.mask)];
results = [from t emit t.subjid, t.flatmapid, dti(t.imgid, t.flatmapid, t.img) as vox];
STORE(results, results);
""", connection=connection)
q.status
