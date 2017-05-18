from myria import *
import numpy
import json
from myria.connection import MyriaConnection
from myria.relation import MyriaRelation
from myria.udf import MyriaPythonFunction
from raco.types import STRING_TYPE, BOOLEAN_TYPE, LONG_TYPE, BLOB_TYPE

connection = MyriaConnection(rest_url='http://localhost:8753',execution_url='http://localhost:8080')
outType= "BLOB_TYPE"

###queries
# data ingested in relation raw

q = MyriaQuery.submit("""
T1 = scan(raw);
processedCCDs = [from T1 emit T1.ccd, T1.visit, processCCD(T1.img) as calexp];

STORE(processedCCDs, calexps);

patchlist = [from T1
            where visit=0288935
            rekeywpatchid(T1.img) as patchid];
STORE(patchlist, patchlist);
""", connection=connection)
q.status


q = MyriaQuery.submit("""
uda createP(visit, patchid, img) {
   [EMPTY_BLOB as coadd];
   [createpatch(visit ,patchid, img )];
   [coadd];
};

uda mergeC(patchid, img){
[EMPTY_BLOB as merge];
[mergeCoadd(patchid, img)];
[merge];
};

T1 = scan(calexps);
T2 = scan(patchlist);
patches = [from T1, T2
        where T1.ccd = T2.ccd,
        emit T1.visit, T2.patchid, createP(T1.visit, T1.patchid, T1.img) as coadd];
mergedCoadds = [from patches as t
     emit t.patchid, mergeC(t.patchid, t.img) as mergedCoadds];
detectSources = [from mergedCoadds as t
                emit t.visit, t.patchid, detectSources(t.mergedCoadds)];
STORE(detectSources, sources);
""", connection=connection)
q.status
