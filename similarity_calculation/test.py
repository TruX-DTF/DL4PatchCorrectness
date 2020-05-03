from sklearn.metrics.pairwise import cosine_similarity
from bert_serving.client import BertClient
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import re
from numpy import dot
from numpy.linalg import norm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


bc = BertClient(check_length=False)
p = r"([^\w_])"

a = ['AbstractIntegrator', '.', 'java', 'final', 'double', '[', ']', 'eventY', '=', 'interpolator', '.', 'getInterpolatedState', '.', 'clone', '(', ')', ';', '/', '/', 'advance', 'all', 'event', 'states', 'to', 'current', 'time', 'for', '(', 'final', 'EventState', 'state', ':', 'eventsStates', ')', '{', 'state', '.', 'stepAccepted', '(', 'eventT', ',', 'eventY', ')', ';', 'isLastStep', '=', 'isLastStep', '|', '|', 'state', '.', 'stop', '(', ')', ';', '}', '/', '/', 'handle', 'the', 'first', 'part', 'of', 'the', 'step', ',', 'up', 'to', 'the', 'event', 'for', '(', 'final', 'StepHandler', 'handler', ':', 'stepHandlers', ')', '{', 'if', '(', 'isLastStep', ')', '{', '/', '/', 'the', 'event', 'asked', 'to', 'stop', 'integration', 'System', '.', 'arraycopy', '(', 'eventY', ',', '0', ',', 'y', ',', '0', ',', 'y', '.', 'length', ')', ';', 'return', 'eventT', ';', '}', 'boolean', 'needReset', '=', 'false', ';', 'for', '(', 'final', 'EventState', 'state', ':', 'eventsStates', ')', '{', 'needReset', '=', 'needReset', '|', '|', 'state', '.', 'reset', '(', 'eventT', ',', 'eventY', ')', ';', '}', 'if', '(', 'needReset', ')', '{', '/', '/', 'some', 'event', 'handler', 'has', 'triggered', 'changes', 'that', '/', '/', 'invalidate', 'the', 'derivatives', ',', 'we', 'need', 'to', 'recompute', 'them', 'System', '.', 'arraycopy', '(', 'eventY', ',', '0', ',', 'y', ',', '0', ',', 'y', '.', 'length', ')', ';', 'computeDerivatives', '(', 'eventT', ',', 'y', ',', 'yDot', ')', ';', 'resetOccurred', '=', 'true', ';', 'return', 'eventT', ';', '}']
b = ['a', '/', 'src', '/', 'main', '/', 'java', '/', 'org', '/', 'apache', '/', 'commons', '/', 'math3', '/', 'ode', '/', 'AbstractIntegrator', '.', 'java', 'b', '/', 'src', '/', 'main', '/', 'java', '/', 'org', '/', 'apache', '/', 'commons', '/', 'math3', '/', 'ode', '/', 'AbstractIntegrator', '.', 'java', 'for', '(', 'final', 'StepHandler', 'handler', ':', 'stepHandlers', ')', '{', 'if', '(', 'org', '.', 'apache', '.', 'commons', '.', 'math3', '.', 'ode', '.', 'AbstractIntegrator', '.', 'this', '.', 'stepHandlers', '.', 'size', '(', ')', '=', '=', 'orderingSign', ')', '{', 'handler', '.', 'handleStep', '(', 'interpolator', ',', 'isLastStep', ')', ';', '}', '}']


a= ["my", "sister", "is", "beautiful"]
# a=["computer", "science", "is", "difficult"]
# a= ["his", "spouse", "is", "lovely"]
b= ["mathematics", "are", "complex", "science"]
# a = [w for w in a if re.search(r'^[a-zA-Z]', w)]
# b = [w for w in b if re.search(r'^[a-zA-Z]', w)]

model = Doc2Vec.load('../data/doc_frag.model')
bug_vec = model.infer_vector(a,alpha=0.025,steps=300).reshape(1,-1)
patched_vec = model.infer_vector(b,alpha=0.025,steps=300).reshape(1,-1)

# bug_vec = bc.encode([a],is_tokenized=True)
# patched_vec = bc.encode([b],is_tokenized=True)

# scaler = StandardScaler()
# scaler.fit_transform(bug_vec)
# scaler.fit_transform(patched_vec)
s1 = dot(bug_vec[0], patched_vec[0])/(norm(bug_vec[0])*norm(patched_vec[0]))
print(s1)

s2 = cosine_similarity(bug_vec, patched_vec)
print(s2)