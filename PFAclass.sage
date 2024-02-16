import itertools
import numpy
import urllib,json
import pickle


#------------------UTILITY FUNCTIONS------------------------

#object: what to dump; name: string without ".pickle"
def pickletofile(object,name):
    thefile = open(name+".pickle",'wb')
    pickle.dump(object,thefile)
    thefile.close()
    return

#filename: string without ".pickle"
def unpicklefile(filename):
    thefile = open(filename+".pickle",'rb')
    theobj = pickle.load(thefile)
    thefile.close()
    return theobj

#NOTE: use dictionary NC for strings of length 2 through 10.
def nfacomplexity(sigma):
    theurl = "http://math.hawaii.edu/~bjoern/complexity-api/?string="+''.join(str(ch) for ch in sigma)
    response = urllib.request.urlopen(theurl)
    data = json.loads(response.read())
    return int(data.get('complexity'))

#returns a string that is everything in atuple concatenated
def tuptostr(atuple):
    return ''.join([str(t) for t in atuple]);

#------------------------------------------------------------------


SC = unpicklefile("stoch_complexity");
NC = unpicklefile("nfa_complexity");
SW = unpicklefile("witnesses");


#------------------------------------------------------------------


class ProbabilisticFiniteAutomaton:
    S2 = {} #set of binary strings of reasonable length
    S2[0] = Set([])
    for i in range(2,15):
        S2[i] = Set([tuptostr(it) for it in itertools.product([0,1],repeat=i)])
        S2[0] = S2[0].union(S2[i])

    # expects matrix_dict to be in usual format (recap here)
    # expects init_states and accept_state to be lists/tuples
    def __init__(self,matrix_dict,init_states,accept_states,ring=QQ):
        n = len(init_states)
        assert n = len(accept_states)
        for m in matrix_dict.keys():
            assert matrix_dict[m].dimensions() == (n,n) # they should all be square mats of same size
        # don't feel rn like allowing different rings: #self.transitions = {}
        self.transitions = matrix_dict
        #TODO: actually check matrix_dict is what it's supposed to be! And other sanity checks
        self.alphabet = list(matrix_dict.keys())
        self.initial_states = matrix(ring,init_states)
        self.accepting_states = matrix(ring,accept_states).transpose()
        self.size = n


    def transition_matrix(self,string):
        A = identity_matrix(self.size)
        for ch in string:
            assert str(ch) in self.alphabet
            A = A*self.transitions[ch]
        return A

    def acceptance_prob(self,string):
        return (self.initial_states*transition_matrix(self,string)*self.accepting_states)[0][0]

    #check if an object is an integer, either int or Sage integer
    def _isint(obj):
        return (isinstance(obj,sage.rings.integer.Integer) or isinstance(obj,int))
